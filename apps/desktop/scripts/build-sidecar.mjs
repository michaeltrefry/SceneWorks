#!/usr/bin/env node
// Builds the sceneworks-rust-api binary (with the embedded web UI) and stages it
// as a Tauri sidecar named for the host target triple. Wired as the
// tauri.conf.json `beforeBuildCommand` so `tauri build` is self-contained.
import { execFileSync, execSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  chmodSync,
  writeFileSync,
  existsSync,
  readdirSync,
  readFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const repoRoot = resolve(desktopDir, "..", ".."); // repository root
const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";

function run(cmd, args, extraEnv = {}) {
  const cmdStr = `${cmd} ${args.join(" ")}`;
  console.log(`> ${cmdStr}`);
  execSync(cmdStr, {
    stdio: "inherit",
    cwd: repoRoot,
    env: { ...process.env, ...extraEnv },
  });
}

// Host target triple, e.g. aarch64-apple-darwin or x86_64-pc-windows-msvc.
const triple = execFileSync("rustc", ["-vV"], { encoding: "utf8" }).match(
  /host:\s*(\S+)/,
)?.[1];
if (!triple) {
  console.error("build-sidecar: could not determine host target triple");
  process.exit(1);
}
const exe = triple.includes("windows") ? ".exe" : "";

// Sign a nested Mach-O for notarization. Tauri signs the .app and the externalBin
// sidecars (sceneworks-api, uv), but NOT the extra binaries we drop into the
// bundle's Resources/ (the static ffmpeg, the onnxruntime dylib). Apple's notary
// service rejects any nested binary that lacks a Developer ID signature, a secure
// timestamp, or (for executables) hardened runtime — so sign them inside-out here,
// before Tauri seals the bundle. No-op unless an identity is configured (the same
// identity Tauri uses for the .app), so plain dev builds are unchanged. The
// identity comes from the APPLE_SIGNING_IDENTITY env var (CI/headless) OR, as a
// fallback, bundle.macOS.signingIdentity in tauri.conf.json — because
// beforeBuildCommand runs before Tauri signs and does NOT inherit the conf value
// as an env var, so a local `tauri build` that sets the identity only in the conf
// would otherwise skip pre-signing and fail notarization on the nested binaries.
// execFileSync (not the shell `run` above) so the identity's spaces/parens in
// "Developer ID Application: Name (TEAMID)" don't need quoting.
function readConfSigningIdentity() {
  try {
    const conf = JSON.parse(
      readFileSync(join(desktopDir, "tauri.conf.json"), "utf8"),
    );
    return conf?.bundle?.macOS?.signingIdentity || "";
  } catch {
    return "";
  }
}
const signingIdentity =
  process.env.APPLE_SIGNING_IDENTITY || readConfSigningIdentity();
function codesignForNotarization(file) {
  if (!signingIdentity || !triple.includes("apple-darwin")) return;
  console.log(`> codesign --force --options runtime --timestamp "${file}"`);
  execFileSync(
    "codesign",
    ["--force", "--sign", signingIdentity, "--options", "runtime", "--timestamp", file],
    { stdio: "inherit" },
  );
  console.log(`build-sidecar: codesigned ${file} for notarization`);
}

// Build the web bundle + API binary with the embedded UI (single source of
// truth for the embedded build). Empty VITE_API_BASE_URL makes the embedded UI
// talk to its own origin (the API serves it), so it works on the dynamic port
// with no CORS.
//
// Opt-in candle (Windows/CUDA) backend (sc-5559): set SCENEWORKS_DESKTOP_CANDLE=1
// to compile the sidecar with `--features embed-web,backend-candle` so the
// desktop's Rust worker runs candle for its eligible surface. CUDA-only (product
// decision: no CPU/AMD); requires the build box to have the CUDA Toolkit 12.9 +
// VS2022 BuildTools MSVC 14.44 toolset on PATH (run from its vcvars64 — CUDA 12.9
// rejects VS2026's 14.51). Default OFF keeps the plain build intact for boxes
// without the CUDA toolkit (windows-latest CI, macOS — candle is Windows-gated).
const candle =
  process.platform === "win32" && process.env.SCENEWORKS_DESKTOP_CANDLE === "1";
if (candle) {
  // CUDA_COMPUTE_CAP=80 builds `compute_80` PTX the driver JITs forward to sm_120
  // (Blackwell) — one binary covers Ampere→Blackwell (per sc-3676). Honor an
  // explicit override (e.g. a single-arch dev build) if the env already set it.
  const candleEnv = { VITE_API_BASE_URL: "" };
  if (!process.env.CUDA_COMPUTE_CAP) {
    candleEnv.CUDA_COMPUTE_CAP = "80";
  }
  console.log(
    `build-sidecar: candle backend ON (CUDA_COMPUTE_CAP=${process.env.CUDA_COMPUTE_CAP ?? "80"})`,
  );
  run(npmCmd, ["run", "api:build:embedded:candle"], candleEnv);
} else {
  run(npmCmd, ["run", "api:build:embedded"], { VITE_API_BASE_URL: "" });
}

const src = join(repoRoot, "target", "release", `sceneworks-rust-api${exe}`);
const outDir = join(desktopDir, "binaries");
mkdirSync(outDir, { recursive: true });
const dest = join(outDir, `sceneworks-api-${triple}${exe}`);
copyFileSync(src, dest);
if (!exe) {
  chmodSync(dest, 0o755);
}
console.log(`build-sidecar: staged ${dest}`);

// The Rust DWPose detector (sc-3487) dlopens onnxruntime at runtime via
// ORT_DYLIB_PATH (set in setup.rs), bundled as a Tauri resource
// (tauri.conf.json `resources` -> `onnxruntime/**/*`) so a packaged, Python-free
// Mac can still detect poses. The `onnxruntime` dir must exist on EVERY platform —
// Tauri errors on a resource glob that matches no files. Only macOS stages the
// real CoreML dylib (pose detection on the Rust worker is macOS-only); other
// platforms ship a placeholder so the glob matches and the build succeeds.
const ortDir = join(desktopDir, "onnxruntime");
mkdirSync(ortDir, { recursive: true });
if (triple.includes("apple-darwin")) {
  const dylibDest = join(ortDir, "libonnxruntime.dylib");
  const py = process.env.PYTHON || "python3";
  run(py, ["apps/desktop/scripts/stage-onnxruntime.py", dylibDest]);
  console.log(`build-sidecar: staged ${dylibDest}`);
  codesignForNotarization(dylibDest);
  // onnxruntime is MIT — ship its license text + notice next to the dylib so the
  // MIT "include the copyright + permission notice" requirement is satisfied at
  // the distribution level (mirrors the ffmpeg GPLv3 §6 staging below). Source of
  // truth: apps/desktop/licenses/onnxruntime/ (tracked); also surfaced in the
  // in-app About → Licenses screen (sc-3778).
  for (const name of ["LICENSE", "NOTICE.txt"]) {
    copyFileSync(
      join(desktopDir, "licenses", "onnxruntime", name),
      join(ortDir, name),
    );
  }
  console.log(`build-sidecar: staged onnxruntime MIT license + notice`);
} else {
  writeFileSync(
    join(ortDir, "README.txt"),
    "onnxruntime CoreML dylib is bundled on macOS only (Rust DWPose detector, sc-3487).\n",
  );
  console.log(`build-sidecar: ${ortDir} placeholder (non-macOS, no DWPose dylib)`);
}

// The Rust worker shells out to ffmpeg (frame sampling, frame extract, timeline
// export, video-gen audio mux) via SCENEWORKS_FFMPEG (set in setup.rs). The
// desktop ships no system ffmpeg, and epic 3482 strips the Python venv it used to
// borrow imageio-ffmpeg from — so we bundle a static ffmpeg as a Tauri resource
// (tauri.conf.json `resources` -> `ffmpeg/**/*`). Like the onnxruntime dir above,
// the `ffmpeg` dir must exist on EVERY platform (Tauri errors on an empty glob);
// only macOS stages the real binary (Windows/Linux desktop + server/Docker use
// PATH ffmpeg), other platforms ship a placeholder. GPLv3 — see
// docs/sc-3767/ffmpeg-bundling.md.
const ffmpegDir = join(desktopDir, "ffmpeg");
mkdirSync(ffmpegDir, { recursive: true });
if (triple.includes("apple-darwin")) {
  const ffmpegDest = join(ffmpegDir, "ffmpeg");
  const py = process.env.PYTHON || "python3";
  run(py, ["apps/desktop/scripts/stage-ffmpeg.py", ffmpegDest]);
  console.log(`build-sidecar: staged ${ffmpegDest}`);
  codesignForNotarization(ffmpegDest);
  // The bundled ffmpeg is GPLv3 — ship its license text + written source offer
  // next to the binary so the distribution satisfies GPLv3 §6 (sc-3767). Source
  // of truth: apps/desktop/licenses/ffmpeg/ (tracked).
  for (const name of ["COPYING.GPLv3", "NOTICE.txt"]) {
    copyFileSync(
      join(desktopDir, "licenses", "ffmpeg", name),
      join(ffmpegDir, name),
    );
  }
  console.log(`build-sidecar: staged ffmpeg GPLv3 license + written offer`);
} else {
  writeFileSync(
    join(ffmpegDir, "README.txt"),
    "Static ffmpeg is bundled on macOS only (sc-3767); Windows/Linux use PATH ffmpeg.\n",
  );
  console.log(`build-sidecar: ${ffmpegDir} placeholder (non-macOS, PATH ffmpeg)`);
}

// The candle (Windows/CUDA) worker links cudarc with dynamic-linking, which
// LoadLibrary's the CUDA runtime redist DLLs by name at runtime. Bundle them as a
// Tauri resource (tauri.conf.json `resources` -> `cuda/**/*`) so a clean NVIDIA
// machine (driver >= 576.02, no CUDA toolkit) runs the candle worker; setup.rs
// prepends this dir to the sidecar's PATH so the loader finds them (sc-5560). Like
// the onnxruntime/ffmpeg dirs above, `cuda` must exist on EVERY platform (Tauri
// errors on an empty glob); only the Windows candle build (SCENEWORKS_DESKTOP_CANDLE
// =1, set above) stages the real DLLs — every other build ships a placeholder.
const cudaDir = join(desktopDir, "cuda");
mkdirSync(cudaDir, { recursive: true });
if (candle) {
  // Resolve the CUDA Toolkit bin dir the redist DLLs are copied from (same
  // CUDA_PATH the cargo build linked against); default to the 12.9 install path.
  const cudaBin = join(
    process.env.CUDA_PATH ??
      "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9",
    "bin",
  );
  // Match by prefix so a minor CUDA point-release (different version suffix) still
  // resolves; the regexes exclude variants like `nvrtc64_120_0.alt.dll`.
  const dllPatterns = [
    /^cudart64_\d+\.dll$/,
    /^cublas64_\d+\.dll$/,
    /^cublasLt64_\d+\.dll$/,
    /^curand64_\d+\.dll$/,
    /^nvrtc64_\d+_\d+\.dll$/,
    /^nvrtc-builtins64_\d+\.dll$/,
  ];
  if (!existsSync(cudaBin)) {
    console.error(
      `build-sidecar: CUDA bin dir not found: ${cudaBin} (set CUDA_PATH to the CUDA 12.9 install for the candle bundle)`,
    );
    process.exit(1);
  }
  const binFiles = readdirSync(cudaBin);
  const staged = [];
  for (const re of dllPatterns) {
    const match = binFiles.find((f) => re.test(f));
    if (!match) {
      console.error(
        `build-sidecar: no CUDA redist DLL matching ${re} in ${cudaBin}`,
      );
      process.exit(1);
    }
    copyFileSync(join(cudaBin, match), join(cudaDir, match));
    staged.push(match);
  }
  // NVIDIA CUDA redistributables ship under the CUDA EULA — stage the tracked
  // notice (CUDA EULA reference + min-driver/NVIDIA requirement) next to the DLLs,
  // mirroring the ffmpeg/onnxruntime license staging above. Same source of truth as
  // the in-app About -> Licenses screen (apps/desktop/licenses/cuda/NOTICE.txt).
  copyFileSync(
    join(desktopDir, "licenses", "cuda", "NOTICE.txt"),
    join(cudaDir, "NOTICE.txt"),
  );
  console.log(
    `build-sidecar: staged ${staged.length} CUDA redist DLLs from ${cudaBin}: ${staged.join(", ")}`,
  );
} else {
  writeFileSync(
    join(cudaDir, "README.txt"),
    "CUDA runtime redist DLLs are bundled only on the Windows candle build (SCENEWORKS_DESKTOP_CANDLE=1, sc-5560).\n",
  );
  console.log(`build-sidecar: ${cudaDir} placeholder (no candle / non-Windows)`);
}
