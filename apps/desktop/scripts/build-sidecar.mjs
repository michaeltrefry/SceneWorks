#!/usr/bin/env node
// Builds the sceneworks-rust-api binary (with the embedded web UI) and stages it
// as a Tauri sidecar named for the host target triple. Wired as the
// tauri.conf.json `beforeBuildCommand` so `tauri build` is self-contained.
import { execFileSync, execSync } from "node:child_process";
import { copyFileSync, mkdirSync, chmodSync, writeFileSync } from "node:fs";
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

// Build the web bundle + API binary with the embedded UI (single source of
// truth for the embedded build). Empty VITE_API_BASE_URL makes the embedded UI
// talk to its own origin (the API serves it), so it works on the dynamic port
// with no CORS.
run(npmCmd, ["run", "api:build:embedded"], { VITE_API_BASE_URL: "" });

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
} else {
  writeFileSync(
    join(ortDir, "README.txt"),
    "onnxruntime CoreML dylib is bundled on macOS only (Rust DWPose detector, sc-3487).\n",
  );
  console.log(`build-sidecar: ${ortDir} placeholder (non-macOS, no DWPose dylib)`);
}
