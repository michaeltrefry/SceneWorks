#!/usr/bin/env node
// Fetches the `uv` binary for the host target triple and stages it as a Tauri
// sidecar (binaries/uv-<triple>) so the packaged app can bootstrap the Python
// venv on first run (sc-1348). Pinned for reproducibility; cached after first
// fetch. Wired into the tauri.conf.json beforeBuildCommand.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, copyFileSync, chmodSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import process from "node:process";

const UV_VERSION = "0.11.15";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const outDir = join(desktopDir, "binaries");

const triple = execFileSync("rustc", ["-vV"], { encoding: "utf8" }).match(
  /host:\s*(\S+)/,
)?.[1];
if (!triple) {
  console.error("fetch-uv: could not determine host target triple");
  process.exit(1);
}
const isWindows = triple.includes("windows");
const exe = isWindows ? ".exe" : "";
const dest = join(outDir, `uv-${triple}${exe}`);

if (existsSync(dest)) {
  console.log(`fetch-uv: ${dest} already present (cached)`);
  process.exit(0);
}

const asset = isWindows ? `uv-${triple}.zip` : `uv-${triple}.tar.gz`;
const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset}`;
const work = join(tmpdir(), `uv-fetch-${process.pid}`);
mkdirSync(work, { recursive: true });
const archive = join(work, asset);

console.log(`fetch-uv: downloading ${url}`);
execFileSync("curl", ["-fsSL", url, "-o", archive], { stdio: "inherit" });
if (isWindows) {
  // Use PowerShell Expand-Archive for the .zip rather than `tar`: depending on
  // PATH the `tar` on Windows may be GNU tar (no zip support — "does not look
  // like a tar archive"), and bsdtar treats a drive-letter path (`C:\...`) as a
  // remote host ("Cannot connect to C: resolve failed"). Expand-Archive is the
  // reliable built-in for zip on any Windows shell.
  execFileSync(
    "powershell",
    [
      "-NoProfile",
      "-Command",
      `Expand-Archive -LiteralPath '${archive}' -DestinationPath '${work}' -Force`,
    ],
    { stdio: "inherit" },
  );
} else {
  // macOS/Linux ship the .tar.gz; bsdtar/GNU tar both extract it fine.
  execFileSync("tar", ["-xf", asset], { cwd: work, stdio: "inherit" });
}

// Find the extracted uv binary (root for zip, uv-<triple>/ for tar.gz).
function findUv(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findUv(full);
      if (found) return found;
    } else if (entry.name === `uv${exe}`) {
      return full;
    }
  }
  return null;
}
const src = findUv(work);
if (!src) {
  console.error("fetch-uv: uv binary not found in archive");
  process.exit(1);
}
mkdirSync(outDir, { recursive: true });
copyFileSync(src, dest);
if (!isWindows) chmodSync(dest, 0o755);
rmSync(work, { recursive: true, force: true });
console.log(`fetch-uv: staged ${dest}`);
