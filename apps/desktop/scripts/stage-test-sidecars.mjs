#!/usr/bin/env node
// Stages minimal files required by Tauri's build-script resource validation for
// desktop crate tests/lints. `tauri build` still runs the real beforeBuildCommand.
import { execFileSync } from "node:child_process";
import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, "..");

const triple = execFileSync("rustc", ["-vV"], { encoding: "utf8" }).match(
  /host:\s*(\S+)/,
)?.[1];

if (!triple) {
  console.error("stage-test-sidecars: could not determine host target triple");
  process.exit(1);
}

const exe = triple.includes("windows") ? ".exe" : "";
const binariesDir = join(desktopDir, "binaries");
mkdirSync(binariesDir, { recursive: true });

for (const name of ["sceneworks-api", "uv"]) {
  const path = join(binariesDir, `${name}-${triple}${exe}`);
  writeFileSync(
    path,
    `SceneWorks CI placeholder for ${name}; replaced by tauri beforeBuildCommand during packaging.\n`,
  );
  if (!exe) {
    chmodSync(path, 0o755);
  }
  console.log(`stage-test-sidecars: staged ${path}`);
}

for (const dir of ["python-src", "onnxruntime", "ffmpeg"]) {
  const path = join(desktopDir, dir);
  mkdirSync(path, { recursive: true });
  writeFileSync(
    join(path, "README.txt"),
    "SceneWorks CI placeholder for desktop crate tests/lints; packaging stages real resources.\n",
  );
  console.log(`stage-test-sidecars: staged ${path}`);
}
