#!/usr/bin/env node
// Stages the Python inference worker (source + requirements) into
// apps/desktop/python-src so Tauri can bundle it as a resource (sc-1348/1347).
// Avoids referencing ../worker directly from Tauri's resource globs. The staged
// dir is gitignored and rebuilt on each `tauri build`.
import { cpSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const repoRoot = resolve(desktopDir, "..", ".."); // repository root
const workerDir = join(repoRoot, "apps", "worker");
const outDir = join(desktopDir, "python-src");

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

// requirements-lens.txt feeds the separate Lens sidecar venv (torch 2.11 /
// transformers 5.8), provisioned alongside the main venv by setup.rs — still used by
// the Lens LoRA/LoKr TRAINING sidecar (lens_train_runner.py + _vendor/lens) until the
// sc-5147 training cutover. Lens INFERENCE moved to the native candle backend and its
// sidecar (lens_runner.py) was retired (sc-5126). scene_worker is copied wholesale below.
for (const file of [
  "requirements.txt",
  "requirements-ltx.txt",
  "requirements-lens.txt",
  // InstantID face-identity extras (insightface/onnxruntime/onnx/peft/einops);
  // installed into the main venv by setup.rs so the instantid_sdxl adapter runs.
  "requirements-instantid.txt",
  // PuLID-FLUX (sc-2012) was retired to the native mlx-gen `pulid_flux` worker
  // target (sc-3344) — its torch extras (requirements-pulid-flux.txt) + vendored
  // stack are gone, so there is nothing to stage here.
  // DWPose whole-body pose-detection extras (rtmlib + the onnxruntime shared with
  // InstantID). Installed into the main venv by setup.rs so the worker
  // advertises the `pose_detect` capability — without it the Pose Library Create
  // tab's detect jobs block ("no active worker supports pose detect"). Epic 2282.
  "requirements-pose.txt",
]) {
  const src = join(workerDir, file);
  if (existsSync(src)) cpSync(src, join(outDir, file));
}

// Copy the scene_worker package, skipping caches.
cpSync(join(workerDir, "scene_worker"), join(outDir, "scene_worker"), {
  recursive: true,
  filter: (src) => !src.includes("__pycache__") && !src.endsWith(".pyc"),
});

// Copy the sceneworks_shared package alongside scene_worker. scene_worker
// imports it at startup (image_adapters/video_adapters); in Docker it's provided
// via PYTHONPATH=/app/packages/shared. Bundling it into python-src keeps the
// packaged worker importable (setup.rs also puts python-src on PYTHONPATH).
cpSync(
  join(repoRoot, "packages", "shared", "sceneworks_shared"),
  join(outDir, "sceneworks_shared"),
  {
    recursive: true,
    filter: (src) => !src.includes("__pycache__") && !src.endsWith(".pyc"),
  },
);

console.log(`stage-python: staged worker source into ${outDir}`);
