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
// transformers 5.8), provisioned alongside the main venv by setup.rs. scene_worker
// (incl. lens_runner.py + _vendor/lens) is copied wholesale below.
for (const file of [
  "requirements.txt",
  "requirements-ltx.txt",
  "requirements-lens.txt",
  // InstantID face-identity extras (insightface/onnxruntime/onnx/peft/einops);
  // installed into the main venv by setup.rs so the instantid_sdxl adapter runs.
  "requirements-instantid.txt",
  // PuLID-FLUX face-identity extras (timm/facexlib/ftfy + shared
  // insightface/onnxruntime). Installed into the main venv by setup.rs so the
  // pulid_flux adapter runs (sc-2012, epic 2003).
  "requirements-pulid-flux.txt",
  // DWPose whole-body pose-detection extras (rtmlib + the onnxruntime shared with
  // InstantID/PuLID). Installed into the main venv by setup.rs so the worker
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
