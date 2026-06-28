# Strict-Control real-weight validation note (sc-8247, epic 8236)

On-device Metal gate for the unified Fun-Controlnet-Union strict-control pipeline. CI never runs these
(no GPU / no weights) — this file is the maintainer's recorded result of the `real_weight_matrix_*`
`#[ignore]` smokes in `src/image_jobs/tests.rs`.

## What each smoke proves

Every smoke drives the FULL worker job seam — `*_control_load` (the `*_control_spec` `LoadSpec` the live
stream builds) → `preprocess_control_entry` (skeleton / auto-canny / auto-depth) →
`build_control_conditioning` → `*_control_generate_one` (the registered native-MLX engine) → decode —
then asserts the control measurably **steers** the render:

- **pose** (`*_pose_directed`) — renders a left-leaning vs a right-leaning standing skeleton plus a
  matched control-free baseline. Each lean must steer off the baseline AND the two leans must differ from
  each other (`meanAbsΔ > 1.0`), proving the pose control is **DIRECTED** (spatial), not merely on/off.
  This is also the post-refactor **pose regression re-proof** per backbone (the pose path now runs through
  the shared `preprocess_control_entry` / `build_control_conditioning` driver after S1/S2).
- **canny** / **depth** — derives a real edge-map / depth-map from a structured synthetic source via the
  shared preprocessor, renders with vs without it, and asserts a structural steer off the control-free
  baseline (`meanAbsΔ > 1.0`) plus a non-degenerate decode (`std > 5.0`).

## Single run entry point

Each smoke loads its own engine (heavy + serial). Run the whole matrix in one shot with the
`real_weight_matrix` filter:

```sh
unset CARGO_TARGET_DIR   # worktrees only

# Weights / env (the HF-cache repo ids resolve automatically unless overridden):
export FLUX1_DEV_DIR=...                 # or SCENEWORKS_FLUX1_DEV_DIR — gated FLUX.1-dev diffusers snapshot
export SCENEWORKS_CONTROLNET_FLUX1=...    # or FLUX1_CONTROL — Shakker FLUX.1-dev-ControlNet-Union-Pro-2.0
export SCENEWORKS_FLUX2_DEV_DIR=...       # converted Q4 FLUX.2-dev dir (default: app-support models/mlx/flux2_dev)
export SCENEWORKS_DEPTH_ANYTHING_V2=...   # Depth-Anything-V2-Small-hf dir (depth modes only; else HF cache)
# Resolved from the HF cache by repo id (no env needed if cached):
#   Tongyi-MAI/Z-Image-Turbo, Tongyi-MAI/Z-Image, Qwen/Qwen-Image
#   alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
#   alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1
#   alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
#   alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union
#   depth-anything/Depth-Anything-V2-Small-hf

cargo test -p sceneworks-worker --lib --release -- --ignored --nocapture real_weight_matrix
```

Run a single cell with a more specific filter, e.g. `... real_weight_matrix_qwen_depth`.

## Result table (fill in on-device)

Record pass/fail + the printed steer metric (`meanAbsΔ vs control-free`; pose also prints the
left-vs-right directed Δ). Date + machine + weight revs in the notes column.

| Backbone (engine id)              | pose (directed Δ / steer) | canny (steer) | depth (steer) | Notes |
|-----------------------------------|---------------------------|---------------|---------------|-------|
| flux1_dev_control                 |                           |               |               |       |
| flux2_dev_control                 |                           |               |               |       |
| z_image_turbo_control             |                           |               |               |       |
| z_image_control (base, full CFG)  |                           |               |               |       |
| qwen_image_control                |                           |               |               |       |

Validated by: ______   Date: ______   Machine / RAM: ______
