# sc-3489 — Port image upscaler (Real-ESRGAN) to Rust (Mac): path-selection spike

**Decision: GO on path (a) — Real-ESRGAN via Rust `ort` (onnxruntime) + CoreML EP.**
Paths (b) mlx-gen upscaler and (c) Apple MetalFX/CoreML super-res are rejected
(reasons below). The chosen path **reuses 100% of the onnxruntime bundling already
shipped by sc-3487** (DWPose) — `ort = 2.0.0-rc.12` (features `coreml`,
`load-dynamic`, `download-binaries`), `resolve_bundled_onnxruntime()` setting
`ORT_DYLIB_PATH` on the API + mlx worker (`apps/desktop/src/setup.rs:888,1214`),
`onnxruntime/**/*` in `tauri.conf.json` resources, and the build-sidecar staging of
`libonnxruntime.dylib`. Net new infra for the upscaler is just the ONNX model file +
the port code.

## What the upscaler is (the thing being ported)

`apps/worker/scene_worker/upscalers.py` runs **Real-ESRGAN (RRDBNet)** in pure torch
(no `basicsr`/`realesrgan` lib): conv-first → 23 RRDB blocks → conv-body residual →
2× nearest-upsample×2 → conv-last. Two shipped variants (manifest
`config/manifests/builtin.models.jsonc` → `imageUpscalers.real-esrgan`):

| factor | weights (`.pth`) | repo | pixel_unshuffle |
|---|---|---|---|
| x2 | `RealESRGAN_x2plus.pth` | `nateraw/real-esrgan` | unshuffle 2 (in_ch×4) |
| x4 | `RealESRGAN_x4plus.pth` | `nateraw/real-esrgan` | none (in_ch×1) |

Inference is **tiled**: `tile_slices(w,h,512)` grid, each tile cropped with
`tile_pad=16` clamp-to-bounds, run, unpadded inner region copied back at factor scale;
RGB f32 in `[0,1]` → clamp → round → u8. (There is also an `AuraSR` engine, 4x-only,
torch+`aura_sr` lib — see "AuraSR" below.)

## The spike

Reproducible artifacts (committed under `scripts/spikes/`):
- `sc3489_export_reference.py` — verbatim RRDBNet port; loads the **exact shipped
  nateraw weights**, exports ONNX (opset 17, dynamic H/W, legacy TorchScript
  exporter), renders a **tiled torch reference**, and cross-checks an onnxruntime-CPU
  pass inside Python (proves the ONNX graph is faithful before the Rust leg).
- `sc3489_ort_upscale/` — standalone Rust crate (detached `[workspace]` so `ort`'s
  internal `unsafe` dodges the workspace `unsafe_code = forbid`, exactly like
  `sc3487_ort_pose`). Ports the tiling + inference and reports pixel parity vs the
  torch reference. Mirrors the worker's eventual `load-dynamic` path (validated by
  sc-3487 on the real worker binary).

Test image: `poses/standing_09.png` (768×768 → 4 tiles at tile=512, so tiling +
edge-tile shapes are exercised). Apple Silicon (M-series), release build.

### Results

**ONNX faithfulness (Python torch vs onnxruntime-CPU, same tiling):**
- x4: `max|Δ|=1  mean|Δ|≈0.00  PSNR=101.0 dB`
- x2: `max|Δ|=1  mean|Δ|≈0.00  PSNR=101.6 dB`

**Rust `ort` vs torch reference (768²):**

| factor | EP | parity vs torch | latency |
|---|---|---|---|
| x4 | CPU    | `max|Δ|=1  PSNR 100.9 dB` (near-exact) | ~40 s (4 tiles, ~10 s/tile) |
| x4 | CoreML | `max|Δ|=2  mean 0.004  PSNR 72.0 dB` (visually identical) | **~4.3 s** (one-time ~2.3 s graph compile on tile0, then ~0.67 s/tile) |
| x2 | CoreML | `max|Δ|=1  PSNR 73.9 dB` | **~1.4 s** |

CoreML's sub-2-LSB drift is execution-provider nondeterminism, **not a port bug**
(same fp32-vs-CoreML pattern sc-3487 saw); a visual side-by-side detail crop
(`/tmp/sc3489/compare_strip.png`) is indistinguishable from the torch output and
strictly sharper than nearest-neighbour. CoreML accepts the **dynamic** tile shapes
(528² and edge tiles) without falling over, and is ~9× faster than CPU.

### Risks retired

- **R1 runs on ort+CoreML** — yes, incl. dynamic per-tile shapes.
- **R2 parity** — CPU near-exact (PSNR 101), CoreML visually identical (PSNR 72–74).
- **R4 latency** — ~0.67 s/tile on CoreML after a one-time per-shape compile;
  acceptable for an on-demand upscale job, ~9× faster than CPU EP.

## Why not (b) mlx-gen or (c) MetalFX

- **(b) mlx-gen upscaler** — would mean re-implementing RRDBNet (conv stack +
  pixel_unshuffle + nearest-upsample) in `mlx-rs` from scratch, plus a `.pth`→MLX
  weight convert, for **zero quality gain** over running the identical graph through
  onnxruntime. More code, more parity surface, no benefit. Rejected.
- **(c) Apple MetalFX / CoreML super-res** — MetalFX is a *temporal/spatial* upscaler
  built for real-time game rendering, not photographic super-resolution; it is a
  **different algorithm** with no weight parity to Real-ESRGAN, so it would change the
  product's output and forfeit the whole point of "keep the dedicated fast upscaler."
  Rejected on quality/parity.

## ONNX provisioning (the one open infra item)

The torch path downloads `.pth`; the Rust path needs the **ONNX**, and the Mac must
get it **without torch**. Resolution order in the port mirrors Python's
(`image_adapters.py:_resolve_*`):

1. env override `SCENEWORKS_REALESRGAN_X{2,4}_ONNX` (and `SCENEWORKS_REALESRGAN_ONNX`)
   — used for dev/E2E now against the locally-exported ONNX;
2. manifest resource (new `onnx` sub-entry under `imageUpscalers.real-esrgan.x{2,4}`);
3. on-disk cache (`<data_dir>/cache/upscale/`);
4. download-on-first-use from a HF repo (parity with sc-3487's rtmlib weights).

**Hosting:** the export is fully reproducible from the committed script, but the two
ONNX files (~67 MB each, fp32) must live at a stable URL for step 4. Consistent with
the manifest convention (every other model is *downloaded*, not bundled) and with
sc-3487, the right home is a **SceneWorks-controlled HF repo** (e.g.
`trefster/sceneworks-real-esrgan-onnx`). This needs a HF **write** token — the token
in this environment is read-only (`HF_READ_TOKEN_MAC`). **Action for Michael:** create
the repo (or grant a write token) so the production default download URL can be wired;
until then the port is fully testable via the env override. fp16 export (~33 MB each)
is a viable size optimisation — CoreML runs fp16 internally anyway — but is left as a
follow-up to keep parity numbers clean.

## AuraSR

`AuraSR` (4x-only, `fal/AuraSR-v2`, torch + `aura_sr` lib, GAN-based) is a *second*
upscaler engine selectable via `engine="aura-sr"`. It is out of scope for this RRDBNet
ort port: it is not RRDBNet, has no clean ONNX export path, and `real-esrgan` is the
default engine. On Mac it stays a tracked gap — the oracle should keep
`engine=aura-sr` upscale jobs reporting `mlx_unsupported` (its own follow-up) while
`real-esrgan` (the default) flips to supported.

## Implementation outline (next)

- `crates/sceneworks-worker/src/upscale_jobs.rs` (macOS-gated `mod`): port
  `tile_slices`/tiling + ort+CoreML inference with a cached `Session`
  (`OnceLock<Mutex<...>>`, amortise the CoreML graph compile across a batch, run inside
  one `spawn_blocking`, CoreML-then-CPU fallback). Result contract **exactly** matches
  `image_adapters.run_image_upscale` (`generationSetId`, `expectedCount`,
  `generationSet`, `assetWrites[fact]` with `mediaPath` under
  `assets/images/{genset}`, `extra.isUpscaled/upscaledFromAssetId/factor/engine`,
  `parents` lineage). `ensure_weights` per the order above.
- `lib.rs`: `#[cfg(macos)] JobType::ImageUpscale => run_image_upscale_job` (mirror
  `PoseDetect`).
- `gpu.rs` `mlx_gpu()`: advertise `WorkerCapability::ImageUpscale`.
- `jobs_store.rs`: flip `mac_rust_supported` `ImageUpscale` `Err(sc-3489)`→`Ok` (for
  `engine=real-esrgan`), update `classify_image_gap` / `model_mac_support` /
  `mac_capabilities.imageUpscale`, oracle tests, `docs/mac-rust-gaps.md` row → Ported.
