# macOS Python-dependency inventory (epic 3482)

The triage table the Python-eradication cutover burns down. Every row is something that, on
**macOS**, still reaches the Python torch/MPS worker — i.e. the in-process Rust/MLX flow can't
run it yet. When the list is empty, the Mac build can stop shipping a Python venv/sidecar
(sc-3492 / sc-3493).

> **This table is code-derived.** Its executable form is
> [`mac_rust_supported(job)`](../crates/sceneworks-core/src/jobs_store.rs) (sc-3484) — the
> inverse of the `*_mlx_eligible` predicates. The same routing constants are the source of
> truth: `MLX_ROUTED_MODELS`, `VIDEO_MLX_ROUTED_MODELS`, `MLX_ROUTED_TRAINING_KERNELS`, and the
> per-family `*_mlx_eligible` gates in `crates/sceneworks-core/src/jobs_store.rs`; the model
> registry is `MODEL_TARGETS` (`apps/worker/scene_worker/image_adapters.py` /
> `video_adapters.py`); training kernels are the builtin targets in
> `crates/sceneworks-core/src/training.rs`. **Keep this file in sync when a surface flips** —
> when a model lands in a `*_ROUTED_*` set or an `*_mlx_eligible` gate opens, move its row to
> *Done* and delete the gap. A row here that no longer matches the predicates is a bug.

**Status legend**

- ✅ **Done** — runs in the Rust/MLX flow on Mac (here for context; not a gap).
- 🔵 **Port-pending (epic/story NNNN)** — has a tracked porting epic or story; ported, dropped on Mac until then.
- 🔵 **Viability / port-or-drop spike (sc-NNNN)** — a spike decides port-vs-drop (the outcome is Michael's call); the model/feature is gated on Mac until it resolves.

**No row is a bare "drop".** Per policy (Michael, 2026-06-07): every model gets a porting epic; every feature gap gets at least a viability spike. A *drop* is only ever the **outcome of a spike**, never a default — so there is no "drop-candidate" or untriaged state here. A code-surfaced gap with no tracked work is a bug in this table.

Rollout reminder: the cutover is staged (epic 3482). The `mlx_unsupported` oracle ships
**warn-only** by default, so flipping `SCENEWORKS_MLX_REQUIRED=1` on a Mac logs every row below
without breaking anything; each surface flips to enforce only once it's ported (its epic
completes) or dropped (UI-gated, sc-3486).

> **UI gating (sc-3486).** The same oracle is surfaced to the web client so a Mac user never
> reaches a `mlx_unsupported` error after submit. `GET /api/v1/models` stamps each model with
> `macSupport { supported, reason, features { pose, reference, edit, lycoris, videoModes } }`
> (`model_mac_support`), and `GET /api/v1/capabilities/mac` carries the master switch
> (`macGatingActive` = `SCENEWORKS_MLX_REQUIRED`), the infra feature gaps below (§5), and the
> supported training kernels (§4). The client (`apps/web/src/macGating.js`) hides torch-only
> models from the studio pickers and disables the feature controls in this table — but only when
> `macGatingActive`, so Windows/Linux (and an observe-mode Mac) are untouched. When a surface
> here flips to *Done*, its `macSupport`/capability flag flips to supported automatically (same
> predicates), and the UI stops gating it — no separate UI change needed.

---

## 1. Torch-only image models

Image models in `MODEL_TARGETS` that are **not** in `MLX_ROUTED_MODELS` → the Python torch
adapter is authoritative on Mac. **Policy (Michael, 2026-06-07): every unported model gets its
own MLX-porting epic and is *dropped on Mac only* (UI-gated, sc-3486) until that port lands —
Windows/Linux keep the torch path.** Nothing here is a permanent drop. `mac_rust_supported` →
`torch_only_image_model_epic(model)` names the specific epic below.

| Model id | Family | Mac disposition | Porting epic |
|---|---|---|---|
| `kolors` | kolors (SDXL UNet + ChatGLM3) | 🟡 Base **T2I** ported (sc-3875); img2img / ControlNet-pose / IP-Adapter-Plus stay torch (per-feature gaps) until later epic-3090 slices | epic 3090 |
| `pulid_flux_dev` | flux (PuLID) | 🔵 Port → drop-on-Mac until then | epic 3069 (engine done; owes SceneWorks routing) |

> **No whole-model torch-only image families remain.** `lens` / `lens_turbo` were the last; they
> are MLX on Mac as of epic 3164 / sc-5105 (see §6). `torch_only_image_model_epic` now names nothing
> — the gap path fires only for a hypothetical unported model id.

> A torch-only image model with **no** porting epic yet → `torch_only_image_model_epic` returns
> `None` and the oracle reports "needs a port epic (epic 3482 policy)"; file one + add it to the
> match. FLUX.2-**dev** is not a Mac `MODEL_TARGETS` entry and is out of mlx-gen scope; third-party
> **LyCORIS** is a feature gap, see §2.

## 2. Image feature gaps on MLX-routed families

Models that ARE MLX-routed but fall back to torch for a specific feature (the `*_mlx_eligible`
exclusions). `mac_rust_supported` names each precisely. **Policy: a feature gap on an
already-ported model gets at least a viability spike (or an epic if large) before it's ported or
dropped — no silent drops.**

| Feature | Affected models | Status | Closing work |
|---|---|---|---|
| Strict-pose ControlNet | `qwen_image` (+ `advanced.poses`) | 🔵 Port-pending | epic 3401 (Qwen ControlNet port) |
| Reference / edit conditioning | base `qwen_image` (reference/`edit_image`) | 🔵 Port-pending | epic 3401 |
| Reference (XLabs IP-Adapter) | `flux_schnell`, `flux_dev` | 🟢 Ported (MLX) | sc-3535 (spike) → epic 3621 (sc-3622–3625) |
| `edit_image` (img2img-edit) | `z_image_turbo`, `z_image_edit` | 🟢 Ported (MLX) | epic 3529 / sc-3923 (engine `Conditioning::Reference` img2img; Turbo weights) |
| reference-without-pose | `z_image_turbo` | 🟢 Ported (MLX) | sc-3536 (spike GO) → sc-3619 |
| Third-party LyCORIS (LoHa / non-peft LoKr) | all families (`networkType=lycoris`) | 🟢 Ported (MLX) | sc-3537 (spike) → epic 3641 (sc-3642/3643/3671 engine + sc-3644 routing) |
| InstantID (identity, 11-view angle set, pose-library mode, face-restore) | `instantid_realvisxl` (`character_image` + `referenceAssetId`) | 🟢 Ported (MLX) | epic 3109 (engine: #153 identity/angle, #193 pose+restore — sc-3117/3380) → sc-3345 (identity+angle integration) + sc-3381 (pose+restore integration). Torch path kept as off-Mac + Mac-fallback (Decision-A); venv strip is the final epic-3482 step. |

> **FLUX.1 `edit_image` is not an eradication gap (sc-3535).** The torch `FluxDiffusersAdapter`
> hard-rejects `edit_image` ("does not support image editing") — FLUX.1 has no edit path on *any*
> platform, so it reaches no Python worker to retire. It's a universal product gap (a future
> FLUX.1-Kontext capability), not a Mac-vs-torch gap, and is intentionally absent from this table.
> Likewise, FLUX.1 reference = the **XLabs IP-Adapter** (not VAE img2img-init), which is why it
> needed a real engine port in epic 3621 (now landed — CLIP-ViT-L encoder + decoupled cross-attn in
> `mlx-gen-flux`, sc-3622–3625) rather than a gate-flip like Z-Image's sc-3619. Both schnell + dev
> route reference to MLX — the Rust engine has no diffusers `load_ip_adapter` schnell limitation.

## 3. Video

`video_generate` `text_to_video`/`image_to_video` on `VIDEO_MLX_ROUTED_MODELS`
(`ltx_2_3`, `ltx_2_3_eros`, `wan_2_2`, `wan_2_2_t2v_14b`, `wan_2_2_i2v_14b`) is ported — **and as of
the epic-3040 / sc-3055 cutover, so are the advanced modes + SVD** (`video_job_is_mlx_eligible` +
`video_mode_is_mlx_eligible` in `jobs_store.rs`). **No video gaps remain.** The rows below are kept
for traceability (all ✅ Ported; see also §6):

| Surface | Status | Shipped by |
|---|---|---|
| `svd` model → `svd_xt` (Stable Video Diffusion, image-to-video) | ✅ Ported (MLX) | sc-3523 ([#493](https://github.com/michaeltrefry/SceneWorks/pull/493)) |
| Advanced `video_generate` modes (`first_last_frame`, `replace_person`) | ✅ Ported (MLX) | sc-3520 ([#466](https://github.com/michaeltrefry/SceneWorks/pull/466)), sc-3521 ([#494](https://github.com/michaeltrefry/SceneWorks/pull/494)) |
| Advanced job types `video_extend`, `video_bridge` | ✅ Ported (MLX) | sc-3522 ([#492](https://github.com/michaeltrefry/SceneWorks/pull/492)) |
| `person_replace` job type (replace_person → native Wan-VACE) + user LoRA/LoKr | ✅ Ported (MLX) | sc-3521, sc-3893 ([#511](https://github.com/michaeltrefry/SceneWorks/pull/511)); real-Mac parity sc-3902 |
| LoKr-on-Wan **inference** (Kronecker adapter on Wan generation) | ✅ Ported (MLX) | sc-3644 (engine `merge_one_lokr` since sc-2393; routing gate flipped — never an engine limit). Wan LoKr *training* stays torch → epic 3039 |
| Third-party LyCORIS on video | ✅ Ported (MLX) | sc-3537 (spike) → epic 3641 (sc-3671 Wan/LTX engine + sc-3644 routing) |

## 4. Training (`lora_train`)

`MLX_ROUTED_TRAINING_KERNELS` = `z_image_lora`, `sdxl_lora`, `kolors_lora`, `lens_lora`, `wan_lora`,
`wan_moe_lora`, `ltx_mlx_lora` (the last is MLX-only). Gaps:

| Kernel | Status | Closing work |
|---|---|---|
| `kolors_lora` (SDXL U-Net + ChatGLM3) | ✅ Ported (native mlx-gen `KolorsTrainer`, LoRA + LoKr) | engine sc-4568, SceneWorks cutover sc-4732 |
| `lens_lora` (gpt-oss MoE + Lens MMDiT) | ✅ Ported (native mlx-gen `LensTrainer`, LoRA + LoKr) | engine sc-5148, SceneWorks cutover sc-5180 (off-Mac keeps the Python sidecar trainer) |
| LoKr-on-Wan (`wan_lora` / `wan_moe_lora` + `networkType=lokr`) | 🔵 Port-pending | epic 3039 |

## 5. Non-model Python infrastructure

Job types / sub-systems that run on the Python worker (onnxruntime / torch / mlx_video) with no
in-process Rust path. Per Michael's 2026-06-07 decision, all four spikes are **port** (not drop).

| Surface | Job type(s) | Python backend | Status | Closing work |
|---|---|---|---|---|
| DWPose pose detection (photo→skeleton) | `pose_detect` | onnxruntime (RTMPose) | ✅ Ported (Rust `ort`/CoreML, macOS MLX worker) | sc-3487 |
| Person detect / track | `person_detect`, `person_track` | YOLO / SAM2 | ✅ Ported (all **native MLX**, macOS MLX worker) | sc-3488 → YOLO detect sc-3633 (native mlx-rs forward, CoreML/ort hangs), ByteTrack track assembly sc-3634, **SAM2 segmenter = MLX engine epic 3704** (spike sc-3635 GO→MLX; CoreML net-negative for the Hiera ViT) + wiring sc-3709 (capability advertise + `mac_rust_supported` flip). maskState active/generated/degraded/missing. **replace_person end-to-end is now complete** — the video-gen/inpaint half (native Wan-VACE) shipped in epic 3040 / sc-3521 (see §3) |
| Image upscaler (standalone) | `image_upscale` | Real-ESRGAN / AuraSR (torch) | ✅ Ported. **Real-ESRGAN** via Rust `ort`/CoreML; **SeedVR2** (`engine=seedvr2`) = native-MLX one-step diffusion super-resolution via in-process `mlx-gen-seedvr2`, run through the single-resident `with_cached_generator` seam (factor→`round_to_16(src×factor)`, optional `--softness` detail knob), **Mac-only** (Windows/Linux SeedVR2 backend = future Candle port **sc-5157**). **AuraSR engine = DROPPED on Mac** (sc-3668 port-or-drop spike: 617M torch-only GigaGAN, no viable Rust path, only a marginal & ~35–50× slower quality difference vs Real-ESRGAN x4) — UI-gated out of the Mac engine picker; stays available on Windows/Linux. SeedVR2 HD spatial tiling (un-tiled image path → large 4× targets can exceed the memory budget) is a tracked mlx-gen follow-up | sc-3489 (Real-ESRGAN); **sc-4815 (SeedVR2, epic 4811)**; **sc-3668 (AuraSR drop)** |
| Video upscaler (standalone) | `video_upscale` | **none (net-new capability — SceneWorks had no video upscaler)** | ✅ Net-new on Mac (**native-MLX SeedVR2**, `mlx-gen-seedvr2`, macOS MLX worker): decode the source clip → one-step super-resolution (temporal chunking + overlap internal to the engine) → re-encode + source-audio passthrough. 3B only here; mac-only (no torch path — Windows/Candle = sc-5157). 7B/int8/HD-tiling = sc-5197/5198/5201 | epic 4811 — engine sc-4813/sc-4814; worker cutover **sc-4816** |
| Dataset captioning | `training_caption` | JoyCaption MLX provider (Python torch fallback off-MLX) | ✅ Ported (macOS MLX worker) | sc-3556 |
| Wan/LTX model conversion | `model_convert` (non-`flux2_klein_diffusers` converter) | `mlx_video.convert_*` (Python) | 🔵 Port-pending | sc-3491 (= sc-3224) |
| Image understanding / interleave | `image_vqa`, `image_interleave` | torch (SenseNova-U1) | ✅ Ported (macOS MLX worker, native `T2iModel`) | epic 3180 / sc-3905 (see §6) |

## 6. Already ported — NOT gaps (context)

Listed so a reviewer doesn't re-file these. All run in the Rust/MLX flow on Mac.

- Image base families: `z_image_turbo`, `flux_schnell`, `flux_dev`, `qwen_image` (txt2img),
  `qwen_image_edit{,_2509,_2511,_2511_lightning}`, `flux2_klein_9b{,_kv,_true_v2}`, `sdxl`,
  `realvisxl` (epic 3018).
- Chroma text-to-image: `chroma1_hd`, `chroma1_base`, `chroma1_flash` (FLUX.1-schnell-derived
  DiT, T5-only true-CFG; `mlx-gen-chroma`) — epic 3531 / sc-3843.
- SenseNova-U1 image modes: `sensenova_u1_8b`, `sensenova_u1_8b_fast` — T2I, instruction edit
  (`edit_image` → Reference), and Character Studio (`character_image` → MultiReference, incl. the
  angle set), base + 8-step distill, Q4/Q8 (`mlx-gen-sensenova`, NEO-Unify; dual CFG: text via
  `guidance`, image via `true_cfg`). No ControlNet (strict pose stays torch), no user LoRA. epic
  3180 / sc-3900.
- SenseNova-U1 understanding + Document Studio: `image_vqa` (image+question → text) and
  `image_interleave` (prompt → ordered text + generated images → `InterleavedDocument`) — served
  in-process via the concrete `T2iModel::{vqa, interleave_gen}` (the `Generator` registry emits
  Images/Video only). Loads dense (no distill LoRA, no quant) for torch parity; the VQA decode is
  bit-identical (no-think primed, sc-3905 engine fix). epic 3180 / sc-3905.
- Lens / Lens-Turbo text-to-image: `lens` (20-step / CFG 5.0), `lens_turbo` (4-step / guidance 1.0)
  — gpt-oss-20b MoE text encoder + 48-layer dual-stream MMDiT + Flux.2 VAE; pure T2I, standard
  guidance + negative prompt, Q4/Q8 (encoder MoE + DiT), LoRA + LoKr at load (`mlx-gen-lens`,
  `mac_only`). Retires the Python `/opt/lens-venv` transformers-5 sidecar on Mac (Win/Linux/Docker
  keep the torch path). LoRA/LoKr *training* is also native MLX now — the `lens_lora` kernel routes to
  the `mlx-gen-lens` Rust trainer on Mac (engine sc-5148, worker cutover sc-5180; off-Mac keeps the
  Python sidecar trainer). epic 3164 / sc-5105.
- SDXL advanced shapes — reference/IP-Adapter, `edit_image`, masked inpaint, outpaint, and
  tile-detail (`image_detail` on `sdxl`/`realvisxl`) — epic 3041 / sc-3060.
- Z-Image img2img-edit: `z_image_edit` + `z_image_turbo` `edit_image` mode (Turbo weights via the
  engine's `Conditioning::Reference` img2img path; `sourceAssetId` + `strength`) — epic 3529 / sc-3923.
- FLUX.2-klein single-file conversion in-process Rust (`flux2_klein_diffusers`, sc-3136).
- Video `text_to_video`/`image_to_video` on Wan2.2 + LTX-2.3 (+ synchronized audio), epic 3018.
- Advanced video — `first_last_frame`, `extend_clip`, `video_bridge`, `replace_person` (→ native
  Wan-VACE, + user LoRA/LoKr), and `svd`→`svd_xt` image-to-video — all on the macOS MLX worker
  (epic 3040 / cutover sc-3055; real-Mac parity sc-3902).
- Training: `z_image_lora`, `sdxl_lora`, `kolors_lora`, `wan_lora`, `wan_moe_lora`, `ltx_mlx_lora` (epic 3039; kolors sc-4732).

---

_Maintained under epic 3482 (sc-3485). Update alongside any change to the routing predicates._
