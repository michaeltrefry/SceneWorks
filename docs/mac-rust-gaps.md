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
| `kolors` | kolors (SDXL UNet + ChatGLM3) | 🔵 Port → drop-on-Mac until then | **epic 3532** |
| `chroma1_hd`, `chroma1_base`, `chroma1_flash` | chroma (FLUX.1-schnell DiT) | 🔵 Port → drop-on-Mac until then | **epic 3531** |
| `z_image_edit` | z-image (edit) | 🔵 Port → drop-on-Mac until then | **epic 3529** |
| `instantid_realvisxl` | sdxl (InstantID) | 🔵 Port → drop-on-Mac until then | epic 3109 |
| `pulid_flux_dev` | flux (PuLID) | 🔵 Port → drop-on-Mac until then | epic 3069 (engine done; owes SceneWorks routing) |
| `sensenova_u1_8b`, `sensenova_u1_8b_fast` | sensenova-u1 | 🔵 Port → drop-on-Mac until then | epic 3180 |
| `lens`, `lens_turbo` | lens (Python sidecar `/opt/lens-venv`) | 🔵 Port → drop-on-Mac until then | epic 3164 |

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
| Reference / IP-Adapter / edit | `flux_schnell`, `flux_dev` | 🔵 Viability spike | sc-3535 |
| `edit_image` (img2img-edit) | `z_image_turbo` | 🔵 Port-pending | epic 3529 (folds into Z-Image-Edit port) |
| reference-without-pose | `z_image_turbo` | 🟢 Ported (MLX) | sc-3536 (spike GO) → sc-3619 |
| Third-party LyCORIS (LoHa / non-peft LoKr) | all families (`networkType=lycoris`) | 🔵 Port-or-drop spike | sc-3537 |

## 3. Video

`video_generate` `text_to_video`/`image_to_video` on `VIDEO_MLX_ROUTED_MODELS`
(`ltx_2_3`, `ltx_2_3_eros`, `wan_2_2`, `wan_2_2_t2v_14b`, `wan_2_2_i2v_14b`) is ported. Gaps:

| Surface | Status | Closing work |
|---|---|---|
| `svd` model (Stable Video Diffusion, `svd_video` adapter — no MLX crate) | 🔵 Port-pending | epic 3040 |
| Advanced `video_generate` modes (`first_last_frame`, `replace_person`) | 🔵 Port-pending | epic 3040 |
| Advanced job types `video_extend`, `video_bridge` | 🔵 Port-pending | epic 3040 |
| `person_replace` job type (replace_person) | 🔵 Port-pending | epic 3040 (+ sc-3488 person track) |
| LoKr-on-Wan (Kronecker adapter on Wan) | 🔵 Port-pending | epic 3040 (LoKr-on-LTX already MLX) |
| Third-party LyCORIS on video | 🔵 Port-or-drop spike | sc-3537 (shared image+video spike) |

## 4. Training (`lora_train`)

`MLX_ROUTED_TRAINING_KERNELS` = `z_image_lora`, `sdxl_lora`, `wan_lora`, `wan_moe_lora`,
`ltx_mlx_lora` (the last is MLX-only). Gaps:

| Kernel | Status | Closing work |
|---|---|---|
| `kolors_lora` (SDXL + ChatGLM3, no mlx-gen trainer) | 🔵 Port-pending | epic 3039 |
| `lens_lora` (Python sidecar trainer) | 🔵 Port-pending | epic 3039 (follows Lens model port, epic 3164) |
| LoKr-on-Wan (`wan_lora` / `wan_moe_lora` + `networkType=lokr`) | 🔵 Port-pending | epic 3039 |

## 5. Non-model Python infrastructure

Job types / sub-systems that run on the Python worker (onnxruntime / torch / mlx_video) with no
in-process Rust path. Per Michael's 2026-06-07 decision, all four spikes are **port** (not drop).

| Surface | Job type(s) | Python backend | Status | Closing work |
|---|---|---|---|---|
| DWPose pose detection (photo→skeleton) | `pose_detect` | onnxruntime (RTMPose) | ✅ Ported (Rust `ort`/CoreML, macOS MLX worker) | sc-3487 |
| Person detect / track | `person_detect`, `person_track` | YOLO / SAM2 | 🔵 Port-pending | sc-3488 |
| Image upscaler (standalone) | `image_upscale` | Real-ESRGAN / AuraSR (torch) | 🔵 Port-pending | sc-3489 |
| Dataset captioning | `training_caption` | JoyCaption MLX provider (Python torch fallback off-MLX) | ✅ Ported (macOS MLX worker) | sc-3556 |
| Wan/LTX model conversion | `model_convert` (non-`flux2_klein_diffusers` converter) | `mlx_video.convert_*` (Python) | 🔵 Port-pending | sc-3491 (= sc-3224) |
| Image understanding / interleave | `image_vqa`, `image_interleave` | torch (SenseNova-U1) | 🔵 Port-pending | epic 3180 (SenseNova port — its understanding surface) |

## 6. Already ported — NOT gaps (context)

Listed so a reviewer doesn't re-file these. All run in the Rust/MLX flow on Mac.

- Image base families: `z_image_turbo`, `flux_schnell`, `flux_dev`, `qwen_image` (txt2img),
  `qwen_image_edit{,_2509,_2511,_2511_lightning}`, `flux2_klein_9b{,_kv,_true_v2}`, `sdxl`,
  `realvisxl` (epic 3018).
- SDXL advanced shapes — reference/IP-Adapter, `edit_image`, masked inpaint, outpaint, and
  tile-detail (`image_detail` on `sdxl`/`realvisxl`) — epic 3041 / sc-3060.
- FLUX.2-klein single-file conversion in-process Rust (`flux2_klein_diffusers`, sc-3136).
- Video `text_to_video`/`image_to_video` on Wan2.2 + LTX-2.3 (+ synchronized audio), epic 3018.
- Training: `z_image_lora`, `sdxl_lora`, `wan_lora`, `wan_moe_lora`, `ltx_mlx_lora` (epic 3039).

---

_Maintained under epic 3482 (sc-3485). Update alongside any change to the routing predicates._
