//! Native MLX image generation jobs — runtime pipeline + Z-Image inference (epic 3018).
//!
//! Parses the job into an [`ImageRequest`], generates `count` images, saves each PNG
//! into the project's `assets/images/`, and reports flat "facts" the Rust API turns
//! into indexed assets. The API's `persist_reported_assets` (apps/rust-api jobs.rs)
//! runs on EVERY progress update — idempotently building each sidecar via
//! `build_image_sidecar_parts` and indexing project.db — so emitting the accumulating
//! `assetWrites` per image is what streams results into the gallery as they land.
//!
//! On macOS, engine-backed families (`z_image_turbo` — sc-3022; `flux_schnell` /
//! `flux_dev` — sc-3023; `qwen_image` — sc-3024 / strict pose sc-3575) run **real**
//! in-process inference via the linked mlx-gen
//! engine; other models (and non-macOS) fall back to a procedural stub (sc-3020), so
//! the pipeline stays cross-platform-testable and each new family just adds a row to
//! the [`crate::engines::MODEL_TABLE`] dispatch table + links its provider crate.

use super::*;
use sceneworks_core::image_request::ImageRequest;

// Force each provider crate to link so its `inventory::submit!` registration survives
// linker GC. Each per-family story adds its provider dep + a matching `use … as _;`.
// See mlx-gen-z-image/tests/registry.rs ("the SceneWorks worker").
// epic 3720 (sc-3724): the backend-neutral contract types come from `gen_core` (the registry
// contract layer mlx-gen re-exports). The `as _;` provider links below stay mlx-gen-specific —
// `cfg(target_os)` decides which backend crates register into the registry, not which contract
// types the worker names.
// Contract types for the generation harness — shared by the macOS MLX path AND the Windows candle
// lane (sc-3675), so broadened from macOS-only. `gen_core` is a direct worker dep on every platform.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Progress, Quant, WeightsSource,
};
// `AdapterKind` (LoRA/LoKr classification) was MLX-only until sc-5126: the candle Lens lane is the
// first candle family to take LoRA/LoKr, so it now classifies adapters too and the import moved into
// the shared block above. `ControlKind` (ControlNet conditioning) stays MLX-only — the candle lane is
// pure txt2img.
#[cfg(target_os = "macos")]
use gen_core::ControlKind;
#[cfg(target_os = "macos")]
use mlx_gen_chroma as _;
#[cfg(target_os = "macos")]
use mlx_gen_flux as _;
#[cfg(target_os = "macos")]
use mlx_gen_flux2 as _;
// Ideogram 4 (epic 4725) — force-link so `inventory::submit!` registers `ideogram_4`.
#[cfg(target_os = "macos")]
use mlx_gen_ideogram as _;
// Boogu-Image-0.1 (epic 6387) — force-link so `inventory::submit!` registers `boogu_image`,
// `boogu_image_turbo`, and `boogu_image_edit` (else linker GC drops their `ModelRegistration` and
// `gen_core::load("boogu_image")` returns "no generator registered").
#[cfg(target_os = "macos")]
use mlx_gen_boogu as _;
#[cfg(target_os = "macos")]
use mlx_gen_kolors as _;
// Krea 2 Turbo (epic 7565) — force-link so `inventory::submit!` registers `krea_2_turbo` (else linker GC
// drops its `ModelRegistration` and `gen_core::load("krea_2_turbo")` returns "no generator registered").
#[cfg(target_os = "macos")]
use mlx_gen_krea as _;
// Stable Diffusion 3.5 (epic 7841 / sc-7871) — force-link so `inventory::submit!` registers
// `sd3_5_large` (true-CFG) + `sd3_5_large_turbo` (ADD-distilled, CFG-off) + `sd3_5_medium` (MMDiT-X);
// else linker GC drops their `ModelRegistration` and `gen_core::load("sd3_5_large")` returns "no
// generator registered". All three reach the generic MODEL_TABLE / `generate_stream` path.
#[cfg(target_os = "macos")]
use mlx_gen_sd3 as _;
// Lens / Lens-Turbo (epic 3164 engine / sc-5105) — an inventory-registered `Generator` under the ids
// `lens` + `lens_turbo`, reached through the generic MODEL_TABLE / `generate_stream` path. Force-link
// or the linker GCs its `ModelRegistration` and `gen_core::load("lens_turbo")` returns "no generator
// registered" (the bug that bit Kolors).
#[cfg(target_os = "macos")]
use mlx_gen_lens as _;
// PuLID-FLUX (sc-3344) IS an inventory-registered `Generator` (engine id `pulid_flux`), unlike the
// bespoke InstantID provider below — so it is force-linked here like the other registry families
// (its `ModelRegistration` is otherwise dropped by linker GC) and reached via `gen_core::load`. The
// reference-face + idWeight/timestepToStartCfg mapping + weight provisioning live in the dedicated
// `generate_pulid_flux_stream` (image_jobs/pulid.rs), not the generic MODEL_TABLE path.
#[cfg(target_os = "macos")]
use mlx_gen_pulid as _;
#[cfg(target_os = "macos")]
use mlx_gen_qwen_image as _;
#[cfg(target_os = "macos")]
use mlx_gen_sdxl as _;
#[cfg(target_os = "macos")]
use mlx_gen_seedvr2 as _;
#[cfg(target_os = "macos")]
use mlx_gen_sensenova as _;
#[cfg(target_os = "macos")]
use mlx_gen_z_image as _;
// Bernini still-image companion (epic 4699 / sc-5424): the full planner+renderer `Generator`
// self-registers under `bernini` (`Modality::Both`); the image path reaches it via
// `gen_core::load("bernini")` (no direct type contact). Force-link here too — the binary already
// links it for the video path (video_jobs.rs), but anchoring the dependency the image surface
// actually uses keeps the `ModelRegistration` if the video path is ever cfg'd out.
#[cfg(target_os = "macos")]
use mlx_gen_bernini as _;
// candle (Windows/CUDA) backend — epic 3672, sc-3675. Mirror of the mlx `use … as _;` anchors above:
// force the candle SDXL provider to link so its `inventory::submit!` (engine id `sdxl`, backend
// `candle`) survives linker GC and resolves through the SAME gen_core registry — no candle-specific
// dispatch, `cfg(target_os)` just decides which backend registers. Gated on the optional
// `backend-candle` build feature too (the dep is pulled only by the CUDA build); whether candle is
// actually USED at runtime is the separate `backend_candle_enabled` setting, not this link anchor.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_sdxl as _;
// The four candle image families wired in sc-5096 (epic 5095). Same force-link anchor pattern as the
// SDXL crate above + the mlx providers: each self-registers its engine id (`z_image_turbo` /
// `flux1_schnell` + `flux1_dev` / `flux2_klein_9b` / `qwen_image`) into the shared gen_core inventory
// registry, and the `as _;` keeps the MSVC release linker from GC-ing the `inventory::submit!`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_flux as _;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_flux2 as _;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_qwen_image as _;
// Candle Stable Diffusion 3.5 (sc-7880, epic 7982): `sd3_5_large` / `sd3_5_large_turbo` / `sd3_5_medium`
// self-register into the shared gen_core inventory; the `as _;` keeps the MSVC release linker from GC-ing
// the `inventory::submit!` registrations (else `gen_core::load("sd3_5_large")` returns "no generator
// registered"). The Windows/CUDA sibling of the `mlx_gen_sd3` anchor above. Pure txt2img; Q4/Q8 quant.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_sd3 as _;
// Candle Ideogram 4 (sc-6596, epic 6561): `ideogram_4` + `ideogram_4_turbo` self-register into the
// shared gen_core inventory; `as _;` keeps the MSVC release linker from GC-ing the registrations.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_ideogram as _;
// Candle Chroma (sc-5484, epic 3692): chroma1_hd / chroma1_base / chroma1_flash self-register into the
// shared gen_core inventory; the `as _;` keeps the MSVC release linker from GC-ing the registrations.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_chroma as _;
// Candle Boogu-Image-0.1 (sc-7524, epic 6831): `boogu_image` / `boogu_image_turbo` / `boogu_image_edit`
// self-register into the shared gen_core inventory; the `as _;` keeps the MSVC release linker from GC-ing
// the `ModelRegistration`s (else `gen_core::load("boogu_image")` returns "no generator registered"). The
// Windows/CUDA sibling of the `mlx_gen_boogu` anchor above.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_boogu as _;
// Candle Krea 2 (sc-7581, epic 7565 P4): `krea_2_turbo` self-registers into the shared gen_core
// inventory; `as _;` keeps the MSVC release linker from GC-ing the `ModelRegistration`. The
// Windows/CUDA sibling of the `mlx_gen_krea` anchor above.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_krea as _;
// Candle Kolors (sc-5576, epic 3692): the `kolors` T2I id self-registers into the shared gen_core
// inventory; `as _;` keeps the MSVC release linker from GC-ing the registration.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_kolors as _;
// Candle SenseNova-U1 (sc-5576, epic 3692): `sensenova_u1_8b` + `sensenova_u1_8b_fast` self-register
// into the shared gen_core inventory; force-linked so the registrations survive linker GC.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_sensenova as _;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_z_image as _;
// Lens / Lens-Turbo (epic 5107 engine / sc-5126 cutover) — the candle Windows/CUDA sibling of the
// `mlx_gen_lens` anchor above, and the 8th candle image family (effectively). Self-registers `lens`
// (20-step/CFG-5) + `lens_turbo` (4-step/g-1.0) into the shared gen_core inventory registry; the
// FIRST candle family to advertise Q4/Q8 quant + LoRA/LoKr. Force-linked so the MSVC release linker
// keeps the `inventory::submit!` (the dead-strip trap that bit Kolors on MLX).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_lens as _;
// Candle SeedVR2 upscaler (sc-5928, epic 4811 / epic 5482) — the Windows/CUDA sibling of the Mac
// `mlx_gen_seedvr2` anchor above. Self-registers the upscaler ids `seedvr2` / `seedvr2_3b` /
// `seedvr2_7b` into the shared gen_core inventory; the image upscale path reaches it via
// `gen_core::load("seedvr2")` from `upscale_jobs::run_seedvr2_upscale`. Force-linked so the MSVC
// release linker keeps the `inventory::submit!` registrations.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_seedvr2 as _;
// CARVE-OUT(epic 3720): backend-specific; absorbed by FaceEmbedder in Phase 3.
// InstantID (sc-3345) is a bespoke provider, not an inventory-registered `Generator`, so it is
// referenced by name (`InstantId::load`) rather than anchored with `as _;` — and the native face
// stack it composes (`mlx-gen-face`, SCRFD + ArcFace) rides in transitively but is anchored here so
// the direct dep the story adds is meaningful + survives any future unused-crate lint. The
// `mlx_gen::weights::Weights` loader and the `mlx_gen_instantid` API stay mlx-gen-typed until the
// bespoke face stack is lifted onto a neutral FaceEmbedder contract.
#[cfg(target_os = "macos")]
use mlx_gen::weights::Weights;
#[cfg(target_os = "macos")]
use mlx_gen_face as _;
#[cfg(target_os = "macos")]
use mlx_gen_instantid::{
    BodyPoint, InstantId, InstantIdPaths, InstantIdRequest, FACE_RESTORE_PROMPT,
};
// The Windows/CUDA sibling: the candle InstantID provider (sc-5491, epic 5480), retiring the Python
// `_vendor/instantid` off-Mac. Same bespoke by-name reference (`InstantId::load`), NOT inventory-
// registered — so no `as _;` force-link anchor (unlike the registered candle families above). The
// SCRFD + ArcFace FaceEmbedder the model composes (`candle-gen-face`, sc-5490) rides in transitively
// via `candle-gen-instantid` and is used directly (not through the registry), so it needs no direct
// worker dep. The candle `with_face` loads the face pair from THEIR DIRECTORY, so there is no
// `Weights::from_file` import on this lane (the MLX `Weights` loader above stays macOS-only).
// `InstantIdPaths`/`InstantIdRequest`/`BodyPoint` resolve to the candle crate's types, but the
// conditioning types they carry (`WeightsSource`, `Image`, `CancelFlag`, `Progress`) are the SHARED
// `gen_core` contract — the single-rev skew gate (sc-4482) is what makes the worker's `gen_core::Image`
// the exact type `InstantId::generate` consumes.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_instantid::{
    BodyPoint, InstantId, InstantIdPaths, InstantIdRequest, FACE_RESTORE_PROMPT,
};
// SDXL IP-Adapter-Plus reference provider (sc-5488, epic 5480) — the candle (Windows/CUDA) reference-
// conditioning sibling of the InstantID lane, living in `candle-gen-sdxl` (it composes that crate's
// IP-Adapter Resampler + the new CLIP ViT-H image encoder + a pure-IP denoise). Candle-only: macOS keeps
// the MLX SDXL IP path (the registry `SdxlSubMode::Ip`), so these named types resolve only off-Mac.
// `candle_gen_sdxl` is already force-link anchored above (the registered txt2img `sdxl`); this is the
// named-type import the bespoke reference route (`image_jobs/sdxl_ipadapter.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_sdxl::{
    IpAdapterSdxl, IpAdapterSdxlPaths, IpAdapterSdxlRequest, SdxlEdit, SdxlEditPaths,
    SdxlEditRequest,
};
// FLUX.2-klein reference / img2img edit provider (sc-5487, epic 5480) — the candle (Windows/CUDA) FLUX.2
// edit lane (the sibling of the SDXL edit lane above), living in `candle-gen-flux2` (Kontext-style
// reference token-concat over the txt2img FLUX.2 stack + the VAE encoder). Candle-only: macOS keeps the
// MLX `flux2_klein_9b_edit` registry path. `candle_gen_flux2` is already force-link anchored above (the
// registered txt2img `flux2_klein_9b`); this is the named-type import the bespoke edit route
// (`image_jobs/flux2_edit_candle.rs`) drives. The same crate carries the bespoke `Flux2Control`
// (FLUX.2-dev Fun-Controlnet-Union strict-pose VACE branch, sc-7460) the candle pose route
// (`image_jobs/flux2_control_candle.rs`, sc-7736) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_flux2::{
    Flux2Control, Flux2ControlPaths, Flux2ControlRequest, Flux2Edit, Flux2EditPaths,
    Flux2EditRequest,
};
// Kolors IP-Adapter-Plus reference provider (sc-5488, epic 5480) — the candle (Windows/CUDA) Kolors
// sibling of the SDXL IP lane, living in `candle-gen-kolors` (it reuses candle-gen-sdxl's vendored IP
// UNet + the CLIP ViT-L/14-336 image encoder, with the Kolors ChatGLM3 conditioning + leading-Euler
// sampler). Candle-only: macOS keeps the MLX Kolors IP path (the registry `Reference` route), so these
// named types resolve only off-Mac. `candle_gen_kolors` is already force-link anchored above (the
// registered txt2img `kolors`); this is the named-type import the bespoke reference route
// (`image_jobs/kolors_ipadapter.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_kolors::{IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest};
// FLUX XLabs IP-Adapter reference provider (sc-5872, epic 5480) — the candle (Windows/CUDA) FLUX sibling
// of the SDXL/Kolors IP lanes, living in `candle-gen-flux` (the forked FLUX DiT with the per-double-block
// XLabs seam + the pooled CLIP-ViT-L image encoder). Candle-only: macOS keeps the MLX FLUX XLabs IP path
// (epic 3621, the registry `Reference` route). `candle_gen_flux` is already force-link anchored above (the
// registered txt2img `flux1_*`); this is the named-type import the bespoke reference route
// (`image_jobs/flux_ipadapter.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_flux::{IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest};
// Qwen-Image ControlNet (strict pose) provider (sc-5489, epic 5480) — the candle (Windows/CUDA)
// strict-pose lane (the first candle ControlNet family beyond the InstantID SDXL path). Candle-only:
// macOS keeps the MLX `qwen_image_control` registry generator. `candle_gen_qwen_image` is already a
// force-link anchor (`use candle_gen_qwen_image as _;`) from the Qwen txt2img wiring; this is the
// named-type import the bespoke pose route (`image_jobs/qwen_control.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_qwen_image::{QwenControl, QwenControlPaths, QwenControlRequest};
// Qwen-Image-Edit provider (sc-5487, epic 5480) — the candle (Windows/CUDA) reference-edit lane (the
// last family of sc-5487; SDXL + FLUX.2-klein edit already shipped). Candle-only: macOS keeps the MLX
// `qwen_image_edit` registry path. The named-type import the bespoke edit route
// (`image_jobs/qwen_edit_candle.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};
// Kolors ControlNet (strict pose) provider (sc-5489, epic 5480) — the candle (Windows/CUDA) Kolors
// sibling of the Qwen strict-pose lane, living in `candle-gen-kolors` (it reuses candle-gen-sdxl's
// vendored UNet + the SDXL `ControlNet`, with the Kolors ChatGLM3 conditioning + leading-Euler sampler).
// Candle-only: macOS keeps the MLX Kolors ControlNet path. `candle_gen_kolors` is already force-link
// anchored above (the registered txt2img `kolors`); this is the named-type import the bespoke pose route
// (`image_jobs/kolors_control.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_kolors::{KolorsControl, KolorsControlPaths, KolorsControlRequest};
// Z-Image Fun-ControlNet (strict pose) provider (sc-5489, epic 5480) — the candle (Windows/CUDA)
// Z-Image sibling of the Qwen/Kolors strict-pose lanes, living in `candle-gen-z-image` (the VACE-style
// dual-injection control on the vendored DiT). Candle-only: macOS keeps the MLX `z_image_turbo_control`
// registry generator. `candle_gen_z_image` is already force-link anchored above (the registered txt2img
// `z_image_turbo`); this is the named-type import the bespoke pose route (`image_jobs/zimage_control.rs`)
// drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_z_image::{ZImageControl, ZImageControlPaths, ZImageControlRequest};
// Z-Image img2img / edit provider (sc-6595, epic 5480) — the candle (Windows/CUDA) sibling of the MLX
// `z_image_turbo` `Conditioning::Reference` img2img route, living in `candle-gen-z-image` (the Turbo DiT
// + a strength-derived source-latent init). Candle-only: macOS keeps the registered MLX generator's
// img2img path. `candle_gen_z_image` is already force-link anchored above; this is the named-type import
// the bespoke edit route (`image_jobs/zimage_edit_candle.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_z_image::{ZImageEdit, ZImageEditPaths, ZImageEditRequest};
// PuLID-FLUX face-identity provider (sc-5492, epic 5480) — the candle (Windows/CUDA) sibling of the
// macOS `pulid_flux` registry generator, living in `candle-gen-pulid` (the EVA02-CLIP tower + IDFormer
// + the 20 PerceiverAttentionCA modules injected into the forked FLUX DiT via the post-block
// `DitImageInjector` seam, composing the gen-core FaceEmbedder + the BiSeNet `face_features_image`).
// Candle-only: macOS keeps the inventory-registered `pulid_flux` MLX generator; the candle `PulidFlux`
// is a bespoke provider referenced BY NAME (like `InstantId`), so no `as _;` anchor is needed — this is
// the named-type import the bespoke route (`image_jobs/pulid_candle.rs`) drives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_pulid::{PulidFlux, PulidFluxPaths, PulidFluxRequest};

/// The stub adapter id recorded on generated assets (matches the contract fixture
/// `tests/fixtures/rust_migration_contracts/sidecars/asset-image.sceneworks.json`).
const STUB_ADAPTER: &str = "procedural_preview";
/// The adapter id recorded on assets produced by the candle (Windows/CUDA) SDXL lane (sc-3678).
/// Used both per-asset (`generate_candle_stream`) and at the generation-set level (`adapter_id`)
/// so the sidecar + result agree on which backend produced the image.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_ADAPTER: &str = "candle_sdxl";
// Shared by the MLX path and the candle Lens lane (sc-5126) — both cap a job's user LoRAs at 3
// (`resolve_adapters`), so the const is available on the Windows candle build too.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MAX_JOB_LORAS: usize = 5;

// The engine dispatch table + its `ModelRow`/`mlx_model` join moved to the all-targets
// `engines` module (sc-3723); the two descriptor-duplicating flags it used to carry
// (`supports_guidance`/`supports_negative_prompt`) are now read from the linked gen_core
// descriptor via `ResolvedModel`. Shared by the macOS MLX path and the Windows candle lane
// (sc-5096) — the join is backend-neutral, so `generate_candle_stream` resolves repo/steps/guidance
// through the same `mlx_model` lookup the MLX path uses.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::engines::{mlx_model, ResolvedModel};
/// Dispatch handler for `JobType::ImageGenerate`: generate, save, and stream image
/// assets through the Rust GPU worker.
pub(crate) async fn run_image_generate_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    tokio::fs::create_dir_all(project_path.join("assets").join("images")).await?;

    // sc-8091: when the Image Studio "Upscale" toggle is on, each generated image also yields a
    // second "(Nx upscaled)" asset, so the generation set expects twice as many images. The inline
    // upscale post-pass only runs where the upscaler engines compile (macOS / candle); the
    // stub-only build keeps the base count.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    let upscale_mult: u32 = if request.upscale.enabled { 2 } else { 1 };
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let upscale_mult: u32 = 1;

    // Resolve the MLX dispatch branch once, then bake that branch's real total into
    // the plan so the generation set + streamed `expectedCount` match what lands in
    // the gallery.
    #[cfg(target_os = "macos")]
    let route = resolve_image_route(&request, settings);
    #[cfg(target_os = "macos")]
    let plan = ImagePlan::with_count(
        &request,
        route.map_or(request.count, |route| route.image_count(&request, settings)) * upscale_mult,
    );
    // Windows/CUDA candle lane: an InstantID angle/pose set produces N images (the active angle
    // collection's length, or the pose count), not `request.count` — bake the real total into the plan
    // so the generation set + streamed `expectedCount` match (sc-5491, mirroring the macOS route's
    // `image_count`). Any other candle job stays `request.count`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let plan = {
        let count = if settings.backend_candle_enabled && instantid_available(&request, settings) {
            instantid_image_count(&request, settings)
        } else {
            request.count
        };
        ImagePlan::with_count(&request, count * upscale_mult)
    };
    #[cfg(all(
        not(target_os = "macos"),
        not(all(not(target_os = "macos"), feature = "backend-candle"))
    ))]
    let plan = ImagePlan::with_count(&request, request.count * upscale_mult);

    // Pre-flight LoRA family-compat guardrail (sc-3027): reject an incompatible LoRA
    // (e.g. a Flux LoRA on an SDXL model, or a Wan 5B LoRA on the 14B base) before any
    // heavy load, with the same message the Python worker raised — instead of failing
    // deep in the engine's strict adapter loader. Network-type handling (peft LoKr AND third-party
    // LyCORIS both apply on MLX now, epic 3641) is done by routing + `classify_adapter` + the engine.
    sceneworks_core::lora_family::validate_lora_compatibility(
        &request.loras,
        Some(plan.family.as_str()),
        adapter_id(&request),
        Some(request.model.as_str()),
    )
    .map_err(WorkerError::InvalidPayload)?;

    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            &format!("Preparing {} image(s).", plan.image_count),
            None,
            backend,
        ),
    )
    .await?;

    let mut asset_writes: Vec<Value> = Vec::with_capacity(plan.image_count as usize);

    // Real in-process MLX inference on macOS for engine-backed models; otherwise the
    // procedural stub (keeps non-macOS + not-yet-ported models working).
    #[cfg(target_os = "macos")]
    let handled = if let Some(route) = route {
        match route {
            ImageRoute::ZImageControl => {
                // Z-Image strict-pose (advanced.poses) → Fun-Controlnet-Union, one image per pose.
                generate_zimage_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::ZImageBaseControl => {
                // Base (full-CFG) Z-Image strict control (advanced.poses on `z_image`) → base
                // Fun-Controlnet-Union, one image per pose (sc-8251).
                generate_zimage_base_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::QwenControl => {
                // Qwen strict-pose (advanced.poses) → InstantX ControlNet-Union, one image per pose.
                generate_qwen_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::KolorsControl => {
                // Kolors strict-pose (advanced.poses + a reference) → the combined pose ControlNet
                // + IP-Adapter identity + img2img pass (sc-4766 / engine sc-5012), one image per pose.
                generate_kolors_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux1DevControl => {
                // FLUX.1-dev strict control (advanced.poses) → Shakker Union-Pro-2.0, one image per pose
                // (pose / canny / depth via advanced.controlMode).
                generate_flux1_dev_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux2DevControl => {
                // FLUX.2-dev strict-pose (advanced.poses) → Fun-Controlnet-Union, one image per pose.
                generate_flux2_dev_control_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Flux2Edit => {
                // FLUX.2-klein edit/reference (mode edit_image or a reference) → edit variant.
                generate_flux2_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::QwenEdit => {
                // Qwen-Image-Edit (mode edit_image / Character-Studio reference / best-effort
                // pose / angle set) → the engine's `qwen_image_edit` model (sc-3397).
                generate_qwen_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::InstantId => {
                // InstantID identity-preserving character image (sc-3345): single identity or
                // grouped angle/pose sets, on RealVisXL + IdentityNet + the native face stack.
                generate_instantid_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::PulidFlux => {
                // PuLID-FLUX face-identity character image (sc-3344): FLUX.1-dev backbone +
                // EVA/IDFormer/CA injection via the native face stack, one image per seed.
                generate_pulid_flux_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::SdxlAdvanced => {
                // SDXL reference (IP-Adapter) / img2img edit / inpaint / outpaint (epic 3041,
                // sc-3060) → the engine's advanced conditioning paths.
                generate_sdxl_advanced_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::SensenovaEdit => {
                // SenseNova-U1 instruction edit + Character Studio on the unified
                // `sensenova_u1_8b` / `_fast` ids (sc-3900).
                generate_sensenova_edit_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Bernini => {
                // Bernini still-image companion (sc-5424): t2i / i2i on the `bernini_image` id,
                // routed to the same `engine_id:"bernini"` planner+renderer with `frames:1`.
                generate_bernini_image_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
            ImageRoute::Mlx => {
                generate_stream(
                    api,
                    settings,
                    job,
                    &plan,
                    &project_path,
                    backend,
                    &mut asset_writes,
                )
                .await?;
            }
        }
        true
    } else {
        false
    };
    // Windows/CUDA candle execution path (sc-3675, epic 3672). The macOS dispatch above is MLX-bound;
    // candle is a narrow txt2img-only lane, so for a candle-engine model (sdxl/realvisxl) with the
    // backend enabled we run `generate_candle_stream` (same neutral assetWrites/progress/cancellation
    // harness). Gated on `backend_candle_enabled` (default off) so production routing is unchanged
    // until parity is accepted — otherwise it stubs exactly like before.
    // InstantID (sc-5491, epic 5480) is the exception to "txt2img-only": the candle InstantID provider
    // gets its own bespoke path (`generate_instantid_stream`, the off-Mac sibling of the macOS
    // `ImageRoute::InstantId` arm) — checked first since `instantid_realvisxl` is not an inventory
    // `is_candle_engine` id.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let handled = if settings.backend_candle_enabled && instantid_available(&request, settings) {
        generate_instantid_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && sdxl_edit_candle_available(&request, settings) {
        // SDXL img2img / inpaint / outpaint edit (sc-5487) — checked BEFORE `is_candle_engine` because
        // `sdxl`/`realvisxl` ARE candle txt2img ids, so without this an `edit_image` job would be caught
        // by the txt2img branch (which can't honor a source/mask). Disjoint from the IP-Adapter lane
        // below (that one is reference-only and not `edit_image`).
        generate_candle_sdxl_edit_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && flux2_edit_candle_available(&request, settings) {
        // FLUX.2-klein reference / img2img edit (sc-5487) — checked BEFORE `is_candle_engine` because
        // `flux2_klein_9b` IS a candle txt2img id, so without this an `edit_image` job would be caught by
        // the txt2img branch (which can't honor a source reference). FLUX.2-klein has no torch path, so
        // before this an off-Mac klein edit had no real lane (it deferred to a torch worker that lacks
        // the model). Disjoint from the IP-Adapter / SDXL-edit lanes (different model family).
        generate_candle_flux2_edit_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && qwen_edit_candle_available(&request, settings) {
        // Qwen-Image-Edit reference / dual-latent edit (sc-5487) — checked BEFORE `is_candle_engine`.
        // `qwen_image_edit` is its OWN model id (not the `qwen_image` candle txt2img id), so it would
        // not be caught by the txt2img branch; routed here (grouped with the edit lanes) to the bespoke
        // candle `QwenEdit` stream. Off-Mac this was a torch fallback; candle now serves it. Disjoint
        // from the Qwen strict-pose control lane below (that one is `qwen_image` + `advanced.poses`).
        generate_candle_qwen_edit_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && zimage_edit_candle_available(&request, settings) {
        // Z-Image img2img / edit (sc-6595) — checked BEFORE `is_candle_engine` because `z_image_turbo` IS
        // a candle txt2img id (and `z_image_edit` is a distinct id the txt2img branch doesn't know), so
        // without this an `edit_image` job would be caught by the txt2img branch (which can't honor a
        // source) or fall through entirely. Grouped with the other edit lanes; disjoint from the Z-Image
        // strict-pose control lane below (that one is `advanced.poses`, not `edit_image`).
        generate_candle_zimage_edit_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled
        && zimage_identity_candle_available(&request, settings)
    {
        // Z-Image identity-init for Image Studio "With Character" (sc-8409, epic 4406) — checked BEFORE
        // `is_candle_engine` because `z_image_turbo` IS a candle txt2img id, so without this a
        // `character_image` + `referenceAssetId` (+ `referenceStrength > 0`) job would be caught by the
        // txt2img branch and silently drop the reference (carrying no identity, hence no score — the gap
        // this story closes). The off-Mac sibling of the macOS generic lane's Z-Image identity img2img
        // (`resolve_zimage_identity_init`); reuses the candle `ZImageEdit` engine with the identity
        // `referenceAssetId` as the source-latent init + wires the sc-4411 face-likeness scorer. Disjoint
        // from the Z-Image edit lane above (that one is `edit_image` + `sourceAssetId`) and the strict-pose
        // control lane below (that one is `advanced.poses`, which this gate excludes). Mirrors the router's
        // `zimage_identity_candle_eligible`.
        generate_candle_zimage_identity_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && sdxl_ipadapter_available(&request, settings) {
        // SDXL IP-Adapter-Plus reference conditioning (sc-5488) — checked BEFORE `is_candle_engine`
        // because `sdxl`/`realvisxl` ARE candle txt2img ids, so without this a reference job would be
        // caught by the txt2img branch and silently drop the reference.
        generate_candle_sdxl_ipadapter_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && kolors_ipadapter_available(&request, settings) {
        // Kolors IP-Adapter-Plus reference conditioning (sc-5488) — checked BEFORE `is_candle_engine`
        // because `kolors` IS a candle txt2img id, so without this a reference job would be caught by
        // the txt2img branch and silently drop the reference (the SDXL IP reasoning, for Kolors).
        generate_candle_kolors_ipadapter_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && flux_ipadapter_available(&request, settings) {
        // FLUX XLabs IP-Adapter reference conditioning (sc-5872) — checked BEFORE `is_candle_engine`
        // because `flux_dev`/`flux_schnell` ARE candle txt2img ids, so without this a reference job
        // would be caught by the txt2img branch and silently drop the reference (the SDXL/Kolors IP
        // reasoning, for FLUX).
        generate_candle_flux_ipadapter_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && pulid_candle_available(&request, settings) {
        // PuLID-FLUX face identity (sc-5492) — `pulid_flux_dev` is its OWN model id (not a candle
        // txt2img id), so it would never be caught by the `is_candle_engine` branch below; routed here
        // (grouped with the reference/identity lanes) to the bespoke candle `PulidFlux` stream. The
        // distinct model id cleanly disambiguates it from the FLUX XLabs IP-Adapter lane above (both
        // condition on a reference image, but that lane is `flux_dev`/`flux_schnell`).
        generate_candle_pulid_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && qwen_control_available(&request, settings) {
        // Qwen-Image strict-pose ControlNet (sc-5489) — checked BEFORE `is_candle_engine` because
        // `qwen_image` IS a candle txt2img id, so without this a `advanced.poses` job would be caught
        // by the txt2img branch and silently drop the poses (the IP-Adapter reasoning, for poses).
        generate_candle_qwen_control_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && kolors_control_available(&request, settings) {
        // Kolors strict-pose ControlNet (sc-5489) — checked BEFORE `is_candle_engine` because `kolors`
        // IS a candle txt2img id, so without this a `advanced.poses` job would be caught by the txt2img
        // branch and silently drop the poses (the Qwen-control reasoning, for the Kolors family).
        generate_candle_kolors_control_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && zimage_control_available(&request, settings) {
        // Z-Image strict-pose Fun-ControlNet (sc-5489) — checked BEFORE `is_candle_engine` because
        // `z_image_turbo` IS a candle txt2img id, so without this a `advanced.poses` job would be caught
        // by the txt2img branch and silently drop the poses (the Qwen/Kolors-control reasoning, for the
        // Z-Image family — the last of the three strict-pose families).
        generate_candle_zimage_control_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled && flux2_control_candle_available(&request, settings)
    {
        // FLUX.2-dev strict-pose Fun-Controlnet-Union (sc-7736, epic 6564) — `flux2_dev` + `advanced.poses`
        // is the bespoke candle `Flux2Control` lane (`generate_candle_flux2_control_stream`), NOT txt2img —
        // the `is_candle_engine` txt2img branch below would silently drop the poses, and the no-pose-lane
        // reject branch would reject them. Branch it out first (the qwen/kolors/z_image-control reasoning,
        // for the FLUX.2-dev family — the 4th wired strict-pose family). Disjoint from the FLUX.2 edit lane
        // above (that one is `edit_image` + a source; this is `advanced.poses`, not edit mode). Mirrors the
        // router's `flux2_dev_control_candle_eligible`.
        generate_candle_flux2_control_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else if settings.backend_candle_enabled
        && is_candle_engine(&request.model)
        && !matches!(
            request.model.as_str(),
            "qwen_image" | "kolors" | "z_image_turbo" | "flux2_dev"
        )
        && request.mode != "edit_image"
        && !pose_entries(&request).is_empty()
    {
        // No-silent-T2I (sc-5968): a strict-pose job on a candle model with NO pose lane (e.g. sdxl)
        // must be REJECTED with a clear error, not silently rendered as plain txt2img (poses dropped)
        // and not bounced to torch. The candle worker CLAIMS these (jobs_store
        // `image_job_candle_pose_reject`) precisely to fail them loudly here, checked BEFORE the
        // `is_candle_engine` txt2img branch below. SDXL identity-pose ships via InstantID; the wired
        // candle pose families are qwen_image / kolors / z_image_turbo / flux2_dev.
        return Err(WorkerError::InvalidPayload(format!(
            "strict pose (advanced.poses) is not supported for model '{}' on the candle backend — \
             refusing rather than silently generating an unconditioned image (wired candle pose \
             families: qwen_image, kolors, z_image_turbo, flux2_dev; SDXL identity-pose runs via \
             InstantID)",
            request.model
        )));
    } else if settings.backend_candle_enabled && is_candle_engine(&request.model) {
        generate_candle_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
        true
    } else {
        false
    };
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let handled = false;

    // An MLX-routed model id whose weights/snapshot didn't resolve must fail
    // loudly with a precise re-download error instead of completing the job
    // with procedural stub output (sc-4176, epic 3482 "unsupported jobs error
    // loudly"). `mlx_available` is the last dispatch arm, so reaching here
    // with a known engine model means exactly that its weights are unusable.
    // Model ids outside the engine families still stub (test models,
    // not-yet-ported families, non-macOS lanes).
    #[cfg(target_os = "macos")]
    if !handled {
        if let Some(gap) = mlx_weights_gap(&request, settings) {
            return Err(WorkerError::InvalidPayload(gap));
        }
    }

    if !handled {
        generate_stub_stream(
            api,
            settings,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
    }

    // sc-8091: Image Studio "Upscale" toggle. The native worker never ported the Python inline-upscale
    // path, so the UI's `upscale` request was silently dropped (images came out at the base size). Mirror
    // Python: after the base images land, upscale each with the selected engine and append a second
    // "(Nx upscaled)" asset. Gated to where the upscaler engines compile (macOS / candle).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    if request.upscale.enabled {
        apply_inline_upscale(
            api,
            settings,
            http_client,
            job,
            &plan,
            &project_path,
            backend,
            &mut asset_writes,
        )
        .await?;
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let _ = http_client;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Generated {} image(s).", plan.image_count),
            Some(streaming_result(&plan, &asset_writes)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Procedural stub generation (sc-3020): a deterministic per-seed gradient per image.
async fn generate_stub_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    for index in 0..request.count as usize {
        check_cancel(api, &job.id, "Image generation canceled by user.").await?;
        let seed = resolve_seed(request, index);
        let pixels = stub_rgb8(request.width, request.height, seed);
        let fact = write_image_asset(
            plan,
            index,
            seed,
            request.width,
            request.height,
            pixels,
            STUB_ADAPTER,
            stub_raw_settings(request),
            project_path,
        )?;
        asset_writes.push(Value::Object(fact));
        let progress = 0.1 + 0.85 * ((index + 1) as f64 / request.count as f64);
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                progress,
                &format!("Generated image {}/{}.", index + 1, request.count),
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }
    Ok(())
}

/// Per-job invariants shared across every image in the generation set.
pub(crate) struct ImagePlan {
    pub(crate) request: ImageRequest,
    pub(crate) genset_id: String,
    pub(crate) created_at: String,
    pub(crate) family: String,
    pub(crate) slug: String,
    pub(crate) generation_set: Value,
    /// Number of images this job produces. Usually `request.count`, but a FLUX.2 angle
    /// set is 11 and a pose set is the pose count (sc-3030) — the generation set's
    /// `count`/`expectedCount` reflect this so the gallery streams against the real
    /// total, not the requested `count`.
    image_count: u32,
}

impl ImagePlan {
    /// Test-only convenience: a plan whose image count is the request count. Production
    /// always goes through [`ImagePlan::with_count`] (the FLUX.2 angle/pose sets need an
    /// effective count that differs from `request.count`).
    #[cfg(test)]
    fn new(request: &ImageRequest) -> Self {
        Self::with_count(request, request.count)
    }

    /// Build a plan whose generation set reports `image_count` images (see the field).
    pub(crate) fn with_count(request: &ImageRequest, image_count: u32) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let created_at = now_rfc3339();
        let family = resolve_family(request);
        let slug = slugify(&request.prompt, "image", Some(42));
        let generation_set = json!({
            "id": genset_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": image_count,
            "createdAt": created_at,
        });
        Self {
            request: request.clone(),
            genset_id,
            created_at,
            family,
            slug,
            generation_set,
            image_count,
        }
    }
}

/// Save image `index` (its RGB8 `pixels`) under `assets/images/` and return the flat
/// fact the API turns into an indexed asset (every key here is consumed by
/// `build_image_sidecar_parts`). Shared by the stub and real paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_image_asset(
    plan: &ImagePlan,
    index: usize,
    seed: i64,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    adapter: &str,
    raw_settings: JsonObject,
    project_path: &Path,
) -> WorkerResult<JsonObject> {
    let request = &plan.request;
    let rgb_image = image::RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| WorkerError::InvalidPayload("image buffer size mismatch".to_owned()))?;

    let filename = format!(
        "{}_{}_{}_{:04}.png",
        &plan.created_at[..10],
        request.model,
        plan.slug,
        index + 1
    );
    let media_rel = format!("assets/images/{}/{filename}", plan.genset_id);
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = media_path.with_extension("tmp.png");
    rgb_image
        .save_with_format(&temp_path, image::ImageFormat::Png)
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = format!(
        "{} #{}",
        if title.is_empty() {
            "Generated image"
        } else {
            title
        },
        index + 1
    );

    let fact = json!({
        "assetId": fresh_asset_id(),
        "type": "image",
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "width": width,
        "height": height,
        "normalizedWidth": request.width,
        "normalizedHeight": request.height,
        "count": plan.image_count,
        "family": plan.family,
        "seed": seed,
        "index": index,
        "displayName": display_name,
        "createdAt": now_rfc3339(),
        "mode": request.mode,
        "model": request.model,
        "adapter": adapter,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "loras": request.loras,
        "stylePreset": request.style_preset,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "sourceAssetId": request.source_asset_id,
        "rawAdapterSettings": raw_settings,
    });
    Ok(fact.as_object().cloned().expect("json! object literal"))
}

/// Normalise the UI's upscale engine id to the canonical worker id. SeedVR2 stays itself;
/// everything else (`real-esrgan` / `realesrgan` / the dropped `aura-sr` / unknown) maps to
/// Real-ESRGAN, so a bad engine string never hard-fails a whole generation.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn normalize_upscale_engine(engine: &str) -> &'static str {
    match engine.trim().to_ascii_lowercase().as_str() {
        "seedvr2" => "seedvr2",
        _ => "real-esrgan",
    }
}

/// Inline upscale post-pass (sc-8091): upscale every base image the generation produced and append a
/// second "(Nx upscaled)" asset, mirroring the Python worker. Reuses the same in-memory upscalers as the
/// standalone `image_upscale` job — Real-ESRGAN via `ort`, SeedVR2 via the registry generator — provisioning
/// weights on first use. Runs after the base images have already been streamed (so they persist even if a
/// late upscale step errors and fails the job).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn apply_inline_upscale(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let factor: u8 = if request.upscale.factor == 4 { 4 } else { 2 };
    let engine_id = normalize_upscale_engine(&request.upscale.engine);
    let softness = request.upscale.softness();
    // The generate payload carries the *generation* model's manifest, not an upscaler one; pass Null
    // so the weight resolvers fall back to the default HF repos (download-on-first-use).
    let manifest = Value::Null;
    let cancel = CancelFlag::new();

    // Snapshot the base image assets (we append the upscaled variants as we go).
    let base_facts: Vec<JsonObject> = asset_writes
        .iter()
        .filter_map(Value::as_object)
        .filter(|fact| fact.get("type").and_then(Value::as_str) == Some("image"))
        .cloned()
        .collect();
    let total = base_facts.len();

    for (i, base_fact) in base_facts.iter().enumerate() {
        check_cancel(api, &job.id, "Image upscale canceled by user.").await?;

        let media_rel = base_fact
            .get("mediaPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                WorkerError::InvalidPayload("upscale source asset missing mediaPath".to_owned())
            })?;
        let source = crate::image_decode::decode_image_any(project_path.join(media_rel))
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("Upscale source could not be loaded: {error}"))
            })?
            .to_rgb8();
        let seed = base_fact.get("seed").and_then(Value::as_i64).unwrap_or(0);

        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Running,
                ProgressStage::Running,
                0.9,
                &format!(
                    "Upscaling image {}/{total} {factor}x with {engine_id}.",
                    i + 1
                ),
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;

        let upscaled = crate::upscale_jobs::upscale_image_in_memory(
            api,
            settings,
            http_client,
            job,
            &manifest,
            engine_id,
            factor,
            softness,
            seed.max(0) as u64,
            source,
            &cancel,
        )
        .await?;

        let fact = write_upscaled_asset(
            plan,
            base_fact,
            &upscaled,
            engine_id,
            factor,
            softness,
            project_path,
        )?;
        asset_writes.push(Value::Object(fact));
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }
    Ok(())
}

/// Write the upscaled variant of a base image as its own asset (sc-8091): same metadata as the base
/// fact, but a fresh `assetId`, the `_up{factor}x` file, the upscaled dimensions, a "(Nx upscaled)"
/// display-name suffix, and a `rawAdapterSettings.upscale` record (so preset-restore reads it back).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_upscaled_asset(
    plan: &ImagePlan,
    base_fact: &JsonObject,
    upscaled: &image::RgbImage,
    engine_id: &str,
    factor: u8,
    softness: f32,
    project_path: &Path,
) -> WorkerResult<JsonObject> {
    let request = &plan.request;
    let index = base_fact.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let (width, height) = (upscaled.width(), upscaled.height());

    let filename = format!(
        "{}_{}_{}_{:04}_up{factor}x.png",
        &plan.created_at[..10],
        request.model,
        plan.slug,
        index + 1
    );
    let media_rel = format!("assets/images/{}/{filename}", plan.genset_id);
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = media_path.with_extension("tmp.png");
    upscaled
        .save_with_format(&temp_path, image::ImageFormat::Png)
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let base_display = base_fact
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("Generated image");
    let display_name = format!("{base_display} ({factor}x upscaled)");

    // rawAdapterSettings: the base settings + an `upscale` record (mirrors the Python worker so the
    // gallery / preset restore can read back the engine/factor/softness).
    let mut raw_settings = base_fact
        .get("rawAdapterSettings")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let upscale_record = if engine_id == "seedvr2" {
        json!({ "enabled": true, "engine": engine_id, "factor": factor, "softness": softness })
    } else {
        json!({ "enabled": true, "engine": engine_id, "factor": factor })
    };
    raw_settings.insert("upscale".to_owned(), upscale_record);

    let mut fact = base_fact.clone();
    fact.insert("assetId".to_owned(), json!(fresh_asset_id()));
    fact.insert("mediaPath".to_owned(), json!(media_rel));
    fact.insert("width".to_owned(), json!(width));
    fact.insert("height".to_owned(), json!(height));
    fact.insert("displayName".to_owned(), json!(display_name));
    fact.insert("createdAt".to_owned(), json!(now_rfc3339()));
    fact.insert("rawAdapterSettings".to_owned(), Value::Object(raw_settings));
    fact.insert(
        "upscaledFrom".to_owned(),
        base_fact.get("assetId").cloned().unwrap_or(Value::Null),
    );
    Ok(fact)
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet`
/// fact drive `persist_reported_assets` (idempotent per progress update).
fn streaming_result(plan: &ImagePlan, asset_writes: &[Value]) -> JsonObject {
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": plan.image_count,
        "adapter": adapter_id(&plan.request),
        "model": plan.request.model,
        "generationSet": plan.generation_set,
        "assetWrites": asset_writes,
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// The adapter id reported for the set (real engine on macOS for a linked family,
/// else the procedural stub).
fn adapter_id(request: &ImageRequest) -> &'static str {
    #[cfg(target_os = "macos")]
    if let Some(model) = mlx_model(&request.model) {
        return model.adapter_label();
    }
    // Windows/CUDA candle lane (sc-3678, per-engine in sc-5096): report the candle adapter for the
    // wired family so the generation-set fact matches the per-asset `adapter` the candle path writes,
    // instead of falling through to the procedural-stub label. Routing (`worker_supports_job`) only
    // lets candle-eligible txt2img jobs reach this worker, so `is_candle_engine` here implies the
    // candle path ran.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if is_candle_engine(&request.model) {
        return candle_adapter_label(&request.model);
    }
    let _ = request;
    STUB_ADAPTER
}

fn stub_raw_settings(request: &ImageRequest) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(false));
    raw
}

/// The asset `family`: the resolved model manifest entry wins (the UI sends it), else
/// the linked mlx-gen descriptor's family on macOS, else empty.
fn resolve_family(request: &ImageRequest) -> String {
    if let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return family.to_owned();
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(family) = gen_core::registry::generators()
            .find(|registration| (registration.descriptor)().id == request.model)
            .map(|registration| (registration.descriptor)().family)
        {
            return family.to_owned();
        }
    }
    String::new()
}

/// Resolve the seed for image `index`, matching the Python worker's `resolve_seed`:
/// a base `seed` (offset by index) wins, else an explicit per-image seed, else a
/// deterministic `sha256("{prompt}:{index}")` so a re-run reproduces.
pub(crate) fn resolve_seed(request: &ImageRequest, index: usize) -> i64 {
    if let Some(base) = request.seed {
        return base.wrapping_add(index as i64);
    }
    if let Some(seed) = request.seeds.get(index) {
        return *seed;
    }
    let digest = Sha256::digest(format!("{}:{}", request.prompt, index).as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as i64
}

/// Progress payload with the worker's real backend label (the shared
/// `progress_payload` hardcodes `cpu`; the MLX worker reports `mlx`).
pub(crate) fn image_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

pub(crate) fn backend_label(gpu_id: &str) -> &str {
    if gpu_id.trim().is_empty() {
        "cpu"
    } else {
        gpu_id
    }
}

/// First-detection handling for the in-loop image cancel poller (sc-5515): trip the
/// engine `CancelFlag` and post a NON-terminal "Cancelling…" update (indeterminate
/// progress; any completed thumbnails stay via the streamed result). The terminal
/// `Canceled` is posted only after the blocking generation actually stops (see
/// `consume_gen_events`), so the worker row — and therefore the next queued job — is
/// not freed until the GPU is genuinely idle, and the UI honestly shows "Cancelling…"
/// until completion. Best-effort: a failed status update here is non-fatal because the
/// post-run terminal write is what ultimately frees the worker.
//
// Gated to where `consume_gen_events` (its only caller) and the `CancelFlag` import live — the
// `include!`d `base.rs` block — so it isn't compiled (referencing the cfg-gated `CancelFlag`) on
// non-macOS / non-candle builds.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn begin_image_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    plan: &ImagePlan,
    asset_writes: &[Value],
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.0,
            "Cancelling — finishing the current image…",
            Some(streaming_result(plan, asset_writes)),
            backend,
        ),
    )
    .await;
}

/// Deterministic placeholder pixels: a vertical gradient from a per-seed base colour
/// to white, exactly `width * height * 3` RGB8 bytes.
fn stub_rgb8(width: u32, height: u32, seed: i64) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let span = height.saturating_sub(1).max(1) as f32;
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for _ in 0..width {
            buffer.extend_from_slice(&row);
        }
    }
    buffer
}

fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Real in-process MLX inference (macOS, via mlx-gen): Z-Image (sc-3022) +
// FLUX.1 schnell/dev (sc-3023), driven by the engines::MODEL_TABLE dispatch table.
// ---------------------------------------------------------------------------

// Neutral generation harness + MLX routing. The streaming helpers (`start_cached_gen_stream` /
// `consume_gen_events` / `generate_one`) and a few resolvers are backend-neutral and shared by the
// Windows candle lane (sc-3675); the MLX-coupled fns inside (`generate_stream`, the `ResolvedModel`
// resolvers) carry their own `#[cfg(target_os = "macos")]`. So these two includes compile on macOS
// AND on the Windows candle build.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// MLX/candle generator stream helpers.
include!("image_jobs/stream.rs");

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// base image routing (MLX) + neutral txt2img generation harness + the candle execution path.
include!("image_jobs/base.rs");
#[cfg(target_os = "macos")]
// Per-generation PiD (pixel-diffusion) super-resolving decoder routing (epic 7840, sc-7849).
include!("image_jobs/pid.rs");
#[cfg(target_os = "macos")]
// Shared strict-control driver (epic 8236, sc-8243): the `(engine_id, control_repo, supported_kinds)`
// single source of truth + the preprocess (pose/canny/depth/user-passthrough) → `Conditioning::Control`
// core the three MLX registry strict-control paths (zimage/flux2/qwen below) route through.
include!("image_jobs/strict_control.rs");
#[cfg(target_os = "macos")]
// Z-Image strict-pose and prompt augmentation helpers.
include!("image_jobs/zimage.rs");
#[cfg(target_os = "macos")]
// FLUX.2 edit routing and conditioning.
include!("image_jobs/flux2.rs");
#[cfg(target_os = "macos")]
// FLUX.1-dev strict-control (Shakker Union-Pro-2.0) routing.
include!("image_jobs/flux1_control.rs");
#[cfg(target_os = "macos")]
// Qwen control/edit routing.
include!("image_jobs/qwen.rs");
#[cfg(target_os = "macos")]
// SenseNova edit routing.
include!("image_jobs/sensenova.rs");
#[cfg(target_os = "macos")]
// Bernini still-image (t2i/i2i) routing.
include!("image_jobs/bernini.rs");
#[cfg(target_os = "macos")]
// SDXL advanced routing.
include!("image_jobs/sdxl.rs");
#[cfg(target_os = "macos")]
// Kolors advanced conditioning (img2img + IP-Adapter-Plus reference).
include!("image_jobs/kolors.rs");
// InstantID native routing — macOS (MLX) + the Windows/CUDA candle lane (sc-5491). The two engines'
// `InstantId` APIs differ only at the load boundary (with_face dir-vs-Weights, quantize, largest_face
// signature), cfg-split inside; the per-item generate/restore loop is backend-neutral over `gen_core`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
include!("image_jobs/instantid.rs");
// SDXL IP-Adapter-Plus reference conditioning — the Windows/CUDA candle lane ONLY (sc-5488). macOS keeps
// the MLX SDXL IP path (sdxl.rs `SdxlSubMode::Ip`); there is no MLX `IpAdapterSdxl`, so this is
// candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/sdxl_ipadapter.rs");
// SDXL img2img / inpaint / outpaint edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps the
// MLX SDXL advanced path (sdxl.rs `SdxlSubMode::{Edit,Inpaint,Outpaint}`); the candle `SdxlEdit` is a
// bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/sdxl_edit_candle.rs");
// FLUX.2-klein reference / img2img edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps the
// MLX FLUX.2 edit path (flux2.rs `generate_flux2_edit_stream`); the candle `Flux2Edit` is a bespoke
// provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux2_edit_candle.rs");
// Qwen-Image-Edit reference / dual-latent edit — the Windows/CUDA candle lane ONLY (sc-5487). macOS keeps
// the MLX Qwen-Image-Edit path (qwen.rs); the candle `QwenEdit` is a bespoke provider, so this is
// candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/qwen_edit_candle.rs");
// Kolors IP-Adapter-Plus reference conditioning — the Windows/CUDA candle lane ONLY (sc-5488). macOS
// keeps the MLX Kolors IP path (kolors.rs, the registry `Reference` route); the candle `IpAdapterKolors`
// is a bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/kolors_ipadapter.rs");
// FLUX XLabs IP-Adapter reference conditioning — the Windows/CUDA candle lane ONLY (sc-5872). macOS keeps
// the MLX FLUX XLabs IP path (epic 3621, the registry `Reference` route); the candle `IpAdapterFlux` is a
// bespoke provider, so this is candle-exclusive.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux_ipadapter.rs");
// Qwen-Image ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the
// MLX `qwen_image_control` registry generator; the candle `QwenControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/qwen_control.rs");
// Kolors ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the MLX
// Kolors ControlNet path; the candle `KolorsControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/kolors_control.rs");
// Z-Image Fun-ControlNet (strict pose) — the Windows/CUDA candle lane ONLY (sc-5489). macOS keeps the
// MLX `z_image_turbo_control` registry generator; the candle `ZImageControl` is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_control.rs");
// FLUX.2-dev Fun-Controlnet-Union (strict pose) — the Windows/CUDA candle lane ONLY (sc-7736, epic 6564).
// macOS keeps the MLX `flux2_dev_control` registry generator (flux2.rs); the candle `Flux2Control` is a
// bespoke provider (the dev VACE control branch over the Q4 dev DiT).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/flux2_control_candle.rs");
// Z-Image img2img / edit — the Windows/CUDA candle lane ONLY (sc-6595). macOS keeps the MLX
// `z_image_turbo` registry generator's `Conditioning::Reference` img2img path; the candle `ZImageEdit`
// is a bespoke provider.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_edit_candle.rs");
// Z-Image identity-init for Image Studio "With Character" — the Windows/CUDA candle lane ONLY (sc-8409,
// epic 4406). macOS keeps the MLX `z_image_turbo` generic-lane identity img2img (`generate_stream` ⇒
// `resolve_zimage_identity_init`); off-Mac this bespoke lane reuses the candle `ZImageEdit` engine with
// the identity `referenceAssetId` as the source-latent init + wires the sc-4411 face-likeness scorer.
// Reuses the sibling `zimage_edit_candle.rs` base/steps helpers, so it is included right after it.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/zimage_identity_candle.rs");
// PuLID-FLUX face identity — the Windows/CUDA candle lane ONLY (sc-5492). macOS keeps the
// inventory-registered `pulid_flux` MLX generator (image_jobs/pulid.rs); the candle `PulidFlux` is a
// bespoke provider, so this file is candle-gated and distinct from the macOS route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
include!("image_jobs/pulid_candle.rs");
#[cfg(target_os = "macos")]
// PuLID-FLUX native routing.
include!("image_jobs/pulid.rs");
#[cfg(target_os = "macos")]
// image detail tile-ControlNet routing.
include!("image_jobs/detail.rs");

/// Off macOS the in-process engine is unavailable; `image_detail` is served by the Python
/// torch worker (the `mlx` worker — the only one advertising this capability — is macOS-only).
#[cfg(not(target_os = "macos"))]
pub(crate) async fn run_image_detail_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_detail runs on the macOS MLX worker, not this worker".to_owned(),
    ))
}

#[cfg(test)]
mod tests;
