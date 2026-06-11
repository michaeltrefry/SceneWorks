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
//! the [`MLX_MODELS`] table + links its provider crate.

use super::*;
use sceneworks_core::image_request::ImageRequest;

// Force each provider crate to link so its `inventory::submit!` registration survives
// linker GC. Each per-family story adds its provider dep + a matching `use … as _;`.
// See mlx-gen-z-image/tests/registry.rs ("the SceneWorks worker").
#[cfg(target_os = "macos")]
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, Conditioning, ControlKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Progress, Quant, WeightsSource,
};
#[cfg(target_os = "macos")]
use mlx_gen_chroma as _;
#[cfg(target_os = "macos")]
use mlx_gen_flux as _;
#[cfg(target_os = "macos")]
use mlx_gen_flux2 as _;
#[cfg(target_os = "macos")]
use mlx_gen_kolors as _;
#[cfg(target_os = "macos")]
use mlx_gen_qwen_image as _;
#[cfg(target_os = "macos")]
use mlx_gen_sdxl as _;
#[cfg(target_os = "macos")]
use mlx_gen_sensenova as _;
#[cfg(target_os = "macos")]
use mlx_gen_z_image as _;
// InstantID (sc-3345) is a bespoke provider, not an inventory-registered `Generator`, so it is
// referenced by name (`InstantId::load`) rather than anchored with `as _;` — and the native face
// stack it composes (`mlx-gen-face`, SCRFD + ArcFace) rides in transitively but is anchored here so
// the direct dep the story adds is meaningful + survives any future unused-crate lint.
#[cfg(target_os = "macos")]
use mlx_gen::weights::Weights;
#[cfg(target_os = "macos")]
use mlx_gen_face as _;
#[cfg(target_os = "macos")]
use mlx_gen_instantid::{
    BodyPoint, InstantId, InstantIdPaths, InstantIdRequest, FACE_RESTORE_PROMPT,
};

/// The stub adapter id recorded on generated assets (matches the contract fixture
/// `tests/fixtures/rust_migration_contracts/sidecars/asset-image.sceneworks.json`).
const STUB_ADAPTER: &str = "procedural_preview";
#[cfg(target_os = "macos")]
const MAX_JOB_LORAS: usize = 3;

/// One engine-backed image family: how a SceneWorks model id maps onto the linked
/// mlx-gen registry, and the per-variant defaults (all chosen for parity with the
/// Python `MODEL_TARGETS` + the per-family MLX adapter). Adding a family = one row
/// here + its provider crate dep + a `use mlx_gen_<x> as _;` above.
#[cfg(target_os = "macos")]
struct MlxModel {
    /// SceneWorks model id (the job payload `model`).
    sceneworks_id: &'static str,
    /// mlx-gen registry id passed to `mlx_gen::load`.
    engine_id: &'static str,
    /// Default HuggingFace repo when the manifest entry omits `repo`.
    default_repo: &'static str,
    /// Default denoise steps (Python `MODEL_TARGETS[...]["steps"]`).
    default_steps: u32,
    /// Whether the variant accepts a guidance scale. Distilled variants
    /// (z-image-turbo, flux schnell) do not — the engine rejects `guidance` on them.
    supports_guidance: bool,
    /// Default guidance when supported and the request omits it.
    default_guidance: f32,
    /// Whether the variant accepts a negative prompt (true CFG). The guidance-distilled
    /// variants do not — the engine rejects `negative_prompt` on them.
    supports_negative_prompt: bool,
    /// The `adapter` id recorded on generated assets (the Python MLX adapter id).
    adapter_label: &'static str,
}

#[cfg(target_os = "macos")]
const MLX_MODELS: &[MlxModel] = &[
    MlxModel {
        sceneworks_id: "z_image_turbo",
        engine_id: "z_image_turbo",
        default_repo: "Tongyi-MAI/Z-Image-Turbo",
        default_steps: 8,
        supports_guidance: false,
        default_guidance: 0.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_z_image",
    },
    // Z-Image-Edit (epic 3529) — img2img/edit. No dedicated Edit checkpoint exists yet, so
    // (like the Python `MODEL_TARGETS` row) it runs the **Turbo weights** through the engine's
    // img2img path (`Conditioning::Reference` — VAE-encode the source + denoise from
    // `init_time_step(steps, strength)`), so it shares the `z_image_turbo` engine model. The
    // `z_image_turbo` `edit_image` mode resolves to the same img2img call (`resolve_zimage_edit_init`).
    MlxModel {
        sceneworks_id: "z_image_edit",
        engine_id: "z_image_turbo",
        default_repo: "Tongyi-MAI/Z-Image-Turbo",
        default_steps: 8,
        supports_guidance: false,
        default_guidance: 0.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_z_image",
    },
    MlxModel {
        sceneworks_id: "flux_schnell",
        engine_id: "flux1_schnell",
        default_repo: "black-forest-labs/FLUX.1-schnell",
        default_steps: 4,
        supports_guidance: false,
        default_guidance: 0.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_flux",
    },
    MlxModel {
        sceneworks_id: "flux_dev",
        engine_id: "flux1_dev",
        default_repo: "black-forest-labs/FLUX.1-dev",
        default_steps: 28,
        supports_guidance: true,
        default_guidance: 3.5,
        supports_negative_prompt: false,
        adapter_label: "mlx_flux",
    },
    MlxModel {
        // Non-distilled true-CFG base: 20 steps + guidance 4.0 + negative prompt
        // (Python MODEL_TARGETS / MlxQwenAdapter). mlx-gen's own default is 4 steps,
        // so steps are passed explicitly. Edit moves to MLX (sc-3397, the `qwen_image_edit`
        // engine model below); base-Qwen strict-pose ControlNet routes to the
        // `qwen_image_control` engine variant when `advanced.poses` is present
        // (epic 3401 / sc-3575).
        sceneworks_id: "qwen_image",
        engine_id: "qwen_image",
        default_repo: "Qwen/Qwen-Image",
        default_steps: 20,
        supports_guidance: true,
        default_guidance: 4.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_qwen",
    },
    // Qwen-Image-Edit (sc-3397) — the three base edit ids all resolve to the engine's
    // single `qwen_image_edit` model (Reference/MultiReference, true CFG, LoRA/LoKr, Q4/Q8);
    // `qwen_image_edit`/`_2509` alias to the 2511 weights (Python MODEL_TARGETS, sc-2160).
    // 40 steps (engine's own default is 4 — passed explicitly, like the txt2img row). The
    // edit path resolves guidance from `trueCfgScale` (4.0), NOT `guidanceScale`; see
    // `resolve_qwen_edit_guidance`. The `_2511_lightning` distill (4-step, CFG-off) shares
    // these weights but adds the `lightning` sampler + the lightx2v distill LoRA — see the
    // row below and [`qwen_edit_lightning`] (sc-3398).
    MlxModel {
        sceneworks_id: "qwen_image_edit",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 40,
        supports_guidance: true,
        default_guidance: 4.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_qwen",
    },
    MlxModel {
        sceneworks_id: "qwen_image_edit_2509",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 40,
        supports_guidance: true,
        default_guidance: 4.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_qwen",
    },
    MlxModel {
        sceneworks_id: "qwen_image_edit_2511",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 40,
        supports_guidance: true,
        default_guidance: 4.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_qwen",
    },
    // Lightning 4-step distill (sc-3398): same `qwen_image_edit` engine model + base
    // Qwen-Image-Edit-2511 weights as the rows above, but the generate path passes the
    // `lightning` sampler (static-shift schedule + CFG-off single forward) and stacks the
    // lightx2v distill LoRA ahead of any user LoRAs (see [`qwen_edit_lightning`] +
    // [`generate_qwen_edit_stream`]). Python parity (MODEL_TARGETS): 4 steps, guidance 1.0,
    // CFG off — so no negative prompt. The distill LoRA is a CFG-distilled adapter, so the
    // engine runs a single forward/step regardless of `default_guidance`.
    MlxModel {
        sceneworks_id: "qwen_image_edit_2511_lightning",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 4,
        supports_guidance: true,
        default_guidance: 1.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_qwen",
    },
    // FLUX.2-klein (sc-3025) — MLX-only family (no torch fallback). All three SceneWorks
    // variants share the engine's single txt2img model `flux2_klein_9b` (edit + KV-cache
    // are the separate `*_edit`/`*_kv_edit` engine models, story sc-3029); the variants
    // differ only in their weights. Distilled klein runs guidance 1.0 (CFG-free) with no
    // negative prompt; the engine accepts guidance but rejects a negative prompt.
    MlxModel {
        sceneworks_id: "flux2_klein_9b",
        engine_id: "flux2_klein_9b",
        default_repo: "black-forest-labs/FLUX.2-klein-9B",
        default_steps: 4,
        supports_guidance: true,
        default_guidance: 1.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_flux2",
    },
    MlxModel {
        // Separately-distilled checkpoint, same architecture — its snapshot carries the
        // full diffusers tree, so txt2img loads through the base `flux2_klein_9b` loader.
        sceneworks_id: "flux2_klein_9b_kv",
        engine_id: "flux2_klein_9b",
        default_repo: "black-forest-labs/FLUX.2-klein-9b-kv",
        default_steps: 4,
        supports_guidance: true,
        default_guidance: 1.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_flux2",
    },
    MlxModel {
        // wikeeyang community fine-tune (sc-2220/2235): UNDISTILLED, so 24 steps. Its raw
        // repo is single-file (GGUF/safetensors) with no diffusers tree, so it loads from a
        // locally-assembled converted dir via the `modelPath` seam (manifest `modelPath`),
        // NOT the source repo below. The convert step is now native Rust/MLX
        // (mlx_gen_flux2::convert_and_assemble, sc-3136; run by the model_convert job).
        sceneworks_id: "flux2_klein_9b_true_v2",
        engine_id: "flux2_klein_9b",
        default_repo: "wikeeyang/Flux2-Klein-9B-True-V2",
        default_steps: 24,
        supports_guidance: true,
        default_guidance: 1.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_flux2",
    },
    // SDXL (sc-3026) — U-Net, real CFG (negative prompt + guidance 7.0), 30 steps.
    // `sdxl` and the `realvisxl` finetune share the engine's single `sdxl` model
    // (identical arch), differing only in weights. Replaces the in-process
    // _vendor/mlx_sd path. The engine supports Q4/Q8 (the Python vendored path had
    // none); Q8 is the default here (engine-validated; saves ~half the U-Net memory).
    MlxModel {
        sceneworks_id: "sdxl",
        engine_id: "sdxl",
        default_repo: "stabilityai/stable-diffusion-xl-base-1.0",
        default_steps: 30,
        supports_guidance: true,
        default_guidance: 7.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_sdxl",
    },
    MlxModel {
        sceneworks_id: "realvisxl",
        engine_id: "sdxl",
        default_repo: "SG161222/RealVisXL_V5.0",
        default_steps: 30,
        supports_guidance: true,
        default_guidance: 7.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_sdxl",
    },
    // Kolors (epic 3090, sc-3875) — Kwai-Kolors SDXL-architecture U-Net + ChatGLM3-6B text
    // encoder + SDXL VAE, EulerDiscrete sampler. Real CFG (negative prompt + guidance 5.0).
    // Python `MODEL_TARGETS` / `KolorsDiffusersAdapter` parity: 25 steps, guidance 5.0. The engine
    // `kolors` model (sc-3874) supports the full surface — img2img / ControlNet-pose /
    // IP-Adapter-Plus / Q8/Q4 / LoRA/LoKr — but this base row drives plain T2I (+ quant + LoRA)
    // through `generate_mlx_stream`; the advanced conditioning modes are gated to torch by
    // `kolors_mlx_eligible` until their dedicated streams land (subsequent epic-3090 slices).
    MlxModel {
        sceneworks_id: "kolors",
        engine_id: "kolors",
        default_repo: "Kwai-Kolors/Kolors-diffusers",
        default_steps: 25,
        supports_guidance: true,
        default_guidance: 5.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_kolors",
    },
    // Chroma (epic 3531, sc-3843) — FLUX.1-schnell-derived DiT, T5-only conditioning. The engine
    // is a TRUE-CFG family: its descriptor advertises `supports_guidance=false` +
    // `supports_negative_prompt=true`, so the CFG scale is forwarded as `true_cfg` (NOT the
    // distilled `guidance` scalar, which the engine rejects) — see [`uses_true_cfg`] /
    // [`resolve_true_cfg`]. HD/Base are full true-CFG (the manifest pre-fills 40 steps + guidance
    // 3.0; the engine's own defaults are 28 steps + 4.0 — the request carries the manifest values).
    // Each SceneWorks id maps 1:1 to the engine registry id of the same name.
    MlxModel {
        sceneworks_id: "chroma1_hd",
        engine_id: "chroma1_hd",
        default_repo: "lodestones/Chroma1-HD",
        default_steps: 40,
        supports_guidance: false,
        default_guidance: 3.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_chroma",
    },
    MlxModel {
        sceneworks_id: "chroma1_base",
        engine_id: "chroma1_base",
        default_repo: "lodestones/Chroma1-Base",
        default_steps: 40,
        supports_guidance: false,
        default_guidance: 3.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_chroma",
    },
    // Flash is the few-step distilled checkpoint: ~8 steps, CFG baked toward 1.0 (single forward —
    // the negative prompt is effectively inert at true_cfg≈1). It shares the true-CFG descriptor,
    // so `true_cfg` still carries the scale (default 1.0).
    MlxModel {
        sceneworks_id: "chroma1_flash",
        engine_id: "chroma1_flash",
        default_repo: "lodestones/Chroma1-Flash",
        default_steps: 8,
        supports_guidance: false,
        default_guidance: 1.0,
        supports_negative_prompt: true,
        adapter_label: "mlx_chroma",
    },
    // SenseNova-U1 (epic 3180, sc-3900) — NEO-Unify: a dense dual-path Qwen3-MoT AR LLM + a
    // flow-matching image generator (no separate VAE / text encoder). Unlike every other family
    // here it uses BOTH CFG knobs: `supports_guidance=true` carries the text CFG via `guidance`
    // (defaults 4.0 base / 1.0 fast), and `supports_true_cfg` carries the it2i image-guidance via
    // `true_cfg` (edit ≈ 1.0 / character ≈ 1.5) — so it is NOT a [`uses_true_cfg`] family (which is
    // for engines that read the *single* CFG knob from `true_cfg`). `supports_negative_prompt=false`
    // (the descriptor advertises no negative prompt). Plain T2I rides [`generate_mlx_stream`]; edit
    // (`Reference`) + Character Studio (`MultiReference`) divert to [`generate_sensenova_edit_stream`]
    // where the dual CFG + reference conditioning are built. `_fast` is the same base weights with
    // the 8-step distill LoRA merged internally at load (`load_fast`); the worker only selects the
    // engine id, the engine resolves + merges the curated distill LoRA itself (no user LoRA slot —
    // `supports_lora=false`). Both ids map 1:1 to the engine registry id of the same name.
    MlxModel {
        sceneworks_id: "sensenova_u1_8b",
        engine_id: "sensenova_u1_8b",
        default_repo: "sensenova/SenseNova-U1-8B-MoT",
        default_steps: 50,
        supports_guidance: true,
        default_guidance: 4.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_sensenova",
    },
    MlxModel {
        sceneworks_id: "sensenova_u1_8b_fast",
        engine_id: "sensenova_u1_8b_fast",
        default_repo: "sensenova/SenseNova-U1-8B-MoT",
        default_steps: 8,
        supports_guidance: true,
        default_guidance: 1.0,
        supports_negative_prompt: false,
        adapter_label: "mlx_sensenova",
    },
];

/// The engine-backed family for a SceneWorks model id, if any.
#[cfg(target_os = "macos")]
fn mlx_model(sceneworks_id: &str) -> Option<&'static MlxModel> {
    MLX_MODELS
        .iter()
        .find(|model| model.sceneworks_id == sceneworks_id)
}

/// Dispatch handler for `JobType::ImageGenerate`: generate, save, and stream image
/// assets through the Rust GPU worker.
pub(crate) async fn run_image_generate_job(
    api: &ApiClient,
    settings: &Settings,
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

    // Resolve the MLX dispatch branch once, then bake that branch's real total into
    // the plan so the generation set + streamed `expectedCount` match what lands in
    // the gallery.
    #[cfg(target_os = "macos")]
    let route = resolve_image_route(&request, settings);
    #[cfg(target_os = "macos")]
    let plan = ImagePlan::with_count(
        &request,
        route.map_or(request.count, |route| route.image_count(&request, settings)),
    );
    #[cfg(not(target_os = "macos"))]
    let plan = ImagePlan::with_count(&request, request.count);

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
            ImageRoute::Mlx => {
                generate_mlx_stream(
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
    #[cfg(not(target_os = "macos"))]
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
        return model.adapter_label;
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
        if let Some(family) = mlx_gen::registry::generators()
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
// FLUX.1 schnell/dev (sc-3023), driven by the MLX_MODELS table.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
// macOS MLX generator stream helpers.
include!("image_jobs/stream.rs");

#[cfg(target_os = "macos")]
// base MLX image routing and txt2img generation.
include!("image_jobs/base.rs");
#[cfg(target_os = "macos")]
// Z-Image strict-pose and prompt augmentation helpers.
include!("image_jobs/zimage.rs");
#[cfg(target_os = "macos")]
// FLUX.2 edit routing and conditioning.
include!("image_jobs/flux2.rs");
#[cfg(target_os = "macos")]
// Qwen control/edit routing.
include!("image_jobs/qwen.rs");
#[cfg(target_os = "macos")]
// SenseNova edit routing.
include!("image_jobs/sensenova.rs");
#[cfg(target_os = "macos")]
// SDXL advanced routing.
include!("image_jobs/sdxl.rs");
#[cfg(target_os = "macos")]
// InstantID native routing.
include!("image_jobs/instantid.rs");
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
        "image_detail runs on the macOS MLX worker or the Python torch worker, not this Rust worker"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests;
