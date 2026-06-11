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

    // A FLUX.2 angle set produces 11 images and a pose set one per pose, regardless of
    // the requested `count` (sc-3030) — bake the real total into the plan so the
    // generation set + streamed `expectedCount` match what lands in the gallery.
    #[cfg(target_os = "macos")]
    let plan = ImagePlan::with_count(&request, grouped_image_count(&request, settings));
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
    let handled = if zimage_control_available(&request, settings) {
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
        true
    } else if qwen_control_available(&request, settings) {
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
        true
    } else if flux2_edit_available(&request, settings) {
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
        true
    } else if qwen_edit_available(&request, settings) {
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
        true
    } else if instantid_available(&request, settings) {
        // InstantID identity-preserving character image (sc-3345): single identity or the
        // 11-view Character-Studio angle set, on RealVisXL + IdentityNet + the native face
        // stack. Pose-library + faceRestore jobs are NOT eligible (kept on the torch
        // adapter) — `instantid_available` gates them out so they fall through to the
        // non-handled path and the torch worker claims them.
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
    } else if sdxl_advanced_available(&request, settings) {
        // SDXL reference (IP-Adapter) / img2img edit / inpaint / outpaint (epic 3041,
        // sc-3060) → the engine's advanced conditioning paths. Plain SDXL txt2img + LoRA
        // stays on the base `mlx_available` path below.
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
        true
    } else if sensenova_edit_available(&request, settings) {
        // SenseNova-U1 instruction edit (edit_image → Reference) + Character Studio
        // (character_image → MultiReference, incl. the angle set) on the unified
        // `sensenova_u1_8b` / `_fast` ids (sc-3900). Plain SenseNova T2I (no reference)
        // falls through to the base `mlx_available` path below.
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
        true
    } else if mlx_available(&request, settings) {
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

/// Events streamed from the blocking generation thread to the async worker.
#[cfg(target_os = "macos")]
enum GenEvent {
    Step {
        index: usize,
        current: u32,
        total: u32,
    },
    Decoding {
        index: usize,
    },
    Image {
        index: usize,
        seed: i64,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
}

/// True when this job can run real in-process inference: the model is a linked,
/// engine-backed family and its weights resolve locally.
/// Fail-loud gate for the stub fallback (sc-4176): Some(message) when the
/// requested model id is a known MLX engine model but its weights snapshot
/// can't be resolved (partially deleted HF cache, stale refs, missing
/// modelPath). None when the model isn't engine-backed (the stub is its
/// intended path) or the weights resolve.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_weights_gap(request: &ImageRequest, settings: &Settings) -> Option<String> {
    let model = mlx_model(&request.model)?;
    match resolve_weights_dir(request, settings) {
        Ok(Some(_)) => return None,
        Err(error) => return Some(error.to_string()),
        Ok(None) => {}
    }
    Some(format!(
        "{}: MLX weights not found or incomplete (Hugging Face repo {}). \
         Re-download the model in Model Manager, then retry.",
        request.model,
        model_repo(request, model),
    ))
}

#[cfg(target_os = "macos")]
fn mlx_available(request: &ImageRequest, settings: &Settings) -> bool {
    mlx_model(&request.model).is_some()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// The HuggingFace repo for the model: the manifest entry's `repo` wins, else the
/// family default.
#[cfg(target_os = "macos")]
fn model_repo(request: &ImageRequest, model: &MlxModel) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(model.default_repo)
        .to_owned()
}

/// Resolve the weights snapshot directory: an explicit `modelPath` dir wins, else the
/// HuggingFace cache snapshot for the model repo. `None` when the model is not a known
/// engine family or its snapshot is absent.
#[cfg(target_os = "macos")]
pub(crate) fn resolve_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "Image modelPath").map(Some);
    }
    let Some(model) = mlx_model(&request.model) else {
        return Ok(None);
    };
    Ok(huggingface_snapshot_dir(
        &settings.data_dir,
        &model_repo(request, model),
    ))
}

#[cfg(target_os = "macos")]
fn quant_int(value: &Value) -> Option<i64> {
    if value.is_boolean() {
        return None;
    }
    value
        .as_i64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

/// Resolve quantization: `advanced.mlxQuantize` → `manifest.mlx.quantize` → Q8
/// default. mlx-gen supports Q4/Q8; map (<=0 → dense, <=4 → Q4, else Q8). Returns the
/// mlx-gen quant + the effective bit count for the recipe (None = dense bf16).
#[cfg(target_os = "macos")]
fn resolve_quant(request: &ImageRequest) -> (Option<Quant>, Option<i64>) {
    let raw = request
        .advanced
        .get("mlxQuantize")
        .and_then(quant_int)
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(|mlx| mlx.get("quantize"))
                .and_then(quant_int)
        });
    match raw {
        None => (Some(Quant::Q8), Some(8)),
        Some(bits) if bits <= 0 => (None, None),
        Some(bits) if bits <= 4 => (Some(Quant::Q4), Some(4)),
        Some(_) => (Some(Quant::Q8), Some(8)),
    }
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) else the family default.
#[cfg(target_os = "macos")]
fn resolve_steps(request: &ImageRequest, model: &MlxModel) -> u32 {
    request
        .advanced
        .get("steps")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|steps| (steps as u32).clamp(1, 80))
        .unwrap_or(model.default_steps)
}

/// Resolve the guidance scale. Distilled variants (z-image-turbo, flux schnell) take
/// no guidance — the engine rejects `Some(_)` on them — so this returns `None`. For a
/// guided variant (flux dev) it is `advanced.guidanceScale` else the family default.
#[cfg(target_os = "macos")]
fn resolve_guidance(request: &ImageRequest, model: &MlxModel) -> Option<f32> {
    if !model.supports_guidance {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance);
    Some(scale)
}

/// True for a TRUE-CFG family whose engine reads the CFG scale from `true_cfg` (with a real
/// negative prompt) and **rejects** the distilled `guidance` scalar — i.e. Chroma (epic 3531),
/// uniquely identified by `supports_guidance=false` + `supports_negative_prompt=true`. The
/// guidance-distilled families (`z_image_turbo`, `flux_schnell`) are `false`/`false` (no CFG at
/// all), and the `guidance`-scalar families (qwen / sdxl / flux2 …) are `true`/*. For a true-CFG
/// family the worker forwards `advanced.guidanceScale` as `true_cfg`, not `guidance`.
#[cfg(target_os = "macos")]
fn uses_true_cfg(model: &MlxModel) -> bool {
    !model.supports_guidance && model.supports_negative_prompt
}

/// Resolve the true-CFG scale for a true-CFG family (Chroma). `None` for every other family
/// (their CFG, if any, flows through [`resolve_guidance`]). The scale is `advanced.guidanceScale`
/// (the same user knob) else the family default — forwarded to the engine as `GenerationRequest.true_cfg`.
#[cfg(target_os = "macos")]
fn resolve_true_cfg(request: &ImageRequest, model: &MlxModel) -> Option<f32> {
    if !uses_true_cfg(model) {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance);
    Some(scale)
}

/// The negative prompt to pass to the engine. `None` for variants without true CFG
/// (the engine rejects `negative_prompt` on the distilled families) and for an empty
/// prompt (the true-CFG engines fall back to their own neutral negative).
#[cfg(target_os = "macos")]
fn resolve_negative_prompt(request: &ImageRequest, model: &MlxModel) -> Option<String> {
    if !model.supports_negative_prompt {
        return None;
    }
    let trimmed = request.negative_prompt.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// First non-empty of installedPath/sourcePath/path/source.path on a LoRA spec.
#[cfg(target_os = "macos")]
pub(crate) fn lora_path(lora: &Value) -> Option<PathBuf> {
    for key in ["installedPath", "sourcePath", "path"] {
        if let Some(value) = lora
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(PathBuf::from(value));
        }
    }
    lora.get("source")
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Classify a LoRA file into the mlx-gen adapter `kind`. SceneWorks peft-LoKr (stamped
/// `networkType: lokr`) → `Lokr` (the engine's metadata-gated `apply_lokr` peft path). Everything
/// else → `Lora`, INCLUDING third-party LyCORIS (LoHa / kohya non-peft LoKr): since epic 3641
/// (sc-3642/3643/3671) the engine's `apply_adapter_specs_autoprefix` detects `lokr_*` / `hada_*`
/// keys by sniff and routes them to its third-party reconstruction regardless of the declared kind,
/// so `Lora` is the correct hint and the worker no longer rejects them. (A LyCORIS algo the engine
/// doesn't implement — e.g. (IA)³/OFT — has no `lokr_*`/`hada_*` keys, so the engine's LoRA loader
/// finds nothing and surfaces a loud "matched nothing" error rather than mis-applying.)
#[cfg(target_os = "macos")]
pub(crate) fn classify_adapter(file: &Path) -> WorkerResult<AdapterKind> {
    let header = read_safetensors_header(file)
        .map_err(|error| WorkerError::InvalidPayload(format!("LoRA header: {error}")))?;
    let network_type = header
        .get("__metadata__")
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    if network_type.as_deref() == Some("lokr") {
        return Ok(AdapterKind::Lokr);
    }
    Ok(AdapterKind::Lora)
}

/// Resolve up to 3 request LoRAs into mlx-gen adapter specs (path + scale + kind).
#[cfg(target_os = "macos")]
fn resolve_adapters(request: &ImageRequest) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = if path.is_dir() {
            first_safetensors_path(&path).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "LoRA has no .safetensors under {}",
                    path.display()
                ))
            })?
        } else {
            path
        };
        if !file.exists() {
            return Err(WorkerError::InvalidPayload(format!(
                "LoRA file is missing: {}",
                file.display()
            )));
        }
        let kind = classify_adapter(&file)?;
        let scale = lora
            .get("weight")
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            .unwrap_or(0.8) as f32;
        specs.push(AdapterSpec::new(file, scale, kind));
    }
    Ok(specs)
}

#[cfg(target_os = "macos")]
fn mlx_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Distilled variants run without CFG (guidance == None → null in the recipe).
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw
}

/// Load the generator for `engine_id` (heavy; once per job).
#[cfg(target_os = "macos")]
fn mlx_load(
    engine_id: &str,
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    mlx_load_with_ip(engine_id, weights_dir, quant, adapters, None)
}

/// As [`mlx_load`], but optionally installing an IP-Adapter from `ip_adapter_dir`
/// (`LoadSpec::with_ip_adapter`). Used by the FLUX.1 XLabs IP-Adapter reference path
/// (epic 3621): the engine then treats a `Conditioning::Reference` as the image prompt.
#[cfg(target_os = "macos")]
fn mlx_load_with_ip(
    engine_id: &str,
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> WorkerResult<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    if let Some(dir) = ip_adapter_dir {
        spec = spec.with_ip_adapter(WeightsSource::Dir(dir));
    }
    mlx_gen::load(engine_id, &spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("{engine_id} load failed: {error}")))
}

/// XLabs FLUX IP-Adapter repos (epic 3621). The torch `flux_dev` path already declares +
/// downloads these (the `ipAdapter` block in `image_adapters`); the MLX path reuses the same
/// HF-cache snapshots — there is no new weight to ship.
#[cfg(target_os = "macos")]
const FLUX_IP_ADAPTER_REPO: &str = "XLabs-AI/flux-ip-adapter";
#[cfg(target_os = "macos")]
const FLUX_IP_IMAGE_ENCODER_REPO: &str = "openai/clip-vit-large-patch14";
/// IP-Adapter scale when the request omits `ipAdapterScale` (XLabs resemblance tier 0.7, matching
/// the torch `FluxDiffusersAdapter`).
#[cfg(target_os = "macos")]
const FLUX_IP_SCALE: f32 = 0.7;
/// `trueCfgScale` default for the FLUX.1-dev IP-Adapter path (real CFG; torch default ~4.0).
#[cfg(target_os = "macos")]
const FLUX_IP_TRUE_CFG: f32 = 4.0;

/// The FLUX.1 engine families that carry the XLabs IP-Adapter (both variants — the Rust engine has
/// no diffusers `load_ip_adapter` schnell limitation).
#[cfg(target_os = "macos")]
fn is_flux_model(model: &str) -> bool {
    matches!(model, "flux_schnell" | "flux_dev")
}

/// The SenseNova-U1 SceneWorks ids (base + 8-step distill), both served by the unified
/// `mlx-gen-sensenova` engine (sc-3900).
#[cfg(target_os = "macos")]
fn is_sensenova_model(model: &str) -> bool {
    matches!(model, "sensenova_u1_8b" | "sensenova_u1_8b_fast")
}

/// Stage the engine's IP-Adapter dir contract from the two cached HF snapshots:
/// `<staged>/ip_adapter.safetensors` (XLabs) + `<staged>/image_encoder/model.safetensors`
/// (openai CLIP-ViT-L). Errors loudly if either snapshot is missing — mirrors the SDXL IP path
/// (`resolve_ip_adapter_dir`); the repos reach the cache via the model-download flow / the torch
/// `flux_dev` path, not a new provisioning step.
#[cfg(target_os = "macos")]
fn resolve_flux_ip_adapter_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    let missing = || {
        WorkerError::InvalidPayload(format!(
            "FLUX IP-Adapter weights not found (download {FLUX_IP_ADAPTER_REPO} + {FLUX_IP_IMAGE_ENCODER_REPO})."
        ))
    };
    let adapter_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_ADAPTER_REPO)
            .ok_or_else(missing)?;
    let clip_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_IMAGE_ENCODER_REPO)
            .ok_or_else(missing)?;
    let ip_file = adapter_snap.join("ip_adapter.safetensors");
    let clip_file = clip_snap.join("model.safetensors");
    if !ip_file.exists() || !clip_file.exists() {
        return Err(missing());
    }
    let staged = settings.data_dir.join("staged").join("flux-ip-adapter");
    let encoder_dir = staged.join("image_encoder");
    std::fs::create_dir_all(&encoder_dir)
        .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter dir: {e}")))?;
    // Re-link each call: the HF-cache targets are immutable, so a stable staged dir is reusable.
    let link = |src: &Path, dst: PathBuf| -> WorkerResult<()> {
        let _ = std::fs::remove_file(&dst);
        std::os::unix::fs::symlink(src, &dst)
            .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter link: {e}")))
    };
    link(&ip_file, staged.join("ip_adapter.safetensors"))?;
    link(&clip_file, encoder_dir.join("model.safetensors"))?;
    Ok(staged)
}

/// Emit an `image_pipeline_load_{start,complete}` event from inside a blocking
/// generation closure (sc-3450), parity with the Python worker's pipeline-load
/// events. On the MLX path `mlx_gen::load` is a single atomic call that also fuses
/// any distill LoRA and applies user LoRAs (`spec.with_adapters`), so there is no
/// separable fuse/apply step to bracket: the adapter total (`adapter_count` =
/// distill + user) is reported here instead of via the torch worker's separate
/// `image_distill_lora_fuse_*` / `image_lora_apply_*` sub-phase events. A `start`
/// with no matching `complete` means the load failed (the error propagates via `?`).
#[cfg(target_os = "macos")]
pub(crate) fn emit_load_event(event: &str, job_id: &str, engine: &str, adapter_count: usize) {
    emit_event(
        event,
        json!({
            "jobId": job_id,
            "engine": engine,
            "adapterCount": adapter_count,
        }),
    );
}

/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
///
/// `reference` is the optional identity img2img-init (sc-3619): `(image, strength)` adds a
/// `Reference` conditioning that seeds the denoise from the reference latents — the plain
/// (no-ControlNet) Z-Image reference-without-pose path, reusing the same engine img2img the
/// strict-pose tier already drives. `None` → plain txt2img.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn mlx_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    true_cfg: Option<f32>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let conditioning = match reference {
        Some((image, strength)) => vec![Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        }],
        None => Vec::new(),
    };
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::InvalidPayload(format!("generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload("generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "generator returned non-image output".to_owned(),
        )),
    }
}

/// Within-image step fraction mapped into the 0.10..0.95 generation band.
#[cfg(target_os = "macos")]
fn step_fraction(index: usize, current: u32, total: u32, count: u32) -> f64 {
    let per = 0.85 / count.max(1) as f64;
    let within = if total > 0 {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.1 + per * (index as f64 + within)).min(0.95)
}

/// Real MLX generation: load once on a blocking thread, generate each image, and
/// stream step/decode/image events back to the async worker (which saves PNGs, emits
/// `assetWrites`, and polls cancel). MLX runs entirely on the blocking thread (the
/// `Box<dyn Generator>` is `!Send` and the MLX device is single-thread).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_mlx_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    let guidance = resolve_guidance(request, model);
    // True-CFG families (Chroma) carry the CFG scale in `true_cfg`, not `guidance` (which their
    // engine rejects); `None` for every other family. The recipe records the effective CFG knob.
    let model_true_cfg = resolve_true_cfg(request, model);
    let negative_prompt = resolve_negative_prompt(request, model);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, model);
    let raw_settings = mlx_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance.or(model_true_cfg),
    );
    let engine_id = model.engine_id;
    let adapter_label = model.adapter_label;
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();
    // Reference conditioning for the base MLX path, resolved once (constant across the set):
    //  • Z-Image reference-identity img2img-init (sc-3619), and
    //  • FLUX.1 XLabs IP-Adapter (epic 3621 — both schnell + dev; `strength = ipAdapterScale`, plus
    //    real CFG via `trueCfgScale` on dev). Qwen/SDXL reference divert to their own advanced
    //    branches before reaching here.
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let (identity_init, flux_ip_dir, flux_true_cfg): (
        Option<(Image, f32)>,
        Option<PathBuf>,
        Option<f32>,
    ) = if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // Z-Image base path: `edit_image` → img2img-edit (sourceAssetId + strength, epic 3529);
        // otherwise the identity-init reference (referenceAssetId + referenceStrength, sc-3619).
        // Both feed the engine's single `Reference` conditioning; only the source + strength
        // keying differs. The strict-pose ControlNet tier diverts earlier (zimage_control_available).
        let init = if request.mode == "edit_image" {
            resolve_zimage_edit_init(request, settings, project_path)?
        } else {
            resolve_zimage_identity_init(request, settings, project_path)?
        };
        (init, None, None)
    } else if is_flux_model(&request.model) && has_reference && request.mode != "edit_image" {
        let reference_id = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &reference_id,
            project_path,
        )?;
        let scale = advanced::f32_clamped(
            &request.advanced,
            "ipAdapterScale",
            FLUX_IP_SCALE,
            0.0..=1.0,
        );
        let ip_dir = resolve_flux_ip_adapter_dir(settings)?;
        // Real CFG only on dev (schnell is distilled — no CFG).
        let true_cfg = (request.model == "flux_dev").then(|| {
            advanced::f32_clamped(
                &request.advanced,
                "trueCfgScale",
                FLUX_IP_TRUE_CFG,
                1.0..=10.0,
            )
        });
        (Some((image, scale)), Some(ip_dir), true_cfg)
    } else {
        (None, None, None)
    };
    // The CFG scale passed to the engine as `true_cfg`: the FLUX.1-dev reference path's scale if
    // present, otherwise the true-CFG family scale (Chroma). `None` for the guidance-scalar and
    // distilled families, which carry CFG (if any) through `guidance` instead.
    let true_cfg = flux_true_cfg.or(model_true_cfg);

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let prompt = request.prompt.clone();
        let (width, height) = (request.width, request.height);
        let seeds = seeds.clone();
        let cancel = cancel.clone();
        let job_id = job.id.clone();
        // `identity_init` + `flux_ip_dir` + `true_cfg` are moved into the closure (the `move`
        // captures them by value).
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let adapter_count = adapters.len();
            emit_load_event(
                "image_pipeline_load_start",
                &job_id,
                engine_id,
                adapter_count,
            );
            let generator = mlx_load_with_ip(engine_id, weights_dir, quant, adapters, flux_ip_dir)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                engine_id,
                adapter_count,
            );
            for (index, seed) in seeds.into_iter().enumerate() {
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (width, height, pixels) = mlx_generate_one(
                    generator.as_ref(),
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    negative_prompt.clone(),
                    identity_init.as_ref(),
                    true_cfg,
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width,
                        height,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Consume the streamed generation events (step / decoding / image) from the blocking
/// thread: write each finished image as an asset fact, stream progress, and poll cancel
/// ~every 2s (draining the channel after a cancel so the blocking sender never blocks).
/// Shared by the base txt2img path ([`generate_mlx_stream`]) and the Z-Image strict-pose
/// control path ([`generate_zimage_control_stream`]). `total` is the number of images
/// the job produces (the request count, or the pose count).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn consume_gen_events(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    adapter_label: &str,
    raw_settings: &JsonObject,
    total: usize,
    mut rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let total_u32 = total as u32;
    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    // Per-image inference lifecycle events (sc-3450), parity with the Python worker's
    // `image_inference_start`/`image_inference_complete`. The first event for an index
    // marks its start; `GenEvent::Image` marks completion. This is the single shared
    // streaming seam, so every MLX image family reports the same phases on mlx-worker.log
    // + the in-app Logs screen.
    let mut started: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut mark_started = |index: usize| {
        if started.insert(index) {
            emit_event(
                "image_inference_start",
                json!({
                    "jobId": job.id,
                    "imageIndex": index,
                    "imageCount": total,
                    "backend": backend,
                }),
            );
        }
    };
    // Heartbeat + cancel-poll on a fixed interval, not only when the blocking
    // thread emits an event. The cold model-load phase (multi-GB load + quantize)
    // emits nothing, so without an interval arm the worker reports no Busy
    // heartbeat and honors no cancel until the first denoise step — long enough
    // for the API's staleness check to think it died (sc-4276 / F-MLXW-12;
    // mirrors the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                if canceled {
                    continue; // drain remaining events so the blocking sender never blocks.
                }
                match event {
            GenEvent::Step {
                index,
                current,
                total: step_total,
            } => {
                mark_started(index);
                if last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    if cancel_requested(api, &job.id, "Image generation canceled by user.").await {
                        cancel.cancel();
                        canceled = true;
                        continue;
                    }
                }
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, current, step_total, total_u32),
                        &format!("Image {}/{total} — step {current}/{step_total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Decoding { index } => {
                mark_started(index);
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, 1, 1, total_u32),
                        &format!("Image {}/{total} — decoding.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Image {
                index,
                seed,
                width,
                height,
                pixels,
            } => {
                let fact = write_image_asset(
                    plan,
                    index,
                    seed,
                    width,
                    height,
                    pixels,
                    adapter_label,
                    raw_settings.clone(),
                    project_path,
                )?;
                asset_writes.push(Value::Object(fact));
                emit_event(
                    "image_inference_complete",
                    json!({
                        "jobId": job.id,
                        "imageIndex": index,
                        "backend": backend,
                    }),
                );
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / total as f64),
                        &format!("Generated image {}/{total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                if !canceled && last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    if cancel_requested(api, &job.id, "Image generation canceled by user.").await {
                        cancel.cancel();
                        canceled = true;
                    }
                }
            }
        }
    }

    let task_result = blocking
        .await
        .map_err(|error| WorkerError::InvalidPayload(format!("generation task join: {error}")))?;
    if canceled {
        // check_cancel already posted the Canceled update; treat the (likely) generate
        // error as the clean cancel.
        return Err(WorkerError::Canceled(
            "Image generation canceled by user.".to_owned(),
        ));
    }
    task_result
}

// ---------------------------------------------------------------------------
// Z-Image strict-pose ControlNet (macOS, sc-3028): the Fun-Controlnet-Union
// `z_image_turbo_control` variant. One image per pose, each driven by a DWPose
// skeleton rendered from the pose's keypoints (see `openpose_skeleton`).
// ---------------------------------------------------------------------------

/// The engine registry id for the Z-Image Fun-Controlnet-Union variant.
#[cfg(target_os = "macos")]
const ZIMAGE_CONTROL_ENGINE_ID: &str = "z_image_turbo_control";
/// Default Fun-Controlnet-Union control-weights repo + file (sc-2257 parity).
#[cfg(target_os = "macos")]
const ZIMAGE_CONTROL_REPO: &str = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1";
#[cfg(target_os = "macos")]
const ZIMAGE_CONTROL_FILE: &str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors";

/// The object-shaped `advanced.poses` entries (the strict-pose tier; empty otherwise).
#[cfg(target_os = "macos")]
fn pose_entries(request: &ImageRequest) -> Vec<&Value> {
    request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .map(|poses| poses.iter().filter(|pose| pose.is_object()).collect())
        .unwrap_or_default()
}

/// True when this is a Z-Image strict-pose job (z-image + ≥1 pose) whose base weights
/// resolve — routed to the Fun-Controlnet-Union control path rather than plain txt2img.
/// Control-weights presence is checked in the stream so a missing checkpoint errors
/// loudly instead of silently dropping the poses to the txt2img path.
#[cfg(target_os = "macos")]
fn zimage_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "z_image_turbo"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the Fun-Controlnet-Union checkpoint (`advanced.controlWeights.{repo,filename}`
/// else defaults) to a single `.safetensors` in the HF cache. `None` when absent (the
/// model-download flow fetches it ahead of generation, like base weights).
#[cfg(target_os = "macos")]
fn resolve_control_weights_for(
    request: &ImageRequest,
    settings: &Settings,
    default_repo: &'static str,
    default_file: &'static str,
) -> Option<PathBuf> {
    let control = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let str_field = |key: &str, default: &'static str| -> String {
        control
            .and_then(|control| control.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    let repo = str_field("repo", default_repo);
    let filename = str_field("filename", default_file);
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, &repo)?;
    let path = snapshot.join(filename);
    path.exists().then_some(path)
}

/// Resolve the Z-Image Fun-Controlnet-Union checkpoint.
#[cfg(target_os = "macos")]
fn resolve_control_weights(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    resolve_control_weights_for(request, settings, ZIMAGE_CONTROL_REPO, ZIMAGE_CONTROL_FILE)
}

/// Pose ControlNet lock strength: `advanced.controlScale` (default 0.9, clamp [0,2]).
#[cfg(target_os = "macos")]
fn resolve_control_scale(request: &ImageRequest) -> f32 {
    request
        .advanced
        .get("controlScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(0.9)
        .clamp(0.0, 2.0)
}

/// A pose's parsed keypoints, ready for [`crate::openpose_skeleton::draw_wholebody`].
#[cfg(target_os = "macos")]
struct PoseInput {
    keypoints: Vec<crate::openpose_skeleton::Keypoint>,
    hands: Option<Vec<crate::openpose_skeleton::Hand>>,
    face: Option<Vec<crate::openpose_skeleton::Keypoint>>,
}

#[cfg(target_os = "macos")]
fn parse_poses(request: &ImageRequest) -> Vec<PoseInput> {
    use crate::openpose_skeleton::{normalize_face, normalize_hands, normalize_keypoints};
    pose_entries(request)
        .into_iter()
        .map(|entry| PoseInput {
            keypoints: entry
                .get("keypoints")
                .map(normalize_keypoints)
                .unwrap_or_else(|| vec![None; 18]),
            hands: entry.get("hands").and_then(normalize_hands),
            face: entry.get("face").and_then(normalize_face),
        })
        .collect()
}

/// Load the Z-Image Fun-Controlnet-Union generator (base snapshot + control overlay).
#[cfg(target_os = "macos")]
fn zimage_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    mlx_gen::load(ZIMAGE_CONTROL_ENGINE_ID, &spec).map_err(|error| {
        WorkerError::InvalidPayload(format!("Z-Image control load failed: {error}"))
    })
}

/// Generate one strict-pose image: the `control` skeleton drives the Fun-Controlnet-Union
/// pose branch at `control_scale`. Z-Image-Turbo is guidance-distilled (no CFG / negative).
///
/// `reference` is the optional identity img2img-init shared across the pose set (sc-3146):
/// `(image, strength)` adds a `Reference` conditioning next to the required `Control`, seeding
/// the denoise from the reference latents. `strength` is the engine's img2img strength (mflux
/// `image_strength` convention: higher = more init kept). `None` → the pose-only tier.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn zimage_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    control: Image,
    control_scale: f32,
    reference: Option<&(Image, f32)>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut conditioning = vec![Conditioning::Control {
        image: control,
        kind: ControlKind::Pose,
        scale: control_scale,
    }];
    if let Some((image, strength)) = reference {
        conditioning.push(Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        });
    }
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::InvalidPayload(format!("control generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload("control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "control generator returned non-image output".to_owned(),
        )),
    }
}

#[cfg(target_os = "macos")]
fn zimage_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Z-Image-Turbo is guidance-distilled — no CFG.
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw
}

/// Real Z-Image strict-pose generation: one image per pose, each conditioned on a DWPose
/// skeleton rendered from the pose keypoints + locked by the Fun-Controlnet-Union branch.
/// Mirrors [`generate_mlx_stream`]'s blocking-thread + streamed-events shape (the MLX
/// generator is `!Send` + single-thread), reusing [`consume_gen_events`].
#[cfg(target_os = "macos")]
async fn generate_zimage_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Identity img2img-init (sc-2328 / sc-3146) — OPT-IN escape hatch, off by default. The
    // Fun-Controlnet-Union pose head denoises the pose FROM NOISE, so seeding from a reference
    // init fights the pose lock on few-step Turbo (validated marginal on 8-step Turbo; no single
    // strength holds BOTH identity and pose). It engages only when advanced.referenceStrength > 0
    // AND a referenceAssetId is present — parity with `MlxZImageAdapter._identity_init_requested`.
    // The reference is shared across the whole pose set (identity is constant; only the per-pose
    // skeleton changes). None → the pose-only tier (the validated sc-2257 default).
    let identity_init = resolve_zimage_identity_init(request, settings, project_path)?;

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Z-Image weights not found".to_owned()))?;
    let control_weights = resolve_control_weights(request, settings).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Z-Image strict-pose control weights not found (download {ZIMAGE_CONTROL_REPO})."
        ))
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let zimage = mlx_model("z_image_turbo")
        .ok_or_else(|| WorkerError::InvalidPayload("z-image model row missing".to_owned()))?;
    let steps = resolve_steps(request, zimage);
    let control_scale = resolve_control_scale(request);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, zimage);
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings =
        zimage_control_raw_settings(request, &repo, steps, quant_bits, control_scale, count);
    // Strict pose shares one seed across the set so noise-derived attributes (hair,
    // wardrobe, lighting) stay constant while only the pose changes (Python parity).
    let seed = resolve_seed(request, 0);

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let prompt = request.prompt.clone();
        let (width, height) = (request.width, request.height);
        let cancel = cancel.clone();
        let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let adapter_count = adapters.len();
            emit_load_event(
                "image_pipeline_load_start",
                &job_id,
                ZIMAGE_CONTROL_ENGINE_ID,
                adapter_count,
            );
            let generator = zimage_control_load(weights_dir, control_weights, quant, adapters)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                ZIMAGE_CONTROL_ENGINE_ID,
                adapter_count,
            );
            let identity_init = identity_init.as_ref();
            for (index, pose) in poses.into_iter().enumerate() {
                let skeleton = crate::openpose_skeleton::draw_wholebody(
                    width,
                    height,
                    &pose.keypoints,
                    pose.hands.as_deref(),
                    pose.face.as_deref(),
                    stickwidth,
                );
                let control = Image {
                    width,
                    height,
                    pixels: skeleton.into_raw(),
                };
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (width, height, pixels) = zimage_control_generate_one(
                    generator.as_ref(),
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    control,
                    control_scale,
                    identity_init,
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width,
                        height,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        ZIMAGE_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// The clamped identity img2img-init strength for the Z-Image strict-pose set, or `None` for the
/// pose-only tier (sc-3146). `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty
/// `referenceAssetId` is present — parity with `MlxZImageAdapter._identity_init_requested`. The
/// strict-pose stream always carries poses (`zimage_control_available`), so the
/// bare-reference-without-poses rejection is handled upstream; here a `referenceStrength` set
/// without an asset simply falls back to pose-only, matching the Python gate rather than erroring.
///
/// `strength` is the user value clamped to `[0.05, 1.0]` and carries the mflux `image_strength`
/// convention **verbatim** (no numeric inversion): the mlx-gen Z-Image control engine and mflux
/// agree — higher strength → later denoise start (`init_time_step`) → output stays closer to the
/// init. Mirrors `MlxZImageAdapter._reference_strength` + the sidecar's verbatim forward. Pure
/// (request only) so the parity-sensitive gate + clamp are unit-testable without asset I/O.
#[cfg(target_os = "macos")]
fn zimage_identity_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// Resolve the optional identity img2img-init for the Z-Image strict-pose set (sc-3146):
/// `Some((image, strength))` when [`zimage_identity_strength`] engages, decoding `referenceAssetId`
/// via [`load_reference_image`]; `None` for the default pose-only tier. The reference is shared
/// across the whole pose set (identity is constant; only the per-pose skeleton changes).
#[cfg(target_os = "macos")]
fn resolve_zimage_identity_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(strength) = zimage_identity_strength(request) else {
        return Ok(None);
    };
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("zimage_identity_strength guarantees a non-empty referenceAssetId");
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(Some((image, strength)))
}

/// Resolve the Z-Image Image-Edit img2img init for `mode == "edit_image"` (epic 3529):
/// `Some((source, strength))` decoding `sourceAssetId` and pre-fitting it to the output W×H
/// (crop/pad/outpaint via [`should_fit_edit_source`]/[`fit_engine_image`] — never stretch an
/// off-aspect source); `None` when not an edit job or no source asset (the caller then falls
/// back to the identity-init reference path / plain txt2img). `strength` is the torch
/// `ZImageImg2ImgPipeline` knob (`advanced.strength`, default 0.6) forwarded verbatim to the
/// engine — its `init_time_step(steps, strength)` matches the diffusers img2img start step.
/// Both `z_image_edit` and `z_image_turbo` (mode `edit_image`) drive this one path (same
/// Turbo-weights img2img call in torch).
#[cfg(target_os = "macos")]
fn resolve_zimage_edit_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    if request.mode != "edit_image" {
        return Ok(None);
    }
    let Some(asset_id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let image = if should_fit_edit_source(request) {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    } else {
        source
    };
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0);
    Ok(Some((image, strength)))
}

/// The asset `adapter` id for Z-Image (strict-pose shares the base z-image label).
#[cfg(target_os = "macos")]
const ZIMAGE_ADAPTER_LABEL: &str = "mlx_z_image";

// ---------------------------------------------------------------------------
// Character-Studio angle set + best-effort pose tier + fit_image (macOS, sc-3030):
// the per-iteration batch orchestration on top of FLUX.2-klein edit. An angle set
// loops the 11 canonical head angles (shared seed, per-angle prompt augment); the
// best-effort pose tier pairs each pose's body skeleton with the reference as a
// `[skeleton, reference]` multi-image set; fit_image pre-fits an Image-Edit source
// to the output W×H (crop/pad/outpaint) so off-aspect edits don't stretch. Faithful
// ports of `character_studio_angles.py` + the `MlxFlux2Adapter` / `fit_image` paths.
// ---------------------------------------------------------------------------

/// The 11 canonical Character-Studio angles, in order. Re-exported from the canonical
/// [`sceneworks_core::angle_kps`] source of truth (the same table the Key Point Library serves
/// as its built-in presets — sc-4434) so the worker and the library can never drift.
#[cfg(target_os = "macos")]
const CHARACTER_ANGLE_SET_ORDER: [&str; 11] = sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER;

/// The per-angle continuation clause appended to the user's prompt (parity with
/// `character_studio_angles.ANGLE_PROMPT_AUGMENTS`). Unknown angle → empty.
#[cfg(target_os = "macos")]
fn angle_prompt_augment(angle: &str) -> &'static str {
    match angle {
        "front" => {
            "frontal portrait, looking directly at the camera, head and shoulders, neutral expression"
        }
        "three_quarter_left" => {
            "three-quarter left profile, head turned slightly to the left, three-quarter view"
        }
        "three_quarter_right" => {
            "three-quarter right profile, head turned slightly to the right, three-quarter view"
        }
        "left_profile" => {
            "full left profile, head turned 90 degrees to the left, side view of the head"
        }
        "right_profile" => {
            "full right profile, head turned 90 degrees to the right, side view of the head"
        }
        "up" => "looking up, head tilted slightly upward toward the sky",
        "down" => "looking down, head tilted slightly downward toward the floor",
        "up_left" => {
            "looking up and to the left, head tilted slightly upward and turned slightly to the left"
        }
        "up_right" => {
            "looking up and to the right, head tilted slightly upward and turned slightly to the right"
        }
        "down_left" => {
            "looking down and to the left, head tilted slightly downward and turned slightly to the left"
        }
        "down_right" => {
            "looking down and to the right, head tilted slightly downward and turned slightly to the right"
        }
        _ => "",
    }
}

/// Strip the user's base prompt for augmentation: trim whitespace, then trailing
/// `,`/`.`/`;` — exactly Python's `(base or "").strip().rstrip(",.;")` (which can
/// leave a trailing space, e.g. `"a . "` → `"a "`).
#[cfg(target_os = "macos")]
fn strip_base_prompt(base: &str) -> &str {
    base.trim().trim_end_matches([',', '.', ';'])
}

/// Append the per-angle clause to the user's base prompt (parity with
/// `augment_prompt_for_angle`). Empty base + unknown angle → empty string.
#[cfg(target_os = "macos")]
fn augment_prompt_for_angle(base: &str, angle: &str) -> String {
    let augment = angle_prompt_augment(angle);
    let base = strip_base_prompt(base);
    if !base.is_empty() && !augment.is_empty() {
        format!("{base}, {augment}")
    } else if !augment.is_empty() {
        augment.to_owned()
    } else {
        base.to_owned()
    }
}

/// The pose-skeleton instruction appended to the prompt for the best-effort pose tier
/// (parity with `character_studio_angles.POSE_SKELETON_PROMPT`).
#[cfg(target_os = "macos")]
const POSE_SKELETON_PROMPT: &str =
    "matching the exact body pose shown in the OpenPose skeleton reference image";

/// Append the pose-skeleton cue to the user's base prompt (parity with
/// `augment_prompt_for_pose`).
#[cfg(target_os = "macos")]
fn augment_prompt_for_pose(base: &str) -> String {
    let base = strip_base_prompt(base);
    if base.is_empty() {
        POSE_SKELETON_PROMPT.to_owned()
    } else {
        format!("{base}, {POSE_SKELETON_PROMPT}")
    }
}

/// How a FLUX.2 edit job batches its iterations.
#[cfg(target_os = "macos")]
enum Flux2Grouping {
    /// `count` independent images (per-image seeds), the plain reference/edit path.
    Plain,
    /// The 11-angle Character-Studio set: shared seed, per-angle prompt augment.
    Angles,
    /// The best-effort pose tier: `n` poses, shared seed, `[skeleton, reference]` sets.
    Poses(usize),
}

/// Decide the grouping for a FLUX.2 edit job (parity with the `MlxFlux2Adapter`
/// decision: pose set > angle set > plain, all gated to `character_image` mode — an
/// `edit_image` job is never grouped). The caller only reaches this with a reference
/// present, so `is_character_image` reduces to the mode check.
#[cfg(target_os = "macos")]
fn flux2_grouping(request: &ImageRequest) -> Flux2Grouping {
    if request.mode != "character_image" {
        return Flux2Grouping::Plain;
    }
    let poses = pose_entries(request).len();
    if poses > 0 {
        return Flux2Grouping::Poses(poses);
    }
    if advanced::flag(&request.advanced, "angleSet") {
        return Flux2Grouping::Angles;
    }
    Flux2Grouping::Plain
}

/// The number of images a FLUX.2 edit job produces: 11 for an angle set, `n` for a
/// pose set, else the request count. `request.count` for any non-FLUX.2-edit job.
/// Threaded into [`ImagePlan`] so the generation set's `count`/`expectedCount` match
/// what is actually generated (the UI streams against it).
#[cfg(target_os = "macos")]
fn flux2_image_count(request: &ImageRequest, settings: &Settings) -> u32 {
    if flux2_edit_available(request, settings) {
        match flux2_grouping(request) {
            Flux2Grouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
            Flux2Grouping::Poses(count) => count as u32,
            Flux2Grouping::Plain => request.count,
        }
    } else {
        request.count
    }
}

/// True when the FLUX.2 Image-Edit source should be pre-fitted to W×H (parity with the
/// `MlxFlux2Adapter` fit gate): `edit_image` mode, a source asset, no character
/// `referenceAssetId`, and a non-`stretch` fit mode. The Character-Studio reference
/// path stays at native resolution.
#[cfg(target_os = "macos")]
fn should_fit_edit_source(request: &ImageRequest) -> bool {
    let has_source = request
        .source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // No character referenceAssetId (absent or empty).
    let no_reference = !request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    request.mode == "edit_image" && has_source && no_reference && request.fit_mode != "stretch"
}

/// Where a `src_w`×`src_h` image lands when contained (long edge fits) and centered in
/// a `width`×`height` box: `(new_w, new_h, left, top)`. Parity with Python `_contain_box`
/// (shared by the pad fit so the kept region lines up). Integer-divides the offsets.
#[cfg(target_os = "macos")]
fn contain_box(src_w: u32, src_h: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
    let ratio = (width as f32 / src_w as f32).min(height as f32 / src_h as f32);
    let new_w = ((src_w as f32 * ratio).round() as u32).max(1);
    let new_h = ((src_h as f32 * ratio).round() as u32).max(1);
    (new_w, new_h, (width - new_w) / 2, (height - new_h) / 2)
}

/// Resize an RGB image to exactly `width`×`height` honoring `mode` without distorting it
/// (parity with Python `fit_image`, RGB path only — no inpaint mask exists on the MLX
/// FLUX.2 edit path, so `outpaint` degrades to `pad` geometry):
///   - `crop`:    scale to COVER (short edge fits), center-crop the overflow.
///   - `pad`/`outpaint`: scale to CONTAIN (long edge fits), center on a black canvas.
///   - `stretch`: legacy non-aspect-preserving resize.
#[cfg(target_os = "macos")]
fn fit_rgb(source: &image::RgbImage, width: u32, height: u32, mode: &str) -> image::RgbImage {
    use image::imageops::FilterType::Lanczos3;
    let width = width.max(1);
    let height = height.max(1);
    let (src_w, src_h) = (source.width(), source.height());
    match mode {
        "stretch" => image::imageops::resize(source, width, height, Lanczos3),
        "crop" => {
            let ratio = (width as f32 / src_w as f32).max(height as f32 / src_h as f32);
            // Ceil so the scaled image always fully covers the target before cropping.
            let new_w = width.max((src_w as f32 * ratio).ceil() as u32);
            let new_h = height.max((src_h as f32 * ratio).ceil() as u32);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let left = (new_w - width) / 2;
            let top = (new_h - height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
        // "pad" / "outpaint": contain + center on a black canvas (letterbox).
        _ => {
            let (new_w, new_h, left, top) = contain_box(src_w, src_h, width, height);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
            canvas
        }
    }
}

/// Fit an engine [`Image`] (RGB8) to `width`×`height` by `mode` via [`fit_rgb`].
#[cfg(target_os = "macos")]
fn fit_engine_image(source: Image, width: u32, height: u32, mode: &str) -> WorkerResult<Image> {
    let rgb =
        image::RgbImage::from_raw(source.width, source.height, source.pixels).ok_or_else(|| {
            WorkerError::InvalidPayload("edit source buffer size mismatch".to_owned())
        })?;
    let fitted = fit_rgb(&rgb, width, height, mode);
    Ok(Image {
        width: fitted.width(),
        height: fitted.height(),
        pixels: fitted.into_raw(),
    })
}

// ---------------------------------------------------------------------------
// FLUX.2-klein edit / reference (macOS, sc-3029): the `flux2_klein_9b_edit` and
// `flux2_klein_9b_kv_edit` variants. FLUX.2-klein is MLX-only (no torch), so this
// is where its edit/reference jobs run. One output per requested count, each
// conditioned on the shared reference image(s); the -kv variant auto-engages the
// reference-K/V cache (~2.4× edit speedup).
// ---------------------------------------------------------------------------

/// The engine edit-variant id for a FLUX.2 SceneWorks model, or `None` if the model
/// has no edit variant. The base 9b + true_v2 share `flux2_klein_9b_edit`; the -kv
/// distill uses `flux2_klein_9b_kv_edit` (reference-K/V cache).
#[cfg(target_os = "macos")]
fn flux2_edit_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "flux2_klein_9b" | "flux2_klein_9b_true_v2" => Some("flux2_klein_9b_edit"),
        "flux2_klein_9b_kv" => Some("flux2_klein_9b_kv_edit"),
        _ => None,
    }
}

/// Reference asset ids for a FLUX.2 edit: the character-flow `referenceAssetId`, else
/// the Image-Edit `sourceAssetId` (edit_image mode). Mirrors the Python
/// `ref_id = referenceAssetId or (sourceAssetId if edit_image)`.
#[cfg(target_os = "macos")]
fn flux2_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    if request.mode == "edit_image" {
        if let Some(id) = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            return vec![id.to_owned()];
        }
    }
    Vec::new()
}

/// True when this is a FLUX.2 edit job (a flux2 edit-capable model + ≥1 reference)
/// whose base weights resolve — routed to the edit variant rather than txt2img.
#[cfg(target_os = "macos")]
fn flux2_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    flux2_edit_engine_id(&request.model).is_some()
        && !flux2_edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve a reference/source asset id to an in-memory RGB8 image (the engine VAE-
/// encodes + resizes it). Uses the indexed `ProjectStore::get_asset` → `file.path`.
#[cfg(target_os = "macos")]
pub(crate) fn load_reference_image(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id} has no media path"))
        })?;
    // The asset's file.path comes from an on-disk sidecar the user can edit, so
    // route it through safe_project_path (rejects `..`/absolute components) rather
    // than a bare join — matching the media-jobs reads and keeping a poisoned
    // sidecar from reading an arbitrary file as the reference (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = image::open(&path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference image {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// One `Reference` (single) or one `MultiReference` (N) edit conditioning from the
/// resolved reference images (cloned per output).
#[cfg(target_os = "macos")]
fn build_edit_conditioning(references: &[Image]) -> Vec<Conditioning> {
    if references.len() == 1 {
        vec![Conditioning::Reference {
            image: references[0].clone(),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference {
            images: references.to_vec(),
        }]
    }
}

/// Generate one FLUX.2 edit image conditioned on `conditioning` (the reference set).
/// Distilled klein: guidance 1.0, no negative prompt.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn flux2_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::InvalidPayload(format!("edit generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload("edit generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "edit generator returned non-image output".to_owned(),
        )),
    }
}

#[cfg(target_os = "macos")]
fn flux2_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    engine_id: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("editEngine".to_owned(), Value::String(engine_id.to_owned()));
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Real FLUX.2 edit generation: load the edit variant once, then `count` outputs each
/// conditioned on the shared reference set. Mirrors [`generate_mlx_stream`]'s blocking-
/// thread + streamed-events shape and reuses [`consume_gen_events`].
#[cfg(target_os = "macos")]
async fn generate_flux2_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let engine_id = flux2_edit_engine_id(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not a FLUX.2 edit model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2 weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    let guidance = resolve_guidance(request, model);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, model);
    let adapter_label = model.adapter_label;

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = flux2_edit_reference_ids(request);
    let mut references = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        references.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires a reference image".to_owned(),
        ));
    }
    // sc-3030 fit_image: pre-fit the Image-Edit source to the output W×H (crop / pad /
    // outpaint→pad) so an off-aspect edit doesn't stretch. Character-Studio references
    // stay native (the `should_fit_edit_source` gate excludes them).
    if should_fit_edit_source(request) {
        references = references
            .into_iter()
            .map(|reference| {
                fit_engine_image(reference, request.width, request.height, &request.fit_mode)
            })
            .collect::<WorkerResult<Vec<_>>>()?;
    }

    // sc-3030 per-iteration grouping: a Character-Studio angle set (11 shared-seed,
    // per-angle prompt) / best-effort pose tier (one per pose, shared seed, each a
    // `[skeleton, reference]` set) / else the plain per-image reference path.
    let grouping = flux2_grouping(request);
    let set_seed = resolve_seed(request, 0);
    let (seeds, prompts, pose_keypoints): (
        Vec<i64>,
        Vec<String>,
        Option<Vec<Vec<crate::openpose_skeleton::Keypoint>>>,
    ) = match &grouping {
        Flux2Grouping::Poses(count) => {
            // Shared seed so only the pose changes across the set (Python parity).
            let keypoints = parse_poses(request)
                .into_iter()
                .map(|pose| pose.keypoints)
                .collect();
            let prompts = vec![augment_prompt_for_pose(&request.prompt); *count];
            (vec![set_seed; *count], prompts, Some(keypoints))
        }
        Flux2Grouping::Angles => {
            // Shared seed so noise-derived attributes (hair, lighting) stay constant
            // across angles — only the head pose changes (sc-2050 InstantID strategy).
            let prompts = CHARACTER_ANGLE_SET_ORDER
                .iter()
                .map(|angle| augment_prompt_for_angle(&request.prompt, angle))
                .collect();
            (
                vec![set_seed; CHARACTER_ANGLE_SET_ORDER.len()],
                prompts,
                None,
            )
        }
        Flux2Grouping::Plain => {
            let count = request.count as usize;
            let seeds = (0..count)
                .map(|index| resolve_seed(request, index))
                .collect();
            (seeds, vec![request.prompt.clone(); count], None)
        }
    };
    let total = seeds.len();

    let mut raw_settings = flux2_edit_raw_settings(
        request,
        &repo,
        engine_id,
        steps,
        quant_bits,
        guidance,
        references.len(),
    );
    match grouping {
        Flux2Grouping::Angles => {
            raw_settings.insert("angleSet".to_owned(), Value::Bool(true));
        }
        Flux2Grouping::Poses(_) => {
            raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        }
        Flux2Grouping::Plain => {}
    }

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let (width, height) = (request.width, request.height);
        let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
        let cancel = cancel.clone();
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let adapter_count = adapters.len();
            emit_load_event(
                "image_pipeline_load_start",
                &job_id,
                engine_id,
                adapter_count,
            );
            let generator = mlx_load(engine_id, weights_dir, quant, adapters)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                engine_id,
                adapter_count,
            );
            for (index, (seed, prompt)) in seeds.into_iter().zip(prompts).enumerate() {
                // Pose tier: pair this pose's body-only skeleton (DWPose body, no
                // hands/face — Python `draw_bodypose`) with the reference as a
                // `[skeleton, reference]` multi-image set; else the plain reference set.
                let conditioning = match &pose_keypoints {
                    Some(keypoints) => {
                        let skeleton = crate::openpose_skeleton::draw_wholebody(
                            width,
                            height,
                            &keypoints[index],
                            None,
                            None,
                            stickwidth,
                        );
                        vec![Conditioning::MultiReference {
                            images: vec![
                                Image {
                                    width,
                                    height,
                                    pixels: skeleton.into_raw(),
                                },
                                references[0].clone(),
                            ],
                        }]
                    }
                    None => build_edit_conditioning(&references),
                };
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (out_w, out_h, pixels) = flux2_edit_generate_one(
                    generator.as_ref(),
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning,
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width: out_w,
                        height: out_h,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Qwen-Image strict-pose ControlNet (macOS, epic 3401 / sc-3575): the InstantX
// `Qwen-Image-ControlNet-Union` variant registered in mlx-gen as `qwen_image_control`.
// One image per library pose, shared seed, true CFG + character LoRA on the base Qwen model.
// ---------------------------------------------------------------------------

/// The engine registry id for the Qwen-Image ControlNet-Union variant.
#[cfg(target_os = "macos")]
const QWEN_CONTROL_ENGINE_ID: &str = "qwen_image_control";
/// Default InstantX Qwen-Image-ControlNet-Union weights (Apache-2.0, DWPose-trained).
#[cfg(target_os = "macos")]
const QWEN_CONTROL_REPO: &str = "InstantX/Qwen-Image-ControlNet-Union";
#[cfg(target_os = "macos")]
const QWEN_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";

/// True when this is the base-Qwen strict-pose tier: `qwen_image` + non-empty object
/// `advanced.poses`, not edit mode, and base weights available. A `referenceAssetId`, if present,
/// is ignored for parity with the Python torch `QwenImageControlNetPipeline` path; identity comes
/// from character LoRA adapters on the base transformer.
#[cfg(target_os = "macos")]
fn qwen_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "qwen_image"
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(target_os = "macos")]
fn resolve_qwen_control_weights(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    resolve_control_weights_for(request, settings, QWEN_CONTROL_REPO, QWEN_CONTROL_FILE)
}

/// Load the Qwen-Image ControlNet-Union generator (base snapshot + InstantX control overlay).
#[cfg(target_os = "macos")]
fn qwen_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    mlx_gen::load(QWEN_CONTROL_ENGINE_ID, &spec).map_err(|error| {
        WorkerError::InvalidPayload(format!("Qwen strict-pose control load failed: {error}"))
    })
}

/// Generate one Qwen strict-pose image: the pose skeleton drives the InstantX control branch at
/// `control_scale`; prompt, true CFG, negative prompt, quant, and LoRA/LoKr mirror base Qwen.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn qwen_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    control: Image,
    control_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance: Some(guidance),
        conditioning: vec![Conditioning::Control {
            image: control,
            kind: ControlKind::Pose,
            scale: control_scale,
        }],
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::InvalidPayload(format!("Qwen strict-pose generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Qwen strict-pose generator produced no image".to_owned(),
                )
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "Qwen strict-pose generator returned non-image output".to_owned(),
        )),
    }
}

#[cfg(target_os = "macos")]
fn qwen_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(QWEN_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

/// Real Qwen strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton. Mirrors the Python `_generate_pose_set` path: shared seed, full body/hands/face
/// skeleton, `advanced.controlScale`, true CFG, and character LoRA identity.
#[cfg(target_os = "macos")]
async fn generate_qwen_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let qwen = mlx_model("qwen_image")
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen model row missing".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen-Image weights not found".to_owned()))?;
    let control_weights = resolve_qwen_control_weights(request, settings).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Qwen strict-pose control weights not found (download {QWEN_CONTROL_REPO})."
        ))
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, qwen);
    let guidance = resolve_guidance(request, qwen).unwrap_or(qwen.default_guidance);
    let negative_prompt = resolve_negative_prompt(request, qwen);
    let control_scale = resolve_control_scale(request);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, qwen);
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = qwen_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        count,
    );
    let seed = resolve_seed(request, 0);

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let prompt = request.prompt.clone();
        let (width, height) = (request.width, request.height);
        let cancel = cancel.clone();
        let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let adapter_count = adapters.len();
            emit_load_event(
                "image_pipeline_load_start",
                &job_id,
                QWEN_CONTROL_ENGINE_ID,
                adapter_count,
            );
            let generator = qwen_control_load(weights_dir, control_weights, quant, adapters)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                QWEN_CONTROL_ENGINE_ID,
                adapter_count,
            );
            for (index, pose) in poses.into_iter().enumerate() {
                let skeleton = crate::openpose_skeleton::draw_wholebody(
                    width,
                    height,
                    &pose.keypoints,
                    pose.hands.as_deref(),
                    pose.face.as_deref(),
                    stickwidth,
                );
                let control = Image {
                    width,
                    height,
                    pixels: skeleton.into_raw(),
                };
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (width, height, pixels) = qwen_control_generate_one(
                    generator.as_ref(),
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    control,
                    control_scale,
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width,
                        height,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        "mlx_qwen",
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Qwen-Image-Edit (macOS, sc-3397): the `qwen_image_edit` / `_2509` / `_2511` ids, all
// served by the engine's single `qwen_image_edit` model (Reference/MultiReference
// dual-latent). This is where Qwen edit/reference jobs run — `edit_image`, the
// Character-Studio reference flow (subject variation), the 11-angle set, and the
// best-effort pose tier (`[reference, skeleton]` multi-image). True CFG (negative prompt
// + guidance from `trueCfgScale`), LoRA/LoKr, Q4/Q8, fit_image. Base-Qwen strict-pose
// ControlNet is handled by the `qwen_image_control` path above; the `_2511_lightning` distill
// (sampler + lightx2v LoRA) is sc-3398.
// ---------------------------------------------------------------------------

/// The engine edit-model id for a Qwen SceneWorks model, or `None` if it has no edit
/// variant. `qwen_image_edit` / `_2509` / `_2511` / `_2511_lightning` all map to the single
/// `qwen_image_edit` engine model; the lightning id differs only in its sampler + distill
/// LoRA (see [`qwen_edit_lightning`]), not in the engine model.
#[cfg(target_os = "macos")]
fn qwen_edit_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => Some("qwen_image_edit"),
        _ => None,
    }
}

/// The Lightning few-step distill for a Qwen edit variant, or `None` for the production
/// (multi-step true-CFG) path. The engine's `lightning` sampler (static-shift schedule +
/// CFG-off single forward, mlx-gen-qwen-image `model_edit.rs`, sc-2909) only produces a
/// clean image when the matching lightx2v distill LoRA is stacked at load time — so a
/// lightning variant carries both. Worker-local: the distill LoRA is fetched lazily into
/// the HF cache on first use (`ensure_distill_lora_cached`), mirroring the Python path's
/// `load_lora_weights` → `fuse_lora` (it is not a manifest install artifact). sc-3398.
#[cfg(target_os = "macos")]
struct LightningDistill {
    /// The engine sampler id passed via `GenerationRequest.sampler`.
    sampler: &'static str,
    /// HuggingFace repo holding the distill LoRA.
    repo: &'static str,
    /// The distill LoRA filename within the repo (the 4-step bf16 variant matches the
    /// 4-step `default_steps` of the lightning model row).
    file: &'static str,
}

/// The Lightning distill config for a SceneWorks model id, or `None` for every
/// production variant. Only `qwen_image_edit_2511_lightning` is a distilled variant today.
#[cfg(target_os = "macos")]
fn qwen_edit_lightning(model: &str) -> Option<LightningDistill> {
    match model {
        "qwen_image_edit_2511_lightning" => Some(LightningDistill {
            sampler: "lightning",
            repo: "lightx2v/Qwen-Image-Edit-2511-Lightning",
            file: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        }),
        _ => None,
    }
}

/// Ensure a single distill-LoRA `file` from HuggingFace `repo` is materialized in the
/// shared HF hub cache, returning its absolute path. Fast-paths when the file is already
/// cached; otherwise fetches just that file into the standard `models--<org>--<name>`
/// layout (deduping with the Python loader and other tools, sc-1904) — the Rust generate
/// path assumes weights are cached, so the lightning path fetches its distill LoRA lazily
/// here, mirroring the Python `load_lora_weights` HF download (sc-3398, decision (b):
/// worker-local, not a cross-cutting model-install artifact).
#[cfg(target_os = "macos")]
async fn ensure_distill_lora_cached(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    file: &str,
) -> WorkerResult<PathBuf> {
    // Fast path: already materialized in the hub cache (the common case after first use).
    if let Some(snapshot_dir) =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
    {
        let candidate = snapshot_dir.join(file);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let repo_dir = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    let revision = "main";
    let client = reqwest::Client::new();
    let snapshot =
        HuggingFaceSnapshot::resolve(&client, settings, repo, revision, &[file.to_owned()]).await?;
    if snapshot.files.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "Distill LoRA {file} not found in Hugging Face repo {repo}."
        )));
    }
    let mut progress = DownloadProgress::new(
        repo,
        directory_size(&repo_dir.join("blobs")).await,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    download_snapshot_into_cache(
        &DownloadContext {
            api,
            client: &client,
            settings,
            job_id: &job.id,
            cancel_message: "Generation canceled while fetching the Lightning distill LoRA.",
            fresh_download: false,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let snapshot_dir = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Hugging Face snapshot for {repo} missing after download."
            ))
        })?;
    let path = snapshot_dir.join(file);
    if !path.is_file() {
        return Err(WorkerError::InvalidPayload(format!(
            "Distill LoRA {file} missing from the {repo} snapshot after download."
        )));
    }
    Ok(path)
}

/// Reference asset ids for a Qwen edit: the character-flow `referenceAssetId`, else the
/// Image-Edit `sourceAssetId` (edit_image mode). Mirrors the Python
/// `ref = referenceAssetId or (sourceAssetId if edit_image)` and the FLUX.2 edit path.
#[cfg(target_os = "macos")]
fn qwen_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    if request.mode == "edit_image" {
        if let Some(id) = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            return vec![id.to_owned()];
        }
    }
    Vec::new()
}

/// True when this is a Qwen edit job (a qwen edit-capable model + ≥1 reference) whose
/// weights resolve — routed to the `qwen_image_edit` engine model rather than txt2img.
#[cfg(target_os = "macos")]
fn qwen_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    qwen_edit_engine_id(&request.model).is_some()
        && !qwen_edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// The number of images this image_generate job produces, accounting for grouped edit
/// sets (Character-Studio angle set = 11, best-effort pose set = n) on either edit
/// family; otherwise `request.count`. Threaded into [`ImagePlan`] so the generation
/// set's `count`/`expectedCount` match what actually lands in the gallery (the UI
/// streams against it). Qwen edit reuses the shared [`flux2_grouping`] decision.
#[cfg(target_os = "macos")]
fn grouped_image_count(request: &ImageRequest, settings: &Settings) -> u32 {
    if instantid_available(request, settings) {
        instantid_image_count(request, settings)
    } else if qwen_edit_available(request, settings) {
        match flux2_grouping(request) {
            Flux2Grouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
            Flux2Grouping::Poses(count) => count as u32,
            Flux2Grouping::Plain => request.count,
        }
    } else if sensenova_edit_available(request, settings) {
        match flux2_grouping(request) {
            Flux2Grouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
            // SenseNova has no strict-pose (ControlNet) path — pose sets are excluded upstream by
            // `sensenova_mlx_eligible`, so any residual grouping is the plain per-image count.
            Flux2Grouping::Poses(_) | Flux2Grouping::Plain => request.count,
        }
    } else {
        flux2_image_count(request, settings)
    }
}

/// Resolve the Qwen edit true-CFG guidance. The engine's `guidance` IS the true CFG
/// (diffusers `true_cfg_scale`), so this reads `advanced.trueCfgScale` (NOT
/// `guidanceScale`, the inert embedded-guidance knob that the Python edit path pins at
/// 1.0) else the family default (4.0). The Character-Studio reference path clamps to
/// [1, 10] (Python `_reference_true_cfg_scale`); `edit_image` floors at 1.0 (the engine
/// needs CFG > 1 to engage).
#[cfg(target_os = "macos")]
fn resolve_qwen_edit_guidance(request: &ImageRequest, model: &MlxModel) -> f32 {
    let raw = request
        .advanced
        .get("trueCfgScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance);
    if request.mode == "character_image" {
        raw.clamp(1.0, 10.0)
    } else {
        raw.max(1.0)
    }
}

/// Flat telemetry for a Qwen edit generation (parity with `flux2_edit_raw_settings`).
#[cfg(target_os = "macos")]
fn qwen_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // The engine guidance is the true CFG — record it under the key the Python path uses.
    raw.insert("trueCfgScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert(
        "editEngine".to_owned(),
        Value::String("qwen_image_edit".to_owned()),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Generate one Qwen edit image conditioned on `conditioning` (the reference set). True
/// CFG: passes the negative prompt + `guidance`. Mirrors [`flux2_edit_generate_one`].
/// `sampler` selects the engine recipe — `Some("lightning")` runs the few-step distilled
/// path (CFG-off single forward), `None` the production multi-step true-CFG path (sc-3398).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn qwen_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    sampler: Option<&str>,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance: Some(guidance),
        sampler: sampler.map(str::to_owned),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::InvalidPayload(format!("edit generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload("edit generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "edit generator returned non-image output".to_owned(),
        )),
    }
}

/// Real Qwen-Image-Edit generation: load the `qwen_image_edit` engine model once, then
/// one output per grouped iteration each conditioned on the shared reference set. Mirrors
/// [`generate_flux2_edit_stream`]'s blocking-thread + streamed-events shape and reuses the
/// shared grouping ([`flux2_grouping`]) and [`consume_gen_events`]; differs in true-CFG
/// guidance (`trueCfgScale`) + the negative prompt, the `[reference, skeleton]` pose order
/// (reference first drives the VL identity prompt), and the body-only pose skeleton.
#[cfg(target_os = "macos")]
async fn generate_qwen_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let engine_id = qwen_edit_engine_id(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not a Qwen edit model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Qwen-Image-Edit weights not found".to_owned())
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    let guidance = resolve_qwen_edit_guidance(request, model);
    let negative_prompt = resolve_negative_prompt(request, model);
    // Lightning few-step distill (sc-3398): the `lightning` sampler engages the engine's
    // CFG-off static-shift recipe, and the matching lightx2v distill LoRA is stacked AHEAD
    // of any user LoRAs (the user's own `loras` still occupy their slots — mirrors the
    // Python fuse-distill-then-stack path). The distill LoRA is fetched lazily into the HF
    // cache on first use. `None`/`Vec::new()` for the production multi-step variants.
    let lightning = qwen_edit_lightning(&request.model);
    let sampler = lightning.as_ref().map(|distill| distill.sampler);
    let mut adapters: Vec<AdapterSpec> = Vec::new();
    if let Some(distill) = &lightning {
        let path =
            ensure_distill_lora_cached(api, settings, job, distill.repo, distill.file).await?;
        adapters.push(AdapterSpec::new(path, 1.0, AdapterKind::Lora));
    }
    adapters.extend(resolve_adapters(request)?);
    let repo = model_repo(request, model);
    let adapter_label = model.adapter_label;

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = qwen_edit_reference_ids(request);
    let mut references = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        references.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Qwen-Image-Edit requires a reference image".to_owned(),
        ));
    }
    // sc-3030 fit_image: pre-fit the Image-Edit source to the output W×H (crop / pad /
    // outpaint→pad) so an off-aspect edit doesn't stretch. Character-Studio references
    // stay native (the `should_fit_edit_source` gate excludes them).
    if should_fit_edit_source(request) {
        references = references
            .into_iter()
            .map(|reference| {
                fit_engine_image(reference, request.width, request.height, &request.fit_mode)
            })
            .collect::<WorkerResult<Vec<_>>>()?;
    }

    // Per-iteration grouping (shared with the FLUX.2 edit path): a Character-Studio angle
    // set (11 shared-seed, per-angle prompt) / best-effort pose tier (one per pose, shared
    // seed, each a `[reference, skeleton]` set) / else the plain per-image reference path.
    let grouping = flux2_grouping(request);
    let set_seed = resolve_seed(request, 0);
    let (seeds, prompts, pose_keypoints): (
        Vec<i64>,
        Vec<String>,
        Option<Vec<Vec<crate::openpose_skeleton::Keypoint>>>,
    ) = match &grouping {
        Flux2Grouping::Poses(count) => {
            // Shared seed so only the pose changes across the set (Python parity).
            let keypoints = parse_poses(request)
                .into_iter()
                .map(|pose| pose.keypoints)
                .collect();
            let prompts = vec![augment_prompt_for_pose(&request.prompt); *count];
            (vec![set_seed; *count], prompts, Some(keypoints))
        }
        Flux2Grouping::Angles => {
            // Shared seed so noise-derived attributes (hair, lighting) stay constant
            // across angles — only the head pose changes.
            let prompts = CHARACTER_ANGLE_SET_ORDER
                .iter()
                .map(|angle| augment_prompt_for_angle(&request.prompt, angle))
                .collect();
            (
                vec![set_seed; CHARACTER_ANGLE_SET_ORDER.len()],
                prompts,
                None,
            )
        }
        Flux2Grouping::Plain => {
            let count = request.count as usize;
            let seeds = (0..count)
                .map(|index| resolve_seed(request, index))
                .collect();
            (seeds, vec![request.prompt.clone(); count], None)
        }
    };
    let total = seeds.len();

    let mut raw_settings = qwen_edit_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        references.len(),
    );
    match grouping {
        Flux2Grouping::Angles => {
            raw_settings.insert("angleSet".to_owned(), Value::Bool(true));
        }
        Flux2Grouping::Poses(_) => {
            raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        }
        Flux2Grouping::Plain => {}
    }
    // Record the Lightning recipe for telemetry/A-B parity (matches the Python `distillLora`
    // key format `repo/file`); absent on the production multi-step variants.
    if let Some(distill) = &lightning {
        raw_settings.insert(
            "sampler".to_owned(),
            Value::String(distill.sampler.to_owned()),
        );
        raw_settings.insert(
            "distillLora".to_owned(),
            Value::String(format!("{}/{}", distill.repo, distill.file)),
        );
    }

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let (width, height) = (request.width, request.height);
        let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
        let cancel = cancel.clone();
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let adapter_count = adapters.len();
            emit_load_event(
                "image_pipeline_load_start",
                &job_id,
                engine_id,
                adapter_count,
            );
            let generator = mlx_load(engine_id, weights_dir, quant, adapters)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                engine_id,
                adapter_count,
            );
            for (index, (seed, prompt)) in seeds.into_iter().zip(prompts).enumerate() {
                // Pose tier: pair the reference with this pose's body-only skeleton (DWPose
                // body, no hands/face — Python `draw_bodypose`) as a `[reference, skeleton]`
                // multi-image set. Reference FIRST: the engine VL-encodes references[0] for
                // the prompt embeds (identity), the skeleton is added dual-latent geometry.
                let conditioning = match &pose_keypoints {
                    Some(keypoints) => {
                        let skeleton = crate::openpose_skeleton::draw_wholebody(
                            width,
                            height,
                            &keypoints[index],
                            None,
                            None,
                            stickwidth,
                        );
                        vec![Conditioning::MultiReference {
                            images: vec![
                                references[0].clone(),
                                Image {
                                    width,
                                    height,
                                    pixels: skeleton.into_raw(),
                                },
                            ],
                        }]
                    }
                    None => build_edit_conditioning(&references),
                };
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (out_w, out_h, pixels) = qwen_edit_generate_one(
                    generator.as_ref(),
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    sampler,
                    conditioning,
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width: out_w,
                        height: out_h,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// SenseNova-U1 it2i (macOS, sc-3900 / epic 3180): instruction edit + Character Studio on the
// unified `sensenova_u1_8b` / `sensenova_u1_8b_fast` ids. The same model does T2I (the base
// `generate_mlx_stream` path) and it2i here; a `Conditioning::Reference` (single, edit_image) or
// `Conditioning::MultiReference` (N, character_image incl. the angle set) drives the
// understanding-path vision encoder. SenseNova uses BOTH CFG knobs: the text CFG via `guidance`
// (`advanced.guidanceScale`, default 4.0 base / 1.0 fast) AND the image-guidance via `true_cfg`
// (`advanced.imageGuidanceScale` → engine `img_cfg_scale`, default 1.0 edit / 1.5 character) — so
// it is NOT a `uses_true_cfg` family. No negative prompt (`supports_negative_prompt=false`). The
// fast variant merges its 8-step distill LoRA internally at load (`load_fast`) — the worker only
// selects the engine id; there is no user-LoRA slot (`supports_lora=false`). SenseNova has no
// ControlNet, so strict pose is excluded by `sensenova_mlx_eligible`. Mirrors
// `generate_qwen_edit_stream`'s blocking-thread + streamed-events shape.
// ---------------------------------------------------------------------------

/// True when this is a SenseNova it2i job: a SenseNova model + ≥1 reference (the character
/// `referenceAssetId`, or the Image-Edit `sourceAssetId` in `edit_image` mode) whose weights
/// resolve. Plain T2I (no reference) is NOT routed here — it rides the base `mlx_available` path.
/// Reuses [`qwen_edit_reference_ids`] (the generic `ref = referenceAssetId or sourceAssetId-if-edit`
/// rule, not Qwen-specific).
#[cfg(target_os = "macos")]
fn sensenova_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sensenova_model(&request.model)
        && !qwen_edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Snap a dimension to SenseNova's 32-pixel cell (the engine rejects off-cell sizes), clamped to
/// the descriptor's [256, 2048] range. SenseNova's trained buckets are already 32-aligned; this
/// guards a hand-set advanced width/height.
#[cfg(target_os = "macos")]
fn sensenova_dim(value: u32) -> u32 {
    let snapped = value.div_ceil(32) * 32;
    snapped.clamp(256, 2048)
}

/// The SenseNova image-conditioning guidance (`true_cfg` → engine `img_cfg_scale`):
/// `advanced.imageGuidanceScale` else the per-mode default — 1.5 for Character Studio
/// (`character_image`, pulls harder toward the reference subject, sc-2015) / 1.0 for instruction
/// edit (the upstream it2i default). Floored at 1.0. Mirrors the Python `_image_guidance_scale`.
#[cfg(target_os = "macos")]
fn resolve_sensenova_img_cfg(request: &ImageRequest) -> f32 {
    let default = if request.mode == "character_image" {
        1.5
    } else {
        1.0
    };
    request
        .advanced
        .get("imageGuidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
        .max(1.0)
}

/// The SenseNova flow-match timestep shift (`scheduler_shift` → engine `timestep_shift`):
/// `advanced.schedulerShift` (or the legacy `timestepShift`) else 3.0; a non-positive value falls
/// back to 3.0. The only sampling knob SenseNova exposes (mirrors the Python adapter).
#[cfg(target_os = "macos")]
fn resolve_sensenova_timestep_shift(request: &ImageRequest) -> f32 {
    let raw = request
        .advanced
        .get("schedulerShift")
        .or_else(|| request.advanced.get("timestepShift"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(3.0);
    if raw > 0.0 {
        raw
    } else {
        3.0
    }
}

/// Flat telemetry for a SenseNova it2i generation (parity with `qwen_edit_raw_settings`).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn sensenova_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    img_cfg: f32,
    timestep_shift: f32,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    if let Some(scale) = guidance {
        raw.insert("guidanceScale".to_owned(), json!(scale));
    }
    raw.insert("imageGuidanceScale".to_owned(), json!(img_cfg));
    raw.insert("schedulerShift".to_owned(), json!(timestep_shift));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert(
        "editEngine".to_owned(),
        Value::String("sensenova_u1".to_owned()),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Generate one SenseNova it2i image conditioned on `conditioning` (the reference set). Dual CFG:
/// `guidance` carries the text CFG, `true_cfg` the image guidance; `scheduler_shift` the
/// flow-match timestep shift. No negative prompt.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn sensenova_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    img_cfg: f32,
    timestep_shift: f32,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg: Some(img_cfg),
        scheduler_shift: Some(timestep_shift),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::InvalidPayload(format!("SenseNova edit generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload("SenseNova edit produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "SenseNova edit returned non-image output".to_owned(),
        )),
    }
}

/// Real SenseNova-U1 it2i generation: load the unified model once (base or distilled `_fast`),
/// then one output per grouped iteration each conditioned on the shared reference set. Mirrors
/// [`generate_qwen_edit_stream`]'s blocking-thread + streamed-events shape; differs in the dual
/// CFG (`guidance` text + `true_cfg` image), no negative prompt, no pose tier (SenseNova has no
/// ControlNet), and no Lightning fetch (the `_fast` distill LoRA is merged inside the engine load).
#[cfg(target_os = "macos")]
async fn generate_sensenova_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let engine_id = model.engine_id;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SenseNova-U1 weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    // Dual CFG: the text CFG flows through `guidance` (Some — SenseNova `supports_guidance`); the
    // image-conditioning guidance through `true_cfg`.
    let guidance = resolve_guidance(request, model);
    let img_cfg = resolve_sensenova_img_cfg(request);
    let timestep_shift = resolve_sensenova_timestep_shift(request);
    let repo = model_repo(request, model);
    let adapter_label = model.adapter_label;
    let (out_w, out_h) = (sensenova_dim(request.width), sensenova_dim(request.height));

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = qwen_edit_reference_ids(request);
    let mut references = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        references.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SenseNova-U1 it2i requires a reference image".to_owned(),
        ));
    }
    // sc-3030 fit_image: pre-fit an off-aspect Image-Edit source to the output W×H (crop / pad /
    // outpaint→pad). Character-Studio references stay native (`should_fit_edit_source` excludes them).
    if should_fit_edit_source(request) {
        references = references
            .into_iter()
            .map(|reference| fit_engine_image(reference, out_w, out_h, &request.fit_mode))
            .collect::<WorkerResult<Vec<_>>>()?;
    }
    let conditioning = build_edit_conditioning(&references);

    // Per-iteration grouping: a Character-Studio angle set (11 shared-seed, per-angle prompt) or the
    // plain per-image reference path. SenseNova has no pose tier (excluded by `sensenova_mlx_eligible`).
    let grouping = flux2_grouping(request);
    let set_seed = resolve_seed(request, 0);
    let (seeds, prompts): (Vec<i64>, Vec<String>) = match &grouping {
        Flux2Grouping::Angles => {
            // Shared seed so noise-derived attributes stay constant across angles.
            let prompts = CHARACTER_ANGLE_SET_ORDER
                .iter()
                .map(|angle| augment_prompt_for_angle(&request.prompt, angle))
                .collect();
            (vec![set_seed; CHARACTER_ANGLE_SET_ORDER.len()], prompts)
        }
        Flux2Grouping::Plain => {
            let count = request.count as usize;
            let seeds = (0..count)
                .map(|index| resolve_seed(request, index))
                .collect();
            (seeds, vec![request.prompt.clone(); count])
        }
        Flux2Grouping::Poses(_) => {
            // Unreachable: strict pose is excluded by `sensenova_mlx_eligible` (no ControlNet).
            return Err(WorkerError::InvalidPayload(
                "SenseNova-U1 has no strict-pose (ControlNet) path".to_owned(),
            ));
        }
    };
    let total = seeds.len();

    let mut raw_settings = sensenova_edit_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        img_cfg,
        timestep_shift,
        references.len(),
    );
    if matches!(grouping, Flux2Grouping::Angles) {
        raw_settings.insert("angleSet".to_owned(), Value::Bool(true));
    }

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let cancel = cancel.clone();
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            emit_load_event("image_pipeline_load_start", &job_id, engine_id, 0);
            let generator = mlx_load(engine_id, weights_dir, quant, Vec::new())?;
            emit_load_event("image_pipeline_load_complete", &job_id, engine_id, 0);
            for (index, (seed, prompt)) in seeds.into_iter().zip(prompts).enumerate() {
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (w, h, pixels) = sensenova_edit_generate_one(
                    generator.as_ref(),
                    &prompt,
                    out_w,
                    out_h,
                    seed,
                    steps,
                    guidance,
                    img_cfg,
                    timestep_shift,
                    conditioning.clone(),
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width: w,
                        height: h,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// SDXL advanced conditioning (macOS, epic 3041 / sc-3060): reference (IP-Adapter),
// img2img edit, masked inpaint, and outpaint on the `sdxl` engine model. The plain
// txt2img + LoRA path stays on `generate_mlx_stream`; this branch handles every SDXL
// shape that used to fall through to the Python torch `SdxlDiffusersAdapter`. The
// engine selects the path from the loaded weights (`ip_adapter`) + conditioning combo
// (mlx-gen-sdxl PRs #137/#138); we just build the right `LoadSpec` + `Conditioning`.
// ---------------------------------------------------------------------------

/// Default h94 IP-Adapter snapshot repo (ViT-H encoder + plus/plus-face SDXL weights).
#[cfg(target_os = "macos")]
const SDXL_IP_ADAPTER_REPO: &str = "h94/IP-Adapter";
/// img2img strength for a plain SDXL edit (torch `SdxlDiffusersAdapter` default 0.6).
#[cfg(target_os = "macos")]
const SDXL_EDIT_STRENGTH: f32 = 0.6;
/// img2img strength for masked inpaint / outpaint (torch default 0.85).
#[cfg(target_os = "macos")]
const SDXL_INPAINT_STRENGTH: f32 = 0.85;
/// IP-Adapter scale when the request omits it — matches the torch plus-face default 0.7
/// (`SdxlDiffusersAdapter._ip_adapter_scale`); the engine's own fallback is 0.6.
#[cfg(target_os = "macos")]
const SDXL_IP_SCALE: f32 = 0.7;

/// Which advanced SDXL path a request maps onto (or `None` for plain txt2img, which stays
/// on [`generate_mlx_stream`]). Outpaint wins over a plain mask when `fit_mode == outpaint`
/// (the torch path checks outpaint first, then unions any user mask into the border).
#[cfg(target_os = "macos")]
enum SdxlSubMode {
    /// Reference image-prompt via IP-Adapter (txt2img + decoupled cross-attn).
    Ip,
    /// Plain img2img edit (Reference init only).
    Edit,
    /// Masked inpaint (Reference init + Mask).
    Inpaint,
    /// Outpaint = inpaint with a generated border mask (+ optional user-mask union).
    Outpaint,
}

#[cfg(target_os = "macos")]
fn non_empty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|id| !id.trim().is_empty())
}

/// The engine-backed SDXL family row for a model id (`sdxl` / `realvisxl`), if any.
#[cfg(target_os = "macos")]
fn sdxl_engine_model(model: &str) -> Option<&'static MlxModel> {
    mlx_model(model).filter(|entry| entry.engine_id == "sdxl")
}

/// Classify an SDXL job into an advanced sub-mode. `None` = plain txt2img (no reference,
/// not an edit) → handled by the base MLX path.
#[cfg(target_os = "macos")]
fn sdxl_sub_mode(request: &ImageRequest) -> Option<SdxlSubMode> {
    if request.mode == "edit_image" {
        if !non_empty(&request.source_asset_id) {
            return None;
        }
        if request.fit_mode == "outpaint" {
            return Some(SdxlSubMode::Outpaint);
        }
        if non_empty(&request.mask_asset_id) {
            return Some(SdxlSubMode::Inpaint);
        }
        return Some(SdxlSubMode::Edit);
    }
    if non_empty(&request.reference_asset_id) {
        return Some(SdxlSubMode::Ip);
    }
    None
}

/// True when this is an SDXL advanced job (sdxl-family model + an advanced sub-mode) whose
/// base weights resolve — routed here rather than to plain txt2img.
#[cfg(target_os = "macos")]
fn sdxl_advanced_available(request: &ImageRequest, settings: &Settings) -> bool {
    sdxl_engine_model(&request.model).is_some()
        && sdxl_sub_mode(request).is_some()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the IP-Adapter snapshot directory (`advanced.ipAdapterRepo` override, else the
/// h94 default). The engine loader finds the ViT-H encoder + plus/plus-face weights inside.
#[cfg(target_os = "macos")]
fn resolve_ip_adapter_dir(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    let repo = request
        .advanced
        .get("ipAdapterRepo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(SDXL_IP_ADAPTER_REPO);
    huggingface_snapshot_dir(&settings.data_dir, repo)
}

/// Resolve a mask asset id to an RGB8 [`Image`] (the engine luma-converts + binarizes it).
#[cfg(target_os = "macos")]
fn load_mask_asset_image(
    settings: &Settings,
    project_id: &str,
    mask_asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    load_reference_image(&settings.data_dir, project_id, mask_asset_id, project_path)
}

/// Composite `source` contained (long edge fits) + centered on a black `width`×`height`
/// canvas, using the **engine's** `contain_box` so the padded source lines up pixel-for-pixel
/// with [`mlx_gen::image::outpaint_border_mask`] (both derive the same kept rect).
#[cfg(target_os = "macos")]
fn sdxl_outpaint_canvas(source: &image::RgbImage, width: u32, height: u32) -> Image {
    use image::imageops::FilterType::Lanczos3;
    let (new_w, new_h, left, top) =
        mlx_gen::image::contain_box(source.width(), source.height(), width, height);
    let resized = image::imageops::resize(source, new_w.max(1), new_h.max(1), Lanczos3);
    let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
    image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
    Image {
        width,
        height,
        pixels: canvas.into_raw(),
    }
}

/// An [`Image`] (RGB8) as an `image::RgbImage` for host-side compositing.
#[cfg(target_os = "macos")]
fn engine_image_to_rgb(image: Image) -> WorkerResult<image::RgbImage> {
    image::RgbImage::from_raw(image.width, image.height, image.pixels)
        .ok_or_else(|| WorkerError::InvalidPayload("image buffer size mismatch".to_owned()))
}

/// Load the SDXL generator for an advanced job. `ip_adapter_dir` (Some only in IP mode) adds
/// the decoupled cross-attn weights at load — the engine then treats a `Reference` as the
/// image prompt rather than an img2img init. Loaded per job (no persistent cache).
#[cfg(target_os = "macos")]
fn sdxl_advanced_load(
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> WorkerResult<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if let Some(ip) = ip_adapter_dir {
        spec = spec.with_ip_adapter(WeightsSource::Dir(ip));
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    mlx_gen::load("sdxl", &spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("sdxl advanced load failed: {error}")))
}

/// Generate one SDXL image conditioned on `conditioning` (Reference[/Mask]). SDXL is true-CFG
/// (negative prompt + guidance honoured). The img2img strength / IP scale ride the Reference
/// `strength` field, so no separate `req.strength` is needed.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn sdxl_advanced_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::InvalidPayload(format!("sdxl advanced generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::InvalidPayload("sdxl advanced produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::InvalidPayload(
            "sdxl advanced returned non-image output".to_owned(),
        )),
    }
}

/// Recipe facts recorded on the assets (the sub-mode + strengths/IP scale that drove it).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn sdxl_advanced_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    mode_tag: &str,
    strength: f32,
    ip_scale: Option<f32>,
) -> JsonObject {
    let mut raw = mlx_raw_settings(request, repo, steps, quant_bits, guidance);
    raw.insert("sdxlMode".to_owned(), Value::String(mode_tag.to_owned()));
    raw.insert("strength".to_owned(), json!(strength));
    if let Some(scale) = ip_scale {
        raw.insert("ipAdapterScale".to_owned(), json!(scale));
    }
    raw
}

/// Real SDXL advanced generation: resolve the conditioning images on the async side, then load
/// once + generate `count` images on the blocking thread (the MLX generator is `!Send`). Reuses
/// [`consume_gen_events`] for streaming + asset writes.
#[cfg(target_os = "macos")]
async fn generate_sdxl_advanced_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = sdxl_engine_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an SDXL engine model".to_owned()))?;
    let sub_mode = sdxl_sub_mode(request)
        .ok_or_else(|| WorkerError::InvalidPayload("not an SDXL advanced job".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SDXL weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    let guidance = resolve_guidance(request, model);
    let negative_prompt = resolve_negative_prompt(request, model);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, model);
    let adapter_label = model.adapter_label;
    let (width, height) = (request.width, request.height);

    // Build the (seed-independent) conditioning + decide whether IP weights load. Images are
    // decoded here on the async side and moved into the blocking task (each cloned per seed).
    let (conditioning, ip_adapter_dir, mode_tag, strength, ip_scale): (
        Vec<Conditioning>,
        Option<PathBuf>,
        &str,
        f32,
        Option<f32>,
    ) = match sub_mode {
        SdxlSubMode::Ip => {
            let reference_id = request
                .reference_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("IP-Adapter requires a reference image".to_owned())
                })?;
            let reference = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                reference_id,
                project_path,
            )?;
            let ip_dir = resolve_ip_adapter_dir(request, settings).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "SDXL IP-Adapter weights not found (download {SDXL_IP_ADAPTER_REPO})."
                ))
            })?;
            let scale = advanced::f32_clamped(
                &request.advanced,
                "ipAdapterScale",
                SDXL_IP_SCALE,
                0.0..=1.0,
            );
            (
                vec![Conditioning::Reference {
                    image: reference,
                    strength: Some(scale),
                }],
                Some(ip_dir),
                "ip_adapter",
                scale,
                Some(scale),
            )
        }
        SdxlSubMode::Edit | SdxlSubMode::Inpaint => {
            let source_id = request
                .source_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("SDXL edit requires a source image".to_owned())
                })?;
            let source = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                source_id,
                project_path,
            )?;
            // Pre-fit the source to the output W×H (crop/pad) so an off-aspect edit doesn't
            // stretch — torch parity with `load_source_image` + `fit_image`.
            let source = fit_engine_image(source, width, height, &request.fit_mode)?;
            let is_inpaint = matches!(sub_mode, SdxlSubMode::Inpaint);
            let strength = advanced::f32_clamped(
                &request.advanced,
                "strength",
                if is_inpaint {
                    SDXL_INPAINT_STRENGTH
                } else {
                    SDXL_EDIT_STRENGTH
                },
                0.0..=1.0,
            );
            let mut conditioning = vec![Conditioning::Reference {
                image: source,
                strength: Some(strength),
            }];
            if is_inpaint {
                let mask_id = request
                    .mask_asset_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("inpaint requires a mask image".to_owned())
                    })?;
                let mask =
                    load_mask_asset_image(settings, &request.project_id, mask_id, project_path)?;
                // Align the mask to the source with the SAME fit geometry.
                let mask = fit_engine_image(mask, width, height, &request.fit_mode)?;
                conditioning.push(Conditioning::Mask { image: mask });
            }
            (
                conditioning,
                None,
                if is_inpaint { "inpaint" } else { "edit" },
                strength,
                None,
            )
        }
        SdxlSubMode::Outpaint => {
            let source_id = request
                .source_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("outpaint requires a source image".to_owned())
                })?;
            let source = engine_image_to_rgb(load_reference_image(
                &settings.data_dir,
                &request.project_id,
                source_id,
                project_path,
            )?)?;
            let (src_w, src_h) = (source.width(), source.height());
            let canvas = sdxl_outpaint_canvas(&source, width, height);
            // White = generate (the padded border), black = keep (the centered source).
            let mut mask = mlx_gen::image::outpaint_border_mask(src_w, src_h, width, height);
            if non_empty(&request.mask_asset_id) {
                // Union the user edit region with the border (white wins) — pad-fit the user
                // mask onto the same contained geometry first.
                let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
                let user_mask =
                    load_mask_asset_image(settings, &request.project_id, mask_id, project_path)?;
                let user_mask = fit_engine_image(user_mask, width, height, "pad")?;
                mask = mlx_gen::image::union_masks(&mask, &user_mask).map_err(|error| {
                    WorkerError::InvalidPayload(format!("outpaint mask union failed: {error}"))
                })?;
            }
            let strength = advanced::f32_clamped(
                &request.advanced,
                "strength",
                SDXL_INPAINT_STRENGTH,
                0.0..=1.0,
            );
            (
                vec![
                    Conditioning::Reference {
                        image: canvas,
                        strength: Some(strength),
                    },
                    Conditioning::Mask { image: mask },
                ],
                None,
                "outpaint",
                strength,
                None,
            )
        }
    };

    let raw_settings = sdxl_advanced_raw_settings(
        request, &repo, steps, quant_bits, guidance, mode_tag, strength, ip_scale,
    );
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();
    let total = seeds.len();

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let prompt = request.prompt.clone();
        let negative_prompt = negative_prompt.clone();
        let cancel = cancel.clone();
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let adapter_count = adapters.len();
            emit_load_event("image_pipeline_load_start", &job_id, "sdxl", adapter_count);
            let generator = sdxl_advanced_load(weights_dir, quant, adapters, ip_adapter_dir)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                "sdxl",
                adapter_count,
            );
            for (index, seed) in seeds.into_iter().enumerate() {
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                let (out_w, out_h, pixels) = sdxl_advanced_generate_one(
                    generator.as_ref(),
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning.clone(),
                    &cancel,
                    &mut on_progress,
                )?;
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width: out_w,
                        height: out_h,
                        pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// InstantID identity-preserving character image (macOS, epic 3109 engine / sc-3345
// integration): the production `instantid_realvisxl` model — InstantID on RealVisXL +
// the stock SDXL IdentityNet ControlNet + the native MLX face stack (SCRFD + ArcFace),
// all in-process with zero Python. Two modes only (torch parity): a single-identity
// `character_image` (the reference's natural head pose) and the 11-view Character-Studio
// angle set. Pose-library mode (`advanced.poses`) + face-restore (`advanced.faceRestore`)
// are NOT handled here — they stay on the torch `InstantIDAdapter` (engine sc-3117 /
// sc-3380 not yet ported), gated out by `instantid_available` so the torch worker claims
// them. fp16 only for now (the validated envelope); Q8/Q4 ride explicit `mlxQuantize`
// (unvalidated at 1024², gated by sc-3329 follow-up). The provider is the bespoke
// `mlx_gen_instantid::InstantId` (not an inventory `Generator`), so this is a dedicated
// stream parallel to `generate_sdxl_advanced_stream`, not an MLX_MODELS row.
// ---------------------------------------------------------------------------

/// The SceneWorks model id for native InstantID (production = InstantID on RealVisXL_V5.0).
#[cfg(target_os = "macos")]
const INSTANTID_MODEL: &str = "instantid_realvisxl";
/// SDXL base for InstantID when the manifest omits `repo` (the photoreal production base).
#[cfg(target_os = "macos")]
const INSTANTID_SDXL_REPO: &str = "SG161222/RealVisXL_V5.0";
/// Stock InstantID checkpoint repo — the IdentityNet `ControlNetModel/` lives here.
#[cfg(target_os = "macos")]
const INSTANTID_CONTROLNET_REPO: &str = "InstantX/InstantID";
/// Converted-weights bundle (download-on-first-use): the MLX `ip-adapter.safetensors`
/// (`tools/convert_instantid.py`) + the native face stack `scrfd_10g.safetensors`
/// (`convert_scrfd.py`) + `arcface_iresnet100.safetensors` (`convert_glintr100.py`). Public
/// repo, mirroring the YOLO11 / SAM2 `SceneWorks/*-mlx` uploads (sc-3633 / sc-3707).
#[cfg(target_os = "macos")]
const INSTANTID_MLX_REPO: &str = "SceneWorks/instantid-mlx";
#[cfg(target_os = "macos")]
const INSTANTID_IP_ADAPTER_FILE: &str = "ip-adapter.safetensors";
#[cfg(target_os = "macos")]
const INSTANTID_SCRFD_FILE: &str = "scrfd_10g.safetensors";
#[cfg(target_os = "macos")]
const INSTANTID_ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";
/// The IdentityNet weight file inside `ControlNetModel/` (a stock diffusers SDXL ControlNet).
#[cfg(target_os = "macos")]
const INSTANTID_CONTROLNET_FILES: [&str; 2] =
    ["config.json", "diffusion_pytorch_model.safetensors"];
/// Torch-parity defaults (the `instantid_realvisxl` MODEL_TARGETS): RealVisXL is tuned for a
/// low CFG; the engine's own `InstantIdRequest::default` guidance (5.0) is for base SDXL.
#[cfg(target_os = "macos")]
const INSTANTID_DEFAULT_STEPS: u32 = 30;
#[cfg(target_os = "macos")]
const INSTANTID_DEFAULT_GUIDANCE: f32 = 3.0;
#[cfg(target_os = "macos")]
const INSTANTID_IP_SCALE: f32 = 0.8;
#[cfg(target_os = "macos")]
const INSTANTID_CONTROLNET_SCALE: f32 = 0.8;
/// xinsir OpenPose-SDXL ControlNet (the pose-mode second branch, sc-3117). Loads via the stock
/// `load_controlnet` (no conversion) — `image_adapters.py:615-617` parity.
#[cfg(target_os = "macos")]
const INSTANTID_OPENPOSE_REPO: &str = "xinsir/controlnet-openpose-sdxl-1.0";
/// Torch-parity default OpenPose lock (`instantid_adapter.py::_openpose_scale`, default 0.7).
#[cfg(target_os = "macos")]
const INSTANTID_OPENPOSE_SCALE: f32 = 0.7;
/// The face-restore re-render side (the engine's production crop size, sc-3380).
#[cfg(target_os = "macos")]
const INSTANTID_FACE_RESTORE_SIDE: u32 = 1024;

/// How an InstantID character job batches its iterations (torch-parity precedence: a pose set
/// wins over an angle set, which wins over plain identity — `instantid_adapter.py:655`).
#[cfg(target_os = "macos")]
enum InstantIdMode {
    /// `count` images at the reference's natural head pose (engine `generate`, W×H letterboxed).
    Identity,
    /// The 11-view Character-Studio set, shared seed (engine `generate_with_kps` from the
    /// worker-owned [`INSTANTID_ANGLE_KPS`] presets, square).
    AngleSet,
    /// `n` pose-library poses, shared seed — MultiControlNet IdentityNet + OpenPose (engine
    /// `generate_pose`, square).
    PoseSet(usize),
}

/// The 11-view Character-Studio angle set flag.
#[cfg(target_os = "macos")]
fn instantid_angle_set(request: &ImageRequest) -> bool {
    advanced::flag(&request.advanced, "angleSet")
}

/// Classify the InstantID iteration mode (pose set > angle set > plain identity).
#[cfg(target_os = "macos")]
fn instantid_mode(request: &ImageRequest) -> InstantIdMode {
    let poses = pose_entries(request).len();
    if poses > 0 {
        InstantIdMode::PoseSet(poses)
    } else if instantid_angle_set(request) {
        InstantIdMode::AngleSet
    } else {
        InstantIdMode::Identity
    }
}

/// Per-image InstantID action (the engine entry point this iteration calls). `Send` (it is moved
/// into the blocking task): `BodyPoint = Option<(f64, f64)>`, `&'static str`, and the unit variant
/// are all `Send`.
#[cfg(target_os = "macos")]
enum InstantIdAction {
    /// `generate` — the reference's natural head pose, W×H letterboxed.
    Identity,
    /// `generate_with_kps` — a Character-Studio view from worker-owned landmark presets (square).
    /// Carries the normalized 5-point kps directly (sc-4424) rather than an angle name, so the
    /// worker owns the framing presets and arbitrary/user-defined kps flow through the same path.
    Angle([(f32, f32); 5]),
    /// `generate_pose` — MultiControlNet IdentityNet + OpenPose on these COCO-18 keypoints (square).
    Pose(Vec<BodyPoint>),
}

/// Bridge the worker's gallery-normalized keypoints (`openpose_skeleton::Keypoint = Option<(f32,
/// f32)>`) to the engine's `BodyPoint = Option<(f64, f64)>`. `parse_poses` already applied the
/// COCO-18 normalize + conf<=0 drop, so this is just the f32→f64 widening.
#[cfg(target_os = "macos")]
fn pose_to_body_points(keypoints: &[crate::openpose_skeleton::Keypoint]) -> Vec<BodyPoint> {
    keypoints
        .iter()
        .map(|point| point.map(|(x, y)| (x as f64, y as f64)))
        .collect()
}

/// Resolve the RealVisXL (SDXL) base snapshot for InstantID: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (default
/// RealVisXL_V5.0). The big base is staged by the normal model-download flow; `None` here
/// means it is not present, so the job is not MLX-runnable (falls through to torch).
#[cfg(target_os = "macos")]
fn resolve_instantid_sdxl_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "InstantID SDXL modelPath")
            .map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(INSTANTID_SDXL_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a native-MLX-eligible InstantID job: the production model in
/// `character_image` mode with a reference face whose SDXL base resolves locally. ALL InstantID
/// modes are now native (sc-3345 identity + angle set; sc-3381 pose mode + face-restore via the
/// #193 engine). Mirrors `jobs_store::instantid_mlx_eligible` so the worker and the router agree.
#[cfg(target_os = "macos")]
fn instantid_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == INSTANTID_MODEL
        && request.mode == "character_image"
        && non_empty(&request.reference_asset_id)
        && matches!(resolve_instantid_sdxl_base(request, settings), Ok(Some(_)))
}

/// The number of images an InstantID job produces: `n` for a pose set, the active angle
/// collection's length for an angle set (sc-4450 — variable N, not fixed 11), else `request.count`.
#[cfg(target_os = "macos")]
fn instantid_image_count(request: &ImageRequest, settings: &Settings) -> u32 {
    match instantid_mode(request) {
        InstantIdMode::PoseSet(count) => count as u32,
        InstantIdMode::AngleSet => active_angle_collection(request, settings).1.len() as u32,
        InstantIdMode::Identity => request.count,
    }
}

/// Resolve the active angle-set collection for this job (sc-4450): the per-generation override
/// (`advanced.keypointCollectionId`) → the user default → the built-in 11. Built-in fallback on
/// any store error so angle generation never hard-fails on a Key Point Library hiccup.
#[cfg(target_os = "macos")]
fn active_angle_collection(
    request: &ImageRequest,
    settings: &Settings,
) -> (
    String,
    Vec<sceneworks_core::project_store::ResolvedAnglePreset>,
) {
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let override_id = advanced::str(&request.advanced, "keypointCollectionId", "");
    let override_id = override_id.trim();
    let override_id = (!override_id.is_empty()).then_some(override_id);
    store
        .resolve_angle_collection(override_id)
        .unwrap_or_else(|_| {
            (
                sceneworks_core::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID.to_owned(),
                builtin_angle_presets(),
            )
        })
}

/// The built-in 11 as resolved angle presets (the worker-side fallback when the store is
/// unreachable, sc-4450).
#[cfg(target_os = "macos")]
fn builtin_angle_presets() -> Vec<sceneworks_core::project_store::ResolvedAnglePreset> {
    use sceneworks_core::{angle_kps, project_store::ResolvedAnglePreset};
    angle_kps::BUILTIN_ANGLE_KPS
        .iter()
        .map(|(angle, kps)| ResolvedAnglePreset {
            preset_id: angle_kps::builtin_preset_id(angle),
            name: angle_kps::builtin_angle_display_name(angle),
            angle: Some((*angle).to_owned()),
            kps: *kps,
        })
        .collect()
}

/// Resolve InstantID denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` →
/// the torch-parity default (30).
#[cfg(target_os = "macos")]
fn instantid_steps(request: &ImageRequest) -> u32 {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    };
    request
        .advanced
        .get("steps")
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get("steps").and_then(parse))
        .map(|steps| steps.clamp(1, 80) as u32)
        .unwrap_or(INSTANTID_DEFAULT_STEPS)
}

/// Resolve InstantID guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the
/// RealVisXL-tuned default (3.0). Clamped to a sane CFG range.
#[cfg(target_os = "macos")]
fn instantid_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(INSTANTID_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Resolve InstantID quantization. **fp16 (dense) is the default** — the validated identity
/// envelope (ArcFace-cosine 0.82 @1024²); Q8/Q4 only on an explicit `advanced.mlxQuantize` /
/// manifest opt-in (identity drops to ~0.64 @512² and full-res quant is unvalidated). Returns
/// the engine `bits` (`Some(4)`/`Some(8)`/`None`) + the recipe bit count.
#[cfg(target_os = "macos")]
fn instantid_quant(request: &ImageRequest) -> (Option<i32>, Option<i64>) {
    let raw = request
        .advanced
        .get("mlxQuantize")
        .and_then(quant_int)
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(|mlx| mlx.get("quantize"))
                .and_then(quant_int)
        });
    match raw {
        Some(bits) if bits > 0 && bits <= 4 => (Some(4), Some(4)),
        Some(bits) if bits > 4 => (Some(8), Some(8)),
        // None / 0 / negative → fp16 (the default + the validated InstantID envelope).
        _ => (None, None),
    }
}

/// Flat telemetry recorded on InstantID assets (parity with `mlx_raw_settings` + the torch
/// `InstantIDAdapter` recipe keys).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn instantid_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    ip_scale: f32,
    controlnet_scale: f32,
    angle_set: bool,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("ipAdapterScale".to_owned(), json!(ip_scale));
    raw.insert(
        "controlnetConditioningScale".to_owned(),
        json!(controlnet_scale),
    );
    raw.insert(
        "instantIdEngine".to_owned(),
        Value::String("mlx_instantid".to_owned()),
    );
    if angle_set {
        raw.insert("angleSet".to_owned(), Value::Bool(true));
    }
    raw
}

/// Resolve a single InstantID weight file: return it if already present in `dir`, else
/// download `url` into `dir` (atomic `.tmp` + rename, so a partial download is never mistaken
/// for a complete one — same shape as `person_segment::ensure_segmenter_weights`).
#[cfg(target_os = "macos")]
async fn ensure_instantid_file(
    client: &reqwest::Client,
    dir: &Path,
    name: &str,
    url: &str,
) -> WorkerResult<PathBuf> {
    ensure_cached_file(client, url, &dir.join(name)).await
}

/// Resolve only the SCRFD detector weights (`scrfd_10g.safetensors`) from the same converted
/// bundle InstantID uses — for the standalone kps-extraction capability (sc-4433), which needs
/// face detection but neither ArcFace nor the SDXL/IdentityNet stack. Shares the env override
/// (`SCENEWORKS_INSTANTID_WEIGHTS`) + app cache + download-on-first-use with
/// [`ensure_instantid_weights`], so a prior InstantID run leaves it already cached.
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_scrfd_weights(settings: &Settings) -> WorkerResult<PathBuf> {
    let client = reqwest::Client::new();
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    let base = format!("https://huggingface.co/{INSTANTID_MLX_REPO}/resolve/main");
    ensure_instantid_file(
        &client,
        &bundle_dir,
        INSTANTID_SCRFD_FILE,
        &format!("{base}/{INSTANTID_SCRFD_FILE}"),
    )
    .await
}

/// Resolve all InstantID weight inputs, downloading the small converted bundle + the stock
/// IdentityNet on first use. Returns `(identitynet_dir, ip_adapter, scrfd, arcface)` — all
/// `Send` paths; the `!Send` MLX load happens on the blocking thread. Resolution order favours
/// an env override / the HF cache before any network fetch.
#[cfg(target_os = "macos")]
async fn ensure_instantid_weights(
    settings: &Settings,
) -> WorkerResult<(WeightsSource, PathBuf, PathBuf, PathBuf)> {
    let client = reqwest::Client::new();

    // Converted bundle (ip-adapter + face stack): an env-pinned dir (pre-staged for local
    // validation) wins, else the app cache (download missing files from SceneWorks/instantid-mlx).
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    let base = format!("https://huggingface.co/{INSTANTID_MLX_REPO}/resolve/main");
    let ip_adapter = ensure_instantid_file(
        &client,
        &bundle_dir,
        INSTANTID_IP_ADAPTER_FILE,
        &format!("{base}/{INSTANTID_IP_ADAPTER_FILE}"),
    )
    .await?;
    let scrfd = ensure_instantid_file(
        &client,
        &bundle_dir,
        INSTANTID_SCRFD_FILE,
        &format!("{base}/{INSTANTID_SCRFD_FILE}"),
    )
    .await?;
    let arcface = ensure_instantid_file(
        &client,
        &bundle_dir,
        INSTANTID_ARCFACE_FILE,
        &format!("{base}/{INSTANTID_ARCFACE_FILE}"),
    )
    .await?;

    // IdentityNet (stock InstantX ControlNetModel): env override → HF cache snapshot →
    // download the two files into the app cache.
    if let Ok(dir) = std::env::var("SCENEWORKS_INSTANTID_CONTROLNET") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok((WeightsSource::Dir(dir), ip_adapter, scrfd, arcface));
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, INSTANTID_CONTROLNET_REPO)
    {
        let controlnet = snapshot.join("ControlNetModel");
        if controlnet
            .join("diffusion_pytorch_model.safetensors")
            .exists()
        {
            return Ok((WeightsSource::Dir(controlnet), ip_adapter, scrfd, arcface));
        }
    }
    let controlnet_dir = settings.data_dir.join("cache").join("instantid-controlnet");
    let cn_base =
        format!("https://huggingface.co/{INSTANTID_CONTROLNET_REPO}/resolve/main/ControlNetModel");
    for file in INSTANTID_CONTROLNET_FILES {
        ensure_instantid_file(&client, &controlnet_dir, file, &format!("{cn_base}/{file}")).await?;
    }
    Ok((
        WeightsSource::Dir(controlnet_dir),
        ip_adapter,
        scrfd,
        arcface,
    ))
}

/// Resolve the xinsir OpenPose-SDXL ControlNet dir for pose mode: env override
/// (`SCENEWORKS_INSTANTID_OPENPOSE`) → HF cache snapshot → download the two files on first use. A
/// stock diffusers SDXL ControlNet (loads via `with_openpose`/`load_controlnet`, no conversion).
#[cfg(target_os = "macos")]
async fn ensure_instantid_openpose(settings: &Settings) -> WorkerResult<WeightsSource> {
    if let Ok(dir) = std::env::var("SCENEWORKS_INSTANTID_OPENPOSE") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok(WeightsSource::Dir(dir));
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, INSTANTID_OPENPOSE_REPO) {
        if snapshot
            .join("diffusion_pytorch_model.safetensors")
            .exists()
        {
            return Ok(WeightsSource::Dir(snapshot));
        }
    }
    let client = reqwest::Client::new();
    let dir = settings.data_dir.join("cache").join("instantid-openpose");
    let base = format!("https://huggingface.co/{INSTANTID_OPENPOSE_REPO}/resolve/main");
    for file in INSTANTID_CONTROLNET_FILES {
        ensure_instantid_file(&client, &dir, file, &format!("{base}/{file}")).await?;
    }
    Ok(WeightsSource::Dir(dir))
}

/// Real InstantID generation: resolve the reference + weights on the async side, then load the
/// bespoke `InstantId` provider once + generate each image on the blocking thread (the MLX
/// model is `!Send`). Three modes (torch parity): single identity (`generate`), the 11-view angle
/// set (`generate_with_kps` from the worker-owned [`INSTANTID_ANGLE_KPS`] presets — sc-4424), and
/// the pose-library set (`generate_pose`, MultiControlNet IdentityNet with xinsir OpenPose —
/// sc-3117). `advanced.faceRestore` adds the ADetailer-style re-render pass (`restore_face`,
/// sc-3380) on each output. The engine `generate*` take the per-job `CancelFlag` (via
/// `InstantIdRequest.cancel`) and a `Progress` callback (sc-4380/sc-4382), so streaming is
/// per-step (`Step`/`Decoding` events) and cancellation is honoured mid-denoise — same contract
/// as the registry families. Reuses [`consume_gen_events`] for the asset writes.
#[cfg(target_os = "macos")]
async fn generate_instantid_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sdxl_base = resolve_instantid_sdxl_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("InstantID base (RealVisXL) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("InstantID requires a reference face image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;

    let (controlnet, ip_adapter, scrfd_path, arcface_path) =
        ensure_instantid_weights(settings).await?;

    let steps = instantid_steps(request);
    let guidance = instantid_guidance(request);
    let (quant_bits, recipe_bits) = instantid_quant(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        INSTANTID_IP_SCALE,
        0.0..=1.0,
    );
    let controlnet_scale = advanced::f32_clamped(
        &request.advanced,
        "controlnetConditioningScale",
        INSTANTID_CONTROLNET_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(INSTANTID_SDXL_REPO)
        .to_owned();
    let mode = instantid_mode(request);
    let angle_set = matches!(mode, InstantIdMode::AngleSet);
    let pose_set = matches!(mode, InstantIdMode::PoseSet(_));
    // The active Key Point Library collection drives the angle set (sc-4450): per-generation
    // override > user default > built-in 11. Resolved once (and only for angle jobs).
    let angle_collection = angle_set.then(|| active_angle_collection(request, settings));
    let openpose_scale = advanced::f32_clamped(
        &request.advanced,
        "openPoseScale",
        INSTANTID_OPENPOSE_SCALE,
        0.0..=2.0,
    );
    let face_restore = advanced::flag(&request.advanced, "faceRestore");
    // Load the xinsir OpenPose ControlNet only for pose mode (it is the MultiControlNet second
    // branch; identity/angle modes don't need it).
    let openpose = if pose_set {
        Some(ensure_instantid_openpose(settings).await?)
    } else {
        None
    };

    let mut raw_settings = instantid_raw_settings(
        request,
        &repo,
        steps,
        recipe_bits,
        guidance,
        ip_scale,
        controlnet_scale,
        angle_set,
    );
    if pose_set {
        raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        raw_settings.insert("openPoseScale".to_owned(), json!(openpose_scale));
    }
    if face_restore {
        raw_settings.insert("faceRestore".to_owned(), Value::Bool(true));
    }
    // Record which collection + ordered presets produced the set, so each asset (by index) maps
    // back to the preset that rendered it (sc-4450).
    if let Some((collection_id, presets)) = &angle_collection {
        raw_settings.insert("keypointCollectionId".to_owned(), json!(collection_id));
        raw_settings.insert(
            "anglePresetIds".to_owned(),
            json!(presets
                .iter()
                .map(|preset| preset.preset_id.clone())
                .collect::<Vec<_>>()),
        );
    }

    // Per-image work items: (seed, prompt, action). Pose + angle sets share one seed (only the
    // pose changes across the set — noise-derived attributes stay constant); single identity is
    // per-seed at the reference's natural pose.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String, InstantIdAction)> = match &mode {
        InstantIdMode::PoseSet(_) => {
            let set_seed = resolve_seed(request, 0);
            parse_poses(request)
                .into_iter()
                .map(|pose| {
                    (
                        set_seed,
                        request.prompt.clone(),
                        InstantIdAction::Pose(pose_to_body_points(&pose.keypoints)),
                    )
                })
                .collect()
        }
        InstantIdMode::AngleSet => {
            let set_seed = resolve_seed(request, 0);
            // One image per preset in the active collection's order (sc-4450). Built-in presets
            // carry their canonical angle so the prompt still gets the per-angle clause; custom
            // presets render to their kps with the base prompt.
            let presets = angle_collection
                .as_ref()
                .map(|(_, presets)| presets.clone())
                .unwrap_or_else(builtin_angle_presets);
            presets
                .into_iter()
                .map(|preset| {
                    let prompt = match &preset.angle {
                        Some(angle) => augment_prompt_for_angle(&request.prompt, angle),
                        None => request.prompt.clone(),
                    };
                    (set_seed, prompt, InstantIdAction::Angle(preset.kps))
                })
                .collect()
        }
        InstantIdMode::Identity => (0..request.count as usize)
            .map(|index| {
                (
                    resolve_seed(request, index),
                    request.prompt.clone(),
                    InstantIdAction::Identity,
                )
            })
            .collect(),
    };
    let total = work.len();

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let negative_prompt = request.negative_prompt.clone();
        let cancel = cancel.clone();
        let job_id = job.id.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            emit_load_event("image_pipeline_load_start", &job_id, "instantid", 0);
            let paths = InstantIdPaths {
                sdxl_base,
                identitynet: controlnet,
                ip_adapter,
            };
            let model = InstantId::load(&paths).map_err(|error| {
                WorkerError::InvalidPayload(format!("InstantID load failed: {error}"))
            })?;
            // Attach OpenPose (pose mode) BEFORE quantize so it quantizes with the stack; quantize
            // before with_face (the engine's documented order).
            let model = match &openpose {
                Some(source) => model.with_openpose(source).map_err(|error| {
                    WorkerError::InvalidPayload(format!("InstantID OpenPose load failed: {error}"))
                })?,
                None => model,
            };
            let model = match quant_bits {
                Some(bits) => model.quantize(bits).map_err(|error| {
                    WorkerError::InvalidPayload(format!("InstantID quantize failed: {error}"))
                })?,
                None => model,
            };
            let scrfd = Weights::from_file(&scrfd_path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "InstantID SCRFD weights {scrfd_path:?}: {error}"
                ))
            })?;
            let arcface = Weights::from_file(&arcface_path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "InstantID ArcFace weights {arcface_path:?}: {error}"
                ))
            })?;
            let model = model.with_face(&scrfd, &arcface).map_err(|error| {
                WorkerError::InvalidPayload(format!("InstantID face stack: {error}"))
            })?;
            // Face-restore needs the reference identity embedding (imposed on the re-rendered
            // crop). Detect it once on the raw reference.
            let restore_embedding = if face_restore {
                Some(
                    model
                        .largest_face(
                            &reference.pixels,
                            reference.height as usize,
                            reference.width as usize,
                        )
                        .map_err(|error| {
                            WorkerError::InvalidPayload(format!(
                                "InstantID face-restore reference: {error}"
                            ))
                        })?
                        .embedding,
                )
            } else {
                None
            };
            emit_load_event("image_pipeline_load_complete", &job_id, "instantid", 0);

            for (index, (seed, prompt, action)) in work.into_iter().enumerate() {
                if cancel.is_cancelled() {
                    break;
                }
                // Per-step progress → GenEvent::Step, so `consume_gen_events` streams step
                // updates, fires `image_inference_start`, and polls the cancel API (sc-4382 —
                // without Step events an InstantID job could never be cancelled).
                let mut on_progress = |progress: Progress| {
                    let event = match progress {
                        Progress::Step { current, total } => GenEvent::Step {
                            index,
                            current,
                            total,
                        },
                        Progress::Decoding => GenEvent::Decoding { index },
                    };
                    let _ = tx.blocking_send(event);
                };
                // Angle + pose sets use a square canvas (the engine forces `req.height =
                // req.width` for the canonical landmark/skeleton — the sc-2009 kps-aspect rule);
                // single identity keeps the requested W×H (the engine letterboxes the reference).
                let req = InstantIdRequest {
                    prompt,
                    negative: negative_prompt.clone(),
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    ip_adapter_scale: ip_scale,
                    controlnet_scale,
                    openpose_scale,
                    seed: seed as u64,
                    cancel: cancel.clone(),
                };
                let result = match &action {
                    InstantIdAction::Identity => model.generate(&req, &reference, &mut on_progress),
                    InstantIdAction::Angle(kps) => {
                        model.generate_with_kps(&req, &reference, kps, &mut on_progress)
                    }
                    InstantIdAction::Pose(keypoints) => {
                        model.generate_pose(&req, &reference, keypoints, &mut on_progress)
                    }
                };
                let mut out = match result {
                    Ok(out) => out,
                    // A cancel tripped mid-denoise surfaces as the engine's cancelled error —
                    // stop cleanly (consume_gen_events posts the Canceled update).
                    Err(_) if cancel.is_cancelled() => break,
                    Err(error) => {
                        return Err(WorkerError::InvalidPayload(format!(
                            "InstantID generation failed: {error}"
                        )))
                    }
                };
                // Optional ADetailer-style face-restore re-render (sc-3380), imposing the
                // reference identity on the cropped face with the gender-neutral restore prompt.
                if let Some(embedding) = &restore_embedding {
                    let restore_req = InstantIdRequest {
                        prompt: FACE_RESTORE_PROMPT.to_owned(),
                        negative: negative_prompt.clone(),
                        width: INSTANTID_FACE_RESTORE_SIDE,
                        height: INSTANTID_FACE_RESTORE_SIDE,
                        steps: steps as usize,
                        guidance,
                        ip_adapter_scale: ip_scale,
                        controlnet_scale,
                        openpose_scale,
                        seed: seed as u64,
                        cancel: cancel.clone(),
                    };
                    out = match model.restore_face(&restore_req, &out, embedding, &mut on_progress)
                    {
                        Ok(out) => out,
                        Err(_) if cancel.is_cancelled() => break,
                        Err(error) => {
                            return Err(WorkerError::InvalidPayload(format!(
                                "InstantID face-restore failed: {error}"
                            )))
                        }
                    };
                }
                if tx
                    .blocking_send(GenEvent::Image {
                        index,
                        seed,
                        width: out.width,
                        height: out.height,
                        pixels: out.pixels,
                    })
                    .is_err()
                {
                    break; // receiver gone — stop generating.
                }
            }
            Ok(())
        })
    };

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        "mlx_instantid",
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Tile-ControlNet detail refine (macOS, epic 3041 / sc-3060): the standalone
// `image_detail` job (Image Editor, epic 2427). Faithful port of the Python
// `run_image_detail` + `_refine_tiled_detail` (image_adapters.py) onto the engine's
// SDXL tile-ControlNet path: each tile is img2img-refined with itself as both the
// init (Reference) and the ControlNet conditioning (Control, control=same), then
// recomposed with a raised-cosine feather over the overlap. Unlike the diffusers
// pipeline, the engine requires width/height ∈ [512, 2048] and multiples of 8, so a
// tile is run at the nearest valid size and the result resized back before blending.
// ---------------------------------------------------------------------------

/// The xinsir tile ControlNet repo (parity with Python `TILE_CONTROLNET_REPO`).
#[cfg(target_os = "macos")]
const TILE_CONTROLNET_REPO: &str = "xinsir/controlnet-tile-sdxl-1.0";
#[cfg(target_os = "macos")]
const DETAIL_DEFAULT_PROMPT: &str = "ultra detailed, sharp focus, fine texture, high quality";
#[cfg(target_os = "macos")]
const DETAIL_DEFAULT_NEGATIVE: &str = "blurry, soft, lowres, smooth, plastic";

/// The locked detail recipe (sc-2437 round-2 spike defaults), resolved from `advanced`.
#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DetailParams {
    strength: f32,
    cn_scale: f32,
    steps: u32,
    guidance: f32,
    tile: u32,
    overlap: u32,
    prompt: String,
    negative: String,
    seed: i64,
}

#[cfg(target_os = "macos")]
fn resolve_detail_params(request: &ImageRequest) -> DetailParams {
    DetailParams {
        strength: advanced::f32_clamped(&request.advanced, "strength", 0.55, 0.2..=1.0),
        cn_scale: advanced::f32_clamped(&request.advanced, "cnScale", 0.7, 0.1..=1.5),
        steps: advanced::u32_clamped(&request.advanced, "steps", 24, 1..=60),
        guidance: advanced::f32_clamped(&request.advanced, "guidanceScale", 5.0, 1.0..=15.0),
        tile: advanced::u32_clamped(&request.advanced, "tile", 1024, 512..=1536),
        overlap: advanced::u32_clamped(&request.advanced, "overlap", 128, 0..=512),
        prompt: advanced::str(&request.advanced, "prompt", DETAIL_DEFAULT_PROMPT),
        negative: advanced::str(&request.advanced, "negativePrompt", DETAIL_DEFAULT_NEGATIVE),
        // Python defaults the detail seed to 7 when the payload omits one.
        seed: request.seed.unwrap_or(7),
    }
}

/// Round a tile dimension up to the nearest multiple of 8 and clamp to the engine's
/// `[512, 2048]` SDXL bounds, so an arbitrary-sized crop can be run through the engine.
#[cfg(target_os = "macos")]
fn engine_dim(value: u32) -> u32 {
    value.div_ceil(8).saturating_mul(8).clamp(512, 2048)
}

/// Raised-cosine alpha ramp over the `overlap` borders so tiles blend seamlessly
/// (parity with Python `_detail_feather`). Row-major `tile_h`×`tile_w` weights.
#[cfg(target_os = "macos")]
fn detail_feather(tile_w: u32, tile_h: u32, overlap: u32) -> Vec<f32> {
    fn ramp(n: u32, overlap: u32) -> Vec<f32> {
        let mut weights = vec![1.0f32; n as usize];
        if overlap > 0 && n > overlap {
            for index in 0..overlap as usize {
                let edge = 0.5
                    - 0.5 * (std::f32::consts::PI * (index as f32 + 0.5) / overlap as f32).cos();
                weights[index] = edge;
                weights[n as usize - 1 - index] = edge;
            }
        }
        weights
    }
    let wx = ramp(tile_w, overlap);
    let wy = ramp(tile_h, overlap);
    let mut out = Vec::with_capacity((tile_w * tile_h) as usize);
    for &vy in &wy {
        for &vx in &wx {
            out.push(vy * vx);
        }
    }
    out
}

/// Load the SDXL generator with the tile ControlNet overlay (per job, no cache).
#[cfg(target_os = "macos")]
fn detail_load(
    weights_dir: PathBuf,
    control_file: PathBuf,
    quant: Option<Quant>,
) -> WorkerResult<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_file));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    mlx_gen::load("sdxl", &spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("sdxl detail load failed: {error}")))
}

/// Refine one tile (already sized to engine-valid `eng_w`×`eng_h`): img2img on the tile
/// with the tile as the ControlNet image (control=same). Returns the refined RGB8 buffer.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn detail_refine_tile(
    generator: &dyn Generator,
    tile: Image,
    eng_w: u32,
    eng_h: u32,
    params: &DetailParams,
    seed: i64,
    cancel: &CancelFlag,
) -> WorkerResult<Vec<u8>> {
    let mut noop = |_progress: Progress| {};
    let request = GenerationRequest {
        prompt: params.prompt.clone(),
        negative_prompt: Some(params.negative.clone()),
        width: eng_w,
        height: eng_h,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(params.steps),
        guidance: Some(params.guidance),
        conditioning: vec![
            Conditioning::Reference {
                image: tile.clone(),
                strength: Some(params.strength),
            },
            Conditioning::Control {
                image: tile,
                kind: ControlKind::Other("tile".to_owned()),
                scale: params.cn_scale,
            },
        ],
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut noop)
        .map_err(|error| WorkerError::InvalidPayload(format!("detail tile failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => Ok(images
            .pop()
            .ok_or_else(|| WorkerError::InvalidPayload("detail tile produced no image".to_owned()))?
            .pixels),
        _ => Err(WorkerError::InvalidPayload(
            "detail tile returned non-image output".to_owned(),
        )),
    }
}

/// Tiled feathered detail refine (parity with Python `_refine_tiled_detail`). Returns the
/// recomposed image + the tile count. Runs on the blocking thread (the generator is `!Send`).
#[cfg(target_os = "macos")]
fn refine_tiled_detail(
    generator: &dyn Generator,
    source: &image::RgbImage,
    params: &DetailParams,
    cancel: &CancelFlag,
    on_tile: &mut dyn FnMut(usize, usize),
) -> WorkerResult<(image::RgbImage, usize)> {
    use image::imageops::FilterType::Lanczos3;
    let (width, height) = (source.width(), source.height());
    let step = params.tile.saturating_sub(params.overlap).max(1);
    let xs: Vec<u32> = (0..width.saturating_sub(params.overlap).max(1))
        .step_by(step as usize)
        .collect();
    let ys: Vec<u32> = (0..height.saturating_sub(params.overlap).max(1))
        .step_by(step as usize)
        .collect();
    let total = xs.len() * ys.len();
    let mut acc = vec![0.0f32; (width * height * 3) as usize];
    let mut wsum = vec![0.0f32; (width * height) as usize];
    let mut done = 0usize;
    for &y in &ys {
        for &x in &xs {
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(
                    "Detail enhancement canceled by user.".to_owned(),
                ));
            }
            let x0 = x.min(width.saturating_sub(params.tile));
            let y0 = y.min(height.saturating_sub(params.tile));
            let tile_w = params.tile.min(width - x0);
            let tile_h = params.tile.min(height - y0);
            let crop = image::imageops::crop_imm(source, x0, y0, tile_w, tile_h).to_image();
            // Run at an engine-valid size (mult-8, ≥512), then resize the refined tile back.
            let (eng_w, eng_h) = (engine_dim(tile_w), engine_dim(tile_h));
            let eng_crop = if (eng_w, eng_h) == (tile_w, tile_h) {
                crop
            } else {
                image::imageops::resize(&crop, eng_w, eng_h, Lanczos3)
            };
            let tile_img = Image {
                width: eng_w,
                height: eng_h,
                pixels: eng_crop.into_raw(),
            };
            let refined_px = detail_refine_tile(
                generator,
                tile_img,
                eng_w,
                eng_h,
                params,
                params.seed + done as i64,
                cancel,
            )?;
            let refined = image::RgbImage::from_raw(eng_w, eng_h, refined_px).ok_or_else(|| {
                WorkerError::InvalidPayload("detail refined tile size mismatch".to_owned())
            })?;
            let refined = if (eng_w, eng_h) == (tile_w, tile_h) {
                refined
            } else {
                image::imageops::resize(&refined, tile_w, tile_h, Lanczos3)
            };
            let feather = detail_feather(tile_w, tile_h, params.overlap);
            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    let f = feather[(ty * tile_w + tx) as usize];
                    let src = refined.get_pixel(tx, ty).0;
                    let gx = x0 + tx;
                    let gy = y0 + ty;
                    let acc_base = ((gy * width + gx) * 3) as usize;
                    acc[acc_base] += src[0] as f32 * f;
                    acc[acc_base + 1] += src[1] as f32 * f;
                    acc[acc_base + 2] += src[2] as f32 * f;
                    wsum[(gy * width + gx) as usize] += f;
                }
            }
            done += 1;
            on_tile(done, total);
        }
    }
    let mut out = image::RgbImage::new(width, height);
    for gy in 0..height {
        for gx in 0..width {
            let w = wsum[(gy * width + gx) as usize].max(1.0);
            let base = ((gy * width + gx) * 3) as usize;
            out.put_pixel(
                gx,
                gy,
                image::Rgb([
                    (acc[base] / w).clamp(0.0, 255.0) as u8,
                    (acc[base + 1] / w).clamp(0.0, 255.0) as u8,
                    (acc[base + 2] / w).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    Ok((out, total))
}

/// Build the detail child-asset fact (lineage to the source) + generation set, matching the
/// Python `run_image_detail` result shape so `persist_reported_assets` indexes it identically.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn detail_result(
    request: &ImageRequest,
    genset_id: &str,
    created_at: &str,
    asset_id: &str,
    media_rel: &str,
    model: &str,
    params: &DetailParams,
    tiles: usize,
    width: u32,
    height: u32,
) -> JsonObject {
    let source_asset_id = request.source_asset_id.clone().unwrap_or_default();
    let detail_settings = json!({
        "enabled": true,
        "backbone": model,
        "controlNet": TILE_CONTROLNET_REPO,
        "strength": params.strength,
        "cnScale": params.cn_scale,
        "steps": params.steps,
        "guidanceScale": params.guidance,
        "tile": params.tile,
        "overlap": params.overlap,
        "tiles": tiles,
        "width": width,
        "height": height,
    });
    let fact = json!({
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "type": "image",
        "width": width,
        "height": height,
        "normalizedWidth": width,
        "normalizedHeight": height,
        "count": 1,
        "seed": params.seed,
        "displayName": "Detail enhanced",
        "createdAt": created_at,
        "mode": "image_detail",
        "model": model,
        "adapter": "mlx_sdxl",
        "prompt": params.prompt,
        "negativePrompt": params.negative,
        "loras": [],
        "stylePreset": "",
        "sourceAssetId": source_asset_id,
        "rawAdapterSettings": { "detail": detail_settings, "realModelInference": true },
        "parents": [source_asset_id],
        "extra": {
            "isDetailEnhanced": true,
            "detailFromAssetId": source_asset_id,
            "backbone": model,
            "strength": params.strength,
            "cnScale": params.cn_scale,
        },
    });
    let generation_set = json!({
        "id": genset_id,
        "mode": "image_detail",
        "model": model,
        "prompt": params.prompt,
        "negativePrompt": params.negative,
        "count": 1,
        "createdAt": created_at,
    });
    json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": "mlx_sdxl",
        "model": model,
        "generationSet": generation_set,
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Native MLX tile-ControlNet detail refine (`JobType::ImageDetail`) on the macOS engine.
#[cfg(target_os = "macos")]
pub(crate) async fn run_image_detail_job(
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
    let model = if request.model.trim().is_empty() {
        "realvisxl".to_owned()
    } else {
        request.model.clone()
    };
    let engine_model = sdxl_engine_model(&model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{model} does not support detail enhancement."))
    })?;
    let source_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Detail-enhance jobs require a source image asset.".to_owned(),
            )
        })?
        .to_owned();

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    tokio::fs::create_dir_all(project_path.join("assets").join("images").join(&genset_id)).await?;
    let backend = backend_label(&settings.gpu_id);

    let params = resolve_detail_params(&request);
    let (quant, _) = resolve_quant(&request);
    // Reuse the model's manifest/modelPath/cache resolution; engine_model gives the default repo.
    let weights_dir = resolve_weights_dir(&request, settings)?
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, engine_model.default_repo));
    let weights_dir = weights_dir
        .ok_or_else(|| WorkerError::InvalidPayload("SDXL detail weights not found".to_owned()))?;
    let control_repo = advanced::str(
        &request.advanced,
        "tileControlNetRepo",
        TILE_CONTROLNET_REPO,
    );
    let control_dir =
        huggingface_snapshot_dir(&settings.data_dir, &control_repo).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "tile ControlNet weights not found (download {control_repo})."
            ))
        })?;
    let control_file = first_safetensors_path(&control_dir).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "no .safetensors under the tile ControlNet snapshot {}",
            control_dir.display()
        ))
    })?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Loading source image.",
            None,
            backend,
        ),
    )
    .await?;

    let source = engine_image_to_rgb(load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &source_id,
        &project_path,
    )?)?;

    let created_at = now_rfc3339();
    let asset_id = fresh_asset_id();
    let filename = format!("{}_detail_{}.png", &created_at[..10], &asset_id[6..14]);
    let media_rel = format!("assets/images/{genset_id}/{filename}");

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, usize)>(16);
    let blocking = {
        let params_ref = params.clone();
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<(image::RgbImage, usize)> {
            let generator = detail_load(weights_dir, control_file, quant)?;
            let mut on_tile = |done: usize, total: usize| {
                let _ = tx.blocking_send((done, total));
            };
            refine_tiled_detail(
                generator.as_ref(),
                &source,
                &params_ref,
                &cancel,
                &mut on_tile,
            )
        })
    };

    let mut last_cancel_check = Instant::now();
    while let Some((done, total)) = rx.recv().await {
        if last_cancel_check.elapsed() >= Duration::from_secs(2) {
            last_cancel_check = Instant::now();
            if cancel_requested(api, &job.id, "Detail enhancement canceled by user.").await {
                cancel.cancel();
            }
        }
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                0.45 + 0.5 * (done as f64 / total.max(1) as f64),
                &format!("Refining detail tile {done}/{total}."),
                None,
                backend,
            ),
        )
        .await?;
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }

    let (refined, tiles) = blocking
        .await
        .map_err(|error| WorkerError::InvalidPayload(format!("detail task join: {error}")))??;
    let (out_w, out_h) = (refined.width(), refined.height());
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");
    refined
        .save_with_format(&temp_path, image::ImageFormat::Png)
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let result = detail_result(
        &request,
        &genset_id,
        &created_at,
        &asset_id,
        &media_rel,
        &model,
        &params,
        tiles,
        out_w,
        out_h,
    );
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Detail enhancement complete.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

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
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> ImageRequest {
        ImageRequest::from_payload(&value.as_object().cloned().unwrap())
    }

    #[test]
    fn render_and_save_writes_png_and_contract_fact() {
        let dir = tempfile::tempdir().unwrap();
        let project_path = dir.path();
        std::fs::create_dir_all(project_path.join("assets").join("images")).unwrap();
        // Distinct dimensions (>= the 256 min, so they survive clamping) also catch a
        // width/height transpose in the encoder.
        let req = request(json!({
            "projectId": "p", "model": "z_image_turbo", "prompt": "Mist over hills",
            "count": 2, "width": 320, "height": 256, "seed": 101,
            "stylePreset": "cinematic", "modelManifestEntry": { "family": "z-image" }
        }));
        let plan = ImagePlan::new(&req);

        let seed = resolve_seed(&req, 0);
        let pixels = stub_rgb8(req.width, req.height, seed);
        let fact = write_image_asset(
            &plan,
            0,
            seed,
            req.width,
            req.height,
            pixels,
            STUB_ADAPTER,
            stub_raw_settings(&req),
            project_path,
        )
        .unwrap();

        let media_rel = fact.get("mediaPath").and_then(Value::as_str).unwrap();
        assert!(media_rel.starts_with(&format!("assets/images/{}/", plan.genset_id)));
        assert!(media_rel.ends_with("_0001.png"));
        let decoded = image::open(project_path.join(media_rel)).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (320, 256));

        for key in [
            "assetId",
            "mediaPath",
            "mimeType",
            "width",
            "height",
            "normalizedWidth",
            "normalizedHeight",
            "count",
            "family",
            "seed",
            "displayName",
            "createdAt",
            "mode",
            "model",
            "adapter",
            "prompt",
            "negativePrompt",
            "loras",
            "stylePreset",
            "characterId",
            "characterLookId",
            "sourceAssetId",
            "rawAdapterSettings",
        ] {
            assert!(fact.contains_key(key), "fact missing key {key}");
        }
        assert_eq!(fact["adapter"], json!("procedural_preview"));
        assert_eq!(fact["family"], json!("z-image"));
        assert_eq!(fact["seed"], json!(101));
        assert_eq!(fact["width"], json!(320));
        assert_eq!(fact["displayName"], json!("Mist over hills #1"));
        assert_eq!(
            fact["rawAdapterSettings"]["realModelInference"],
            json!(false)
        );
    }

    #[test]
    fn resolve_seed_matches_python_precedence() {
        // base seed wins (seed + index), even over an explicit seeds list.
        let base = request(json!({ "projectId": "p", "seed": 100, "seeds": [7, 8] }));
        assert_eq!(resolve_seed(&base, 0), 100);
        assert_eq!(resolve_seed(&base, 2), 102);
        // explicit per-image seeds when no base seed.
        let listed = request(json!({ "projectId": "p", "seeds": [7, 8] }));
        assert_eq!(resolve_seed(&listed, 1), 8);
        // deterministic hash fallback (same prompt+index -> same seed).
        let none = request(json!({ "projectId": "p", "prompt": "hello" }));
        assert_eq!(resolve_seed(&none, 0), resolve_seed(&none, 0));
        assert_ne!(resolve_seed(&none, 0), resolve_seed(&none, 1));
    }

    #[test]
    fn distinct_seeds_produce_distinct_pixels() {
        let a = stub_rgb8(8, 8, 1);
        let b = stub_rgb8(8, 8, 5000);
        assert_eq!(a.len(), 8 * 8 * 3);
        assert_ne!(a, b);
    }

    #[test]
    fn streaming_result_carries_facts_for_api_persistence() {
        let plan = ImagePlan::new(&request(
            json!({ "projectId": "p", "prompt": "x", "count": 1 }),
        ));
        let writes = vec![json!({ "assetId": "a1" })];
        let result = streaming_result(&plan, &writes);
        assert_eq!(result["generationSetId"], json!(plan.genset_id));
        assert_eq!(result["assetWrites"].as_array().map(Vec::len), Some(1));
        assert!(result.contains_key("generationSet"));
    }

    #[test]
    fn backend_label_defaults_empty_to_cpu() {
        assert_eq!(backend_label("mlx"), "mlx");
        assert_eq!(backend_label(""), "cpu");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn quant_mapping_defaults_to_q8_and_maps_bits() {
        use mlx_gen::Quant;
        let default = request(json!({ "projectId": "p" }));
        assert!(matches!(
            resolve_quant(&default),
            (Some(Quant::Q8), Some(8))
        ));
        let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert!(matches!(resolve_quant(&q4), (Some(Quant::Q4), Some(4))));
        let dense = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert!(matches!(resolve_quant(&dense), (None, None)));
        let six = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 6 } }));
        assert!(matches!(resolve_quant(&six), (Some(Quant::Q8), Some(8))));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn steps_default_is_family_default_and_clamps() {
        let zimage = mlx_model("z_image_turbo").unwrap();
        let schnell = mlx_model("flux_schnell").unwrap();
        let dev = mlx_model("flux_dev").unwrap();
        // Family defaults (Python MODEL_TARGETS parity): z-image 8, schnell 4, dev 28.
        assert_eq!(
            resolve_steps(&request(json!({ "projectId": "p" })), zimage),
            8
        );
        assert_eq!(
            resolve_steps(&request(json!({ "projectId": "p" })), schnell),
            4
        );
        assert_eq!(
            resolve_steps(&request(json!({ "projectId": "p" })), dev),
            28
        );
        // advanced.steps overrides, clamped to 1..=80.
        assert_eq!(
            resolve_steps(
                &request(json!({ "projectId": "p", "advanced": { "steps": 200 } })),
                dev
            ),
            80
        );
        assert_eq!(
            resolve_steps(
                &request(json!({ "projectId": "p", "advanced": { "steps": 12 } })),
                schnell
            ),
            12
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_model_table_maps_known_families() {
        assert_eq!(
            mlx_model("z_image_turbo").unwrap().engine_id,
            "z_image_turbo"
        );
        assert_eq!(
            mlx_model("flux_schnell").unwrap().engine_id,
            "flux1_schnell"
        );
        assert_eq!(mlx_model("flux_dev").unwrap().engine_id, "flux1_dev");
        assert_eq!(mlx_model("flux_dev").unwrap().adapter_label, "mlx_flux");
        let qwen = mlx_model("qwen_image").unwrap();
        assert_eq!(qwen.engine_id, "qwen_image");
        assert_eq!(qwen.adapter_label, "mlx_qwen");
        assert_eq!(qwen.default_steps, 20);
        assert!(qwen.supports_guidance && qwen.supports_negative_prompt);
        // All three FLUX.2-klein variants share the engine's single txt2img model.
        for id in [
            "flux2_klein_9b",
            "flux2_klein_9b_kv",
            "flux2_klein_9b_true_v2",
        ] {
            let m = mlx_model(id).unwrap();
            assert_eq!(m.engine_id, "flux2_klein_9b");
            assert_eq!(m.adapter_label, "mlx_flux2");
            assert!(m.supports_guidance && !m.supports_negative_prompt);
        }
        // Distilled variants are 4-step; the undistilled true_v2 is 24-step.
        assert_eq!(mlx_model("flux2_klein_9b").unwrap().default_steps, 4);
        assert_eq!(mlx_model("flux2_klein_9b_kv").unwrap().default_steps, 4);
        assert_eq!(
            mlx_model("flux2_klein_9b_true_v2").unwrap().default_steps,
            24
        );
        // SDXL + the realvisxl finetune share the single `sdxl` engine model (real CFG).
        for id in ["sdxl", "realvisxl"] {
            let m = mlx_model(id).unwrap();
            assert_eq!(m.engine_id, "sdxl");
            assert_eq!(m.adapter_label, "mlx_sdxl");
            assert_eq!(m.default_steps, 30);
            assert!(m.supports_guidance && m.supports_negative_prompt);
        }
        assert!(mlx_model("instantid_sdxl").is_none());

        // SenseNova-U1 (sc-3900): base + 8-step distill `_fast`, each its own engine id. Uses
        // text guidance (4.0 base / 1.0 fast) AND image guidance (true_cfg) but advertises NO
        // negative prompt — so it is NOT a `uses_true_cfg` family (see `uses_true_cfg`).
        let base = mlx_model("sensenova_u1_8b").unwrap();
        assert_eq!(base.engine_id, "sensenova_u1_8b");
        assert_eq!(base.default_repo, "sensenova/SenseNova-U1-8B-MoT");
        assert_eq!(base.default_steps, 50);
        assert_eq!(base.default_guidance, 4.0);
        assert_eq!(base.adapter_label, "mlx_sensenova");
        assert!(base.supports_guidance && !base.supports_negative_prompt);
        assert!(!uses_true_cfg(base), "dual-CFG, not a true-CFG-only family");
        let fast = mlx_model("sensenova_u1_8b_fast").unwrap();
        assert_eq!(fast.engine_id, "sensenova_u1_8b_fast");
        assert_eq!(fast.default_steps, 8);
        assert_eq!(fast.default_guidance, 1.0);
        assert_eq!(fast.adapter_label, "mlx_sensenova");
        assert!(fast.supports_guidance && !fast.supports_negative_prompt);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sensenova_dual_cfg_and_shift_resolve_per_mode() {
        // Image guidance (true_cfg): edit default 1.0, character default 1.5, override via
        // imageGuidanceScale, floored at 1.0.
        assert_eq!(
            resolve_sensenova_img_cfg(&request(json!({ "projectId": "p", "mode": "edit_image" }))),
            1.0
        );
        assert_eq!(
            resolve_sensenova_img_cfg(&request(
                json!({ "projectId": "p", "mode": "character_image" })
            )),
            1.5
        );
        assert_eq!(
            resolve_sensenova_img_cfg(&request(json!({
                "projectId": "p", "mode": "character_image",
                "advanced": { "imageGuidanceScale": 2.5 }
            }))),
            2.5
        );
        assert_eq!(
            resolve_sensenova_img_cfg(&request(json!({
                "projectId": "p", "mode": "edit_image",
                "advanced": { "imageGuidanceScale": 0.2 }
            }))),
            1.0,
            "img cfg is floored at 1.0"
        );
        // Timestep shift: default 3.0, schedulerShift (or legacy timestepShift) overrides,
        // non-positive falls back to 3.0.
        assert_eq!(
            resolve_sensenova_timestep_shift(&request(json!({ "projectId": "p" }))),
            3.0
        );
        assert_eq!(
            resolve_sensenova_timestep_shift(&request(json!({
                "projectId": "p", "advanced": { "schedulerShift": 4.5 }
            }))),
            4.5
        );
        assert_eq!(
            resolve_sensenova_timestep_shift(&request(json!({
                "projectId": "p", "advanced": { "timestepShift": 2.0 }
            }))),
            2.0
        );
        assert_eq!(
            resolve_sensenova_timestep_shift(&request(json!({
                "projectId": "p", "advanced": { "schedulerShift": 0.0 }
            }))),
            3.0
        );
        // 32-cell snap, clamped to [256, 2048].
        assert_eq!(sensenova_dim(1536), 1536); // already 32-aligned
        assert_eq!(sensenova_dim(1000), 1024); // rounds up to the next multiple of 32
        assert_eq!(sensenova_dim(100), 256); // clamps to the minimum
        assert_eq!(sensenova_dim(5000), 2048); // clamps to the maximum
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sensenova_edit_available_needs_a_reference() {
        // Plain T2I (no reference) is NOT the edit path — it rides the base mlx path.
        assert!(!sensenova_edit_available(
            &request(json!({ "projectId": "p", "model": "sensenova_u1_8b", "prompt": "a fox" })),
            &Settings::from_env()
        ));
        // edit_image needs a source; character_image needs a reference. (Weights may be
        // absent in CI, so only assert the negative/structural cases here.)
        assert!(qwen_edit_reference_ids(&request(json!({
            "projectId": "p", "model": "sensenova_u1_8b", "mode": "edit_image", "sourceAssetId": "s"
        })))
        .contains(&"s".to_owned()));
        assert!(qwen_edit_reference_ids(&request(json!({
            "projectId": "p", "model": "sensenova_u1_8b", "mode": "character_image",
            "referenceAssetId": "r"
        })))
        .contains(&"r".to_owned()));
    }

    /// Real-weights smoke: SenseNova-U1 it2i. Loads `sensenova_u1_8b` (the ~35GB
    /// `sensenova/SenseNova-U1-8B-MoT` snapshot) and generates one image conditioned on a synthetic
    /// reference via the worker's dual-CFG it2i path (text `guidance` + image `true_cfg` +
    /// `scheduler_shift`). The worker-level entry for the sc-3900 parity gate (component +
    /// early-step + coherence, not pixel bit-parity — the port runs f32 vs the bf16 reference).
    /// Needs the HF cache + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_it2i_real_weights`.
    /// Uses 8 steps + 512² for speed (the production base default is 50 steps).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + Metal device"]
    fn sensenova_it2i_real_weights_generates_one_image() {
        let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
        let generator = mlx_load(
            mlx_model("sensenova_u1_8b").unwrap().engine_id,
            snapshot,
            Some(mlx_gen::Quant::Q8),
            Vec::new(),
        )
        .unwrap();
        let reference = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: stub_rgb8(512, 512, 7),
        };
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = sensenova_edit_generate_one(
            generator.as_ref(),
            "make it a watercolor painting",
            512,
            512,
            42,
            8,
            Some(4.0), // text CFG
            1.0,       // image CFG (edit default)
            3.0,       // timestep shift
            build_edit_conditioning(std::slice::from_ref(&reference)),
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_negative_prompt_only_for_true_cfg_families() {
        let qwen = mlx_model("qwen_image").unwrap();
        let flux = mlx_model("flux_dev").unwrap();
        // qwen (true CFG) passes a non-empty negative prompt; empty → None (fallback).
        assert_eq!(
            resolve_negative_prompt(
                &request(json!({ "projectId": "p", "negativePrompt": "blurry" })),
                qwen
            ),
            Some("blurry".to_owned())
        );
        assert_eq!(
            resolve_negative_prompt(
                &request(json!({ "projectId": "p", "negativePrompt": "  " })),
                qwen
            ),
            None
        );
        // Non-true-CFG families never pass a negative prompt (the engine rejects it).
        assert_eq!(
            resolve_negative_prompt(
                &request(json!({ "projectId": "p", "negativePrompt": "blurry" })),
                flux
            ),
            None
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_guidance_none_for_distilled_set_for_dev() {
        let schnell = mlx_model("flux_schnell").unwrap();
        let dev = mlx_model("flux_dev").unwrap();
        let zimage = mlx_model("z_image_turbo").unwrap();
        // Distilled variants take no guidance (the engine rejects Some on them).
        assert_eq!(
            resolve_guidance(&request(json!({ "projectId": "p" })), schnell),
            None
        );
        assert_eq!(
            resolve_guidance(&request(json!({ "projectId": "p" })), zimage),
            None
        );
        // flux dev defaults to 3.5, overridable via advanced.guidanceScale.
        assert_eq!(
            resolve_guidance(&request(json!({ "projectId": "p" })), dev),
            Some(3.5)
        );
        assert_eq!(
            resolve_guidance(
                &request(json!({ "projectId": "p", "advanced": { "guidanceScale": 2.0 } })),
                dev
            ),
            Some(2.0)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn adapter_id_reports_per_family_mlx_label() {
        assert_eq!(
            adapter_id(&request(json!({ "model": "z_image_turbo" }))),
            "mlx_z_image"
        );
        assert_eq!(
            adapter_id(&request(json!({ "model": "flux_schnell" }))),
            "mlx_flux"
        );
        assert_eq!(
            adapter_id(&request(json!({ "model": "flux_dev" }))),
            "mlx_flux"
        );
        assert_eq!(adapter_id(&request(json!({ "model": "sdxl" }))), "mlx_sdxl");
        // Kolors base T2I is MLX-routed now (sc-3875) → records the mlx_kolors adapter label.
        assert_eq!(
            adapter_id(&request(json!({ "model": "kolors" }))),
            "mlx_kolors"
        );
        // A torch-only model with no mlx-gen engine records the procedural stub adapter.
        assert_eq!(
            adapter_id(&request(json!({ "model": "pulid_flux_dev" }))),
            "procedural_preview"
        );
    }

    /// The Z-Image + FLUX.1 + Qwen-Image providers linked into the worker
    /// self-registered via inventory.
    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_engine_registry_links_image_families() {
        let ids: Vec<&str> = mlx_gen::registry::generators()
            .map(|reg| (reg.descriptor)().id)
            .collect();
        for id in [
            "z_image_turbo",
            "flux1_schnell",
            "flux1_dev",
            "qwen_image",
            "qwen_image_control",
            "qwen_image_edit",
            "flux2_klein_9b",
            "sdxl",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "kolors",
        ] {
            assert!(ids.contains(&id), "registry missing {id}");
        }
    }

    /// Resolve a HuggingFace cache snapshot dir for `models--<dir>` (test helper).
    #[cfg(target_os = "macos")]
    fn hf_snapshot(model_dir: &str) -> std::path::PathBuf {
        std::fs::read_dir(dirs_home().join(format!(".cache/huggingface/hub/{model_dir}/snapshots")))
            .expect("HF cache snapshots dir")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .expect("a snapshot dir")
    }

    /// Load + generate one small image through the public mlx-gen path (test helper).
    /// Keyed by SceneWorks model id — the engine id + step default come from the table,
    /// so several SceneWorks ids can share one engine id (e.g. the FLUX.2 variants).
    #[cfg(target_os = "macos")]
    fn smoke_generate_one(
        sceneworks_id: &str,
        snapshot: std::path::PathBuf,
        guidance: Option<f32>,
        negative_prompt: Option<String>,
    ) {
        let model = mlx_model(sceneworks_id).unwrap();
        let generator = mlx_load(
            model.engine_id,
            snapshot,
            Some(mlx_gen::Quant::Q8),
            Vec::new(),
        )
        .unwrap();
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let steps = model.default_steps;
        let (w, h, pixels) = mlx_generate_one(
            generator.as_ref(),
            "a serene mountain lake at dawn",
            512,
            512,
            42,
            steps,
            guidance,
            negative_prompt,
            None,
            None,
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        // Not a flat image.
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    /// Real-weights smoke: load + generate one small Z-Image image. Needs the HF cache
    /// (`Tongyi-MAI/Z-Image-Turbo`) + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored zimage_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Z-Image weights + Metal device"]
    fn zimage_real_weights_generates_one_image() {
        smoke_generate_one(
            "z_image_turbo",
            hf_snapshot("models--Tongyi-MAI--Z-Image-Turbo"),
            None,
            None,
        );
    }

    /// Real-weights smoke: load + generate one small FLUX.1-schnell image (4-step,
    /// guidance-distilled). Needs the HF cache (`black-forest-labs/FLUX.1-schnell`) +
    /// a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored flux_schnell_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real FLUX.1-schnell weights + Metal device"]
    fn flux_schnell_real_weights_generates_one_image() {
        smoke_generate_one(
            "flux_schnell",
            hf_snapshot("models--black-forest-labs--FLUX.1-schnell"),
            None,
            None,
        );
    }

    /// Real-weights smoke: load + generate one small FLUX.1-dev image (guided, 28-step).
    /// Needs the HF cache (`black-forest-labs/FLUX.1-dev`) + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored flux_dev_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real FLUX.1-dev weights + Metal device"]
    fn flux_dev_real_weights_generates_one_image() {
        smoke_generate_one(
            "flux_dev",
            hf_snapshot("models--black-forest-labs--FLUX.1-dev"),
            Some(3.5),
            None,
        );
    }

    /// Real-weights smoke: load + generate one small Qwen-Image image (true CFG,
    /// guidance 4.0 + a negative prompt). Needs the HF cache (`Qwen/Qwen-Image`) + a
    /// Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored qwen_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Qwen-Image weights + Metal device"]
    fn qwen_real_weights_generates_one_image() {
        smoke_generate_one(
            "qwen_image",
            hf_snapshot("models--Qwen--Qwen-Image"),
            Some(4.0),
            Some("blurry, low quality".to_owned()),
        );
    }

    /// Real-weights smoke: FLUX.2-klein-9b (4-step distilled, guidance 1.0, no negative).
    /// Needs the HF cache (`black-forest-labs/FLUX.2-klein-9B`) + a Metal device; run on
    /// demand: `cargo test -p sceneworks-worker --lib -- --ignored flux2_klein_9b_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real FLUX.2-klein-9b weights + Metal device"]
    fn flux2_klein_9b_real_weights_generates_one_image() {
        smoke_generate_one(
            "flux2_klein_9b",
            hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b"),
            Some(1.0),
            None,
        );
    }

    /// Real-weights smoke: FLUX.2-klein-9b-kv txt2img (the separately-distilled checkpoint
    /// loaded through the base txt2img loader). Needs the HF cache
    /// (`black-forest-labs/FLUX.2-klein-9b-kv`) + a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real FLUX.2-klein-9b-kv weights + Metal device"]
    fn flux2_klein_9b_kv_real_weights_generates_one_image() {
        smoke_generate_one(
            "flux2_klein_9b_kv",
            hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b-kv"),
            Some(1.0),
            None,
        );
    }

    /// Real-weights smoke: FLUX.2-klein-9b-true_v2 (wikeeyang undistilled fine-tune,
    /// 24-step). Loads the locally-assembled converted diffusers dir under the SceneWorks
    /// data dir (`models/mlx/flux2_klein_9b_true_v2`) via the modelPath seam — verifying
    /// the converted-dir layout passthrough on the base `flux2_klein_9b` loader. Needs a
    /// previously-converted dir + a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs a converted true_v2 dir + Metal device"]
    fn flux2_klein_9b_true_v2_real_weights_generates_one_image() {
        let dir = dirs_home()
            .join("Library/Application Support/SceneWorks/data/models/mlx/flux2_klein_9b_true_v2");
        smoke_generate_one("flux2_klein_9b_true_v2", dir, Some(1.0), None);
    }

    /// Real-weights smoke: SDXL base (real CFG, guidance 7.0 + a negative prompt,
    /// 30-step, Q8). Verifies the engine's SDXL quant default works (the Python
    /// vendored path had no quant). Needs the HF cache
    /// (`stabilityai/stable-diffusion-xl-base-1.0`) + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sdxl_real_weights`.
    /// SDXL native is 1024²; min is 512 — this smoke uses 512² for speed.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SDXL weights + Metal device"]
    fn sdxl_real_weights_generates_one_image() {
        smoke_generate_one(
            "sdxl",
            hf_snapshot("models--stabilityai--stable-diffusion-xl-base-1.0"),
            Some(7.0),
            Some("blurry, low quality".to_owned()),
        );
    }

    /// Real-weights smoke: the RealVisXL finetune through the same `sdxl` engine model.
    /// Needs the HF cache (`SG161222/RealVisXL_V5.0`) + a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real RealVisXL weights + Metal device"]
    fn realvisxl_real_weights_generates_one_image() {
        smoke_generate_one(
            "realvisxl",
            hf_snapshot("models--SG161222--RealVisXL_V5.0"),
            Some(7.0),
            Some("blurry, low quality".to_owned()),
        );
    }

    /// L2-normalized cosine similarity between two ArcFace embeddings (test helper).
    #[cfg(target_os = "macos")]
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Real-weights validation for sc-4424/sc-4427: render the worker-owned InstantID angle
    /// presets through the engine `generate_with_kps` pass-in path and assert, per view, that
    /// (a) **the engine honours the pass-in landmarks** — the detected face lands where the
    /// preset's nose keypoint says it should, proving the worker (not the engine's retired
    /// hardcoded table) now owns the framing — and (b) **identity holds** — ArcFace cosine vs the
    /// reference stays well above floor on the measurable frontal-ish views. Profiles are ArcFace
    /// identity-N/A (frontal-only metric, see the likeness-score memo) so they assert placement
    /// only, and a missing detection on a profile is tolerated (recorded, not failed).
    ///
    /// This is the framing-fill goal from epic 4422: the presets pull head-and-shoulders up into
    /// the frame instead of the old lower-half framing, so LoRA training gets more character pixels.
    ///
    /// Needs: RealVisXL (`SG161222/RealVisXL_V5.0`) + InstantID IdentityNet (`InstantX/InstantID`)
    /// in the HF cache, the converted bundle (`scrfd_10g`/`arcface_iresnet100`/`ip-adapter`) in the
    /// app cache (`SCENEWORKS_INSTANTID_WEIGHTS` overrides), and a reference face
    /// (`SCENEWORKS_TEST_FACE` overrides). Metal device. On demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored instantid_angle_kps_real_weights --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real InstantID weights + Metal device"]
    fn instantid_angle_kps_real_weights_fills_frame_and_holds_identity() {
        // --- resolve weights (HF cache + app bundle) ---
        let sdxl_base = hf_snapshot("models--SG161222--RealVisXL_V5.0");
        let identitynet = hf_snapshot("models--InstantX--InstantID").join("ControlNetModel");
        let bundle = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                dirs_home().join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
            });
        let scrfd_path = bundle.join(INSTANTID_SCRFD_FILE);
        let arcface_path = bundle.join(INSTANTID_ARCFACE_FILE);
        let ip_adapter = bundle.join(INSTANTID_IP_ADAPTER_FILE);
        for p in [
            &sdxl_base,
            &identitynet,
            &scrfd_path,
            &arcface_path,
            &ip_adapter,
        ] {
            assert!(p.exists(), "missing InstantID weight: {}", p.display());
        }
        let face_path = std::env::var("SCENEWORKS_TEST_FACE").unwrap_or_else(|_| {
            "/Users/michael/Library/Application Support/SceneWorks/data/projects/ab.sceneworks/assets/images/genset_e6b07eb5b5374627af1bf47083bac305/2026-06-10_qwen_image_edit_2511_lightning_22-year-old-woman-with-fair-complexion-a-p_0001.png".to_owned()
        });
        let decoded = image::open(&face_path)
            .unwrap_or_else(|e| panic!("reference face {face_path}: {e}"))
            .to_rgb8();
        let reference = Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        };

        // --- load model + native face stack (production load order) ---
        let paths = InstantIdPaths {
            sdxl_base,
            identitynet: WeightsSource::Dir(identitynet),
            ip_adapter,
        };
        let model = InstantId::load(&paths).expect("InstantID load");
        let scrfd = Weights::from_file(&scrfd_path).expect("SCRFD weights");
        let arcface = Weights::from_file(&arcface_path).expect("ArcFace weights");
        let model = model.with_face(&scrfd, &arcface).expect("face stack");

        // Reference identity embedding (frontal source).
        let ref_face = model
            .largest_face(
                &reference.pixels,
                reference.height as usize,
                reference.width as usize,
            )
            .expect("reference face detected");

        // Square canvas (the engine forces square for kps; sc-2009 aspect rule).
        let side: u32 = 1024;
        // Views where ArcFace identity is meaningful (frontal-ish; profiles are N/A — the metric is
        // frontal-only, see the likeness-score memo).
        let identity_views = ["front", "three_quarter_left", "three_quarter_right"];
        // Strict profiles occlude the far eye, so SCRFD's landmark regression there is unreliable —
        // placement is recorded but not hard-asserted (identity is N/A too).
        let profile_views = ["left_profile", "right_profile"];
        // Identity floor — well below the ~0.83-0.87 seen in sc-3345/sc-3365 validation, so the
        // assertion catches a regression without being brittle to seed/quant jitter.
        const IDENTITY_FLOOR: f32 = 0.50;
        // Mean per-landmark distance (square-fraction) between the GENERATED face's detected kps and
        // the preset kps the engine was told to draw — the direct "did `generate_with_kps` honour the
        // pass-in landmarks" check. IdentityNet conditions on these points, so the realized face's
        // landmarks track them tightly; the tolerance absorbs SCRFD/seed jitter.
        const PLACEMENT_TOL: f32 = 0.10;

        let mut failures: Vec<String> = Vec::new();
        println!("\n  view                 kps-dist   area%   id-cos   verdict");
        for &angle in CHARACTER_ANGLE_SET_ORDER.iter() {
            let kps = sceneworks_core::angle_kps::angle_kps(angle).expect("built-in angle kps");
            let req = InstantIdRequest {
                prompt: augment_prompt_for_angle("a portrait photo of a woman", angle),
                negative: "blurry, low quality, deformed".to_owned(),
                width: side,
                height: side,
                steps: INSTANTID_DEFAULT_STEPS as usize,
                guidance: INSTANTID_DEFAULT_GUIDANCE,
                ip_adapter_scale: INSTANTID_IP_SCALE,
                controlnet_scale: INSTANTID_CONTROLNET_SCALE,
                seed: 12345,
                ..InstantIdRequest::default()
            };
            let out = model
                .generate_with_kps(&req, &reference, &kps, &mut |_| {})
                .unwrap_or_else(|e| panic!("{angle}: generate_with_kps failed: {e}"));
            assert_eq!((out.width, out.height), (side, side), "{angle}: canvas");

            let is_identity_view = identity_views.contains(&angle);
            let is_profile = profile_views.contains(&angle);
            match model.largest_face(&out.pixels, out.height as usize, out.width as usize) {
                Ok(face) => {
                    // Mean distance between the generated face's detected landmarks and the preset.
                    let kps_dist: f32 = (0..5)
                        .map(|i| {
                            let dx = face.kps[i][0] / side as f32 - kps[i].0;
                            let dy = face.kps[i][1] / side as f32 - kps[i].1;
                            (dx * dx + dy * dy).sqrt()
                        })
                        .sum::<f32>()
                        / 5.0;
                    let area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                        / (side as f32 * side as f32);
                    let id = cosine(&ref_face.embedding, &face.embedding);
                    // Placement is hard-asserted on non-profile views (reliable landmarks).
                    let placed = is_profile || kps_dist <= PLACEMENT_TOL;
                    let id_ok = !is_identity_view || id >= IDENTITY_FLOOR;
                    let verdict = if placed && id_ok { "ok" } else { "FAIL" };
                    println!(
                        "  {angle:<20} {kps_dist:>6.3}   {:>5.1}  {id:>6.3}  {verdict}",
                        area * 100.0,
                    );
                    if !is_profile && kps_dist > PLACEMENT_TOL {
                        failures.push(format!(
                            "{angle}: realized landmarks {kps_dist:.3} > {PLACEMENT_TOL} from preset"
                        ));
                    }
                    if is_identity_view && id < IDENTITY_FLOOR {
                        failures.push(format!(
                            "{angle}: identity {id:.3} < floor {IDENTITY_FLOOR}"
                        ));
                    }
                }
                Err(e) => {
                    println!("  {angle:<20} no-detect ({e})");
                    // A frontal-ish view that fails to detect is a real failure; a profile is N/A.
                    if is_identity_view {
                        failures.push(format!("{angle}: no face detected (frontal view)"));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "preset validation failures:\n  {}",
            failures.join("\n  ")
        );
    }

    /// Load + generate one small image for a TRUE-CFG family (Chroma): the CFG scale rides
    /// `true_cfg` (not the distilled `guidance` scalar the engine rejects), mirroring
    /// [`generate_mlx_stream`]'s wiring. Sibling of [`smoke_generate_one`].
    #[cfg(target_os = "macos")]
    fn smoke_generate_one_true_cfg(
        sceneworks_id: &str,
        snapshot: std::path::PathBuf,
        true_cfg: Option<f32>,
        negative_prompt: Option<String>,
    ) {
        let model = mlx_model(sceneworks_id).unwrap();
        let generator = mlx_load(
            model.engine_id,
            snapshot,
            Some(mlx_gen::Quant::Q8),
            Vec::new(),
        )
        .unwrap();
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let steps = model.default_steps;
        let (w, h, pixels) = mlx_generate_one(
            generator.as_ref(),
            "a serene mountain lake at dawn",
            512,
            512,
            42,
            steps,
            None,
            negative_prompt,
            None,
            true_cfg,
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    /// Real-weights smoke: Chroma1-HD full true-CFG (sc-3843). Exercises the true-CFG worker
    /// path (`true_cfg` carries the scale; the engine rejects `guidance`). Needs the HF cache
    /// (`lodestones/Chroma1-HD`) + a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Chroma1-HD weights + Metal device"]
    fn chroma_hd_real_weights_generates_one_image() {
        smoke_generate_one_true_cfg(
            "chroma1_hd",
            hf_snapshot("models--lodestones--Chroma1-HD"),
            Some(3.0),
            Some("blurry, low quality".to_owned()),
        );
    }

    /// Real-weights smoke: Chroma1-Base (beta-sigma schedule, same true-CFG path as HD).
    /// Needs the HF cache (`lodestones/Chroma1-Base`) + a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Chroma1-Base weights + Metal device"]
    fn chroma_base_real_weights_generates_one_image() {
        smoke_generate_one_true_cfg(
            "chroma1_base",
            hf_snapshot("models--lodestones--Chroma1-Base"),
            Some(3.0),
            Some("blurry, low quality".to_owned()),
        );
    }

    /// Real-weights smoke: Chroma1-Flash few-step distilled (true_cfg≈1, negative inert).
    /// Needs the HF cache (`lodestones/Chroma1-Flash`) + a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Chroma1-Flash weights + Metal device"]
    fn chroma_flash_real_weights_generates_one_image() {
        smoke_generate_one_true_cfg(
            "chroma1_flash",
            hf_snapshot("models--lodestones--Chroma1-Flash"),
            Some(1.0),
            None,
        );
    }

    // --- Z-Image strict-pose control path (sc-3028) ---

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_control_scale_defaults_and_clamps() {
        assert_eq!(
            resolve_control_scale(&request(json!({ "projectId": "p" }))),
            0.9
        );
        assert_eq!(
            resolve_control_scale(&request(
                json!({ "projectId": "p", "advanced": { "controlScale": 0.65 } })
            )),
            0.65
        );
        // Clamp to [0, 2].
        assert_eq!(
            resolve_control_scale(&request(
                json!({ "projectId": "p", "advanced": { "controlScale": 5.0 } })
            )),
            2.0
        );
        assert_eq!(
            resolve_control_scale(&request(
                json!({ "projectId": "p", "advanced": { "controlScale": -1.0 } })
            )),
            0.0
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn pose_entries_filters_to_objects() {
        let req = request(json!({
            "projectId": "p",
            "advanced": { "poses": [{ "id": "a" }, "not-an-object", { "id": "b" }] }
        }));
        assert_eq!(pose_entries(&req).len(), 2);
        // No poses → empty (not a strict-pose job).
        assert!(pose_entries(&request(json!({ "projectId": "p" }))).is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_poses_extracts_keypoints_hands_face() {
        let req = request(json!({
            "projectId": "p",
            "advanced": { "poses": [{
                "id": "a",
                "keypoints": [[0.5, 0.2], [0.5, 0.35]],
                "hands": [[[0.1, 0.1]], [[0.2, 0.2]]],
                "face": [[0.5, 0.18]]
            }] }
        }));
        let poses = parse_poses(&req);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].keypoints.len(), 18); // padded
        assert_eq!(poses[0].keypoints[0], Some((0.5, 0.2)));
        assert!(poses[0].hands.is_some());
        assert!(poses[0].face.is_some());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_control_raw_settings_records_control_recipe() {
        let req = request(json!({
            "projectId": "p",
            "model": "qwen_image",
            "advanced": {
                "poses": [{ "id": "pose_1" }],
                "controlScale": 0.5
            }
        }));
        let raw = qwen_control_raw_settings(&req, "Qwen/Qwen-Image", 20, Some(4), 4.0, 0.5, 1);
        assert_eq!(
            raw.get("controlEngine").and_then(Value::as_str),
            Some(QWEN_CONTROL_ENGINE_ID)
        );
        assert_eq!(raw.get("controlScale"), Some(&json!(0.5)));
        assert_eq!(raw.get("poseCount"), Some(&json!(1)));
        assert_eq!(raw.get("guidanceScale"), Some(&json!(4.0)));
        assert_eq!(raw.get("mlxQuantize"), Some(&json!(4)));
    }

    /// sc-3031 A/B dump (NOT a CI test): generate ONE image through the **real new-adapter
    /// path** — the production resolvers (`model_repo` / `resolve_steps` / `resolve_guidance` /
    /// `resolve_negative_prompt` / `resolve_quant` / `resolve_adapters` / `resolve_weights_dir` /
    /// `resolve_seed`) + the `mlx_load` + `mlx_generate_one` core that `generate_mlx_stream`
    /// drives — and write it to `$SC3031_OUT` for head-to-head comparison against the Python
    /// `Mlx*Adapter` output. Covers the txt2img families (z-image / flux / qwen / flux2 / sdxl).
    /// Env: `SC3031_PAYLOAD` (job-payload JSON object), `SC3031_OUT` (.png path); set
    /// `SCENEWORKS_DATA_DIR` + `HF_HOME` so weights resolve. Run:
    /// `SC3031_PAYLOAD=… SC3031_OUT=… cargo test -p sceneworks-worker --lib -- --ignored --exact \
    ///   image_jobs::tests::sc3031_ab_dump_txt2img`
    #[cfg(target_os = "macos")]
    #[ignore = "sc-3031 A/B dump harness: drive via SC3031_PAYLOAD + SC3031_OUT"]
    #[test]
    fn sc3031_ab_dump_txt2img() {
        let payload: Value =
            serde_json::from_str(&std::env::var("SC3031_PAYLOAD").expect("SC3031_PAYLOAD"))
                .expect("SC3031_PAYLOAD is JSON");
        let out = std::env::var("SC3031_OUT").expect("SC3031_OUT");
        let req = request(payload);
        let settings = Settings::from_env(); // honors SCENEWORKS_DATA_DIR + HF_HOME

        let model = mlx_model(&req.model).expect("an MLX txt2img model id");
        let _repo = model_repo(&req, model);
        let steps = resolve_steps(&req, model);
        let guidance = resolve_guidance(&req, model);
        let negative = resolve_negative_prompt(&req, model);
        let (quant, _bits) = resolve_quant(&req);
        let weights = resolve_weights_dir(&req, &settings)
            .expect("weights resolve")
            .expect("weights in HF cache");
        let adapters = resolve_adapters(&req).expect("adapters");
        let seed = resolve_seed(&req, 0);
        let generator = mlx_load(model.engine_id, weights, quant, adapters).expect("load");

        let cancel = CancelFlag::new();
        let (w, h, pixels) = mlx_generate_one(
            generator.as_ref(),
            &req.prompt,
            req.width,
            req.height,
            seed,
            steps,
            guidance,
            negative,
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .expect("generate");
        image::RgbImage::from_raw(w, h, pixels)
            .expect("rgb buffer")
            .save(&out)
            .expect("save png");
        eprintln!(
            "sc3031 rust dump: model={} {w}x{h} seed={seed} steps={steps} guidance={guidance:?} -> {out}",
            req.model
        );
    }

    /// sc-3031 A/B dump (pose): generate ONE strict-pose image through the **real new-adapter
    /// path** — the production `resolve_control_weights` / `resolve_control_scale` / `parse_poses`,
    /// the `draw_wholebody` skeleton render, and `zimage_control_load` / `zimage_control_generate_one`
    /// (the core that `generate_zimage_control_stream` drives) — and write it to `$SC3031_OUT` for
    /// head-to-head comparison against the Python `MlxZImageAdapter` strict-pose tier (drive the
    /// Python side with the SAME `advanced.poses` payload). Env: `SC3031_PAYLOAD` (must carry
    /// `advanced.poses`), `SC3031_OUT`; set `SCENEWORKS_DATA_DIR` + `HF_HOME`.
    #[cfg(target_os = "macos")]
    #[ignore = "sc-3031 A/B dump harness (pose): drive via SC3031_PAYLOAD + SC3031_OUT"]
    #[test]
    fn sc3031_ab_dump_pose() {
        let payload: Value =
            serde_json::from_str(&std::env::var("SC3031_PAYLOAD").expect("SC3031_PAYLOAD"))
                .expect("SC3031_PAYLOAD is JSON");
        let out = std::env::var("SC3031_OUT").expect("SC3031_OUT");
        let req = request(payload);
        let settings = Settings::from_env();

        let weights = resolve_weights_dir(&req, &settings)
            .expect("z-image weights resolve")
            .expect("z-image weights in HF cache");
        let control_weights =
            resolve_control_weights(&req, &settings).expect("Fun-Controlnet-Union weights");
        let (quant, _bits) = resolve_quant(&req);
        let zimage = mlx_model("z_image_turbo").expect("z-image model row");
        let steps = resolve_steps(&req, zimage);
        let control_scale = resolve_control_scale(&req);
        let adapters = resolve_adapters(&req).expect("adapters");
        let seed = resolve_seed(&req, 0);

        let pose = parse_poses(&req)
            .into_iter()
            .next()
            .expect("advanced.poses");
        let (w, h) = (req.width, req.height);
        let skeleton = crate::openpose_skeleton::draw_wholebody(
            w,
            h,
            &pose.keypoints,
            pose.hands.as_deref(),
            pose.face.as_deref(),
            crate::openpose_skeleton::body_stickwidth(w, h),
        );
        let control = Image {
            width: w,
            height: h,
            pixels: skeleton.into_raw(),
        };

        let generator =
            zimage_control_load(weights, control_weights, quant, adapters).expect("load");
        let cancel = CancelFlag::new();
        let (ow, oh, pixels) = zimage_control_generate_one(
            generator.as_ref(),
            &req.prompt,
            w,
            h,
            seed,
            steps,
            control,
            control_scale,
            None,
            &cancel,
            &mut |_| {},
        )
        .expect("generate");
        image::RgbImage::from_raw(ow, oh, pixels)
            .expect("rgb buffer")
            .save(&out)
            .expect("save png");
        eprintln!(
            "sc3031 rust pose dump: {ow}x{oh} seed={seed} steps={steps} control_scale={control_scale} -> {out}"
        );
    }

    /// sc-3031 A/B (pose, no model): render JUST the control skeleton from
    /// `$SC3031_PAYLOAD`'s `advanced.poses` via the production `parse_poses` + `draw_wholebody`
    /// and write it to `$SC3031_OUT`, to compare the Rust skeleton renderer against the Python
    /// one for the same keypoints (separates skeleton-render parity from the engine/schedule).
    /// CPU-only, instant (no weights / no Metal). macOS-gated because it uses `parse_poses`
    /// (part of the macOS strict-pose path); the Linux workspace-check lane configures that out.
    #[cfg(target_os = "macos")]
    #[ignore = "sc-3031 skeleton-render dump: drive via SC3031_PAYLOAD + SC3031_OUT"]
    #[test]
    fn sc3031_dump_skeleton() {
        let payload: Value =
            serde_json::from_str(&std::env::var("SC3031_PAYLOAD").expect("SC3031_PAYLOAD"))
                .expect("SC3031_PAYLOAD is JSON");
        let out = std::env::var("SC3031_OUT").expect("SC3031_OUT");
        let req = request(payload);
        let pose = parse_poses(&req)
            .into_iter()
            .next()
            .expect("advanced.poses");
        let (w, h) = (req.width, req.height);
        let skeleton = crate::openpose_skeleton::draw_wholebody(
            w,
            h,
            &pose.keypoints,
            pose.hands.as_deref(),
            pose.face.as_deref(),
            crate::openpose_skeleton::body_stickwidth(w, h),
        );
        image::RgbImage::from_raw(w, h, skeleton.into_raw())
            .expect("rgb buffer")
            .save(&out)
            .expect("save png");
        eprintln!("sc3031 rust skeleton dump: {w}x{h} -> {out}");
    }

    /// Identity img2img-init gate + clamp (sc-3146): the parity-sensitive decision the
    /// strict-pose stream makes before loading the reference image. Mirrors the Python
    /// `MlxZImageAdapter._identity_init_requested` + `_reference_strength` semantics.
    #[cfg(target_os = "macos")]
    #[test]
    fn zimage_identity_strength_gate_and_clamp() {
        let with = |adv: Value, asset: Value| {
            let mut payload = json!({
                "projectId": "p", "model": "z_image_turbo", "prompt": "a knight"
            });
            let obj = payload.as_object_mut().unwrap();
            obj.insert("advanced".to_owned(), adv);
            if !asset.is_null() {
                obj.insert("referenceAssetId".to_owned(), asset);
            }
            zimage_identity_strength(&request(payload))
        };
        let approx = |got: Option<f32>, want: f32| match got {
            Some(value) => assert!((value - want).abs() < 1e-6, "got {value}, want {want}"),
            None => panic!("expected Some({want}), got None"),
        };

        // Pose-only tiers → None: no referenceStrength; referenceStrength == 0 (parity:
        // the Python gate requires > 0); referenceStrength > 0 but no/blank asset (a bare
        // reference has no MLX home, so it falls back to pose-only rather than erroring).
        assert_eq!(with(json!({}), json!("ref_1")), None);
        assert_eq!(
            with(json!({ "referenceStrength": 0.0 }), json!("ref_1")),
            None
        );
        assert_eq!(with(json!({ "referenceStrength": 0.6 }), Value::Null), None);
        assert_eq!(
            with(json!({ "referenceStrength": 0.6 }), json!("   ")),
            None
        );

        // Engaged: strength forwarded verbatim (no inversion) and clamped to [0.05, 1.0].
        approx(
            with(json!({ "referenceStrength": 0.6 }), json!("ref_1")),
            0.6,
        );
        approx(
            with(json!({ "referenceStrength": "0.45" }), json!("ref_1")),
            0.45,
        );
        assert_eq!(
            with(json!({ "referenceStrength": 1.8 }), json!("ref_1")),
            Some(1.0)
        );
        assert_eq!(
            with(json!({ "referenceStrength": 0.01 }), json!("ref_1")),
            Some(0.05)
        );
    }

    /// Real-weights smoke: Z-Image strict-pose ControlNet. Loads the base
    /// `Tongyi-MAI/Z-Image-Turbo` snapshot + the cached Fun-Controlnet-Union checkpoint,
    /// renders a DWPose skeleton, and generates one pose image. Needs both in the HF
    /// cache + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored zimage_control_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Z-Image + Fun-Controlnet-Union weights + Metal device"]
    fn zimage_control_real_weights_generates_one_pose() {
        let base = hf_snapshot("models--Tongyi-MAI--Z-Image-Turbo");
        let control = std::fs::read_dir(dirs_home().join(
            ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1/snapshots",
        ))
        .expect("control snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .map(|dir| dir.join(super::ZIMAGE_CONTROL_FILE))
        .filter(|path| path.exists())
        .expect("control weights file");

        let generator =
            zimage_control_load(base, control, Some(mlx_gen::Quant::Q8), Vec::new()).unwrap();

        // A minimal standing skeleton at 512².
        let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
            [0.5, 0.2],
            [0.5, 0.35],
            [0.42, 0.35],
            [0.40, 0.5],
            [0.40, 0.65],
            [0.58, 0.35],
            [0.60, 0.5],
            [0.60, 0.65],
            [0.45, 0.6],
            [0.45, 0.8],
            [0.45, 0.95],
            [0.55, 0.6],
            [0.55, 0.8],
            [0.55, 0.95],
            [0.48, 0.18],
            [0.52, 0.18],
            [0.46, 0.2],
            [0.54, 0.2]
        ]));
        let skeleton = crate::openpose_skeleton::draw_wholebody(
            512,
            512,
            &kp,
            None,
            None,
            crate::openpose_skeleton::body_stickwidth(512, 512),
        );
        let control = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: skeleton.into_raw(),
        };

        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = zimage_control_generate_one(
            generator.as_ref(),
            "a person standing in a meadow",
            512,
            512,
            42,
            8,
            control,
            0.9,
            None,
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    /// Real-weights smoke: Qwen-Image strict-pose ControlNet. Loads the base
    /// `Qwen/Qwen-Image` snapshot + the cached InstantX ControlNet-Union checkpoint,
    /// renders one DWPose skeleton, and generates one image through `qwen_image_control`.
    /// Needs both in the HF cache + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored qwen_control_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Qwen-Image + InstantX ControlNet weights + Metal device"]
    fn qwen_control_real_weights_generates_one_pose() {
        let base = hf_snapshot("models--Qwen--Qwen-Image");
        let control = hf_snapshot("models--InstantX--Qwen-Image-ControlNet-Union")
            .join(super::QWEN_CONTROL_FILE);
        assert!(
            control.exists(),
            "Qwen control weights missing: {control:?}"
        );

        let generator =
            qwen_control_load(base, control, Some(mlx_gen::Quant::Q8), Vec::new()).unwrap();

        let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
            [0.5, 0.2],
            [0.5, 0.35],
            [0.42, 0.35],
            [0.40, 0.5],
            [0.40, 0.65],
            [0.58, 0.35],
            [0.60, 0.5],
            [0.60, 0.65],
            [0.45, 0.6],
            [0.45, 0.8],
            [0.45, 0.95],
            [0.55, 0.6],
            [0.55, 0.8],
            [0.55, 0.95],
            [0.48, 0.18],
            [0.52, 0.18],
            [0.46, 0.2],
            [0.54, 0.2]
        ]));
        let skeleton = crate::openpose_skeleton::draw_wholebody(
            512,
            512,
            &kp,
            None,
            None,
            crate::openpose_skeleton::body_stickwidth(512, 512),
        );
        let control = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: skeleton.into_raw(),
        };

        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = qwen_control_generate_one(
            generator.as_ref(),
            "a person standing in a meadow",
            Some("blurry, low quality".to_owned()),
            512,
            512,
            42,
            4,
            4.0,
            control,
            0.9,
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    // --- FLUX.2 edit path (sc-3029) ---

    #[cfg(target_os = "macos")]
    #[test]
    fn flux2_edit_engine_id_maps_variants() {
        assert_eq!(
            flux2_edit_engine_id("flux2_klein_9b"),
            Some("flux2_klein_9b_edit")
        );
        assert_eq!(
            flux2_edit_engine_id("flux2_klein_9b_true_v2"),
            Some("flux2_klein_9b_edit")
        );
        assert_eq!(
            flux2_edit_engine_id("flux2_klein_9b_kv"),
            Some("flux2_klein_9b_kv_edit")
        );
        assert_eq!(flux2_edit_engine_id("z_image_turbo"), None);
        assert_eq!(flux2_edit_engine_id("sdxl"), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn flux2_edit_reference_ids_prefers_reference_then_source() {
        // referenceAssetId (character flow) wins.
        assert_eq!(
            flux2_edit_reference_ids(&request(json!({
                "projectId": "p", "referenceAssetId": "ref_1", "sourceAssetId": "src_1"
            }))),
            vec!["ref_1".to_owned()]
        );
        // sourceAssetId only in edit_image mode.
        assert_eq!(
            flux2_edit_reference_ids(&request(json!({
                "projectId": "p", "mode": "edit_image", "sourceAssetId": "src_1"
            }))),
            vec!["src_1".to_owned()]
        );
        // sourceAssetId without edit_image mode is ignored (it's the txt2img path).
        assert!(flux2_edit_reference_ids(&request(json!({
            "projectId": "p", "sourceAssetId": "src_1"
        })))
        .is_empty());
        assert!(flux2_edit_reference_ids(&request(json!({ "projectId": "p" }))).is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_edit_conditioning_single_vs_multi() {
        let img = |seed| mlx_gen::Image {
            width: 8,
            height: 8,
            pixels: stub_rgb8(8, 8, seed),
        };
        match build_edit_conditioning(std::slice::from_ref(&img(1))).as_slice() {
            [mlx_gen::Conditioning::Reference { .. }] => {}
            other => panic!("expected one Reference, got {other:?}"),
        }
        match build_edit_conditioning(&[img(1), img(2)]).as_slice() {
            [mlx_gen::Conditioning::MultiReference { images }] => assert_eq!(images.len(), 2),
            other => panic!("expected MultiReference, got {other:?}"),
        }
    }

    /// Real-weights smoke: FLUX.2-klein edit. Loads `flux2_klein_9b_edit` (base 9B
    /// snapshot) and generates one image conditioned on a synthetic reference. Needs
    /// the HF cache (`black-forest-labs/FLUX.2-klein-9B`) + a Metal device; run on
    /// demand: `cargo test -p sceneworks-worker --lib -- --ignored flux2_edit_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real FLUX.2-klein-9b weights + Metal device"]
    fn flux2_edit_real_weights_generates_one_image() {
        let snapshot = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b");
        let generator = mlx_load(
            "flux2_klein_9b_edit",
            snapshot,
            Some(mlx_gen::Quant::Q8),
            Vec::new(),
        )
        .unwrap();
        let reference = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: stub_rgb8(512, 512, 7),
        };
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = flux2_edit_generate_one(
            generator.as_ref(),
            "make it a watercolor painting",
            512,
            512,
            42,
            4,
            Some(1.0),
            build_edit_conditioning(std::slice::from_ref(&reference)),
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    // --- Angle set / pose tier / fit_image (sc-3030) ---

    #[cfg(target_os = "macos")]
    #[test]
    fn character_angle_set_is_eleven_ordered_angles() {
        assert_eq!(CHARACTER_ANGLE_SET_ORDER.len(), 11);
        assert_eq!(CHARACTER_ANGLE_SET_ORDER[0], "front");
        // Every angle has a non-empty augment clause.
        for angle in CHARACTER_ANGLE_SET_ORDER {
            assert!(
                !angle_prompt_augment(angle).is_empty(),
                "no augment for {angle}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn augment_prompt_for_angle_appends_clause_and_strips_punctuation() {
        assert_eq!(
            augment_prompt_for_angle("a knight", "front"),
            "a knight, frontal portrait, looking directly at the camera, head and shoulders, neutral expression"
        );
        // Trailing punctuation on the base is stripped before the comma join.
        assert_eq!(
            augment_prompt_for_angle("a knight.", "left_profile"),
            "a knight, full left profile, head turned 90 degrees to the left, side view of the head"
        );
        // Empty base → the augment clause alone.
        assert_eq!(
            augment_prompt_for_angle("", "down"),
            "looking down, head tilted slightly downward toward the floor"
        );
        // Unknown angle (no clause) → the base prompt unchanged.
        assert_eq!(augment_prompt_for_angle("a knight", "sideways"), "a knight");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn augment_prompt_for_pose_appends_cue() {
        assert_eq!(
            augment_prompt_for_pose("a hero"),
            "a hero, matching the exact body pose shown in the OpenPose skeleton reference image"
        );
        assert_eq!(augment_prompt_for_pose("  "), POSE_SKELETON_PROMPT);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn flux2_grouping_poses_over_angles_over_plain() {
        // Pose set wins even when angleSet is also set.
        let poses = request(json!({
            "projectId": "p", "mode": "character_image", "referenceAssetId": "ref",
            "advanced": { "angleSet": true, "poses": [{ "id": "a" }, { "id": "b" }] }
        }));
        assert!(matches!(flux2_grouping(&poses), Flux2Grouping::Poses(2)));
        // angleSet without poses → the 11-angle set.
        let angles = request(json!({
            "projectId": "p", "mode": "character_image", "referenceAssetId": "ref",
            "advanced": { "angleSet": true }
        }));
        assert!(matches!(flux2_grouping(&angles), Flux2Grouping::Angles));
        // character_image with neither → plain.
        let plain = request(json!({
            "projectId": "p", "mode": "character_image", "referenceAssetId": "ref"
        }));
        assert!(matches!(flux2_grouping(&plain), Flux2Grouping::Plain));
        // edit_image never groups, even with angleSet (mode gate).
        let edit = request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src",
            "advanced": { "angleSet": true }
        }));
        assert!(matches!(flux2_grouping(&edit), Flux2Grouping::Plain));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn should_fit_edit_source_only_for_off_aspect_edit_image() {
        // edit_image + source + no reference + non-stretch → fit.
        assert!(should_fit_edit_source(&request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src", "fitMode": "crop"
        }))));
        // A character reference present → the reference path stays native.
        assert!(!should_fit_edit_source(&request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src",
            "referenceAssetId": "ref", "fitMode": "crop"
        }))));
        // stretch keeps the legacy naive resize.
        assert!(!should_fit_edit_source(&request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src", "fitMode": "stretch"
        }))));
        // character_image is never the edit-source fit path.
        assert!(!should_fit_edit_source(&request(json!({
            "projectId": "p", "mode": "character_image", "referenceAssetId": "ref"
        }))));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn contain_box_centers_the_contained_rect() {
        // Wide source contained in a square: full width, centered vertically.
        assert_eq!(contain_box(100, 50, 50, 50), (50, 25, 0, 12));
        // Tall source: full height, centered horizontally.
        assert_eq!(contain_box(50, 100, 50, 50), (25, 50, 12, 0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fit_rgb_crop_pad_stretch_produce_exact_dims_and_geometry() {
        // 100×50 solid white source.
        let source = image::RgbImage::from_pixel(100, 50, image::Rgb([255, 255, 255]));

        // crop → cover + center-crop, exact target dims, no black bars (all white).
        let cropped = fit_rgb(&source, 50, 50, "crop");
        assert_eq!((cropped.width(), cropped.height()), (50, 50));
        assert_eq!(cropped.get_pixel(0, 0), &image::Rgb([255, 255, 255]));
        assert_eq!(cropped.get_pixel(25, 25), &image::Rgb([255, 255, 255]));

        // pad → contain + letterbox: black top/bottom bars, white band in the middle.
        let padded = fit_rgb(&source, 50, 50, "pad");
        assert_eq!((padded.width(), padded.height()), (50, 50));
        assert_eq!(padded.get_pixel(0, 0), &image::Rgb([0, 0, 0])); // top bar
        assert_eq!(padded.get_pixel(25, 24), &image::Rgb([255, 255, 255])); // content band

        // outpaint degrades to pad geometry (same letterbox).
        assert_eq!(
            fit_rgb(&source, 50, 50, "outpaint").into_raw(),
            padded.into_raw()
        );

        // stretch → exact target dims (aspect not preserved).
        let stretched = fit_rgb(&source, 40, 30, "stretch");
        assert_eq!((stretched.width(), stretched.height()), (40, 30));
    }

    // --- Qwen-Image-Edit path (sc-3397) ---

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_edit_model_table_rows() {
        for id in [
            "qwen_image_edit",
            "qwen_image_edit_2509",
            "qwen_image_edit_2511",
        ] {
            let m = mlx_model(id).unwrap();
            assert_eq!(m.engine_id, "qwen_image_edit");
            assert_eq!(m.default_repo, "Qwen/Qwen-Image-Edit-2511");
            assert_eq!(m.default_steps, 40);
            assert_eq!(m.default_guidance, 4.0);
            assert_eq!(m.adapter_label, "mlx_qwen");
            assert!(m.supports_guidance && m.supports_negative_prompt);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_edit_lightning_model_row_is_cfg_off_4step_distill() {
        // sc-3398: shares the engine model + base weights with the production edit rows
        // but runs the 4-step CFG-off recipe (no negative prompt) + the lightx2v distill.
        let m = mlx_model("qwen_image_edit_2511_lightning").unwrap();
        assert_eq!(m.engine_id, "qwen_image_edit");
        assert_eq!(m.default_repo, "Qwen/Qwen-Image-Edit-2511");
        assert_eq!(m.default_steps, 4);
        assert_eq!(m.default_guidance, 1.0);
        assert_eq!(m.adapter_label, "mlx_qwen");
        assert!(!m.supports_negative_prompt, "lightning runs CFG-off");

        // The lightning lookup carries the engine sampler + the lightx2v 4-step distill LoRA;
        // the production edit ids carry none.
        let distill = qwen_edit_lightning("qwen_image_edit_2511_lightning").unwrap();
        assert_eq!(distill.sampler, "lightning");
        assert_eq!(distill.repo, "lightx2v/Qwen-Image-Edit-2511-Lightning");
        assert_eq!(
            distill.file,
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        );
        assert!(qwen_edit_lightning("qwen_image_edit_2511").is_none());
        assert!(qwen_edit_lightning("qwen_image_edit").is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_edit_engine_id_maps_variants() {
        for id in [
            "qwen_image_edit",
            "qwen_image_edit_2509",
            "qwen_image_edit_2511",
            // The Lightning distill maps to the same engine model (sc-3398); only its
            // sampler + distill LoRA differ.
            "qwen_image_edit_2511_lightning",
        ] {
            assert_eq!(qwen_edit_engine_id(id), Some("qwen_image_edit"));
        }
        // Base txt2img Qwen and other families have no edit variant.
        assert_eq!(qwen_edit_engine_id("qwen_image"), None);
        assert_eq!(qwen_edit_engine_id("flux2_klein_9b"), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen_edit_reference_ids_prefers_reference_then_source() {
        // referenceAssetId (character flow) wins over a source.
        assert_eq!(
            qwen_edit_reference_ids(&request(json!({
                "projectId": "p", "referenceAssetId": "ref_1", "sourceAssetId": "src_1"
            }))),
            vec!["ref_1".to_owned()]
        );
        // sourceAssetId only in edit_image mode.
        assert_eq!(
            qwen_edit_reference_ids(&request(json!({
                "projectId": "p", "mode": "edit_image", "sourceAssetId": "src_1"
            }))),
            vec!["src_1".to_owned()]
        );
        // sourceAssetId without edit_image mode is ignored (the txt2img path).
        assert!(qwen_edit_reference_ids(&request(json!({
            "projectId": "p", "sourceAssetId": "src_1"
        })))
        .is_empty());
        assert!(qwen_edit_reference_ids(&request(json!({ "projectId": "p" }))).is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_qwen_edit_guidance_reads_true_cfg_scale_not_guidance_scale() {
        let model = mlx_model("qwen_image_edit_2511").unwrap();
        // Default is the family true-CFG default (4.0).
        assert_eq!(
            resolve_qwen_edit_guidance(
                &request(json!({ "projectId": "p", "mode": "edit_image" })),
                model
            ),
            4.0
        );
        // guidanceScale (the inert embedded-guidance knob the Python edit path pins at 1.0)
        // is IGNORED — only trueCfgScale drives the engine's true CFG.
        assert_eq!(
            resolve_qwen_edit_guidance(
                &request(json!({
                    "projectId": "p", "mode": "edit_image",
                    "advanced": { "guidanceScale": 1.0 }
                })),
                model
            ),
            4.0
        );
        // trueCfgScale overrides.
        assert_eq!(
            resolve_qwen_edit_guidance(
                &request(json!({
                    "projectId": "p", "mode": "edit_image",
                    "advanced": { "trueCfgScale": 6.0 }
                })),
                model
            ),
            6.0
        );
        // The character reference path clamps to [1, 10].
        assert_eq!(
            resolve_qwen_edit_guidance(
                &request(json!({
                    "projectId": "p", "mode": "character_image",
                    "advanced": { "trueCfgScale": 50.0 }
                })),
                model
            ),
            10.0
        );
        assert_eq!(
            resolve_qwen_edit_guidance(
                &request(json!({
                    "projectId": "p", "mode": "character_image",
                    "advanced": { "trueCfgScale": 0.5 }
                })),
                model
            ),
            1.0
        );
        // edit_image floors at 1.0 (the engine needs CFG > 1 to engage).
        assert_eq!(
            resolve_qwen_edit_guidance(
                &request(json!({
                    "projectId": "p", "mode": "edit_image",
                    "advanced": { "trueCfgScale": 0.5 }
                })),
                model
            ),
            1.0
        );
    }

    /// Real-weights smoke: Qwen-Image-Edit. Loads `qwen_image_edit` (Qwen-Image-Edit-2511
    /// snapshot) and generates one image conditioned on a synthetic reference — true CFG
    /// (guidance 4.0 + a negative prompt). Needs the HF cache (`Qwen/Qwen-Image-Edit-2511`)
    /// and a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored qwen_edit_real_weights`.
    /// Uses 4 steps + 512² for speed (the production default is 40 steps).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Qwen-Image-Edit-2511 weights + Metal device"]
    fn qwen_edit_real_weights_generates_one_image() {
        let snapshot = hf_snapshot("models--Qwen--Qwen-Image-Edit-2511");
        let generator = mlx_load(
            "qwen_image_edit",
            snapshot,
            Some(mlx_gen::Quant::Q8),
            Vec::new(),
        )
        .unwrap();
        let reference = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: stub_rgb8(512, 512, 7),
        };
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = qwen_edit_generate_one(
            generator.as_ref(),
            "make it a watercolor painting",
            Some("blurry, low quality".to_owned()),
            512,
            512,
            42,
            4,
            4.0,
            None,
            build_edit_conditioning(std::slice::from_ref(&reference)),
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    /// Real-weights smoke: Qwen-Image-Edit **Lightning** (sc-3398). Loads `qwen_image_edit`
    /// (Qwen-Image-Edit-2511 snapshot) with the lightx2v 4-step distill LoRA stacked on, then
    /// generates one image via the `lightning` sampler (CFG-off single forward, no negative
    /// prompt) at 4 steps. Needs the HF cache for BOTH `Qwen/Qwen-Image-Edit-2511` and the
    /// distill LoRA `lightx2v/Qwen-Image-Edit-2511-Lightning` + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored qwen_edit_lightning_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Qwen-Image-Edit-2511 + lightx2v distill LoRA weights + Metal device"]
    fn qwen_edit_lightning_real_weights_generates_one_image() {
        let distill = qwen_edit_lightning("qwen_image_edit_2511_lightning").unwrap();
        // The distill LoRA lives in the HF cache (download it once via the lightning
        // generate path, or `huggingface-cli download <repo>`); resolve its snapshot file.
        let snapshot_dir = crate::model_jobs::huggingface_snapshot_dir(
            &Settings::from_env().data_dir,
            distill.repo,
        )
        .expect("lightx2v distill LoRA must be cached for this smoke");
        let lora_path = snapshot_dir.join(distill.file);
        assert!(
            lora_path.is_file(),
            "distill LoRA missing in cache: {}",
            lora_path.display()
        );

        let snapshot = hf_snapshot("models--Qwen--Qwen-Image-Edit-2511");
        let generator = mlx_load(
            "qwen_image_edit",
            snapshot,
            Some(mlx_gen::Quant::Q8),
            vec![AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora)],
        )
        .unwrap();
        let reference = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: stub_rgb8(512, 512, 7),
        };
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = qwen_edit_generate_one(
            generator.as_ref(),
            "make it a watercolor painting",
            // Lightning runs CFG-off: no negative prompt, guidance 1.0 (engine ignores it).
            None,
            512,
            512,
            42,
            4,
            1.0,
            Some("lightning"),
            build_edit_conditioning(std::slice::from_ref(&reference)),
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    /// Real-weights smoke: the best-effort pose tier — a `[skeleton, reference]`
    /// `MultiReference` edit through `flux2_klein_9b_edit`. Verifies the engine accepts
    /// the multi-image pose conditioning on real weights (the single-reference smoke
    /// above does not). Needs the HF cache (`black-forest-labs/FLUX.2-klein-9B`) + a
    /// Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored flux2_pose_tier_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real FLUX.2-klein-9b weights + Metal device"]
    fn flux2_pose_tier_real_weights_generates_one_image() {
        let snapshot = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b");
        let generator = mlx_load(
            "flux2_klein_9b_edit",
            snapshot,
            Some(mlx_gen::Quant::Q8),
            Vec::new(),
        )
        .unwrap();
        // A minimal standing skeleton (body only — the best-effort tier uses no
        // hands/face) + a synthetic reference, paired as the pose multi-image set.
        let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
            [0.5, 0.2],
            [0.5, 0.35],
            [0.42, 0.35],
            [0.40, 0.5],
            [0.40, 0.65],
            [0.58, 0.35],
            [0.60, 0.5],
            [0.60, 0.65],
            [0.45, 0.6],
            [0.45, 0.8],
            [0.45, 0.95],
            [0.55, 0.6],
            [0.55, 0.8],
            [0.55, 0.95],
            [0.48, 0.18],
            [0.52, 0.18],
            [0.46, 0.2],
            [0.54, 0.2]
        ]));
        let skeleton = crate::openpose_skeleton::draw_wholebody(
            512,
            512,
            &kp,
            None,
            None,
            crate::openpose_skeleton::body_stickwidth(512, 512),
        );
        let skeleton_img = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: skeleton.into_raw(),
        };
        let reference = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: stub_rgb8(512, 512, 7),
        };
        let conditioning = vec![mlx_gen::Conditioning::MultiReference {
            images: vec![skeleton_img, reference],
        }];
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (w, h, pixels) = flux2_edit_generate_one(
            generator.as_ref(),
            &augment_prompt_for_pose("a knight standing in a courtyard"),
            512,
            512,
            42,
            4,
            Some(1.0),
            conditioning,
            &cancel,
            &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((w, h), (512, 512));
        assert_eq!(pixels.len(), 512 * 512 * 3);
        assert!(steps_seen >= 1, "expected denoise step progress");
        assert!(pixels.windows(2).any(|w| w[0] != w[1]));
    }

    #[cfg(target_os = "macos")]
    fn dirs_home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sdxl_sub_mode_classifies_advanced_shapes() {
        // Plain txt2img (no reference, not an edit) → not an advanced job.
        assert!(sdxl_sub_mode(&request(json!({ "model": "sdxl", "prompt": "a fox" }))).is_none());
        // Reference (not edit) → IP-Adapter.
        assert!(matches!(
            sdxl_sub_mode(&request(
                json!({ "model": "sdxl", "referenceAssetId": "ref_1" })
            )),
            Some(SdxlSubMode::Ip)
        ));
        // edit_image + source → plain img2img edit.
        assert!(matches!(
            sdxl_sub_mode(&request(
                json!({ "model": "sdxl", "mode": "edit_image", "sourceAssetId": "src_1" })
            )),
            Some(SdxlSubMode::Edit)
        ));
        // edit_image + source + mask → inpaint.
        assert!(matches!(
            sdxl_sub_mode(&request(json!({
                "model": "sdxl", "mode": "edit_image",
                "sourceAssetId": "src_1", "maskAssetId": "mask_1"
            }))),
            Some(SdxlSubMode::Inpaint)
        ));
        // fit_mode outpaint wins over a user mask (the torch path checks outpaint first,
        // then unions the user mask into the generated border).
        assert!(matches!(
            sdxl_sub_mode(&request(json!({
                "model": "sdxl", "mode": "edit_image", "sourceAssetId": "src_1",
                "fitMode": "outpaint", "maskAssetId": "mask_1"
            }))),
            Some(SdxlSubMode::Outpaint)
        ));
        // edit_image without a source → nothing to do (falls through, not advanced).
        assert!(
            sdxl_sub_mode(&request(json!({ "model": "sdxl", "mode": "edit_image" }))).is_none()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn engine_dim_rounds_up_to_mult8_and_clamps() {
        assert_eq!(engine_dim(1024), 1024); // already valid
        assert_eq!(engine_dim(1000), 1000); // already a multiple of 8
        assert_eq!(engine_dim(1001), 1008); // rounds up to the next multiple of 8
        assert_eq!(engine_dim(500), 512); // clamps to the engine minimum
        assert_eq!(engine_dim(3000), 2048); // clamps to the engine maximum
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn detail_feather_ramps_over_overlap() {
        // No overlap → a flat field of ones (every pixel contributes fully).
        let flat = detail_feather(8, 8, 0);
        assert_eq!(flat.len(), 64);
        assert!(flat.iter().all(|&w| (w - 1.0).abs() < 1e-6));

        // With overlap, the borders ramp down (raised cosine) while the center stays 1.0.
        let f = detail_feather(16, 16, 4);
        assert_eq!(f.len(), 256);
        let at = |x: usize, y: usize| f[y * 16 + x];
        assert!((at(8, 8) - 1.0).abs() < 1e-6, "center is full weight");
        assert!(at(0, 0) < at(8, 8), "corner is feathered below center");
        // Symmetric across the tile.
        assert!((at(0, 8) - at(15, 8)).abs() < 1e-6);
        assert!((at(8, 0) - at(8, 15)).abs() < 1e-6);
    }

    /// sc-3625 real-Mac E2E (epic 3621): drive the WORKER's FLUX.1 XLabs IP-Adapter reference path
    /// end to end on real weights — `resolve_flux_ip_adapter_dir` staging from the real HF cache +
    /// `mlx_load_with_ip` + a real `Conditioning::Reference` dev `true_cfg` render against the
    /// pinned mlx-gen engine. Guards the worker plumbing the engine-side A/B can't: the staged-dir
    /// contract + the dev reference render NOT regressing to the pre-#173 saturation (which
    /// collapsed `true_cfg=4` to a near-uniform white frame). Run (needs FLUX.1-dev +
    /// `XLabs-AI/flux-ip-adapter` + `openai/clip-vit-large-patch14` in the HF cache):
    /// ```text
    /// HF_HUB_CACHE=$HOME/.cache/huggingface/hub \
    ///   cargo test -p sceneworks-worker --release flux_ip_reference_worker_e2e -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-Mac E2E: loads FLUX.1-dev + XLabs IP-Adapter + CLIP-ViT-L from the HF cache"]
    fn flux_ip_reference_worker_e2e() {
        fn hf_cache() -> String {
            std::env::var("HF_HUB_CACHE").unwrap_or_else(|_| {
                format!("{}/.cache/huggingface/hub", std::env::var("HOME").unwrap())
            })
        }
        fn hf_snapshot(repo: &str, needs: &str) -> PathBuf {
            let safe = repo.replace('/', "--");
            let snaps = PathBuf::from(hf_cache())
                .join(format!("models--{safe}"))
                .join("snapshots");
            std::fs::read_dir(&snaps)
                .unwrap_or_else(|_| panic!("HF snapshot for {repo}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir() && p.join(needs).exists())
                .unwrap_or_else(|| panic!("a complete {repo} snapshot with {needs}"))
        }

        // Point the worker's HF-cache resolver + Settings at the real cache + a temp data dir.
        std::env::set_var("HF_HUB_CACHE", hf_cache());
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("SCENEWORKS_DATA_DIR", data.path());
        let settings = Settings::from_env();

        // (1) The worker's OWN staging fn, against the real cache (the net-new sc-3625 fs logic).
        let staged = resolve_flux_ip_adapter_dir(&settings).expect("stage flux ip-adapter dir");
        assert!(
            staged.join("ip_adapter.safetensors").exists(),
            "staged ip_adapter.safetensors"
        );
        assert!(
            staged.join("image_encoder/model.safetensors").exists(),
            "staged image_encoder/model.safetensors"
        );

        // (2) Load FLUX.1-dev through the worker loader with the staged IP dir, resolving the
        // engine id from the MLX_MODELS table exactly as the real dispatch does (model "flux_dev"
        // → engine "flux1_dev").
        let engine_id = mlx_model("flux_dev")
            .expect("flux_dev in MLX_MODELS")
            .engine_id;
        let flux_dev = hf_snapshot("black-forest-labs/FLUX.1-dev", "transformer");
        let generator = mlx_load_with_ip(engine_id, flux_dev, None, vec![], Some(staged))
            .unwrap_or_else(|e| panic!("mlx_load_with_ip {engine_id} + ip: {e}"));

        // (3) Reference render through the dev `true_cfg` path (white-dot garbage pre-#173).
        let reference = {
            let p = "/tmp/flux_ab/reference.png";
            if std::path::Path::new(p).exists() {
                let img = image::open(p).unwrap().to_rgb8();
                Image {
                    width: img.width(),
                    height: img.height(),
                    pixels: img.into_raw(),
                }
            } else {
                // Synthetic fallback: a solid orange field (still drives the IP branch).
                Image {
                    width: 64,
                    height: 64,
                    pixels: [255u8, 140, 0]
                        .iter()
                        .cycle()
                        .take(64 * 64 * 3)
                        .copied()
                        .collect(),
                }
            }
        };
        let req = |conditioning, true_cfg| GenerationRequest {
            prompt: "an oil painting in the bold swirling brushstroke style of Van Gogh".into(),
            width: 512,
            height: 512,
            seed: Some(2),
            steps: Some(16),
            true_cfg,
            conditioning,
            ..Default::default()
        };
        let run = |r: &GenerationRequest| match generator.generate(r, &mut |_| {}).unwrap() {
            GenerationOutput::Images(mut v) => v.remove(0),
            _ => unreachable!(),
        };

        let ref_out = run(&req(
            vec![Conditioning::Reference {
                image: reference,
                strength: Some(0.7),
            }],
            Some(4.0),
        ));
        let plain = run(&req(vec![], None));

        // Non-degenerate: the pre-#173 saturation collapsed dev true_cfg=4 to a near-uniform white.
        let n = ref_out.pixels.len() as f64;
        let mean = ref_out.pixels.iter().map(|&b| b as f64).sum::<f64>() / n;
        let var = ref_out
            .pixels
            .iter()
            .map(|&b| (b as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(
            var > 200.0,
            "reference render near-uniform (var={var:.1}) — true_cfg saturation regression"
        );
        assert!(
            mean < 245.0,
            "reference render near-white (mean={mean:.1}) — true_cfg saturation regression"
        );

        // The reference actually changed the image vs plain txt2img (IP branch is applied).
        let diff = ref_out
            .pixels
            .iter()
            .zip(&plain.pixels)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff > ref_out.pixels.len() / 10,
            "reference barely changed the render ({diff} px)"
        );

        if let Ok(p) = std::env::var("FLUX_IP_WORKER_OUT") {
            image::RgbImage::from_raw(ref_out.width, ref_out.height, ref_out.pixels.clone())
                .unwrap()
                .save(&p)
                .unwrap();
            println!("[worker-e2e] wrote {p}");
        }
        println!("[worker-e2e] OK: staged dir contract + flux_dev load + dev true_cfg=4 reference render — var={var:.1} mean={mean:.1} diff-vs-txt2img={diff}px");
    }
}
