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
    all(target_os = "windows", feature = "backend-candle")
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
#[cfg(target_os = "macos")]
use mlx_gen_kolors as _;
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
// candle (Windows/CUDA) backend — epic 3672, sc-3675. Mirror of the mlx `use … as _;` anchors above:
// force the candle SDXL provider to link so its `inventory::submit!` (engine id `sdxl`, backend
// `candle`) survives linker GC and resolves through the SAME gen_core registry — no candle-specific
// dispatch, `cfg(target_os)` just decides which backend registers. Gated on the optional
// `backend-candle` build feature too (the dep is pulled only by the CUDA build); whether candle is
// actually USED at runtime is the separate `backend_candle_enabled` setting, not this link anchor.
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
use candle_gen_sdxl as _;
// The four candle image families wired in sc-5096 (epic 5095). Same force-link anchor pattern as the
// SDXL crate above + the mlx providers: each self-registers its engine id (`z_image_turbo` /
// `flux1_schnell` + `flux1_dev` / `flux2_klein_9b` / `qwen_image`) into the shared gen_core inventory
// registry, and the `as _;` keeps the MSVC release linker from GC-ing the `inventory::submit!`.
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
use candle_gen_flux as _;
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
use candle_gen_flux2 as _;
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
use candle_gen_qwen_image as _;
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
use candle_gen_z_image as _;
// Lens / Lens-Turbo (epic 5107 engine / sc-5126 cutover) — the candle Windows/CUDA sibling of the
// `mlx_gen_lens` anchor above, and the 8th candle image family (effectively). Self-registers `lens`
// (20-step/CFG-5) + `lens_turbo` (4-step/g-1.0) into the shared gen_core inventory registry; the
// FIRST candle family to advertise Q4/Q8 quant + LoRA/LoKr. Force-linked so the MSVC release linker
// keeps the `inventory::submit!` (the dead-strip trap that bit Kolors on MLX).
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
use candle_gen_lens as _;
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

/// The stub adapter id recorded on generated assets (matches the contract fixture
/// `tests/fixtures/rust_migration_contracts/sidecars/asset-image.sceneworks.json`).
const STUB_ADAPTER: &str = "procedural_preview";
/// The adapter id recorded on assets produced by the candle (Windows/CUDA) SDXL lane (sc-3678).
/// Used both per-asset (`generate_candle_stream`) and at the generation-set level (`adapter_id`)
/// so the sidecar + result agree on which backend produced the image.
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
const CANDLE_ADAPTER: &str = "candle_sdxl";
// Shared by the MLX path and the candle Lens lane (sc-5126) — both cap a job's user LoRAs at 3
// (`resolve_adapters`), so the const is available on the Windows candle build too.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
const MAX_JOB_LORAS: usize = 3;

// The engine dispatch table + its `ModelRow`/`mlx_model` join moved to the all-targets
// `engines` module (sc-3723); the two descriptor-duplicating flags it used to carry
// (`supports_guidance`/`supports_negative_prompt`) are now read from the linked gen_core
// descriptor via `ResolvedModel`. Shared by the macOS MLX path and the Windows candle lane
// (sc-5096) — the join is backend-neutral, so `generate_candle_stream` resolves repo/steps/guidance
// through the same `mlx_model` lookup the MLX path uses.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
use crate::engines::{mlx_model, ResolvedModel};
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
    #[cfg(all(target_os = "windows", feature = "backend-candle"))]
    let handled = if settings.backend_candle_enabled && is_candle_engine(&request.model) {
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
        all(target_os = "windows", feature = "backend-candle")
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
        return model.adapter_label();
    }
    // Windows/CUDA candle lane (sc-3678, per-engine in sc-5096): report the candle adapter for the
    // wired family so the generation-set fact matches the per-asset `adapter` the candle path writes,
    // instead of falling through to the procedural-stub label. Routing (`worker_supports_job`) only
    // lets candle-eligible txt2img jobs reach this worker, so `is_candle_engine` here implies the
    // candle path ran.
    #[cfg(all(target_os = "windows", feature = "backend-candle"))]
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
    all(target_os = "windows", feature = "backend-candle")
))]
// MLX/candle generator stream helpers.
include!("image_jobs/stream.rs");

#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
// base image routing (MLX) + neutral txt2img generation harness + the candle execution path.
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
// Kolors advanced conditioning (img2img + IP-Adapter-Plus reference).
include!("image_jobs/kolors.rs");
#[cfg(target_os = "macos")]
// InstantID native routing.
include!("image_jobs/instantid.rs");
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
        "image_detail runs on the macOS MLX worker or the Python torch worker, not this Rust worker"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests;
