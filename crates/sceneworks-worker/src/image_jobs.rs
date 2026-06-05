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
//! `flux_dev` — sc-3023) run **real** in-process inference via the linked mlx-gen
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
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Progress, Quant, WeightsSource,
};
#[cfg(target_os = "macos")]
use mlx_gen_flux as _;
#[cfg(target_os = "macos")]
use mlx_gen_z_image as _;

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
        adapter_label: "mlx_z_image",
    },
    MlxModel {
        sceneworks_id: "flux_schnell",
        engine_id: "flux1_schnell",
        default_repo: "black-forest-labs/FLUX.1-schnell",
        default_steps: 4,
        supports_guidance: false,
        default_guidance: 0.0,
        adapter_label: "mlx_flux",
    },
    MlxModel {
        sceneworks_id: "flux_dev",
        engine_id: "flux1_dev",
        default_repo: "black-forest-labs/FLUX.1-dev",
        default_steps: 28,
        supports_guidance: true,
        default_guidance: 3.5,
        adapter_label: "mlx_flux",
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

    let plan = ImagePlan::new(&request);
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            &format!("Preparing {} image(s).", request.count),
            None,
            backend,
        ),
    )
    .await?;

    let mut asset_writes: Vec<Value> = Vec::with_capacity(request.count as usize);

    // Real in-process MLX inference on macOS for engine-backed models; otherwise the
    // procedural stub (keeps non-macOS + not-yet-ported models working).
    #[cfg(target_os = "macos")]
    let handled = if mlx_available(&request, settings) {
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
            &format!("Generated {} image(s).", request.count),
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
struct ImagePlan {
    request: ImageRequest,
    genset_id: String,
    created_at: String,
    family: String,
    slug: String,
    generation_set: Value,
}

impl ImagePlan {
    fn new(request: &ImageRequest) -> Self {
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
            "count": request.count,
            "createdAt": created_at,
        });
        Self {
            request: request.clone(),
            genset_id,
            created_at,
            family,
            slug,
            generation_set,
        }
    }
}

/// Save image `index` (its RGB8 `pixels`) under `assets/images/` and return the flat
/// fact the API turns into an indexed asset (every key here is consumed by
/// `build_image_sidecar_parts`). Shared by the stub and real paths.
#[allow(clippy::too_many_arguments)]
fn write_image_asset(
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
    let media_rel = format!("assets/images/{filename}");
    let media_path = project_path.join(&media_rel);
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
        "count": request.count,
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
        "expectedCount": plan.request.count,
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
fn resolve_seed(request: &ImageRequest, index: usize) -> i64 {
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
fn image_progress(
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
        extra: BTreeMap::new(),
    }
}

fn backend_label(gpu_id: &str) -> &str {
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
#[cfg(target_os = "macos")]
fn mlx_available(request: &ImageRequest, settings: &Settings) -> bool {
    mlx_model(&request.model).is_some() && resolve_weights_dir(request, settings).is_some()
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
fn resolve_weights_dir(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        if path.is_dir() {
            return Some(path);
        }
    }
    let model = mlx_model(&request.model)?;
    huggingface_snapshot_dir(&settings.data_dir, &model_repo(request, model))
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

/// First non-empty of installedPath/sourcePath/path/source.path on a LoRA spec.
#[cfg(target_os = "macos")]
fn lora_path(lora: &Value) -> Option<PathBuf> {
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

/// Classify a LoRA file into the mlx-gen adapter kind. SceneWorks peft-LoKr (stamped
/// `networkType: lokr`) → `Lokr`; third-party LyCORIS (LoHa / kohya LoKr) is not
/// reconstructable here → rejected; everything else → `Lora`.
#[cfg(target_os = "macos")]
fn classify_adapter(file: &Path) -> WorkerResult<AdapterKind> {
    let header = read_safetensors_header(file)
        .map_err(|error| WorkerError::InvalidPayload(format!("LoRA header: {error}")))?;
    let metadata = header.get("__metadata__");
    let network_type = metadata
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    if network_type.as_deref() == Some("lokr") {
        return Ok(AdapterKind::Lokr);
    }
    let module = metadata
        .and_then(|meta| meta.get("ss_network_module"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let lycoris_keys = header
        .as_object()
        .map(|object| {
            object
                .keys()
                .any(|key| key.contains("lokr_") || key.contains("hada_"))
        })
        .unwrap_or(false);
    if module.contains("lycoris") || lycoris_keys {
        return Err(WorkerError::InvalidPayload(
            "Third-party LyCORIS LoRA (LoHa / kohya LoKr) is not supported on the MLX path."
                .to_owned(),
        ));
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
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    mlx_gen::load(engine_id, &spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("{engine_id} load failed: {error}")))
}

/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
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
    let weights_dir = resolve_weights_dir(request, settings)
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    let guidance = resolve_guidance(request, model);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, model);
    let raw_settings = mlx_raw_settings(request, &repo, steps, quant_bits, guidance);
    let engine_id = model.engine_id;
    let adapter_label = model.adapter_label;
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<GenEvent>(64);

    let blocking = {
        let prompt = request.prompt.clone();
        let (width, height) = (request.width, request.height);
        let seeds = seeds.clone();
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let generator = mlx_load(engine_id, weights_dir, quant, adapters)?;
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

    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    while let Some(event) = rx.recv().await {
        if canceled {
            continue; // drain remaining events so the blocking sender never blocks.
        }
        match event {
            GenEvent::Step {
                index,
                current,
                total,
            } => {
                if last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    if check_cancel(api, &job.id, "Image generation canceled by user.")
                        .await
                        .is_err()
                    {
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
                        step_fraction(index, current, total, request.count),
                        &format!(
                            "Image {}/{} — step {current}/{total}.",
                            index + 1,
                            request.count
                        ),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Decoding { index } => {
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, 1, 1, request.count),
                        &format!("Image {}/{} — decoding.", index + 1, request.count),
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
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / request.count as f64),
                        &format!("Generated image {}/{}.", index + 1, request.count),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
        }
    }

    let task_result = blocking
        .await
        .map_err(|error| WorkerError::InvalidPayload(format!("Z-Image task join: {error}")))?;
    if canceled {
        // check_cancel already posted the Canceled update; treat the (likely) generate
        // error as the clean cancel.
        return Err(WorkerError::Canceled(
            "Image generation canceled by user.".to_owned(),
        ));
    }
    task_result
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
        assert!(media_rel.starts_with("assets/images/"));
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
        assert!(mlx_model("sdxl").is_none());
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
        assert_eq!(
            adapter_id(&request(json!({ "model": "sdxl" }))),
            "procedural_preview"
        );
    }

    /// The Z-Image + FLUX.1 providers linked into the worker self-registered via inventory.
    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_engine_registry_links_image_families() {
        let ids: Vec<&str> = mlx_gen::registry::generators()
            .map(|reg| (reg.descriptor)().id)
            .collect();
        for id in ["z_image_turbo", "flux1_schnell", "flux1_dev"] {
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
    #[cfg(target_os = "macos")]
    fn smoke_generate_one(engine_id: &str, snapshot: std::path::PathBuf, guidance: Option<f32>) {
        let generator =
            mlx_load(engine_id, snapshot, Some(mlx_gen::Quant::Q8), Vec::new()).unwrap();
        let cancel = mlx_gen::CancelFlag::new();
        let mut steps_seen = 0u32;
        let steps = mlx_model(match engine_id {
            "flux1_schnell" => "flux_schnell",
            "flux1_dev" => "flux_dev",
            _ => "z_image_turbo",
        })
        .unwrap()
        .default_steps;
        let (w, h, pixels) = mlx_generate_one(
            generator.as_ref(),
            "a serene mountain lake at dawn",
            512,
            512,
            42,
            steps,
            guidance,
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
            "flux1_schnell",
            hf_snapshot("models--black-forest-labs--FLUX.1-schnell"),
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
            "flux1_dev",
            hf_snapshot("models--black-forest-labs--FLUX.1-dev"),
            Some(3.5),
        );
    }

    #[cfg(target_os = "macos")]
    fn dirs_home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
    }
}
