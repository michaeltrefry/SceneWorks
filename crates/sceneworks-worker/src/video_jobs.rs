//! Native video-generation jobs — runtime pipeline + procedural stub (epic 3018, sc-3033).
//!
//! Parses the job into a [`VideoRequest`], produces a single video (one mp4 asset,
//! unlike images which batch `count`), and reports a flat "fact" the Rust API turns
//! into an indexed asset (mirroring `video_generation_result` in the Python worker's
//! `video_adapters.py`). The shared encode pipeline takes the engine's video output
//! shape — RGB8 `frames` + `fps` + an optional synchronized `audio` track — writes
//! the frames to an mp4 (libx264), muxes a 16-bit-PCM WAV as AAC when audio is present
//! (`-shortest`), remuxes `+faststart` (WKWebView range-seek), and extracts a poster
//! frame. It reuses [`crate::media_jobs::run_ffmpeg`] (binary resolution + the
//! periodic-heartbeat / cooperative-cancel loop).
//!
//! sc-3033 ships only the **procedural stub** generator (a moving gradient + a quiet
//! synchronized tone for the LTX family, mirroring the engine: LTX emits audio, Wan
//! does not). The real in-process MLX video models — Wan2.2 (sc-3034) and LTX-2.3 +
//! audio (sc-3035) — link the `mlx-gen-wan` / `mlx-gen-ltx` provider crates and decode
//! `gen_core::GenerationOutput::Video { frames, fps, audio }` into the same
//! [`DecodedVideo`], so the encode/mux/poster path below is unchanged for them.

use std::f32::consts::PI;
use std::path::Path;

use sceneworks_core::video_request::{is_ltx_model, VideoRequest};

use super::*;
use crate::media_jobs::{run_ffmpeg, FfmpegContext};

// Real MLX Wan2.2 generation (macOS, sc-3034). The provider crate self-registers its
// three models via `inventory` only when linked + referenced (`use mlx_gen_wan as _;`,
// the same link-time pattern as the image families in `image_jobs.rs`).
#[cfg(target_os = "macos")]
use crate::image_jobs::{classify_adapter, load_reference_image, lora_path};
// epic 3720 (sc-3724): the backend-neutral generation contract types come from `gen_core`; the
// `as _;` provider links below stay mlx-gen-specific (they register the video engines into the
// registry). `cfg(target_os)` decides which backend crates link, not which contract types this
// module names.
// Backend-neutral contract types shared by the macOS MLX video path AND the Windows candle video
// lane (sc-5097): the streaming driver (`generate_video`), the output decode
// (`run_loaded_video_generation`), and `VideoGenInput`/`video_load_spec` are all backend-neutral, so
// these types compile on both lanes. `cfg(target_os)` decides which provider crate registered the
// video engine, not which contract types this module names.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Precision, Progress, Quant, WeightsSource,
};
// MLX-only contract types (LoRA classification, MoE experts) — the candle video lane uses none of these.
#[cfg(target_os = "macos")]
use gen_core::{AdapterKind, MoeExpert};
// VACE conditioning + replacement types are shared by the MLX Wan-VACE path and the candle Wan-VACE
// lane (sc-5494), so they are available on both the macos and windows+backend-candle builds.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{Image, ReplacementMode};
#[cfg(target_os = "macos")]
use mlx_gen_ltx as _;
#[cfg(target_os = "macos")]
use mlx_gen_seedvr2 as _;
#[cfg(target_os = "macos")]
use mlx_gen_svd as _;
#[cfg(target_os = "macos")]
use mlx_gen_wan as _;
// Bernini (epic 4699): the full planner+renderer `Generator` registers under `bernini` via
// `inventory::submit!`; force-link so the registration survives the linker (reached only through
// `gen_core::load("bernini")`, no direct type contact — the "no generator registered" trap).
#[cfg(target_os = "macos")]
use mlx_gen_bernini as _;
// SCAIL-2 (epic 5439): the character-animation `Generator` registers under `scail2_14b` via
// `inventory::submit!`; force-link so the registration survives the linker (reached only through
// `gen_core::load("scail2_14b")`, no direct type contact — the "no generator registered" trap).
#[cfg(target_os = "macos")]
use mlx_gen_scail2 as _;
// Candle (Windows/CUDA) video providers (sc-5097; sc-5493 adds svd) — force-link anchors so their
// `inventory::submit!` registrations (`wan2_2_ti2v_5b` / `ltx_2_3_distilled` / `svd_xt`) survive the
// MSVC release linker, mirroring the image providers in image_jobs.rs.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_ltx as _;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_svd as _;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_wan as _;
// Candle SeedVR2 video upscaler (sc-5928, epic 4811 / epic 5482) — the Windows/CUDA sibling of the
// Mac `mlx_gen_seedvr2` anchor above. Self-registers `seedvr2_3b` (+ `seedvr2` / `seedvr2_7b`) into
// the shared gen_core inventory; the `video_upscale` path reaches it via `gen_core::load` from
// `run_video_upscale_job`. Force-linked so the MSVC release linker keeps the `inventory::submit!`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_seedvr2 as _;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::character_store::CharacterStore;
// Frame-count stride coercion (Wan needs frames ≡ 1 mod 4; LTX snaps to 8k+1) — used by the MLX path
// and the candle entry (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::video_request::{ltx_frame_count, wan_frame_count};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use std::time::{Duration, Instant};

/// Stub adapter id recorded on generated assets — matches the Python
/// `ProceduralVideoAdapter.id` so the asset sidecar reads identically.
const STUB_ADAPTER: &str = "procedural_video";
const CANCEL_MESSAGE: &str = "Video generation canceled by user.";

/// Decoded video ready for muxing — the worker-side shape both the procedural stub
/// (sc-3033) and the real engine output (`gen_core::GenerationOutput::Video`,
/// sc-3034/3035) feed into [`encode_media`]. Mirrors the engine contract: `frames`
/// are RGB8, `audio` is `Some` for LTX (a synchronized track) and `None` for Wan.
/// The frames are held in memory (the engine returns them that way); the duration
/// clamp (≤30s) in [`VideoRequest`] bounds the footprint.
struct DecodedVideo {
    frames: Vec<RgbFrame>,
    fps: u32,
    audio: Option<AudioTrack>,
}

/// One RGB8 frame, row-major, `pixels.len() == width * height * 3` (the engine's
/// `Image`).
struct RgbFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Interleaved PCM audio — the engine's `AudioTrack` (LTX-2.3 synchronized audio).
struct AudioTrack {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

/// Dispatch handler for `JobType::VideoGenerate`: generate, encode, and stream a
/// single video asset through the Rust GPU worker.
pub(crate) async fn run_video_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = VideoRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let plan = VideoPlan::new(&request, &project_path);
    if let Some(parent) = plan.media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing video.",
            None,
            backend,
        ),
    )
    .await?;

    // sc-3033 ships the procedural stub only; the real MLX video models (Wan sc-3034,
    // LTX+audio sc-3035) decode `GenerationOutput::Video` into a `DecodedVideo` here.
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    let seed = resolve_video_seed(&request);
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.2,
            "Rendering frames.",
            None,
            backend,
        ),
    )
    .await?;
    // sc-3459 (epic 3456): Wan2.2 VACE-Fun A14B is a NEW dual-expert VACE engine
    // (`wan2_2_vace_fun_14b`; native MLX sc-6604, native candle sc-6605). Until those engines
    // merge and the worker's mlx-gen/candle-gen pin is bumped, the runtime cannot serve it — and
    // replace_person must NEVER fall through to the Wan2.1 `generate_wan_vace` backend below, which
    // would silently render with the WRONG checkpoint (the exact failure the epic forbids). Fail
    // honestly instead. When the engine pin lands, this guard is replaced by the real per-platform
    // `generate_wan_vace_fun` dispatch (resolve the dual-expert snapshot → the `wan2_2_vace_fun_14b`
    // engine).
    if request.model == "wan_2_2_vace_fun_14b" {
        return Err(WorkerError::InvalidPayload(
            "wan_2_2_vace_fun_14b: the native Wan2.2 VACE-Fun engine is not yet available in this \
             build (pending the mlx-gen/candle-gen integration, sc-6604/sc-6605). The job will not \
             be routed to the Wan2.1 VACE backend. Choose another model, or update once the engine \
             ships."
                .to_owned(),
        ));
    }
    // Generate: real MLX on macOS for Wan (sc-3034) / LTX+audio (sc-3035) models with
    // resolvable weights, else the procedural stub (non-macOS or missing weights = stub).
    // replace_person (sc-3521) always routes to the native Wan-VACE provider regardless of the
    // user-picked (replace-capable) model — the native equivalent of the torch `WanVACEPipeline`
    // path. It errors clearly if the VACE snapshot is unprovisioned rather than degrading to the
    // procedural stub (a stubbed person-replace would be meaningless). It also reports the honest
    // `replacementStatus` the asset sidecar folds in (project_store::build_video_sidecar_parts).
    #[cfg(target_os = "macos")]
    let (decoded, adapter, raw_settings, replacement_status) = if request.mode == "replace_person" {
        // sc-5452: SCAIL-2 is a higher-quality cross-identity replacement backend behind the same
        // YOLO11 → ByteTrack → SAM3 person-track pipeline. A `scail2_14b` person-replace job routes
        // to SCAIL-2 (the tracked person's masks + the character reference → the engine's
        // replacement conditioning, `replace_flag = true`); every other replace-capable model keeps
        // native Wan-VACE (sc-3521). Routed by model id, not weight availability: like the Wan-VACE
        // path, `generate_scail2_replace` resolves-or-errors loudly if the snapshot is unprovisioned
        // (a person-replace must never silently degrade to a different backend or the stub). Both
        // report the honest `replacementStatus` the asset sidecar folds in.
        if let Some(engine_id) = scail2_engine_id(&request.model) {
            let (decoded, status) = generate_scail2_replace(
                api,
                settings,
                job,
                &request,
                &project_path,
                engine_id,
                backend,
            )
            .await?;
            (
                // The replace_person path doesn't resolve user LoRAs (sc-5452), so no lightning recipe.
                decoded,
                SCAIL2_ADAPTER,
                scail2_raw_settings(&request, false),
                Some(status),
            )
        } else {
            let (decoded, status) =
                generate_wan_vace(api, settings, job, &request, &project_path, backend).await?;
            (
                decoded,
                WAN_VACE_ADAPTER,
                wan_vace_raw_settings(&request),
                Some(status),
            )
        }
    } else if matches!(request.mode.as_str(), "extend_clip" | "video_bridge")
        && wan_engine_id(&request.model) == Some("wan2_2_ti2v_5b")
        && wan_available(&request, settings)
    {
        // sc-3812 (tier C): route Wan extend/bridge to native Wan-VACE for genuine motion
        // continuity (the model attends to real source frames, not one boundary still). Falls
        // back to the sc-3357 single-frame TI2V-5B keyframe path when the VACE snapshot is
        // unprovisioned, so the mode keeps working on the weights the user already has. The
        // engine substitution under the `wan_2_2` pick is recorded honestly in raw-settings.
        match resolve_wan_vace_model_dir(settings) {
            Ok(model_dir) => (
                generate_wan_vace_extend_bridge(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    backend,
                    model_dir,
                )
                .await?,
                WAN_VACE_ADAPTER,
                wan_vace_extend_raw_settings(&request),
                None,
            ),
            Err(_) => {
                let engine_id = "wan2_2_ti2v_5b";
                (
                    generate_wan(
                        api,
                        settings,
                        job,
                        &request,
                        &project_path,
                        engine_id,
                        backend,
                    )
                    .await?,
                    WAN_ADAPTER,
                    wan_raw_settings(&request, engine_id),
                    None,
                )
            }
        }
    } else if let Some(engine_id) =
        wan_engine_id(&request.model).filter(|_| wan_available(&request, settings))
    {
        (
            generate_wan(
                api,
                settings,
                job,
                &request,
                &project_path,
                engine_id,
                backend,
            )
            .await?,
            WAN_ADAPTER,
            wan_raw_settings(&request, engine_id),
            None,
        )
    } else if let Some(engine_id) =
        ltx_engine_id(&request.model).filter(|_| ltx_available(&request, settings))
    {
        (
            generate_ltx(
                api,
                settings,
                job,
                &request,
                &project_path,
                engine_id,
                backend,
            )
            .await?,
            LTX_ADAPTER,
            ltx_raw_settings(&request),
            None,
        )
    } else if let Some(engine_id) =
        svd_engine_id(&request.model).filter(|_| svd_available(&request, settings))
    {
        (
            generate_svd(
                api,
                settings,
                job,
                &request,
                &project_path,
                engine_id,
                backend,
            )
            .await?,
            SVD_ADAPTER,
            svd_raw_settings(&request),
            None,
        )
    } else if let Some(engine_id) =
        bernini_engine_id(&request.model).filter(|_| bernini_available(&request, settings))
    {
        // Bernini (epic 4699 / sc-4707 + sc-4703): the full Qwen2.5-VL planner + Wan2.2-T2V-A14B
        // renderer. Serves text_to_video + the editing/reference video modes (video_to_video /
        // reference_to_video / reference_video_to_video); `generate_bernini` maps the SceneWorks mode
        // to the engine guidance task and resolves the source media into the planner conditioning.
        // (t2i/i2i image companion = a separate image-typed catalog id, tracked under epic 4699.)
        (
            generate_bernini(
                api,
                settings,
                job,
                &request,
                &project_path,
                engine_id,
                backend,
            )
            .await?,
            BERNINI_ADAPTER,
            bernini_raw_settings(&request),
            None,
        )
    } else if let Some(engine_id) =
        scail2_engine_id(&request.model).filter(|_| scail2_available(&request, settings))
    {
        // SCAIL-2 (epic 5439 / sc-5448): Wan2.1-14B I2V character animation. `generate_scail2`
        // segments the reference image + driving frames with native SAM3, paints the color-coded
        // masks, and maps the SceneWorks mode to the engine task (animate_character → animation;
        // replace_person → replacement is wired in sc-5452). No torch path (mac-only engine).
        (
            generate_scail2(
                api,
                settings,
                job,
                &request,
                &project_path,
                engine_id,
                backend,
            )
            .await?,
            SCAIL2_ADAPTER,
            // Best-effort lightning detection for the asset's effective-recipe record (the adapter
            // file is already resolved by generate_scail2 above; re-reading its header is cheap).
            scail2_raw_settings(
                &request,
                scail2_adapters_have_lightning(
                    &resolve_scail2_adapters(settings, &request).unwrap_or_default(),
                ),
            ),
            None,
        )
    } else {
        // An MLX-routed video model whose snapshot didn't resolve must fail
        // loudly with the resolver's precise error instead of completing with
        // procedural stub output (sc-4176, epic 3482 "unsupported jobs error
        // loudly"). replace_person above already follows this rule; the stub
        // remains only for ids outside the engine families (test models,
        // not-yet-ported families).
        ensure_video_engine_weights(&request, settings)?;
        (
            generate_stub_video(&request, seed),
            STUB_ADAPTER,
            stub_raw_settings(&request),
            None,
        )
    };
    // Windows/CUDA candle video lane (sc-5097): a real wan/ltx txt2video job runs through
    // `generate_candle_video` (the same neutral encode/mux path as MLX + the stub); anything the
    // candle lane doesn't serve stubs exactly as before. Gated on `backend_candle_enabled` (default
    // off → routing unchanged until parity). Conditioning shapes never reach here — the router's
    // `video_job_is_candle_eligible` confines the candle worker to txt2video.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (decoded, adapter, raw_settings, replacement_status) = if settings.backend_candle_enabled
        && request.mode == "replace_person"
    {
        // Candle Wan-VACE person replacement (sc-5494) — the candle equivalent of the MLX `wan_vace`
        // path. The router (`video_request_candle_eligible`) already confirmed the candle-VACE model +
        // the source clip + person track before this job reached the candle worker; `generate_candle_
        // wan_vace` resolves-or-errors loudly (a person-replace must never silently degrade to a stub).
        let (decoded, status) =
            generate_candle_wan_vace(api, settings, job, &request, &project_path, backend).await?;
        (
            decoded,
            CANDLE_WAN_VACE_ADAPTER,
            wan_vace_raw_settings(&request),
            Some(status),
        )
    } else if settings.backend_candle_enabled
        && matches!(request.mode.as_str(), "extend_clip" | "video_bridge")
    {
        // Candle Wan-VACE extend/bridge (sc-5494): real source frames pinned at the kept positions +
        // a generated span (the candle equivalent of the MLX `generate_wan_vace_extend_bridge`).
        let decoded = generate_candle_wan_vace_extend_bridge(
            api,
            settings,
            job,
            &request,
            &project_path,
            backend,
        )
        .await?;
        (
            decoded,
            CANDLE_WAN_VACE_ADAPTER,
            wan_vace_extend_raw_settings(&request),
            None::<Value>,
        )
    } else if settings.backend_candle_enabled && is_candle_video_engine(&request.model) {
        let (decoded, adapter, raw_settings) =
            generate_candle_video(api, settings, job, &request, &project_path, backend).await?;
        (decoded, adapter, raw_settings, None::<Value>)
    } else {
        (
            generate_stub_video(&request, seed),
            STUB_ADAPTER,
            stub_raw_settings(&request),
            None::<Value>,
        )
    };
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let (decoded, adapter, raw_settings, replacement_status) = (
        generate_stub_video(&request, seed),
        STUB_ADAPTER,
        stub_raw_settings(&request),
        None::<Value>,
    );
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.6,
            "Encoding video.",
            None,
            backend,
        ),
    )
    .await?;
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    encode_media(&plan.media_path, decoded, Some(ctx)).await?;

    let fact = video_asset_fact(&plan, seed, adapter, raw_settings, replacement_status);
    let result = streaming_result(&plan, &fact, adapter);
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Generated video.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Per-job invariants for the single video this job produces.
struct VideoPlan {
    request: VideoRequest,
    genset_id: String,
    asset_id: String,
    created_at: String,
    family: String,
    /// `assets/videos/<genset>/<date>_<model>_<slug>.mp4` (project-relative).
    media_rel: String,
    /// Absolute path to the media file.
    media_path: PathBuf,
}

impl VideoPlan {
    fn new(request: &VideoRequest, project_path: &Path) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let asset_id = fresh_asset_id();
        let created_at = now_rfc3339();
        let family = resolve_family(request);
        let slug = slugify(&request.prompt, "video", Some(42));
        // Nest under the per-generation id so two renders sharing date+model+slug
        // cannot collide on a flat path (mirrors the image + Python video adapters).
        let media_rel = format!(
            "assets/videos/{genset_id}/{}_{}_{slug}.mp4",
            &created_at[..10],
            request.model
        );
        let media_path = project_path.join(&media_rel);
        Self {
            request: request.clone(),
            genset_id,
            asset_id,
            created_at,
            family,
            media_rel,
            media_path,
        }
    }
}

/// Resolve the seed, matching the Python `resolve_seed(seed, prompt)`: an explicit
/// seed wins, else the first 4 bytes of `sha256(prompt)` (its `hexdigest()[:8]`).
fn resolve_video_seed(request: &VideoRequest) -> i64 {
    if let Some(seed) = request.seed {
        return seed;
    }
    let digest = Sha256::digest(request.prompt.as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as i64
}

/// The asset's video family, from the resolved manifest entry when present, else
/// inferred from the model id (parity with the Python `VIDEO_MODEL_TARGETS` family).
fn resolve_family(request: &VideoRequest) -> String {
    if let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
    {
        if !family.trim().is_empty() {
            return family.to_owned();
        }
    }
    if is_ltx_model(&request.model) {
        "ltx-video".to_owned()
    } else if request.model.starts_with("wan") {
        "wan-video".to_owned()
    } else {
        "video".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Procedural stub generator (sc-3033). Real MLX models land in sc-3034/3035.
// ---------------------------------------------------------------------------

/// Build a deterministic placeholder clip: `frame_count` moving-gradient frames at
/// the request fps, plus a quiet synchronized tone for the LTX family (the engine
/// emits audio for LTX and none for Wan, so the stub mirrors that split — exercising
/// both the audio-mux and video-only encode paths).
fn generate_stub_video(request: &VideoRequest, seed: i64) -> DecodedVideo {
    let frame_count = request.frame_count();
    let fps = request.fps.max(1);
    let (width, height) = (request.width, request.height);
    let frames = (0..frame_count)
        .map(|index| RgbFrame {
            width,
            height,
            pixels: stub_video_rgb8(width, height, seed, index, frame_count),
        })
        .collect();
    let audio = is_ltx_model(&request.model).then(|| stub_audio_track(frame_count, fps));
    DecodedVideo { frames, fps, audio }
}

/// Deterministic per-frame pixels: a vertical gradient from a per-seed base colour to
/// white, with a bright vertical band that sweeps left→right across the clip so frames
/// differ (visible motion). Exactly `width * height * 3` RGB8 bytes.
fn stub_video_rgb8(width: u32, height: u32, seed: i64, index: u32, frame_count: u32) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let v_span = height.saturating_sub(1).max(1) as f32;
    // The sweeping band's centre column for this frame.
    let progress = index as f32 / frame_count.max(1) as f32;
    let band_centre = progress * width.saturating_sub(1).max(1) as f32;
    let band_half = (width as f32 * 0.06).max(1.0);
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / v_span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for x in 0..width {
            let dist = (x as f32 - band_centre).abs();
            if dist <= band_half {
                // Brighten toward white inside the band (1.0 at centre → 0 at edge).
                let highlight = 1.0 - dist / band_half;
                buffer.push(lerp(row[0], highlight));
                buffer.push(lerp(row[1], highlight));
                buffer.push(lerp(row[2], highlight));
            } else {
                buffer.extend_from_slice(&row);
            }
        }
    }
    buffer
}

fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

/// A quiet 220 Hz mono tone matching the clip length (`frame_count / fps` seconds) at
/// 48 kHz — enough to exercise the WAV-write + AAC-mux + `-shortest` path end to end.
fn stub_audio_track(frame_count: u32, fps: u32) -> AudioTrack {
    let sample_rate = 48_000u32;
    let duration = frame_count as f32 / fps.max(1) as f32;
    let n = (sample_rate as f32 * duration).round().max(1.0) as usize;
    let freq = 220.0f32;
    let samples = (0..n)
        .map(|i| (2.0 * PI * freq * (i as f32 / sample_rate as f32)).sin() * 0.2)
        .collect();
    AudioTrack {
        samples,
        sample_rate,
        channels: 1,
    }
}

// ---------------------------------------------------------------------------
// Encode pipeline: frames → mp4 (+ optional AAC audio) → faststart → poster.
// Reuses `media_jobs::run_ffmpeg`. Pure of the API except the optional `ctx`
// (heartbeat/cancel), so it is exercisable in tests with a real ffmpeg.
// ---------------------------------------------------------------------------

/// Write `decoded` to `media_path` as an mp4: frames → libx264, an optional 16-bit
/// PCM WAV muxed as AAC (`-shortest`), then a best-effort `+faststart` remux and
/// `.poster.jpg`. `media_path` is created (atomically renamed from a temp) only on
/// success; all intermediates are removed regardless of outcome.
async fn encode_media(
    media_path: &Path,
    decoded: DecodedVideo,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let frames_dir = media_path.with_extension("frames");
    let enc_tmp = media_path.with_extension("enc.mp4");
    let wav_tmp = media_path.with_extension("audio.wav");
    let mux_tmp = media_path.with_extension("mux.mp4");
    let result = encode_inner(
        media_path,
        decoded,
        ctx,
        &frames_dir,
        &enc_tmp,
        &wav_tmp,
        &mux_tmp,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let _ = tokio::fs::remove_file(&enc_tmp).await;
    let _ = tokio::fs::remove_file(&wav_tmp).await;
    let _ = tokio::fs::remove_file(&mux_tmp).await;
    if result.is_err() {
        // A failure (or cancel) before the atomic rename leaves no media_path; if the
        // rename itself half-completed, drop the partial so the asset never points at it.
        let _ = tokio::fs::remove_file(media_path).await;
    }
    result
}

#[allow(clippy::too_many_arguments)]
async fn encode_inner(
    media_path: &Path,
    decoded: DecodedVideo,
    ctx: Option<FfmpegContext<'_>>,
    frames_dir: &Path,
    enc_tmp: &Path,
    wav_tmp: &Path,
    mux_tmp: &Path,
) -> WorkerResult<()> {
    let fps = decoded.fps.max(1);
    let audio = decoded.audio;
    let frames = decoded.frames;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "video generation produced no frames".to_owned(),
        ));
    }

    // 1. Write the frame sequence (blocking PNG encodes off the async runtime).
    tokio::fs::create_dir_all(frames_dir).await?;
    let dir = frames_dir.to_path_buf();
    tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        for (index, frame) in frames.into_iter().enumerate() {
            let RgbFrame {
                width,
                height,
                pixels,
            } = frame;
            let image = image::RgbImage::from_raw(width, height, pixels).ok_or_else(|| {
                WorkerError::InvalidPayload("video frame buffer size mismatch".to_owned())
            })?;
            let path = dir.join(format!("frame_{index:05}.png"));
            image
                .save_with_format(&path, image::ImageFormat::Png)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        }
        Ok(())
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    // 2. Frames → mp4 (libx264, yuv420p — request dims are multiples of 32, so even).
    let pattern = frames_dir.join("frame_%05d.png");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-framerate".to_owned(),
            fps.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            pattern.to_string_lossy().into_owned(),
            "-c:v".to_owned(),
            "libx264".to_owned(),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            "-r".to_owned(),
            fps.to_string(),
            enc_tmp.to_string_lossy().into_owned(),
        ],
        ctx,
    )
    .await?;

    // 3. Mux the audio track (LTX) as AAC, else the video-only mp4 is the result.
    let finished_tmp = if let Some(audio) = audio {
        write_wav_pcm16(&audio, wav_tmp)?;
        run_ffmpeg(
            vec![
                "ffmpeg".to_owned(),
                "-nostdin".to_owned(),
                "-y".to_owned(),
                "-i".to_owned(),
                enc_tmp.to_string_lossy().into_owned(),
                "-i".to_owned(),
                wav_tmp.to_string_lossy().into_owned(),
                "-c:v".to_owned(),
                "copy".to_owned(),
                "-c:a".to_owned(),
                "aac".to_owned(),
                "-shortest".to_owned(),
                mux_tmp.to_string_lossy().into_owned(),
            ],
            ctx,
        )
        .await?;
        mux_tmp
    } else {
        enc_tmp
    };

    // 4. Publish atomically, then best-effort faststart + poster (mirrors Python).
    tokio::fs::rename(finished_tmp, media_path).await?;
    faststart_mp4(media_path).await;
    write_poster_frame(media_path).await;
    Ok(())
}

/// Peak-normalize the f32 PCM to 16-bit and write a canonical WAV. Silence (peak 0)
/// stays silent rather than dividing by zero.
fn write_wav_pcm16(audio: &AudioTrack, path: &Path) -> WorkerResult<()> {
    let peak = audio
        .samples
        .iter()
        .fold(0.0f32, |max, &sample| max.max(sample.abs()));
    let scale = if peak > 0.0 {
        i16::MAX as f32 / peak
    } else {
        0.0
    };
    let mut pcm = Vec::with_capacity(audio.samples.len() * 2);
    for &sample in &audio.samples {
        let value = (sample * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        pcm.extend_from_slice(&value.to_le_bytes());
    }

    let channels = audio.channels.max(1);
    let bits_per_sample = 16u16;
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = audio.sample_rate * block_align as u32;
    let data_len = pcm.len() as u32;

    let mut buffer = Vec::with_capacity(44 + pcm.len());
    buffer.extend_from_slice(b"RIFF");
    buffer.extend_from_slice(&(36 + data_len).to_le_bytes());
    buffer.extend_from_slice(b"WAVE");
    buffer.extend_from_slice(b"fmt ");
    buffer.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buffer.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buffer.extend_from_slice(&channels.to_le_bytes());
    buffer.extend_from_slice(&audio.sample_rate.to_le_bytes());
    buffer.extend_from_slice(&byte_rate.to_le_bytes());
    buffer.extend_from_slice(&block_align.to_le_bytes());
    buffer.extend_from_slice(&bits_per_sample.to_le_bytes());
    buffer.extend_from_slice(b"data");
    buffer.extend_from_slice(&data_len.to_le_bytes());
    buffer.extend_from_slice(&pcm);
    std::fs::write(path, buffer)?;
    Ok(())
}

/// Best-effort `+faststart` remux (moov atom to the front so WKWebView can start
/// playback without a tail byte-range seek). A missing/failing ffmpeg leaves the
/// original untouched — the API's byte-range support is the load-bearing guarantee.
async fn faststart_mp4(media_path: &Path) {
    if !media_path.exists() {
        return;
    }
    let remuxed = media_path.with_extension("faststart.mp4");
    let ok = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            "-movflags".to_owned(),
            "+faststart".to_owned(),
            remuxed.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .is_ok();
    if ok {
        let _ = tokio::fs::rename(&remuxed, media_path).await;
    } else {
        let _ = tokio::fs::remove_file(&remuxed).await;
    }
}

/// Best-effort poster extraction to `<name>.poster.jpg` (WKWebView does not paint a
/// `<video>`'s first frame on its own). A missing/failing ffmpeg leaves no poster.
async fn write_poster_frame(media_path: &Path) {
    if !media_path.exists() {
        return;
    }
    let poster = media_path.with_extension("poster.jpg");
    let ok = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-q:v".to_owned(),
            "3".to_owned(),
            poster.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .is_ok();
    if !ok {
        let _ = tokio::fs::remove_file(&poster).await;
    }
}

// ---------------------------------------------------------------------------
// Asset fact + streaming result (mirrors `video_generation_result`).
// ---------------------------------------------------------------------------

/// The flat per-asset fact the Rust API turns into an indexed video asset (every key
/// is consumed by the API's video sidecar builder). Mirrors `video_generation_result`.
/// `adapter` is the generating adapter id (`procedural_video` stub / `mlx_wan` real)
/// and `raw_settings` its recorded knobs.
fn video_asset_fact(
    plan: &VideoPlan,
    seed: i64,
    adapter: &str,
    raw_settings: Value,
    replacement_status: Option<Value>,
) -> Value {
    let request = &plan.request;
    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = if title.is_empty() {
        "Generated video".to_owned()
    } else {
        title.to_owned()
    };
    let timeline_context = request
        .advanced
        .get("timelineContext")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut fact = json!({
        "type": "video",
        "assetId": plan.asset_id,
        "mediaPath": plan.media_rel,
        "mimeType": "video/mp4",
        "width": request.width,
        "height": request.height,
        "duration": request.duration,
        "fps": request.fps,
        "quality": request.quality,
        "family": plan.family,
        "seed": seed,
        "displayName": display_name,
        "createdAt": plan.created_at,
        "mode": request.mode,
        "model": request.model,
        "adapter": adapter,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "loras": request.loras,
        "rawAdapterSettings": raw_settings,
        "sourceAssetId": request.source_asset_id,
        "lastFrameAssetId": request.last_frame_asset_id,
        "sourceClipAssetId": request.source_clip_asset_id,
        "bridgeRightClipAssetId": request.bridge_right_clip_asset_id,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "personTrackId": request.person_track_id,
        "replacementMode": request.replacement_mode,
        "timelineContext": timeline_context,
    });
    // replace_person reports its honest mask/track provenance (mirrors the torch
    // `video_generation_result` `replacementStatus` fold; sc-3521).
    if let (Some(status), Some(object)) = (replacement_status, fact.as_object_mut()) {
        object.insert("replacementStatus".to_owned(), status);
    }
    fact
}

fn stub_raw_settings(request: &VideoRequest) -> Value {
    json!({
        "model": request.model,
        "frameCount": request.frame_count(),
        "fps": request.fps,
        "duration": request.duration,
        "quality": request.quality,
        "stub": true,
    })
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet`
/// fact. A video job always reports exactly one asset (`expectedCount` 1).
fn streaming_result(plan: &VideoPlan, fact: &Value, adapter: &str) -> JsonObject {
    let request = &plan.request;
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": 1,
        "adapter": adapter,
        "model": request.model,
        "generationSet": {
            "id": plan.genset_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": 1,
            "createdAt": plan.created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Progress payload with the worker's real backend label (mirrors `image_progress`).
fn video_progress(
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

// ---------------------------------------------------------------------------
// SeedVR2 video upscale (epic 4811, sc-4816): the net-new `video_upscale` job —
// SceneWorks' first video upscaler. Decode the source clip -> native-MLX SeedVR2
// one-step super-resolution (temporal chunking + overlap is internal to the engine)
// -> encode + source-audio passthrough. macOS-only (no torch path). Reuses the shared
// encode pipeline (`encode_media`) + the streaming engine driver (`generate_video`).
// ---------------------------------------------------------------------------

/// HF repo hosting the raw SeedVR2 checkpoint (`numz/SeedVR2_comfyUI`); the engine converts it
/// in-memory at load (no Python). Override the staged dir with `SCENEWORKS_SEEDVR2_DIR`.
// SeedVR2 video upscale runs on Mac (native MLX) AND the Windows/CUDA candle lane (sc-5928); these
// constants/helpers are backend-neutral (gen_core + ffmpeg + the shared streaming driver).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_REPO: &str = "numz/SeedVR2_comfyUI";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_VAE_FILE: &str = "ema_vae_fp16.safetensors";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_DIT_3B_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";
/// The engine registry id wired for video upscale (3B; 7B = sc-5197 / sc-5927).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_ENGINE_ID: &str = "seedvr2_3b";
/// Adapter id recorded on the result asset for provenance (mirrors the other `mlx_*` video adapters;
/// SeedVR2 itself takes no LoRA — this is metadata only).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_ADAPTER: &str = "mlx_seedvr2";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_CANCEL_MESSAGE: &str = "Video upscale canceled by user.";

/// Snap a dimension to the SeedVR2 VAE/patch stride (a multiple of 16, the engine's hard
/// requirement), rounding to nearest and clamping to the engine's `[16, 4096]` size range.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn snap_seedvr2_dim(value: u32) -> u32 {
    let rounded = value.saturating_add(8) / 16 * 16;
    rounded.clamp(16, 4096)
}

/// Resolve a project-relative asset path safely under `project_path` (reject `..` / absolute
/// components — same guard as `upscale_jobs::resolve_source`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn safe_join(project_path: &Path, rel: &str) -> Option<PathBuf> {
    let mut path = project_path.to_path_buf();
    for component in Path::new(rel).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => return None,
        }
    }
    Some(path)
}

/// Provision the SeedVR2 checkpoint dir: an env-pinned dir (pre-staged for local validation) wins,
/// else the app cache (download the VAE + 3B DiT from `numz/SeedVR2_comfyUI` on first use). Returns
/// the dir to hand the engine as `WeightsSource::Dir`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_seedvr2_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let dir = std::env::var("SCENEWORKS_SEEDVR2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("seedvr2-mlx"));
    let client = reqwest::Client::new();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Video upscale canceled while fetching SeedVR2 weights.",
        fresh_download: false,
    };
    for file in [SEEDVR2_VAE_FILE, SEEDVR2_DIT_3B_FILE] {
        crate::downloads::ensure_hf_cached_file(
            &context,
            SEEDVR2_REPO,
            "main",
            file,
            &dir.join(file),
        )
        .await?;
    }
    Ok(dir)
}

/// Decode every frame of `source` to an engine [`Image`] sequence (native resolution — the engine
/// bicubic-upscales internally to the target). Uses the bundled ffmpeg (`run_ffmpeg`) into PNGs in a
/// temp dir, then loads them in order. `-fps_mode passthrough` keeps the exact source frame count.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn decode_seedvr2_source_frames(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    source: &Path,
) -> WorkerResult<Vec<Image>> {
    let frames_dir = std::env::temp_dir().join(format!("sceneworks_seedvr2_src_{job_id}"));
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    tokio::fs::create_dir_all(&frames_dir).await?;
    let ctx = FfmpegContext::new(api, settings, job_id, SEEDVR2_CANCEL_MESSAGE);
    let decode = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-fps_mode".to_owned(),
            "passthrough".to_owned(),
            frames_dir
                .join("in_%05d.png")
                .to_string_lossy()
                .into_owned(),
        ],
        Some(ctx),
    )
    .await;
    let loaded = match decode {
        Ok(()) => {
            let dir = frames_dir.clone();
            tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
                let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
                    .collect();
                paths.sort();
                let mut frames = Vec::with_capacity(paths.len());
                for path in paths {
                    let image = crate::image_decode::decode_image_any(&path)
                        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
                        .to_rgb8();
                    frames.push(rgb_image_to_engine(image));
                }
                Ok(frames)
            })
            .await
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
        }
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let frames = loaded?;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "source video produced no frames to upscale".to_owned(),
        ));
    }
    Ok(frames)
}

/// Dispatch handler for `JobType::VideoUpscale`: decode the source clip, run the SeedVR2 upscaler
/// (native MLX on Mac / candle CUDA on Windows, sc-5928), re-encode, pass the source audio through,
/// and stream a single upscaled video asset.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_video_upscale_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let req: sceneworks_core::contracts::VideoUpscaleRequest =
        serde_json::from_value(Value::Object(job.payload.clone())).map_err(|error| {
            WorkerError::InvalidPayload(format!("Invalid video_upscale payload: {error}"))
        })?;
    if req.source_asset_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Video upscale jobs require a source video asset.".to_owned(),
        ));
    }
    let engine = req.engine.trim().to_ascii_lowercase();
    if !matches!(engine.as_str(), "" | "seedvr2" | "seedvr2_3b") {
        return Err(WorkerError::InvalidPayload(format!(
            "Rust video upscaler supports only engine=seedvr2 (got {engine})."
        )));
    }
    let project_id = req
        .project_id
        .clone()
        .or_else(|| job.project_id.clone())
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload("Missing payload.projectId".to_owned()))?;
    let factor: u32 = if req.factor == 4 { 4 } else { 2 };
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Loading source video.",
            None,
            backend,
        ),
    )
    .await?;

    // Resolve the source video asset (on-disk path + fps + display name) from its sidecar.
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&project_id)?;
    let project_path = PathBuf::from(project.path);
    let asset = store
        .get_asset(&project_id, &req.source_asset_id)
        .map_err(|_| WorkerError::InvalidPayload("Source video asset not found.".to_owned()))?;
    let file = asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset has no media file.".to_owned()))?;
    let rel = file.get("path").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload("Source asset media path missing.".to_owned())
    })?;
    let source_path = safe_join(&project_path, rel)
        .filter(|path| path.exists())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Source media file is unavailable.".to_owned())
        })?;
    let source_fps = file
        .get("fps")
        .and_then(Value::as_f64)
        .map(|fps| fps.round() as u32)
        .filter(|fps| *fps > 0)
        .unwrap_or(24);
    let source_display = asset
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);

    check_cancel(api, &job.id, SEEDVR2_CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Fetching SeedVR2 weights.",
            None,
            backend,
        ),
    )
    .await?;
    let weights_dir = ensure_seedvr2_weights(api, settings, job).await?;

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.18,
            "Decoding source frames.",
            None,
            backend,
        ),
    )
    .await?;
    let source_frames = decode_seedvr2_source_frames(api, settings, &job.id, &source_path).await?;
    let src_w = source_frames[0].width;
    let src_h = source_frames[0].height;
    let frame_count = source_frames.len() as u32;
    let (target_w, target_h) = match (req.target_width, req.target_height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => (src_w.saturating_mul(factor), src_h.saturating_mul(factor)),
    };
    let target_w = snap_seedvr2_dim(target_w);
    let target_h = snap_seedvr2_dim(target_h);
    let seed = req.seed.unwrap_or(0);

    // Run the SeedVR2 upscale through the shared streaming driver (generator cache + stall
    // watchdog + cancel + Generating-stage progress). The VideoClip conditioning carries the LR
    // source frame sequence; `width`/`height` are the target output size (÷16).
    let input = VideoGenInput {
        engine_id: SEEDVR2_ENGINE_ID,
        model_dir: weights_dir,
        conditioning: vec![Conditioning::VideoClip {
            frames: source_frames,
            frame_idx: 0,
            strength: 1.0,
        }],
        width: target_w,
        height: target_h,
        frames: frame_count,
        fps: source_fps,
        seed,
        softness: Some(req.softness),
        ..Default::default()
    };
    let decoded = generate_video(api, settings, job, backend, input).await?;
    let out_fps = decoded.fps;
    let out_count = decoded.frames.len();
    let (out_w, out_h) = decoded
        .frames
        .first()
        .map(|frame| (frame.width, frame.height))
        .unwrap_or((target_w, target_h));
    let duration = out_count as f64 / out_fps.max(1) as f64;

    // Plan the output asset path (nested under the per-generation id, like VideoPlan).
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let media_rel = format!(
        "assets/videos/{genset_id}/{}_seedvr2_upscale.mp4",
        &created_at[..10]
    );
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.6,
            "Encoding upscaled video.",
            None,
            backend,
        ),
    )
    .await?;
    // Encode the upscaled frames to a (silent) mp4 + poster + faststart.
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    encode_media(&media_path, decoded, Some(ctx)).await?;

    // Source-audio passthrough: remux the source's audio onto the upscaled video. `-map 1:a:0?`
    // makes audio optional, so a source with no audio yields a clean video-only file (no error);
    // `-c:v copy` keeps the upscaled video stream untouched, `+faststart` is preserved.
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.85,
            "Muxing source audio.",
            None,
            backend,
        ),
    )
    .await?;
    let mux_tmp = media_path.with_extension("audiomux.mp4");
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            source_path.to_string_lossy().into_owned(),
            "-map".to_owned(),
            "0:v:0".to_owned(),
            "-map".to_owned(),
            "1:a:0?".to_owned(),
            "-c:v".to_owned(),
            "copy".to_owned(),
            "-c:a".to_owned(),
            "aac".to_owned(),
            "-movflags".to_owned(),
            "+faststart".to_owned(),
            "-shortest".to_owned(),
            mux_tmp.to_string_lossy().into_owned(),
        ],
        Some(ctx),
    )
    .await?;
    tokio::fs::rename(&mux_tmp, &media_path).await?;

    let display_name = req
        .display_name
        .clone()
        .unwrap_or_else(|| match &source_display {
            Some(name) => format!("{name} ({factor}x upscaled)"),
            None => format!("Upscaled video ({factor}x)"),
        });
    let raw_settings = json!({
        "engine": "seedvr2",
        "model": req.model,
        "factor": factor,
        "softness": req.softness,
        "sourceAssetId": req.source_asset_id,
        "sourceWidth": src_w,
        "sourceHeight": src_h,
        "targetWidth": out_w,
        "targetHeight": out_h,
        "frameCount": out_count,
    });
    let fact = json!({
        "type": "video",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "video/mp4",
        "width": out_w,
        "height": out_h,
        "duration": duration,
        "fps": out_fps,
        "quality": "best",
        "family": "video",
        "seed": seed as i64,
        "displayName": display_name,
        "createdAt": created_at,
        "mode": "video_upscale",
        "model": req.model,
        "adapter": SEEDVR2_ADAPTER,
        "prompt": "",
        "negativePrompt": Value::Null,
        "loras": [],
        "rawAdapterSettings": raw_settings,
        "sourceAssetId": req.source_asset_id,
        "timelineContext": json!({}),
    });
    let result = json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": SEEDVR2_ADAPTER,
        "model": req.model,
        "generationSet": {
            "id": genset_id,
            "mode": "video_upscale",
            "model": req.model,
            "prompt": "",
            "negativePrompt": Value::Null,
            "count": 1,
            "createdAt": created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal");

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Video upscale complete.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Real MLX Wan2.2 generation (macOS, via mlx-gen-wan, sc-3034): T2V/TI2V (5B
// dense, z48 VAE), T2V/I2V (A14B dual-expert MoE) + MoE/Lightning LoRA. Decodes
// the engine's `GenerationOutput::Video { frames, fps, audio: None }` into a
// `DecodedVideo` and reuses the [`encode_media`] pipeline above. LTX (sc-3035) and
// every other model keep the procedural stub.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Wan asset (mirrors the image `mlx_*` convention).
#[cfg(target_os = "macos")]
const WAN_ADAPTER: &str = "mlx_wan";

/// At most 3 user LoRAs per job (mirrors the image path's `MAX_JOB_LORAS`).
#[cfg(target_os = "macos")]
const MAX_JOB_LORAS: usize = 3;

/// Raw-settings recorded on a real MLX Wan asset: the request's `advanced` knobs plus
/// the real-inference markers (mirrors the image `mlx_raw_settings`). Also records the
/// effective sampler the worker actually dispatched (sc-4997) — the 5B interim default / the
/// 14B Lightning preset — so the chosen steps/CFG is inspectable on the asset, not silent.
#[cfg(target_os = "macos")]
fn wan_raw_settings(request: &VideoRequest, engine_id: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("frameCount".to_owned(), json!(request.frame_count()));
    raw.insert("fps".to_owned(), json!(request.fps));
    let (steps, guidance) = wan_sampling(engine_id, request);
    if let Some(steps) = steps {
        raw.insert("effectiveSteps".to_owned(), json!(steps));
    }
    if let Some(guidance) = guidance {
        raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
    }
    Value::Object(raw)
}

/// SceneWorks Wan model id → mlx-gen registry id, or `None` if `model` is not a Wan
/// family id this worker serves.
#[cfg(target_os = "macos")]
fn wan_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
        "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
        _ => None,
    }
}

/// Whether the linked Wan engine can serve this request now: a Wan model id with
/// resolvable on-disk weights. Off macOS / non-Wan / weights-absent → the stub
/// (mirrors the image `mlx_available` weights gate).
/// Fail-loud gate for the stub fallback (sc-4176): when the requested model id
/// maps to an MLX video engine family (Wan/LTX/SVD) but its weights/snapshot
/// can't be resolved, surface the resolver's precise re-download error instead
/// of silently degrading the job to procedural stub output. Non-engine model
/// ids pass through (the stub is their intended path).
#[cfg(target_os = "macos")]
pub(crate) fn ensure_video_engine_weights(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    if let Some(engine_id) = wan_engine_id(&request.model) {
        resolve_wan_model_dir(settings, &request.model, engine_id)?;
    }
    if ltx_engine_id(&request.model).is_some() {
        resolve_ltx_model_dir(settings, request)?;
    }
    if svd_engine_id(&request.model).is_some() {
        if request.source_asset_id.is_none() {
            return Err(WorkerError::InvalidPayload(
                "SVD image-to-video requires a source image asset.".to_owned(),
            ));
        }
        resolve_svd_model_dir(settings)?;
    }
    if bernini_engine_id(&request.model).is_some() {
        resolve_bernini_model_dir(settings)?;
    }
    if scail2_engine_id(&request.model).is_some() {
        resolve_scail2_model_dir(settings)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn wan_available(request: &VideoRequest, settings: &Settings) -> bool {
    match wan_engine_id(&request.model) {
        Some(engine_id) => resolve_wan_model_dir(settings, &request.model, engine_id).is_ok(),
        None => false,
    }
}

/// Resolve the converted MLX snapshot directory for a Wan model (mirrors the Python
/// `_resolve_wan_mlx`): an env override, then the app-managed `<data>/models/mlx/<id>`,
/// then (T2V-14B only) the turnkey HF MLX snapshot. Errors clearly if none is present.
#[cfg(target_os = "macos")]
fn resolve_wan_model_dir(
    settings: &Settings,
    model: &str,
    _engine_id: &str,
) -> WorkerResult<PathBuf> {
    let (env, local_id, hf_repo): (&str, &str, Option<&str>) = match model {
        "wan_2_2" => (
            "SCENEWORKS_MLX_WAN5B_DIR",
            "wan_2_2",
            Some("SceneWorks/wan2.2-ti2v-5b-mlx"),
        ),
        "wan_2_2_t2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_T2V_DIR",
            "wan_2_2_t2v_14b",
            Some("SceneWorks/wan2.2-t2v-a14b-mlx"),
        ),
        "wan_2_2_i2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_I2V_DIR",
            "wan_2_2_i2v_14b",
            Some("SceneWorks/wan2.2-i2v-a14b-mlx"),
        ),
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "not a Wan model: {other}"
            )))
        }
    };
    if let Some(dir) = local_mlx_dir(settings, env, local_id) {
        return Ok(dir);
    }
    if let Some(repo) = hf_repo {
        if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, repo) {
            return Ok(dir);
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{model}: no MLX weights found. Convert/download the Wan checkpoint into {}{}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join(local_id)
            .display(),
        hf_repo
            .map(|repo| format!(" (or download the turnkey repo {repo})"))
            .unwrap_or_default(),
    )))
}

/// A locally-converted MLX dir for the model (env override, then
/// `<data>/models/mlx/<id>`), counted only when it holds a `config.json` — mirrors the
/// Python `_local_mlx_dir`, so a locally-quantized conversion supersedes a turnkey download.
#[cfg(target_os = "macos")]
fn local_mlx_dir(settings: &Settings, env: &str, local_id: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(override_dir) = std::env::var(env) {
        let trimmed = override_dir.trim();
        if !trimmed.is_empty() {
            candidates.push(PathBuf::from(trimmed));
        }
    }
    candidates.push(settings.data_dir.join("models").join("mlx").join(local_id));
    candidates
        .into_iter()
        .find(|dir| dir.join("config.json").is_file())
}

/// The 4-step Lightning distill LoRA pair (high/low) for an A14B MoE model
/// (`lightx2v/Wan2.2-Lightning`, the rank-64 Seko distill). The subdir is architecture-specific:
/// T2V-A14B (V1.1) and I2V-A14B (V1) ship distinct LoRAs that are NOT cross-compatible (sc-4997).
/// Errors if not downloaded / the per-architecture subdir is missing.
#[cfg(target_os = "macos")]
fn resolve_lightning_loras(
    settings: &Settings,
    engine_id: &str,
) -> WorkerResult<(PathBuf, PathBuf)> {
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, "lightx2v/Wan2.2-Lightning")
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id}: the Lightning distill LoRA (lightx2v/Wan2.2-Lightning) is not \
                 downloaded — fetch it via the model manager"
            ))
        })?;
    let base = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "{other}: no Lightning distill LoRA — only the A14B MoE models bake Lightning"
            )))
        }
    };
    let high = snapshot.join(base).join("high_noise_model.safetensors");
    let low = snapshot.join(base).join("low_noise_model.safetensors");
    for file in [&high, &low] {
        if !file.is_file() {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id}: Lightning LoRA file missing: {}",
                file.display()
            )));
        }
    }
    Ok((high, low))
}

/// The `.low_noise.safetensors` sibling of a Wan A14B MoE high-noise LoRA file, or
/// `None` when the file is not the high-noise half of a pair (port of the Python
/// `wan_moe_low_noise_sibling`; case-insensitive `.high_noise.safetensors` suffix).
#[cfg(target_os = "macos")]
fn wan_moe_low_noise_sibling(primary: &Path) -> Option<PathBuf> {
    const HIGH: &str = ".high_noise.safetensors";
    let name = primary.file_name()?.to_str()?;
    if !name.to_ascii_lowercase().ends_with(HIGH) {
        return None;
    }
    let stem = &name[..name.len() - HIGH.len()];
    let sibling = primary.with_file_name(format!("{stem}.low_noise.safetensors"));
    sibling.is_file().then_some(sibling)
}

/// Resolve a LoRA spec's file (a directory → its first `.safetensors`), verifying it exists.
/// The `path` originates from attacker-controllable job payload, so it is first confined to an
/// app-managed root (sc-5723 / WKA-002) before any on-disk use.
#[cfg(target_os = "macos")]
fn resolve_lora_file(settings: &Settings, path: PathBuf) -> WorkerResult<PathBuf> {
    let path = crate::normalize_app_managed_lora_path(settings, &path)?;
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
    Ok(file)
}

/// A LoRA spec's strength (`weight`, default 0.8 — matches the image path).
#[cfg(target_os = "macos")]
fn lora_scale(lora: &Value) -> f32 {
    lora.get("weight")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(0.8) as f32
}

/// Build the adapter specs for a Wan generation (sc-3034): the Lightning distill pair
/// (both A14B MoE models — T2V + I2V — tagged high/low, sc-4997) followed by the user LoRAs.
/// On the MoE models a user
/// `*.high_noise.safetensors` with a `.low_noise` sibling tags high→High / low→Low; a
/// single-file LoRA is shared (both experts on MoE, the single model on the 5B). peft LoKr AND
/// third-party LyCORIS (LoHa / non-peft LoKr) both apply on the MLX Wan/LTX paths now (epic 3641,
/// sc-3671) — `classify_adapter` returns `Lora` for third-party and the engine detects + merges it.
#[cfg(target_os = "macos")]
fn resolve_wan_adapters(
    settings: &Settings,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let is_moe = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    let mut specs: Vec<AdapterSpec> = Vec::new();

    // Lightning distill (both A14B MoE models — T2V + I2V, sc-4997): 4-step, applied
    // per-expert at strength 1.0. The subdir is resolved per architecture (not cross-compatible).
    if is_moe {
        let (high, low) = resolve_lightning_loras(settings, engine_id)?;
        specs.push(moe_adapter(high, 1.0, AdapterKind::Lora, MoeExpert::High));
        specs.push(moe_adapter(low, 1.0, AdapterKind::Lora, MoeExpert::Low));
    }

    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(settings, path)?;
        let kind = classify_adapter(&file)?;
        let scale = lora_scale(lora);
        match (is_moe, wan_moe_low_noise_sibling(&file)) {
            (true, Some(low)) => {
                // A MoE pair → high half to the high-noise expert, the sibling to the low.
                let low_kind = classify_adapter(&low)?;
                specs.push(moe_adapter(file, scale, kind, MoeExpert::High));
                specs.push(moe_adapter(low, scale, low_kind, MoeExpert::Low));
            }
            _ => {
                // Single-file → shared (both experts on MoE; the dense single model on the 5B).
                specs.push(AdapterSpec {
                    path: file,
                    scale,
                    kind,
                    pass_scales: None,
                    moe_expert: None,
                });
            }
        }
    }
    Ok(specs)
}

#[cfg(target_os = "macos")]
fn moe_adapter(path: PathBuf, scale: f32, kind: AdapterKind, expert: MoeExpert) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind,
        pass_scales: None,
        moe_expert: Some(expert),
    }
}

/// Build the adapter specs for a Wan-VACE generation (sc-3893 worker routing). Unlike the base Wan
/// path, VACE-1.3B is a **single dense** transformer: no Lightning distill, no MoE high/low experts.
/// So every user LoRA/LoKr is applied shared with `moe_expert: None` — the engine `wan_vace` provider
/// merges diffusers-named LoRA/LoKr (mlx-gen #184) and rejects `moe_expert` tags. `classify_adapter`
/// tags SceneWorks peft LoKr as `Lokr` and everything else (incl. third-party LyCORIS LoHa / non-peft
/// LoKr) as `Lora`, which the engine then detects + merges by key sniff (epic 3641).
#[cfg(target_os = "macos")]
fn resolve_wan_vace_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs: Vec<AdapterSpec> = Vec::new();
    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(settings, path)?;
        let kind = classify_adapter(&file)?;
        specs.push(AdapterSpec {
            path: file,
            scale: lora_scale(lora),
            kind,
            pass_scales: None,
            moe_expert: None,
        });
    }
    Ok(specs)
}

/// Build the adapter specs for a SCAIL-2 generation (sc-5451 inference LoRA path, mlx-gen #462).
/// SCAIL-2 is a single **dense** Wan2.1-14B-I2V transformer — like Wan-VACE, no Lightning distill and
/// no MoE high/low experts — so every LoRA is applied shared with `moe_expert: None`. The engine
/// installs a standard `lora_down/up` (PEFT/diffusers/kohya/LoKr) adapter as a forward-time residual
/// over the (Q4/Q8) base; `classify_adapter` tags SceneWorks peft LoKr as `Lokr` and everything else
/// (incl. third-party LyCORIS) as `Lora`. This carries both a user-selected SCAIL-2 LoRA and the
/// bundled Bias-Aware DPO quality LoRA (both surface through `request.loras`). A lightx2v diff-patch
/// "lightning" LoRA installs via the engine's in-place diff-patch merge (sc-5684); selecting it makes
/// the worker apply the step-distill recipe (`scail2_sampling`, sc-5700).
#[cfg(target_os = "macos")]
fn resolve_scail2_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs: Vec<AdapterSpec> = Vec::new();
    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(settings, path)?;
        let kind = classify_adapter(&file)?;
        specs.push(AdapterSpec {
            path: file,
            scale: lora_scale(lora),
            kind,
            pass_scales: None,
            moe_expert: None,
        });
    }
    Ok(specs)
}

/// The first-frame conditioning for a Wan generation: required for I2V-14B, optional for
/// the TI2V-5B (present → image-conditioned mask-blend, absent → pure T2V), and ignored
/// by the T2V-14B (text-only). Loads `source_asset_id` to an in-memory RGB8 image.
#[cfg(target_os = "macos")]
fn resolve_wan_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    // first_last_frame is Wan-native only on the TI2V-5B mask-blend keyframe path (sc-3357);
    // the routing gate (`video_mode_is_mlx_eligible`) already restricts FLF to `wan_2_2`, but
    // guard here too so a mis-routed 14B MoE job fails clearly instead of silently dropping it.
    if request.mode == "first_last_frame" {
        if engine_id != "wan2_2_ti2v_5b" {
            return Err(WorkerError::InvalidPayload(format!(
                "first_last_frame is only supported on wan_2_2 (TI2V-5B), not {engine_id}."
            )));
        }
        return resolve_keyframe_conditioning(settings, request, project_path);
    }
    let required = engine_id == "wan2_2_i2v_14b";
    let accepts = required || engine_id == "wan2_2_ti2v_5b";
    if !accepts {
        return Ok(Vec::new());
    }
    match request.source_asset_id.as_deref() {
        Some(asset_id) => {
            let image = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?;
            // Pre-fit to the output W×H by the chosen crop/pad mode (sc-6139) — see
            // `resolve_ltx_conditioning`; without it the provider VAE-encodes a stretched
            // first frame into its channel-concat `y`.
            let image = crate::image_jobs::fit_engine_image(
                image,
                request.width,
                request.height,
                &request.fit_mode,
            )?;
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None if required => Err(WorkerError::InvalidPayload(
            "wan_2_2_i2v_14b: image-to-video requires a source image (sourceAssetId).".to_owned(),
        )),
        None => Ok(Vec::new()),
    }
}

/// Which boundary frame of a source clip to extract for Wan-native clip conditioning (sc-3357).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ClipFramePosition {
    /// The clip's first decoded frame (the right-side clip's head for `video_bridge`).
    First,
    /// The clip's last decoded frame (the source tail for `extend_clip` / the left-side clip).
    Last,
}

/// Build the Wan-native boundary [`Conditioning::Keyframe`] set for extend_clip / video_bridge
/// (sc-3357). Wan TI2V-5B has no in-context clip-append path (LTX's IC-LoRA `VideoClip`); its only
/// clip primitive is the single-frame mask-blend `Keyframe` (the same one Wan FLF rides). So the
/// faithful Wan-native form — matching the torch Wan reference, which routed these modes to plain
/// i2v (`_pipeline_kind` → `"image"`, never IC-LoRA/VACE) — pins the clip *boundary* frame(s):
/// - **extend_clip** → the source clip's last frame pinned at latent frame `0` (continue from it),
///   strength `videoConditioningStrength`.
/// - **video_bridge** → the left clip's last frame at `0` (`videoConditioningStrength`) + the right
///   clip's first frame at latent frame `-1` (the engine's negative-from-end index), strength
///   `bridgeRightVideoConditioningStrength`. Mechanically identical to first_last_frame.
///
/// Both strengths default to `1.0` (fully pinned), mirroring [`build_video_clip_conditioning`] and
/// the torch `_advanced_float` defaults. This is the single-frame fidelity ceiling for Wan; richer
/// motion-tail continuity is the LTX IC-LoRA path or native Wan-VACE (sc-3385 routing matrix).
#[cfg(target_os = "macos")]
fn build_wan_boundary_conditioning(
    request: &VideoRequest,
    left_frame: Image,
    right_frame: Option<Image>,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = vec![Conditioning::Keyframe {
        image: left_frame,
        frame_idx: 0,
        strength: advanced::f32(&request.advanced, "videoConditioningStrength", 1.0),
    }];
    if request.mode == "video_bridge" {
        let right = right_frame.ok_or_else(|| {
            WorkerError::InvalidPayload(
                "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                    .to_owned(),
            )
        })?;
        conditioning.push(Conditioning::Keyframe {
            image: right,
            frame_idx: -1,
            strength: advanced::f32(
                &request.advanced,
                "bridgeRightVideoConditioningStrength",
                1.0,
            ),
        });
    }
    Ok(conditioning)
}

/// Resolve extend_clip / video_bridge into Wan-native boundary [`Conditioning::Keyframe`]s
/// (sc-3357). Wan-native clip conditioning is **only** the TI2V-5B mask-blend keyframe path, so
/// guard the engine (the routing gate `video_mode_is_mlx_eligible` already restricts these to
/// `wan_2_2`, but fail clearly here too if a 14B MoE job is mis-routed). Extracts the boundary
/// frame(s) — the source clip's last frame (+ the right clip's first frame for bridge) — then maps
/// them via [`build_wan_boundary_conditioning`]. Unlike the LTX path this needs **no** IC-LoRA.
#[cfg(target_os = "macos")]
async fn resolve_wan_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    if engine_id != "wan2_2_ti2v_5b" {
        return Err(WorkerError::InvalidPayload(format!(
            "{} is only supported on wan_2_2 (TI2V-5B), not {engine_id}.",
            request.mode.replace('_', " ")
        )));
    }
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_frame = extract_clip_boundary_frame(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        ClipFramePosition::Last,
    )
    .await?;
    let right_frame = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            extract_clip_boundary_frame(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    build_wan_boundary_conditioning(request, left_frame, right_frame)
}

/// Decode a single boundary frame (first or last) of a source clip into an [`Image`], fit to the
/// output `width`×`height` by contain+pad (letterbox, `FRAME_PAD_COLOR`) so a clip whose aspect
/// differs from the output is not distorted — sc-6229, matching the `load_source_video_frames`
/// recipe (sc-3357, the Wan boundary-keyframe conditioning input). The last frame
/// uses ffmpeg `-sseof` to seek near the end + `-update 1` so each decoded frame overwrites the lone
/// output, leaving the final frame; the first frame is a plain `-frames:v 1`. Extracted via the
/// shared [`run_ffmpeg`] (binary resolution + heartbeat/cancel), then loaded off the async runtime.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn extract_clip_boundary_frame(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    position: ClipFramePosition,
) -> WorkerResult<Image> {
    let clip_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let frames_dir = project_path
        .join("assets")
        .join(".cond_clips")
        .join(Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&frames_dir).await?;
    let out = frames_dir.join("boundary.png");
    let mut args = vec!["ffmpeg".to_owned(), "-nostdin".to_owned(), "-y".to_owned()];
    if position == ClipFramePosition::Last {
        // Seek to ~2s before EOF; short clips clamp to the start (whole clip decoded). `-update 1`
        // overwrites the single output per frame, so the final decoded frame is what remains.
        args.push("-sseof".to_owned());
        args.push("-2".to_owned());
    }
    args.push("-i".to_owned());
    args.push(clip_path.display().to_string());
    args.push("-vf".to_owned());
    // Contain+pad (letterbox) to the output dims so a source clip whose aspect differs from the
    // requested W×H is not stretched (sc-6229); reuses the `FRAME_PAD_COLOR` recipe.
    args.push(format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24"
    ));
    if position == ClipFramePosition::Last {
        args.push("-update".to_owned());
        args.push("1".to_owned());
    } else {
        args.push("-frames:v".to_owned());
        args.push("1".to_owned());
    }
    args.push(out.display().to_string());
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let result = run_ffmpeg(args, Some(ctx)).await;
    let load = async {
        result?;
        let path = out.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<Image> {
            let decoded = crate::image_decode::decode_image_any(&path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "boundary conditioning frame {}: {error}",
                        path.display()
                    ))
                })?
                .to_rgb8();
            Ok(Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            })
        })
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
    };
    let frame = load.await;
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    frame
}

/// Map `advanced.mlxQuantize` to a quant level (≤0 → dense, ≤4 → Q4, else Q8). Absent →
/// `None`: dense bf16, or the engine builds it quantized from a pre-quantized snapshot.
#[cfg(target_os = "macos")]
fn resolve_wan_quant(request: &VideoRequest) -> Option<Quant> {
    let bits = request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })?;
    match bits {
        b if b <= 0 => None,
        b if b <= 4 => Some(Quant::Q4),
        _ => Some(Quant::Q8),
    }
}

/// Interim step count for the dense TI2V-5B until a 5B distill LoRA ships (sc-4999): half the
/// engine's 40-step default, so an out-of-the-box 1280×720 job no longer runs the ~40-min /
/// GPU-wedging 40-step+CFG schedule that wedged the GPU (sc-4986 / sc-4997). CFG is retained
/// (no 5B distill exists, so dropping it would hurt prompt adherence); the user can still dial
/// `steps`/`guidanceScale` lower from VideoStudio, and the engine pre-flight guard (sc-4986) is
/// the memory backstop. The full few-step / no-CFG preset lands once the 5B distill LoRA exists.
#[cfg(target_os = "macos")]
const WAN5B_INTERIM_STEPS: u32 = 20;

/// An optional positive-integer `advanced` knob (`steps`); accepts a number or a numeric string.
/// Shared by the MLX path and the candle video lane (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_opt_u32(request: &VideoRequest, key: &str) -> Option<u32> {
    request.advanced.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    })
}

/// An optional float `advanced` knob (`guidanceScale`); accepts a number or a numeric string.
/// Shared by the MLX path and the candle video lane (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_opt_f32(request: &VideoRequest, key: &str) -> Option<f32> {
    request.advanced.get(key).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    })
}

/// Per-model sampling for the base Wan path (sc-3034 / sc-4997). Both A14B MoE models (T2V + I2V)
/// bake the 4-step Lightning distill → forced 4 steps / CFG-off (guide 1.0); the distill is
/// mandatory, so a user `steps`/`guidanceScale` can't break it. The dense TI2V-5B has no distill
/// LoRA yet (sc-4999): honor an explicit user `steps`/`guidanceScale`, else apply the interim
/// default ([`WAN5B_INTERIM_STEPS`], CFG retained). `None` ⇒ the engine config default.
#[cfg(target_os = "macos")]
fn wan_sampling(engine_id: &str, request: &VideoRequest) -> (Option<u32>, Option<f32>) {
    if engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b" {
        return (Some(4), Some(1.0));
    }
    // wan2_2_ti2v_5b (dense): user override wins, else the interim default; CFG left to the
    // engine (guide 5.0) unless the user disables it via `guidanceScale`.
    let steps = advanced_opt_u32(request, "steps").or(Some(WAN5B_INTERIM_STEPS));
    let guidance = advanced_opt_f32(request, "guidanceScale");
    (steps, guidance)
}

/// The lightx2v lightning step-distill recipe (sc-5684 / sc-5700): 8 steps, CFG off, scheduler shift 1.
#[cfg(target_os = "macos")]
const SCAIL2_LIGHTNING_STEPS: u32 = 8;
#[cfg(target_os = "macos")]
const SCAIL2_LIGHTNING_GUIDANCE: f32 = 1.0;
#[cfg(target_os = "macos")]
const SCAIL2_LIGHTNING_SHIFT: f32 = 1.0;

/// SCAIL-2 sampling recipe `(steps, guidance, scheduler_shift)`. When a lightx2v diff-patch
/// "lightning" LoRA is selected (`lightning`), apply the step-distill recipe so the toggle yields the
/// ~10× fewer-DiT-passes speedup: CFG off (guidance 1.0 → the engine short-circuits to a single DiT
/// forward per step) and scheduler shift 1.0 are the lightning invariants (forced), and the step count
/// defaults to 8 but honors an explicit user `advanced.steps` override. Without a lightning LoRA, return
/// all-`None` so the engine's quality defaults (40 steps, guide 5.0, shift 5.0) stand exactly as before
/// — this path is unchanged. The chosen knobs are recorded as `effective*` in [`scail2_raw_settings`]
/// so what actually ran is inspectable on the asset (mirrors [`wan_raw_settings`]).
#[cfg(target_os = "macos")]
fn scail2_sampling(
    request: &VideoRequest,
    lightning: bool,
) -> (Option<u32>, Option<f32>, Option<f32>) {
    if !lightning {
        return (None, None, None);
    }
    (
        advanced_opt_u32(request, "steps").or(Some(SCAIL2_LIGHTNING_STEPS)),
        Some(SCAIL2_LIGHTNING_GUIDANCE),
        Some(SCAIL2_LIGHTNING_SHIFT),
    )
}

/// `true` if any resolved adapter is a lightx2v diff-patch ("lightning") LoRA — the engine's own
/// detector (a file carrying full-rank `.diff`/`.diff_b` tensors), so the recipe keys off the actual
/// format, not a catalog id or filename. A file that can't be read is treated as non-lightning (the
/// engine surfaces the real load error downstream).
#[cfg(target_os = "macos")]
fn scail2_adapters_have_lightning(adapters: &[AdapterSpec]) -> bool {
    adapters
        .iter()
        .any(|a| mlx_gen_scail2::has_diff_patch_keys(&a.path).unwrap_or(false))
}

/// The resolved inputs for one video generation (engine load + request build), shared by
/// Wan (sc-3034) and LTX (sc-3035) — split out so the engine call is unit-testable on real
/// weights without the API/job plumbing. The LTX-only knobs (`video_mode` no_audio,
/// prompt-enhance) default off for Wan; the Wan-only `moe_expert` rides on `adapters`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct VideoGenInput {
    engine_id: &'static str,
    model_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    conditioning: Vec<Conditioning>,
    prompt: String,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    steps: Option<u32>,
    guidance: Option<f32>,
    /// Flow-matching scheduler shift (`req.scheduler_shift`); `None` ⇒ the engine default. Set by the
    /// SCAIL-2 lightning recipe (shift 1.0, sc-5700); the other models leave it at the engine default.
    scheduler_shift: Option<f32>,
    seed: u64,
    /// Per-request control-clip conditioning scale (Wan-VACE `conditioning_scale`, sc-3441 /
    /// sc-3521); `None` ⇒ the engine default (1.0). Unused by the non-control paths.
    control_scale: Option<f32>,
    // LTX-only knobs (sc-3035); left at defaults by Wan + the other models.
    video_mode: Option<String>,
    enhance_prompt: bool,
    use_uncensored_enhancer: bool,
    enhance_max_tokens: Option<u32>,
    enhance_temperature: Option<f32>,
    // SVD-only micro-conditioning knobs (sc-3523); `None` on the other models.
    motion_bucket_id: Option<f32>,
    noise_aug_strength: Option<f32>,
    decode_chunk_size: Option<u32>,
    // SVD motion-conditioning fps, decoupled from the output `fps` (sc-3764); `None` elsewhere.
    conditioning_fps: Option<u32>,
    // SeedVR2 input pre-blur (sc-4816); `None` on the other models.
    softness: Option<f32>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Default for VideoGenInput {
    fn default() -> Self {
        Self {
            engine_id: "",
            model_dir: PathBuf::new(),
            quant: None,
            adapters: Vec::new(),
            conditioning: Vec::new(),
            prompt: String::new(),
            negative_prompt: None,
            width: 0,
            height: 0,
            frames: 0,
            fps: 0,
            steps: None,
            guidance: None,
            scheduler_shift: None,
            seed: 0,
            control_scale: None,
            video_mode: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_load_spec(input: &VideoGenInput) -> LoadSpec {
    LoadSpec {
        weights: WeightsSource::Dir(input.model_dir.clone()),
        quantize: input.quant,
        precision: Precision::Bf16,
        control: None,
        // MultiControlNet (sc-3378) is image-only; video providers ignore it.
        extra_controls: Vec::new(),
        ip_adapter: None,
        adapters: input.adapters.clone(),
    }
}

/// Run one generation to a [`DecodedVideo`] (RGB8 frames + fps + optional audio) against an already
/// loaded video generator, streaming denoise progress via `on_progress` and honoring `cancel`.
/// The engine fills the audio track (LTX) or leaves it `None` (Wan).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn run_loaded_video_generation(
    generator: &dyn Generator,
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let req = GenerationRequest {
        prompt: input.prompt,
        negative_prompt: input.negative_prompt,
        width: input.width,
        height: input.height,
        frames: Some(input.frames),
        fps: Some(input.fps),
        steps: input.steps,
        guidance: input.guidance,
        scheduler_shift: input.scheduler_shift,
        seed: Some(input.seed),
        conditioning: input.conditioning,
        control_scale: input.control_scale,
        video_mode: input.video_mode,
        enhance_prompt: input.enhance_prompt,
        use_uncensored_enhancer: input.use_uncensored_enhancer,
        enhance_max_tokens: input.enhance_max_tokens,
        enhance_temperature: input.enhance_temperature,
        motion_bucket_id: input.motion_bucket_id,
        noise_aug_strength: input.noise_aug_strength,
        decode_chunk_size: input.decode_chunk_size,
        conditioning_fps: input.conditioning_fps,
        softness: input.softness,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&req, on_progress)
        .map_err(|error| WorkerError::Engine(format!("video generation failed: {error}")))?;
    match output {
        GenerationOutput::Video { frames, fps, audio } => Ok(DecodedVideo {
            frames: frames
                .into_iter()
                .map(|image| RgbFrame {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels,
                })
                .collect(),
            fps,
            audio: audio.map(|track| AudioTrack {
                samples: track.samples,
                sample_rate: track.sample_rate,
                channels: track.channels,
            }),
        }),
        GenerationOutput::Images(_) => Err(WorkerError::Engine(
            "video model returned images, expected video frames".to_owned(),
        )),
    }
}

#[cfg(all(target_os = "macos", test))]
fn load_video_generation_for_tests(input: &VideoGenInput) -> WorkerResult<Box<dyn Generator>> {
    let spec = LoadSpec {
        weights: WeightsSource::Dir(input.model_dir.clone()),
        quantize: input.quant,
        precision: Precision::Bf16,
        control: None,
        // MultiControlNet (sc-3378) is image-only; video providers ignore it.
        extra_controls: Vec::new(),
        ip_adapter: None,
        adapters: input.adapters.clone(),
    };
    gen_core::load(input.engine_id, &spec)
        .map_err(|error| WorkerError::Engine(format!("video load failed: {error}")))
}

#[cfg(all(target_os = "macos", test))]
fn run_video_generation(
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let generator = load_video_generation_for_tests(&input)?;
    run_loaded_video_generation(generator.as_ref(), input, cancel, on_progress)
}

/// Forward-progress watchdog: if the engine emits no progress event (no denoise `Step`, no
/// `Decoding`) for this long — covering both the silent cold model-load phase and the gap
/// between steps — the generation is treated as wedged and the job is failed with a clear
/// error instead of heartbeating indefinitely. Tuned well above any legitimate single load or
/// step on the current video models; override via `SCENEWORKS_VIDEO_STALL_SECS` for an
/// unusually large/slow model or disk.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const VIDEO_STALL_TIMEOUT: Duration = Duration::from_secs(600);

/// Grace period granted after a stall is detected and engine cancellation is requested, before
/// the still-running blocking task is abandoned. A cooperative engine bails between steps well
/// within this window (the manual-cancel path proves it honors the flag); the abandon escape
/// only matters for a hard Metal wedge that never re-checks cancel, and keeps the watchdog from
/// itself re-hanging on the join.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const VIDEO_STALL_GRACE: Duration = Duration::from_secs(60);

/// The effective forward-progress stall timeout: `SCENEWORKS_VIDEO_STALL_SECS` (a positive
/// integer number of seconds) when set, else [`VIDEO_STALL_TIMEOUT`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_stall_timeout() -> Duration {
    parse_stall_timeout(std::env::var("SCENEWORKS_VIDEO_STALL_SECS").ok())
}

/// Parse the `SCENEWORKS_VIDEO_STALL_SECS` override (a positive integer number of seconds),
/// falling back to [`VIDEO_STALL_TIMEOUT`] when unset, blank, non-numeric, or zero.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn parse_stall_timeout(raw: Option<String>) -> Duration {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or(VIDEO_STALL_TIMEOUT)
}

/// First-detection handling for the in-loop video cancel poller (sc-5516): trip the engine
/// `CancelFlag` and post a NON-terminal "Cancelling…" update (indeterminate progress bar —
/// `running` + fraction 0.0 renders the "Working" animation, not a backward jump). The terminal
/// `Canceled` is posted only after the blocking generation actually stops (see `generate_video`),
/// so the worker row — and therefore the next queued job — is not freed until the GPU is genuinely
/// idle, and the UI honestly shows "Cancelling…" until completion. Best-effort: a failed status
/// update here is non-fatal because the post-run terminal write is what ultimately frees the
/// worker. Mirrors the image path's `begin_image_cancel` (sc-5515).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn begin_video_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.0,
            "Cancelling — finishing the current step…",
            None,
            backend,
        ),
    )
    .await;
}

/// Drive a `run_video_generation` on a blocking thread, forwarding its streamed denoise
/// progress to the async worker (Generating stage ~0.25..0.58) + polling cancel ~every 2s.
/// The shared blocking + mpsc + cancel plumbing for Wan and LTX. A forward-progress watchdog
/// ([`video_stall_timeout`]) fails a wedged job loudly rather than letting it look alive forever.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn generate_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    input: VideoGenInput,
) -> WorkerResult<DecodedVideo> {
    let cancel = CancelFlag::new();
    let stall_timeout = video_stall_timeout();
    let log_engine_id = input.engine_id;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Progress>(64);
    let blocking = {
        let cancel = cancel.clone();
        let spec = video_load_spec(&input);
        let engine_id = input.engine_id;
        tokio::spawn(async move {
            crate::generator_cache::with_cached_generator(
                engine_id,
                spec,
                "video load failed",
                move |generator| {
                    let mut on_progress = |progress: Progress| {
                        let _ = tx.blocking_send(progress);
                    };
                    run_loaded_video_generation(generator, input, &cancel, &mut on_progress)
                },
            )
            .await
        })
    };

    let mut canceled = false;
    // Set when the watchdog (not the user) tripped, so the job is failed with a stall error
    // rather than reported as a clean user cancellation.
    let mut stalled = false;
    // Once a stall is detected we request engine cancel and wait at most `VIDEO_STALL_GRACE`
    // for the blocking task to unwind; past this deadline we abandon it (a hard Metal wedge)
    // so the watchdog never re-hangs on the join.
    let mut abandon_deadline: Option<Instant> = None;
    let mut abandoned = false;
    let mut last_cancel = Instant::now();
    // Time of the most recent progress event; the forward-progress watchdog fails the job if
    // this goes stale for `stall_timeout` (covers both the silent load phase and step-to-step).
    let mut last_progress = Instant::now();
    // Interval arm so the cold model-load phase (gen_core::load emits no progress)
    // still heartbeats and polls cancel, instead of looking dead to the API's
    // staleness check until the first denoise step (sc-4276 / F-MLXW-12; mirrors
    // the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(crate::progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            maybe_progress = rx.recv() => {
                let Some(progress) = maybe_progress else {
                    break;
                };
                last_progress = Instant::now(); // forward progress — reset the stall watchdog.
                if canceled {
                    continue; // drain so the blocking sender never blocks.
                }
        if last_cancel.elapsed() >= Duration::from_secs(2) {
            last_cancel = Instant::now();
            if cancel_requested_peek(api, &job.id).await {
                begin_video_cancel(api, &job.id, &cancel, backend).await;
                canceled = true;
                continue;
            }
            heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
        }
        let (fraction, message) = match progress {
            Progress::Step { current, total } => (
                0.25 + 0.30 * (current as f64 / total.max(1) as f64),
                format!("Generating frames — step {current}/{total}."),
            ),
            Progress::Decoding => (0.58, "Decoding frames.".to_owned()),
        };
        update_job(
            api,
            &job.id,
            video_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                fraction,
                &message,
                None,
                backend,
            ),
        )
        .await?;
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                if !canceled && last_cancel.elapsed() >= Duration::from_secs(2) {
                    last_cancel = Instant::now();
                    if cancel_requested_peek(api, &job.id).await {
                        begin_video_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                    }
                }
                // Forward-progress watchdog: a wedged engine keeps this async loop heartbeating
                // (the block runs on a separate thread), so the API sees a healthy job forever.
                // If no progress has arrived for `stall_timeout`, request engine cancel and start
                // the abandon countdown.
                if !canceled && last_progress.elapsed() >= stall_timeout {
                    tracing::warn!(
                        event = "rust_worker_video_stalled",
                        jobId = %job.id,
                        engine = %log_engine_id,
                        stallSeconds = stall_timeout.as_secs(),
                        "no progress within the stall window — requesting engine cancel"
                    );
                    cancel.cancel();
                    canceled = true;
                    stalled = true;
                    abandon_deadline = Some(Instant::now() + VIDEO_STALL_GRACE);
                }
                if let Some(deadline) = abandon_deadline {
                    if Instant::now() >= deadline {
                        abandoned = true;
                        break;
                    }
                }
            }
        }
    }

    if abandoned {
        // The engine never honored the cancel flag within the grace window (a hard Metal wedge).
        // Detach the still-running blocking task instead of awaiting it — awaiting would re-hang
        // the very failure path this watchdog exists to break. The thread (and the GPU it holds)
        // leaks until the worker is restarted by the supervisor.
        tracing::error!(
            event = "rust_worker_video_abandoned",
            jobId = %job.id,
            engine = %log_engine_id,
            graceSeconds = VIDEO_STALL_GRACE.as_secs(),
            "engine did not respond to cancellation within the grace window — exiting the worker \
             so the supervisor can recover the wedged GPU task"
        );
        blocking.abort();
        std::process::exit(70);
    }
    let result = blocking
        .await
        .map_err(|error| task_join_error("video task join", error))?;
    if stalled {
        return Err(WorkerError::Engine(format!(
            "Video generation stalled: no progress for {}s. The job was canceled.",
            stall_timeout.as_secs()
        )));
    }
    if canceled {
        // Reached only on a genuine user cancel — the stall/abandon watchdog returns above.
        // Generation has actually stopped now, so post the TERMINAL Canceled here (not at the
        // earlier cancel poll, which only tripped the flag + showed "Cancelling…"). This terminal
        // write is what frees the worker row (`jobs_store::update_job_progress`), so it lands as
        // the worker returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free (sc-5516; mirrors the image path sc-5515).
        update_job(
            api,
            &job.id,
            video_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                CANCEL_MESSAGE,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    result
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) video lane (sc-5097, epic 5095). The candle wan/ltx providers serve a narrow
// **txt2video-only** first slice (no image/VACE conditioning, audio, LoRA, or quant). This is the
// video sibling of the candle image lane (image_jobs.rs `generate_candle_stream`): it builds a
// `VideoGenInput` and drives the SAME neutral streaming harness (`generate_video` →
// `run_loaded_video_generation` → the registry-resolved candle generator), reusing the shared
// encode/mux/poster path. Reached only when `backend_candle_enabled` (default off).
// ---------------------------------------------------------------------------

/// Per-asset adapter ids for the candle video engines (`candle_<family>`), the candle siblings of
/// the MLX `mlx_wan` / `mlx_ltx` labels (sc-5097).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_ADAPTER: &str = "candle_wan";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_ADAPTER: &str = "candle_ltx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SVD_ADAPTER: &str = "candle_svd";

/// Default HuggingFace repos the candle video providers load (overridable via the manifest `repo`).
/// The candle wan providers read a Wan2.2 diffusers snapshot — the TI2V-5B, or the T2V-A14B /
/// I2V-A14B 14B MoE (sc-5175); ltx reads the LTX-2.3 checkpoint plus a separate Gemma-3-12B encoder
/// snapshot (`LTX_GEMMA_DIR`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_5B_REPO: &str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_T2V_14B_REPO: &str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_I2V_14B_REPO: &str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_REPO: &str = "Lightricks/LTX-2.3";
// The `ltx_2_3_eros` weights repo (sc-5495): a full dense LTX-2.3 fine-tune (the candle provider
// loads its `10Eros_v1_bf16.safetensors` like the base; same architecture, same Gemma encoder).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_EROS_REPO: &str = "TenStrip/LTX2.3-10Eros";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_GEMMA_REPO: &str = "google/gemma-3-12b-it";

/// Per-asset adapter id for the candle Wan-VACE controllable-video lane (sc-5494) — the candle sibling
/// of the MLX `mlx_wan_vace` label.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_VACE_ADAPTER: &str = "candle_wan_vace";

/// The diffusers Wan2.1-VACE-14B snapshot the candle `wan_vace` provider reads (`transformer/` +
/// `text_encoder/` + `vae/` + `tokenizer/`). Overridable via `SCENEWORKS_CANDLE_WAN_VACE_DIR`. The 14B
/// repo matches the provider's `WanVaceConfig::vace_14b` dims (dim 5120, 40 layers).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_VACE_REPO: &str = "Wan-AI/Wan2.1-VACE-14B-diffusers";

/// SceneWorks video model id → candle registry engine id, or `None` for an id the candle video lane
/// does not serve. Note ltx maps to `ltx_2_3_distilled` (the candle provider's id), not the MLX
/// `ltx_2_3`. Covers the base txt2video ids (5B + ltx) plus the Wan2.2 **14B** dual-expert MoE pair
/// (sc-5174 / sc-5175): `wan_2_2_t2v_14b` (text→video) and `wan_2_2_i2v_14b` (image→video), plus `svd`
/// (image→video, sc-5493 / epic 5481). `ltx_2_3_eros` (sc-5495) maps to the same `ltx_2_3_distilled`
/// engine as the base — it's a full dense LTX-2.3 fine-tune — differing only in the weights repo.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
        "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
        // Base + eros both load the one candle LTX-2.3 engine (the eros merge is a full dense LTX-2.3
        // checkpoint, sc-5495); they differ only in the weights repo (see `candle_video_repo`).
        "ltx_2_3" | "ltx_2_3_eros" => Some("ltx_2_3_distilled"),
        // SVD-XT image→video (sc-5493 / epic 5481): the candle-gen-svd provider's `svd_xt` engine.
        "svd" => Some("svd_xt"),
        _ => None,
    }
}

/// Whether `model` is served by the candle video lane (sc-5097).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_candle_video_engine(model: &str) -> bool {
    candle_video_engine_id(model).is_some()
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_adapter_label(engine_id: &str) -> &'static str {
    match engine_id {
        "ltx_2_3_distilled" => CANDLE_LTX_ADAPTER,
        "svd_xt" => CANDLE_SVD_ADAPTER,
        _ => CANDLE_WAN_ADAPTER,
    }
}

/// The candle default weights repo for a video engine id (the per-variant Wan2.2 diffusers snapshot,
/// or the LTX-2.3 checkpoint). Used when the manifest entry omits `repo`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_default_repo(engine_id: &str) -> &'static str {
    match engine_id {
        "ltx_2_3_distilled" => CANDLE_LTX_REPO,
        "svd_xt" => SVD_REPO,
        "wan2_2_t2v_14b" => CANDLE_WAN_T2V_14B_REPO,
        "wan2_2_i2v_14b" => CANDLE_WAN_I2V_14B_REPO,
        // `wan2_2_ti2v_5b` (and any other wan id) → the 5B TI2V snapshot.
        _ => CANDLE_WAN_5B_REPO,
    }
}

/// The candle weights repo for a video engine: the manifest `repo` wins, else `ltx_2_3_eros` selects
/// its own fine-tune repo (it shares the `ltx_2_3_distilled` engine id with the base), else the candle
/// default repo for the engine.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_repo(request: &VideoRequest, engine_id: &str) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if request.model == "ltx_2_3_eros" {
                CANDLE_LTX_EROS_REPO.to_owned()
            } else {
                candle_video_default_repo(engine_id).to_owned()
            }
        })
}

/// Resolve the candle weights snapshot dir for `repo`. Errors loudly (no procedural-stub fallback)
/// when the snapshot is absent, so a missing model surfaces a re-download error.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_snapshot_dir(settings: &Settings, repo: &str) -> WorkerResult<PathBuf> {
    huggingface_snapshot_dir(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "candle video weights snapshot not found for {repo}"
        ))
    })
}

/// Point the candle LTX provider at the Gemma-3-12B encoder snapshot via `LTX_GEMMA_DIR` (the env var
/// the provider reads; it otherwise falls back to `<checkpoint>/text_encoder`). Best-effort: if the
/// Gemma snapshot isn't in the HF cache we leave `LTX_GEMMA_DIR` unset so the provider tries its
/// `<root>/text_encoder` fallback and emits its own clear "set LTX_GEMMA_DIR …" error. The worker runs
/// video jobs sequentially, so the process-global env set is race-free.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn ensure_ltx_gemma_dir(settings: &Settings) {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return; // honor an explicit operator override.
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, CANDLE_LTX_GEMMA_REPO) {
        std::env::set_var("LTX_GEMMA_DIR", dir);
    }
}

/// Raw-settings recorded on a candle video asset (mirrors `wan_raw_settings`, trimmed to the
/// txt2video surface): the request `advanced` knobs plus the real-inference markers.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_raw_settings(request: &VideoRequest, repo: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("frameCount".to_owned(), json!(request.frame_count()));
    raw.insert("fps".to_owned(), json!(request.fps));
    Value::Object(raw)
}

/// Per-request conditioning for a candle video generation. The Wan2.2 **14B I2V** engine
/// (sc-5174 / sc-5175) and **SVD-XT** (sc-5493) are conditioned: each requires a source image, loaded
/// to a single [`Conditioning::Reference`] — for Wan, the channel-concat first frame the provider
/// VAE-encodes into its `y` (`in_dim=36`); for SVD, the CLIP-encoded + noise-aug VAE-encoded driving
/// frame. The candle analog of the MLX i2v / SVD conditioning ([`resolve_wan_conditioning`]).
/// Every other candle video engine (5B, T2V-14B, ltx) is txt2video-only, so this returns an empty set.
/// The router's `video_request_candle_eligible` already guarantees the i2v shape carries a source and
/// the txt2video ids do not, but the source is required here too so a mis-routed job fails clearly.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_video_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    // The Wan2.2 14B I2V engine (sc-5175) and SVD-XT (sc-5493) condition on a single source image; every
    // other candle video engine (5B, T2V-14B, ltx) is txt2video-only (empty conditioning).
    if engine_id != "wan2_2_i2v_14b" && engine_id != "svd_xt" {
        return Ok(Vec::new());
    }
    let asset_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id}: image-to-video requires a source image (sourceAssetId)."
            ))
        })?;
    let image = crate::image_jobs::load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    // Pre-fit the source to the output W×H by the chosen crop/pad mode (sc-6139), the
    // same as the macOS LTX/Wan paths, so the Windows/CUDA candle I2V engine conditions
    // on an undistorted frame instead of an internal stretch.
    let image = crate::image_jobs::fit_engine_image(
        image,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    Ok(vec![Conditioning::Reference {
        image,
        strength: None,
    }])
}

/// Windows/CUDA candle video path (sc-5097 txt2video; sc-5175 adds the Wan2.2 14B MoE T2V + I2V).
/// Resolves the engine + weights, provisions the LTX Gemma encoder, resolves any i2v source-image
/// conditioning, builds a `VideoGenInput`, and runs it through the shared [`generate_video`] streaming
/// driver. Returns the decoded clip + the candle adapter label.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let engine_id = candle_video_engine_id(&request.model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{} is not a candle video engine", request.model))
    })?;
    let adapter = candle_video_adapter_label(engine_id);
    let repo = candle_video_repo(request, engine_id);
    let model_dir = candle_video_snapshot_dir(settings, &repo)?;
    // ltx needs the separate Gemma-3-12B encoder (its only conditioning input).
    let is_ltx = engine_id == "ltx_2_3_distilled";
    if is_ltx {
        ensure_ltx_gemma_dir(settings);
    }
    // Wan 14B I2V conditions on a source image (`Conditioning::Reference`); every other candle video
    // engine is txt2video-only (empty conditioning).
    let conditioning =
        resolve_candle_video_conditioning(settings, request, project_path, engine_id)?;

    // SVD-XT (sc-5493): image→video only — no prompt / negative / guidance (the engine uses its
    // frame-wise CFG ramp), a model-fixed burst (≤25 frames), the user `fps` as the playback cadence,
    // and the motion micro-conditioning knobs (motion_bucket_id / noise_aug_strength / decode_chunk /
    // conditioning_fps). Mirrors the MLX `generate_svd`; the conditioning is the source `Reference`
    // resolved above.
    if engine_id == "svd_xt" {
        let input = VideoGenInput {
            engine_id,
            model_dir,
            conditioning,
            width: request.width,
            height: request.height,
            frames: svd_i32(request, "numFrames", "numFrames", 25, 1, 25) as u32,
            fps: request.fps,
            steps: Some(svd_steps(request)),
            seed: resolve_video_seed(request) as u64,
            motion_bucket_id: Some(
                svd_i32(request, "motionBucketId", "motionBucketId", 127, 1, 255) as f32,
            ),
            noise_aug_strength: Some(svd_f32(
                request,
                "noiseAugStrength",
                "noiseAugStrength",
                0.02,
            )),
            decode_chunk_size: Some(
                svd_i32(request, "decodeChunkSize", "decodeChunkSize", 8, 1, 64) as u32,
            ),
            conditioning_fps: Some(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30) as u32),
            ..VideoGenInput::default()
        };
        let mut raw_settings = svd_raw_settings(request);
        if let Value::Object(map) = &mut raw_settings {
            map.insert("repo".to_owned(), Value::String(repo.clone()));
        }
        let decoded = generate_video(api, settings, job, backend, input).await?;
        return Ok((decoded, adapter, raw_settings));
    }

    // Descriptor-narrowed sampling surface: wan (5B + 14B) takes guidance + a negative prompt; the
    // distilled ltx takes neither (single-stage, no CFG). Steps/guidance default to the provider's own
    // constants when the request omits them.
    let steps = advanced_opt_u32(request, "steps");
    let (guidance, negative_prompt) = if is_ltx {
        (None, None)
    } else {
        let guidance = advanced_opt_f32(request, "guidanceScale");
        let trimmed = request.negative_prompt.trim();
        let negative = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        (guidance, negative)
    };
    // Coerce the requested frame count onto each engine's temporal stride (wan: ≡1 mod 4; ltx: 8k+1).
    let frames = if is_ltx {
        ltx_frame_count(request.raw_frame_count())
    } else {
        wan_frame_count(request.raw_frame_count())
    };
    let input = VideoGenInput {
        engine_id,
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        ..VideoGenInput::default()
    };
    let raw_settings = candle_video_raw_settings(request, &repo);
    let decoded = generate_video(api, settings, job, backend, input).await?;
    Ok((decoded, adapter, raw_settings))
}

/// Resolve the candle Wan-VACE diffusers snapshot dir (sc-5494): `SCENEWORKS_CANDLE_WAN_VACE_DIR`
/// override (when it holds a `transformer/config.json`), else the HF [`CANDLE_WAN_VACE_REPO`] snapshot.
/// Errors loudly when absent (no stub fallback — a missing VACE checkpoint surfaces a re-download error).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_wan_vace_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_CANDLE_WAN_VACE_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("transformer").join("config.json").is_file() {
            return Ok(path);
        }
    }
    candle_video_snapshot_dir(settings, CANDLE_WAN_VACE_REPO)
}

/// Windows/CUDA candle Wan-VACE `replace_person` (sc-5494): the candle sibling of the MLX
/// [`generate_wan_vace`]. Resolves the diffusers VACE snapshot, extracts the source-clip control frames,
/// builds the per-frame person mask from the saved track + the character references, and runs the
/// `wan_vace` engine. Person detect/track/segment stays upstream (the masks are pre-saved); the
/// conditioning builders are shared with the MLX path. No quant / LoRA (the candle VACE provider rejects
/// them). Returns the decoded clip + the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_wan_vace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_candle_wan_vace_model_dir(settings)?;
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let frames =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let (masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frames.len(),
    )?;
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let frame_total = frames.len();

    let masking_strength = advanced::f32(&request.advanced, "maskingStrength", 1.0);
    let conditioning = build_vace_conditioning(
        frames,
        masks,
        references,
        masking_strength,
        replacement_mode_from(&request.replacement_mode),
    )?;
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let input = VideoGenInput {
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, input).await?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        CANDLE_WAN_VACE_ADAPTER,
    );
    Ok((decoded, status))
}

/// Windows/CUDA candle Wan-VACE `extend_clip` / `video_bridge` (sc-5494): the candle sibling of the MLX
/// [`generate_wan_vace_extend_bridge`]. Loads the real source-clip anchor frames (the left clip's tail
/// for extend; both clips' boundaries for bridge), builds the source-at-kept-positions + generated-span
/// ControlClip, and runs the `wan_vace` engine. No reference images, no quant / LoRA.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_wan_vace_extend_bridge(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let model_dir = resolve_candle_wan_vace_model_dir(settings)?;
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let anchor = extend_anchor_frames(request, frame_count);
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_anchor = load_clip_anchor_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        anchor,
        ClipFramePosition::Last,
    )
    .await?;
    let right_anchor = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            load_clip_anchor_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                anchor,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    let conditioning = build_extend_bridge_vace_conditioning(
        request,
        request.width,
        request.height,
        frame_count,
        left_anchor,
        right_anchor,
    )?;
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let input = VideoGenInput {
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

/// Resolve a Wan request into a [`VideoGenInput`] and run it (sc-3034).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_wan(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let (steps, guidance) = wan_sampling(engine_id, request);
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    // extend_clip / video_bridge build single-frame boundary `Keyframe` conditioning from the
    // source clip(s) (async ffmpeg frame extraction, sc-3357); every other mode resolves
    // keyframe/reference conditioning synchronously from images.
    let conditioning = match request.mode.as_str() {
        "extend_clip" | "video_bridge" => {
            resolve_wan_clip_conditioning(api, settings, job, request, project_path, engine_id)
                .await?
        }
        _ => resolve_wan_conditioning(settings, request, project_path, engine_id)?,
    };
    let input = VideoGenInput {
        engine_id,
        model_dir: resolve_wan_model_dir(settings, &request.model, engine_id)?,
        quant: resolve_wan_quant(request),
        adapters: resolve_wan_adapters(settings, request, engine_id)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

// ---------------------------------------------------------------------------
// Real MLX Bernini generation (macOS, via mlx-gen-bernini, epic 4699 / sc-4707 + sc-4703 + sc-5425):
// the full Qwen2.5-VL semantic planner + Wan2.2-T2V-A14B dual-expert renderer. Serves the planner
// video task surface: text_to_video (t2v), video_to_video (v2v — source-clip edit),
// reference_to_video (r2v — subject references → video), reference_video_to_video (rv2v — source clip
// + references), multi_video_to_video (mv2v — multiple source clips), ads2v (source video + reference
// video + references). The SceneWorks mode maps to the engine `video_mode` task string and the source
// media is resolved into the planner's `VideoClip` / `MultiReference` conditioning. Q4 default / Q8
// opt-in at load. The turnkey `SceneWorks/bernini-mlx` snapshot is self-contained. (t2i/i2i image
// companion = a separate image-typed catalog id, tracked under epic 4699.)
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Bernini asset.
#[cfg(target_os = "macos")]
const BERNINI_ADAPTER: &str = "mlx_bernini";

/// SceneWorks Bernini model id → mlx-gen registry id, or `None` if `model` is not the Bernini family.
#[cfg(target_os = "macos")]
fn bernini_engine_id(model: &str) -> Option<&'static str> {
    (model == "bernini").then_some("bernini")
}

/// Whether the linked Bernini engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
fn bernini_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_bernini_model_dir(settings).is_ok()
}

/// Resolve the Bernini MLX snapshot dir: env override (`SCENEWORKS_MLX_BERNINI_DIR`) → app-managed
/// `<data>/models/mlx/bernini` → the turnkey download-on-first-use `SceneWorks/bernini-mlx` snapshot
/// (mirrors `resolve_wan_model_dir`). Errors clearly if none is present (no stub fallback).
#[cfg(target_os = "macos")]
pub(crate) fn resolve_bernini_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(settings, "SCENEWORKS_MLX_BERNINI_DIR", "bernini") {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, "SceneWorks/bernini-mlx") {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "bernini: no MLX weights found. Download the turnkey SceneWorks/bernini-mlx snapshot via the \
         Model Manager, set $SCENEWORKS_MLX_BERNINI_DIR, or place a converted snapshot at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("bernini")
            .display(),
    )))
}

/// MLX quantization for a Bernini load: Q4 default (the validated 128 GB-fitting tier, sc-4709 ~44 GB
/// peak), Q8 opt-in via the advanced `mlxQuantize: 8` control, explicit `<= 0` ⇒ bf16 (power users
/// with ample RAM). Never defaults to bf16 — the snapshot is ~93 GB at bf16.
#[cfg(target_os = "macos")]
fn resolve_bernini_quant(request: &VideoRequest) -> Option<Quant> {
    match request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }) {
        Some(bits) if bits <= 0 => None,
        Some(bits) if bits <= 4 => Some(Quant::Q4),
        Some(_) => Some(Quant::Q8),
        None => Some(Quant::Q4),
    }
}

/// Raw-settings recorded on a real MLX Bernini asset (mirrors `wan_raw_settings`).
#[cfg(target_os = "macos")]
fn bernini_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("frameCount".to_owned(), json!(request.frame_count()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine guidance task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "berniniTask".to_owned(),
        Value::String(bernini_engine_video_mode(&request.mode).to_owned()),
    );
    Value::Object(raw)
}

/// Map a SceneWorks video mode to the Bernini engine `video_mode` task string (which selects the
/// renderer guidance mode). The engine also infers the mode from the supplied conditioning, but the
/// explicit task keeps the mapping unambiguous. Unknown / `text_to_video` ⇒ plain `t2v`.
#[cfg(target_os = "macos")]
fn bernini_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "video_to_video" => "v2v",
        "reference_to_video" => "r2v",
        "reference_video_to_video" => "rv2v",
        // Multi-source-video modes (sc-5425): mv2v (multiple source clips) and ads2v
        // (source video + reference video + reference images). Both resolve to the
        // engine's `V2vApg` guidance via `task_to_vit_mode`; they differ only in the
        // supplied media.
        "multi_video_to_video" => "mv2v",
        "ads2v" => "ads2v",
        _ => "t2v",
    }
}

/// Resolve the source media for a Bernini editing/reference request into the planner conditioning:
/// source clips → [`Conditioning::VideoClip`] (the edit structure, VAE/ViT-encoded by the engine)
/// and subject reference images → [`Conditioning::MultiReference`]. `text_to_video` needs none.
/// The single-clip modes (v2v / rv2v / ads2v) use `sourceClipAssetId`; mv2v supplies several via
/// `sourceClipAssetIds`; ads2v additionally appends the reference video (`referenceClipAssetId`) as a
/// second clip after the source clip (sc-5425). Clips are emitted videos-first, in submission order,
/// then images — matching the engine's `collect_conditioning` / `assign_source_ids` ordering. Each
/// mode's required media is enforced here (defense in depth — the API validates the same), so a
/// mis-built request fails loudly instead of silently rendering an unconditioned clip. Every clip is
/// decoded to the output frame count (the engine resamples to its `target_fps` grid).
#[cfg(target_os = "macos")]
async fn resolve_bernini_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let mode = request.mode.as_str();

    // Source video clips, in the order the engine assigns source ids (videos first; for ads2v the
    // source clip leads the reference video).
    let mut clip_ids: Vec<&str> = Vec::new();
    match mode {
        "video_to_video" | "reference_video_to_video" | "ads2v" => {
            let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "bernini {} requires a source clip (sourceClipAssetId).",
                    request.mode.replace('_', " ")
                ))
            })?;
            clip_ids.push(clip_id);
        }
        "multi_video_to_video" => {
            if request.source_clip_asset_ids.len() < 2 {
                return Err(WorkerError::InvalidPayload(
                    "bernini multi video to video requires at least two source clips \
                     (sourceClipAssetIds)."
                        .to_owned(),
                ));
            }
            clip_ids.extend(request.source_clip_asset_ids.iter().map(String::as_str));
        }
        _ => {}
    }
    if mode == "ads2v" {
        let ref_clip_id = request.reference_clip_asset_id.as_deref().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "bernini ads2v requires a reference video (referenceClipAssetId).".to_owned(),
            )
        })?;
        clip_ids.push(ref_clip_id);
    }

    let mut conditioning = Vec::new();
    for clip_id in clip_ids {
        let frames = extract_clip_frames(
            api,
            settings,
            job,
            &request.project_id,
            project_path,
            clip_id,
            request.width,
            request.height,
            wan_frame_count(request.raw_frame_count()),
        )
        .await?;
        conditioning.push(Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        });
    }

    // Subject reference images → MultiReference (r2v / rv2v / ads2v).
    let needs_refs = matches!(
        mode,
        "reference_to_video" | "reference_video_to_video" | "ads2v"
    );
    if needs_refs {
        if request.reference_asset_ids.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "bernini {} requires at least one reference image (referenceAssetIds).",
                request.mode.replace('_', " ")
            )));
        }
        let mut images = Vec::with_capacity(request.reference_asset_ids.len());
        for asset_id in &request.reference_asset_ids {
            images.push(load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?);
        }
        conditioning.push(Conditioning::MultiReference { images });
    }

    Ok(conditioning)
}

/// Real MLX Bernini video generation (epic 4699 / sc-4707 + sc-4703 + sc-5425): build the
/// `VideoGenInput` and run the shared `generate_video` path. The SceneWorks mode resolves to the
/// engine `video_mode` task ([`bernini_engine_video_mode`]) and the source media into the planner
/// conditioning ([`resolve_bernini_conditioning`]) — empty for t2v, one or more `VideoClip`s for
/// v2v/mv2v/rv2v/ads2v, and `MultiReference` for r2v/rv2v/ads2v. No LoRA (the engine reports
/// `supports_lora=false`); steps/guidance stay at the engine defaults. Frame count uses the Wan
/// 1-mod-4 stride coercion (the renderer is Wan).
#[cfg(target_os = "macos")]
async fn generate_bernini(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let conditioning =
        resolve_bernini_conditioning(api, settings, job, request, project_path).await?;
    let input = VideoGenInput {
        engine_id,
        model_dir: resolve_bernini_model_dir(settings)?,
        quant: resolve_bernini_quant(request),
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(bernini_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

// ---------------------------------------------------------------------------
// Real MLX SCAIL-2 generation (macOS, via mlx-gen-scail2, epic 5439 / sc-5448): end-to-end character
// animation — a reference character image + a driving video → an animated clip of the character
// performing the driving motion. The worker paints the color-coded segmentation masks the engine
// needs from native SAM3 (no user masks): the reference image and the driving frames are each
// segmented (every person → a distinct palette color, left-to-right) and painted onto the
// whose-world-to-keep background (animation: driving bg black, ref bg white). Conditioning =
// `Reference` (the character) + `Mask` (its color mask) + `ControlClip{frames, mask}` (driving video +
// per-frame color masks). `replace_person` (cross-identity, replace_flag=true) is the same engine,
// wired in sc-5452; multi-character (paired ref+mask) awaits the engine request-contract extension.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX SCAIL-2 asset.
#[cfg(target_os = "macos")]
const SCAIL2_ADAPTER: &str = "mlx_scail2";

/// SceneWorks SCAIL-2 model id → mlx-gen registry id, or `None` if `model` is not SCAIL-2.
#[cfg(target_os = "macos")]
fn scail2_engine_id(model: &str) -> Option<&'static str> {
    (model == "scail2_14b").then_some("scail2_14b")
}

/// Whether the linked SCAIL-2 engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
fn scail2_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_scail2_model_dir(settings).is_ok()
}

/// Resolve the SCAIL-2 MLX snapshot dir: env override (`SCENEWORKS_MLX_SCAIL2_DIR`) → app-managed
/// `<data>/models/mlx/scail2` → the turnkey download-on-first-use `SceneWorks/scail2-mlx` snapshot
/// (mirrors `resolve_bernini_model_dir`). Errors clearly if none is present (no stub fallback).
#[cfg(target_os = "macos")]
fn resolve_scail2_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(settings, "SCENEWORKS_MLX_SCAIL2_DIR", "scail2") {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, "SceneWorks/scail2-mlx") {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "scail2: no MLX weights found. Download the turnkey SceneWorks/scail2-mlx snapshot via the \
         Model Manager, set $SCENEWORKS_MLX_SCAIL2_DIR, or place a converted snapshot at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("scail2")
            .display(),
    )))
}

/// MLX quantization for a SCAIL-2 load: Q4 default, Q8 opt-in via the advanced `mlxQuantize: 8`
/// control, explicit `<= 0` ⇒ bf16 (power users with ample RAM). Mirrors `resolve_bernini_quant`.
#[cfg(target_os = "macos")]
fn resolve_scail2_quant(request: &VideoRequest) -> Option<Quant> {
    match request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }) {
        Some(bits) if bits <= 0 => None,
        Some(bits) if bits <= 4 => Some(Quant::Q4),
        Some(_) => Some(Quant::Q8),
        None => Some(Quant::Q4),
    }
}

/// Raw-settings recorded on a real MLX SCAIL-2 asset (mirrors `bernini_raw_settings`). When the
/// lightx2v lightning LoRA is applied (`lightning`, sc-5700), records the effective step-distill recipe
/// the worker dispatched — so the chosen steps/CFG/shift is inspectable on the asset, not silent
/// (mirrors `wan_raw_settings`).
#[cfg(target_os = "macos")]
fn scail2_raw_settings(request: &VideoRequest, lightning: bool) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("frameCount".to_owned(), json!(request.frame_count()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "scail2Task".to_owned(),
        Value::String(scail2_engine_video_mode(&request.mode).to_owned()),
    );
    if lightning {
        let (steps, guidance, shift) = scail2_sampling(request, true);
        raw.insert("scail2Lightning".to_owned(), Value::Bool(true));
        if let Some(steps) = steps {
            raw.insert("effectiveSteps".to_owned(), json!(steps));
        }
        if let Some(guidance) = guidance {
            raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
        }
        if let Some(shift) = shift {
            raw.insert("effectiveSchedulerShift".to_owned(), json!(shift));
        }
    }
    Value::Object(raw)
}

/// Map a SceneWorks video mode to the SCAIL-2 engine `video_mode` task string. `replace_person`
/// (cross-identity, sc-5452) flips the engine `replace_flag`; everything else (`animate_character`)
/// is plain animation.
#[cfg(target_os = "macos")]
fn scail2_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "replace_person" => "replacement",
        _ => "animation",
    }
}

/// Resolve a SCAIL-2 request into the engine conditioning: load the reference character image and the
/// driving clip, segment both with native SAM3 (every person → a palette color), paint the color-coded
/// masks (animation background convention), and assemble `Reference` + `Mask` + `ControlClip`. The
/// segmentation + painting run on the blocking pool (GPU inference). The reference is
/// `referenceAssetIds[0]` (preferred) or `sourceAssetId`; the driving clip is `sourceClipAssetId`.
#[cfg(target_os = "macos")]
async fn resolve_scail2_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // The character: a reference image (referenceAssetIds first, else the i2v sourceAssetId).
    let ref_id = request
        .reference_asset_ids
        .first()
        .map(String::as_str)
        .or(request.source_asset_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "scail2 animate_character requires a reference character image (referenceAssetIds \
                 or sourceAssetId)."
                    .into(),
            )
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        ref_id,
        project_path,
    )?;

    // The driving video → frames at the output size (the engine re-resizes internally).
    let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "scail2 animate_character requires a driving video (sourceClipAssetId).".into(),
        )
    })?;
    let driving = extract_clip_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        clip_id,
        request.width,
        request.height,
        wan_frame_count(request.raw_frame_count()),
    )
    .await?;

    // SAM3 segmenter weights (download-on-first-use), shared by both segmentation passes.
    let client = reqwest::Client::new();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SCAIL-2 canceled while fetching the SAM3 segmenter weights.",
        fresh_download: false,
    };
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3::ensure_segmenter_weights(settings, &context).await?;

    // Decode the engine `Image`s to `RgbImage`s for SAM3 (it normalizes RGB internally).
    let ref_rgb =
        image::RgbImage::from_raw(reference.width, reference.height, reference.pixels.clone())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("scail2 reference image is malformed".into())
            })?;
    let driving_rgb: Vec<image::RgbImage> = driving
        .iter()
        .map(|f| image::RgbImage::from_raw(f.width, f.height, f.pixels.clone()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| WorkerError::InvalidPayload("scail2 driving frame is malformed".into()))?;

    // Segment + paint the reference mask (animation keeps the reference's world → white background).
    let (rm, rt) = (sam_model.clone(), sam_tokenizer.clone());
    let ref_mask = tokio::task::spawn_blocking(move || -> WorkerResult<Image> {
        let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
            &rm,
            &rt,
            std::slice::from_ref(&ref_rgb),
        )?;
        crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_WHITE)
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    // Segment + paint the per-frame driving masks (animation → black background).
    let driving_mask = tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
            &sam_model,
            &sam_tokenizer,
            &driving_rgb,
        )?;
        Ok(crate::scail2_masks::paint_driving_masks(
            &masks,
            crate::scail2_masks::BG_BLACK,
        ))
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    Ok(vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_mask,
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ])
}

/// Real MLX SCAIL-2 generation (epic 5439 / sc-5448): build the `VideoGenInput` and run the shared
/// `generate_video` path. The SceneWorks mode resolves to the engine `video_mode` task
/// ([`scail2_engine_video_mode`]) and the source media into the SAM3-painted conditioning
/// ([`resolve_scail2_conditioning`]). A user-selected SCAIL-2 LoRA and the bundled Bias-Aware DPO
/// quality LoRA install as forward-time residuals over the Q4 base ([`resolve_scail2_adapters`],
/// sc-5451 / mlx-gen #462); steps/guidance stay at the engine defaults. Frame count uses the Wan
/// 1-mod-4 stride coercion (the
/// renderer is Wan2.1); the engine stitches > 81-frame clips into overlapping segments internally.
#[cfg(target_os = "macos")]
async fn generate_scail2(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let conditioning =
        resolve_scail2_conditioning(api, settings, job, request, project_path).await?;
    // Selecting a lightx2v diff-patch "lightning" LoRA flips the worker to the step-distill recipe
    // (8 steps, CFG off, shift 1.0) so the toggle yields the speedup; otherwise steps/guidance/shift
    // stay `None` and the engine's quality defaults stand (sc-5700).
    let adapters = resolve_scail2_adapters(settings, request)?;
    let (steps, guidance, scheduler_shift) =
        scail2_sampling(request, scail2_adapters_have_lightning(&adapters));
    let input = VideoGenInput {
        engine_id,
        model_dir: resolve_scail2_model_dir(settings)?,
        quant: resolve_scail2_quant(request),
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        adapters,
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

/// Resolve a `replace_person` request into SCAIL-2 cross-identity replacement conditioning (sc-5452,
/// the **integrated** surface). Unlike the standalone `animate_character` path
/// ([`resolve_scail2_conditioning`], which segments a fresh driving clip), this reuses the masks
/// SceneWorks already computed: the saved person track (native YOLO11 → ByteTrack → SAM3,
/// corrections applied) supplies the per-frame driving masks, and the character's approved reference
/// image is the identity. Driving frames come from the source clip exactly as the Wan-VACE backend
/// loads them ([`load_source_video_frames`]), so the resampled track masks stay frame-aligned 1:1.
/// Replacement keeps the **driving** clip's world (driving mask bg white, reference mask bg black);
/// `video_mode = "replacement"` flips the engine `replace_flag`. SCAIL-2 is a full-character model —
/// it replaces the whole tracked person, so the face_only/full_person `replacementMode` knob and
/// `maskingStrength` are inert (the engine reads only the color masks). Multi-character (the extra
/// references) awaits the engine request-contract extension (sc-5583), so only the first reference
/// is used. Returns the conditioning plus the honest `replacementStatus` for the asset sidecar.
#[cfg(target_os = "macos")]
async fn resolve_scail2_replace_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, Value)> {
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Driving frames + their per-frame binary person masks — the same source the Wan-VACE backend
    // consumes, loaded identically so the resampled masks align 1:1 with the frames.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let driving =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let frame_total = driving.len();
    let (binary_masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frame_total,
    )?;
    // The tracked person → blue (person 0); replacement keeps the driving's world → white bg.
    let driving_masks = crate::scail2_masks::paint_track_driving_masks(
        &binary_masks,
        crate::scail2_masks::BG_WHITE,
    );

    // The character identity: the first approved reference image (multi-ref = sc-5583).
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let reference = references.into_iter().next().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        )
    })?;

    // The reference color mask: a fresh native-SAM3 pass on the reference image → the primary person
    // painted blue on a black background (replacement discards the reference's surrounding world).
    let client = reqwest::Client::new();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SCAIL-2 canceled while fetching the SAM3 segmenter weights.",
        fresh_download: false,
    };
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3::ensure_segmenter_weights(settings, &context).await?;
    let ref_rgb =
        image::RgbImage::from_raw(reference.width, reference.height, reference.pixels.clone())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("scail2 reference image is malformed".into())
            })?;
    let ref_mask = tokio::task::spawn_blocking(move || -> WorkerResult<Image> {
        let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
            &sam_model,
            &sam_tokenizer,
            std::slice::from_ref(&ref_rgb),
        )?;
        crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_BLACK)
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    let conditioning = vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_masks,
            // masking_strength / start_frame / mode are inert for SCAIL-2 (it reads only the color
            // masks); carried at neutral defaults for the shared ControlClip contract.
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ];
    // maskingStrength is recorded as 1.0 — SCAIL-2 always does a full-character replacement, so the
    // Wan-VACE partial-mask knob does not apply.
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        1.0,
        reference_count,
        frame_total,
        SCAIL2_ADAPTER,
    );
    Ok((conditioning, status))
}

/// Real MLX SCAIL-2 cross-identity replacement (epic 5439 / sc-5452): the integrated backend behind
/// the existing `replace_person` pipeline. Builds the replacement conditioning from the saved person
/// track + character reference ([`resolve_scail2_replace_conditioning`]) and runs the shared
/// `generate_video` path with `video_mode = "replacement"` (engine `replace_flag = true`). Returns
/// the decoded video plus the honest `replacementStatus` (adapter `mlx_scail2`). Mirrors
/// [`generate_wan_vace`]'s return shape so the dispatch folds the status identically.
#[cfg(target_os = "macos")]
async fn generate_scail2_replace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let (conditioning, status) =
        resolve_scail2_replace_conditioning(api, settings, job, request, project_path).await?;
    let input = VideoGenInput {
        engine_id,
        model_dir: resolve_scail2_model_dir(settings)?,
        quant: resolve_scail2_quant(request),
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, input).await?;
    Ok((decoded, status))
}

// ---------------------------------------------------------------------------
// Real MLX LTX-2.3 generation (macOS, via mlx-gen-ltx, sc-3035): T2V/I2V with
// SYNCHRONIZED AUDIO (the 2-stage distilled A/V pipeline; CFG forced 1.0). One
// engine model `ltx_2_3` serves both `ltx_2_3` + `ltx_2_3_eros` (the checkpoint dir
// selects quant via split_model.json). The Gemma-3 text encoder is resolved by the
// engine ($LTX_GEMMA_DIR / the HF cache). Audio rides the sc-3033 WAV→AAC mux path.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX LTX asset.
#[cfg(target_os = "macos")]
const LTX_ADAPTER: &str = "mlx_ltx";

/// SceneWorks LTX model id → mlx-gen registry id (one engine model serves both), or
/// `None` if not an LTX family id.
#[cfg(target_os = "macos")]
fn ltx_engine_id(model: &str) -> Option<&'static str> {
    matches!(model, "ltx_2_3" | "ltx_2_3_eros").then_some("ltx_2_3")
}

/// Whether the linked LTX engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
fn ltx_available(request: &VideoRequest, settings: &Settings) -> bool {
    ltx_engine_id(&request.model).is_some() && resolve_ltx_model_dir(settings, request).is_ok()
}

/// The turnkey SceneWorks LTX-2.3 MLX bundle (sc-5608, epic 5594; replaces the third-party
/// `notapalindrome/ltx23-mlx-av-q4` + `mlx-community/gemma-3-12b-it-bf16` mirrors). One repo with
/// the LTX `q4/` (default) + `q8/` (opt-in) checkpoint subdirs — each the full audio+I2V component
/// set — plus the bundled `gemma/` text encoder the engine reads via `$LTX_GEMMA_DIR`.
#[cfg(target_os = "macos")]
const LTX_BUNDLE_REPO: &str = "SceneWorks/ltx-2.3-mlx";

/// Whether `dir` is a converted LTX snapshot **complete for the current engine** — it must
/// carry the audio `vocoder` + I2V `vae_encoder` + single `upsampler`/`vae_decoder` the
/// engine `load()` reads. Older conversions (`spatial_/temporal_upscaler_*`, no vocoder)
/// fail this, so a stale local dir is skipped in favour of the turnkey snapshot.
#[cfg(target_os = "macos")]
fn ltx_dir_is_complete(dir: &Path) -> bool {
    [
        "connector.safetensors",
        "transformer.safetensors",
        "upsampler.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]
    .iter()
    .all(|file| dir.join(file).is_file())
}

/// Whether the request opts into the higher-quality Q8 LTX checkpoint (`advanced.mlxQuantize: 8`,
/// accepted as int or string). The default is Q4 (sc-5608).
#[cfg(target_os = "macos")]
fn ltx_wants_q8(request: &VideoRequest) -> bool {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|bits| bits >= 8)
        .unwrap_or(false)
}

/// Pick the engine-complete `q4/`/`q8/` checkpoint subdir of a SceneWorks LTX bundle `root`,
/// preferring the requested quant (sc-5608). Returns the first **complete** ([`ltx_dir_is_complete`])
/// subdir — so a partially-downloaded bundle falls through rather than half-loading — or `None`.
#[cfg(target_os = "macos")]
fn ltx_bundle_subdir(root: &Path, wants_q8: bool) -> Option<PathBuf> {
    let order: &[&str] = if wants_q8 {
        &["q8", "q4"]
    } else {
        &["q4", "q8"]
    };
    order
        .iter()
        .map(|sub| root.join(sub))
        .find(|dir| ltx_dir_is_complete(dir))
}

/// Resolve the converted LTX MLX snapshot dir. Env override (`SCENEWORKS_MLX_LTX_DIR` /
/// `…_EROS_DIR`) → `<data>/models/mlx/<candidate>` → (base only) the turnkey SceneWorks bundle
/// [`LTX_BUNDLE_REPO`], descending into its `q4/`/`q8/` subdir. Only a dir **complete for the
/// current engine** ([`ltx_dir_is_complete`]) counts, so a stale local conversion is skipped. For
/// the base model the Q4 checkpoint is the default (`mlxQuantize: 8` prefers the Q8 one); the engine
/// reads the actual bits from `split_model.json`, so this only picks *which* dir to load.
#[cfg(target_os = "macos")]
fn resolve_ltx_model_dir(settings: &Settings, request: &VideoRequest) -> WorkerResult<PathBuf> {
    let eros = request.model == "ltx_2_3_eros";
    let env = if eros {
        "SCENEWORKS_MLX_LTX_EROS_DIR"
    } else {
        "SCENEWORKS_MLX_LTX_DIR"
    };
    if let Ok(override_dir) = std::env::var(env) {
        let path = PathBuf::from(override_dir.trim());
        if ltx_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let wants_q8 = ltx_wants_q8(request);
    let candidates: &[&str] = if eros {
        &["ltx_2_3_eros"]
    } else if wants_q8 {
        &["ltx_2_3_base_q8", "ltx_2_3_base_q4", "ltx_2_3"]
    } else {
        &["ltx_2_3_base_q4", "ltx_2_3_base_q8", "ltx_2_3"]
    };
    for id in candidates {
        let dir = settings.data_dir.join("models").join("mlx").join(id);
        if ltx_dir_is_complete(&dir) {
            return Ok(dir);
        }
    }
    // Turnkey SceneWorks bundle for the base model (sc-5608): one repo with `q4/` + `q8/` LTX
    // subdirs (+ a bundled `gemma/` the engine reads via $LTX_GEMMA_DIR). Pick the quant subdir;
    // the engine reads the actual bits from split_model.json, so this only selects which to load.
    if !eros {
        if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) {
            if let Some(dir) = ltx_bundle_subdir(&root, wants_q8) {
                return Ok(dir);
            }
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}: no complete converted LTX MLX weights found under {} (expected one of {candidates:?} \
         with the audio vocoder + i2v vae_encoder; or the turnkey {LTX_BUNDLE_REPO} q4/ or q8/ \
         subdir; or set ${env})",
        request.model,
        settings.data_dir.join("models").join("mlx").display(),
    )))
}

/// The Gemma-3 text encoder bundled beside a resolved LTX dir, if present: `<parent>/gemma`
/// (sc-5608). The SceneWorks bundle ships it as a sibling of the `q4/`/`q8/` checkpoint dir; a
/// local/legacy conversion has none (→ the engine falls back to the HF-cache gemma snapshot).
#[cfg(target_os = "macos")]
fn bundled_ltx_gemma_dir(model_dir: &Path) -> Option<PathBuf> {
    let gemma = model_dir.parent()?.join("gemma");
    gemma.is_dir().then_some(gemma)
}

/// Point the engine at the Gemma-3 text encoder bundled beside the resolved LTX dir (sc-5608).
/// Setting `$LTX_GEMMA_DIR` (the var [`mlx-gen-ltx`'s `resolve_gemma_dir`] honors on every OS) makes
/// a fresh install self-contained — no separate `mlx-community/gemma` download. Best-effort +
/// non-destructive: honors an explicit operator `$LTX_GEMMA_DIR`, and skips when no bundled `gemma/`
/// sibling exists ([`bundled_ltx_gemma_dir`]). The worker runs video jobs sequentially, so the env
/// set is race-free.
#[cfg(target_os = "macos")]
fn ensure_bundled_ltx_gemma_dir(model_dir: &Path) {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return; // honor an explicit operator override.
    }
    if let Some(gemma) = bundled_ltx_gemma_dir(model_dir) {
        std::env::set_var("LTX_GEMMA_DIR", gemma);
    }
}

/// On-demand fetch of the bundle's `q8/` subdir (sc-5679). The macOS default download is lean
/// (`q4/` + `gemma/`); when a job opts into Q8 ([`ltx_wants_q8`]) and the bundle's `q8/` isn't already
/// complete, pull just `q8/*` from [`LTX_BUNDLE_REPO`] into the HF cache so [`resolve_ltx_model_dir`]
/// can load it. Base model only (eros has its own single-dir conversion). No-op when Q8 isn't
/// requested, the bundle snapshot isn't downloaded yet (resolve surfaces the clear "download the
/// bundle" error), or `q8/` is already present. Fails loud on a real download error — fast, before
/// any compute; a missing `hf` CLI leaves `q8/` absent so resolve gracefully falls back to Q4.
/// Mirrors the eros [`ensure_ltx_upscaler_cached`] on-demand fetch.
#[cfg(target_os = "macos")]
async fn ensure_ltx_q8_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model == "ltx_2_3_eros" || !ltx_wants_q8(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) else {
        return Ok(());
    };
    if ltx_dir_is_complete(&root.join("q8")) {
        return Ok(());
    }
    let scratch = settings
        .data_dir
        .join("cache")
        .join(format!(".ltx-q8-fetch-{}", job.id));
    tokio::fs::create_dir_all(&scratch).await?;
    let files = vec!["q8/*".to_owned()];
    let result = crate::model_jobs::download_model_with_hf_cli(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        "main",
        &files,
        &scratch,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    result.map(|_| ())
}

/// User LoRAs for an LTX generation (sc-3035): each at a uniform per-pass strength
/// (`pass_scales` left `None` → the engine applies `scale` on every distilled stage; a
/// per-stage schedule is parity-plus). No distill/Lightning prepend — the 2-stage distill
/// is baked into the checkpoint. peft LoKr allowed (engine residual), LyCORIS rejected.
#[cfg(target_os = "macos")]
fn resolve_ltx_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
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
        let file = resolve_lora_file(settings, path)?;
        let kind = classify_adapter(&file)?;
        specs.push(AdapterSpec::new(file, lora_scale(lora), kind));
    }
    Ok(specs)
}

/// Optional I2V conditioning for LTX: a `source_asset_id` → a single `Reference` image
/// (image→video); absent → pure text→video. `first_last_frame` → two `Keyframe`s (sc-3055).
/// (Audio is produced either way.)
#[cfg(target_os = "macos")]
fn resolve_ltx_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    if request.mode == "first_last_frame" {
        return resolve_keyframe_conditioning(settings, request, project_path);
    }
    match request.source_asset_id.as_deref() {
        Some(asset_id) => {
            let image = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?;
            // Pre-fit the starting image to the output W×H by the chosen crop/pad mode
            // (sc-6139) — without this the engine resizes it internally = stretch. Reuses
            // the image-edit lane's helper; a pre-fit-to-exact-dims reference is a no-op
            // for any further internal resize.
            let image = crate::image_jobs::fit_engine_image(
                image,
                request.width,
                request.height,
                &request.fit_mode,
            )?;
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None => Ok(Vec::new()),
    }
}

/// Read an `advanced` boolean flag (JSON bool), default `false` (Python `bool(.get(k))`).
/// First/last-frame conditioning (sc-3055 cutover): two [`Conditioning::Keyframe`]s — the source
/// image pinned at latent frame 0 and the last-frame image at latent frame `-1` (the engine's
/// Python-style negative-from-end index, so the worker needs no latent-frame math; the engine
/// bounds-checks it). Mirrors the torch `_ltx_conditioning_images` first_last_frame path: first @
/// `imageConditioningStrength`, last @ `lastFrameConditioningStrength` (both default 1.0 = fully
/// pinned). Shared by LTX (`ltx_2_3`) and Wan TI2V-5B (`wan_2_2`), the engines whose providers
/// advertise `Keyframe`. `imageFrameIndex` (default 0) is forwarded as the first keyframe's latent
/// index — for the universal FLF case (0) latent 0 == output 0.
#[cfg(target_os = "macos")]
fn resolve_keyframe_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let first_id = request.source_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "first_last_frame requires a source image (sourceAssetId).".to_owned(),
        )
    })?;
    let last_id = request.last_frame_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "first_last_frame requires a last-frame image (lastFrameAssetId).".to_owned(),
        )
    })?;
    let first = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        first_id,
        project_path,
    )?;
    let last = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        last_id,
        project_path,
    )?;
    // Fit both keyframes to the output W×H by the chosen crop/pad mode (sc-6139) so a
    // square first/last frame letterboxes (pad) or fills+trims (crop) into an off-aspect
    // clip instead of the engine stretching each internally.
    let first = crate::image_jobs::fit_engine_image(
        first,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    let last = crate::image_jobs::fit_engine_image(
        last,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    Ok(vec![
        Conditioning::Keyframe {
            image: first,
            frame_idx: advanced::i32(&request.advanced, "imageFrameIndex", 0),
            strength: advanced::f32(&request.advanced, "imageConditioningStrength", 1.0),
        },
        Conditioning::Keyframe {
            image: last,
            frame_idx: -1,
            strength: advanced::f32(&request.advanced, "lastFrameConditioningStrength", 1.0),
        },
    ])
}

/// Whether the job's LoRA set includes an IC-LoRA — the in-context conditioning adapter the
/// LTX extend/bridge keyframe-append path needs (without it the appended clip tokens are inert,
/// per the engine `apply_ltx_adapters` seam). Port of the torch `lora_looks_like_ic_lora`
/// (lora_adapters.py): an explicit `icLora`/`isIcLora` flag, a `conditioningRole: ic_lora`, or an
/// "ic-lora" / "ltx-2-3-ic-" marker anywhere in the id / name / path / file list. The IC-LoRA is a
/// user-installed LoRA flowing through `request.loras` (not an auto-provisioned fixed repo), so it
/// rides the existing [`resolve_ltx_adapters`] seam with no new adapter-loading code.
#[cfg(target_os = "macos")]
fn loras_contain_ic_lora(loras: &[Value]) -> bool {
    loras.iter().any(lora_looks_like_ic_lora)
}

#[cfg(target_os = "macos")]
fn lora_looks_like_ic_lora(lora: &Value) -> bool {
    let Some(obj) = lora.as_object() else {
        // A bare string lora id: sniff the string itself.
        return lora
            .as_str()
            .map(|id| ic_lora_marker(&id.to_lowercase().replace('_', "-")))
            .unwrap_or(false);
    };
    if obj.get("icLora") == Some(&Value::Bool(true))
        || obj.get("isIcLora") == Some(&Value::Bool(true))
    {
        return true;
    }
    if let Some(role) = obj.get("conditioningRole").and_then(Value::as_str) {
        if role.trim().to_lowercase().replace('-', "_") == "ic_lora" {
            return true;
        }
    }
    let source = obj.get("source").and_then(Value::as_object);
    // Gather every id/name/path/file string the torch heuristic inspects.
    let mut haystacks: Vec<String> = Vec::new();
    for key in [
        "id",
        "loraId",
        "name",
        "displayName",
        "installedPath",
        "sourcePath",
        "path",
    ] {
        if let Some(value) = obj.get(key).and_then(Value::as_str) {
            haystacks.push(value.to_owned());
        }
    }
    if let Some(source) = source {
        for key in ["repo", "file", "path"] {
            if let Some(value) = source.get(key).and_then(Value::as_str) {
                haystacks.push(value.to_owned());
            }
        }
    }
    // `files` (or `source.files`) may be a list or a single string.
    let files = source
        .and_then(|s| s.get("files"))
        .or_else(|| obj.get("files"));
    match files {
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(value) = item.as_str() {
                    haystacks.push(value.to_owned());
                }
            }
        }
        Some(Value::String(value)) => haystacks.push(value.clone()),
        _ => {}
    }
    let text = haystacks.join(" ").to_lowercase().replace('_', "-");
    ic_lora_marker(&text)
}

/// The torch `lora_looks_like_ic_lora` text test (already `_`→`-` normalised + lowercased).
#[cfg(target_os = "macos")]
fn ic_lora_marker(text: &str) -> bool {
    text.contains("ic-lora") || text.contains("ltx-2-3-ic-")
}

/// Build the in-context [`Conditioning::VideoClip`] set for extend_clip / video_bridge (sc-3522).
/// Source-of-truth = torch `_ltx_video_conditioning` (video_adapters.py) + the engine consumer
/// `mlx_gen_ltx::build_clips`: each source clip's frames are appended as IC-LoRA in-context tokens
/// at an output **latent** frame index, with a `1 − strength` denoise mask.
/// - **extend_clip** → one clip pinned at latent frame `0`, strength `videoConditioningStrength`.
/// - **video_bridge** → a left clip at `0` (strength `videoConditioningStrength`) + a right clip at
///   latent frame `-1` (the engine's negative-from-end index, `lf + idx`, so the worker needs no
///   latent-frame math), strength `bridgeRightVideoConditioningStrength`.
///
/// Both strengths default to `1.0` (fully pinned), mirroring the torch `_advanced_float` defaults.
#[cfg(target_os = "macos")]
fn build_video_clip_conditioning(
    request: &VideoRequest,
    left_frames: Vec<Image>,
    right_frames: Option<Vec<Image>>,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = vec![Conditioning::VideoClip {
        frames: left_frames,
        frame_idx: 0,
        strength: advanced::f32(&request.advanced, "videoConditioningStrength", 1.0),
    }];
    if request.mode == "video_bridge" {
        let right = right_frames.ok_or_else(|| {
            WorkerError::InvalidPayload(
                "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                    .to_owned(),
            )
        })?;
        conditioning.push(Conditioning::VideoClip {
            frames: right,
            frame_idx: -1,
            strength: advanced::f32(
                &request.advanced,
                "bridgeRightVideoConditioningStrength",
                1.0,
            ),
        });
    }
    Ok(conditioning)
}

/// Resolve an asset id to its on-disk media file path (the source clip mp4), mirroring the asset
/// lookup in [`load_reference_image`] but returning the path for ffmpeg frame extraction (the
/// Rust equivalent of the torch `source_asset_media_path`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_clip_media_path(
    settings: &Settings,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<PathBuf> {
    let asset = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("source clip asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("source clip asset {asset_id} has no media path"))
        })?;
    // file.path is sidecar-sourced (user-editable on disk), so guard it through
    // safe_project_path instead of a bare join so a poisoned sidecar can't escape
    // the project to read an arbitrary file as the source clip (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    if !path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip file is missing for asset {asset_id}: {}",
            path.display()
        )));
    }
    Ok(path)
}

/// Decode the first `count` frames of a source clip into [`Image`]s for in-context conditioning.
/// Mirrors the torch reference `decode_video_by_frame(starting_frame=0, frame_cap=num_frames)` /
/// `video_preprocess` (ltx_pipelines): **sequential** frames from the start at the clip's native
/// cadence (no fps resample), fit to the output `width`×`height` by contain+pad (letterbox,
/// `FRAME_PAD_COLOR`) so a clip whose aspect differs from the output is not distorted (sc-6229;
/// the engine `build_clips` LANCZOS-downsizes each frame to stage-1 half-res, so this only bounds
/// memory). `count` is the
/// generation's snapped frame count (`8k+1`); a clip shorter than `count` yields fewer frames,
/// which the engine VAE encode accepts. Extracted via the shared [`run_ffmpeg`] (binary
/// resolution + heartbeat/cancel), then loaded off the async runtime.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn extract_clip_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    count: u32,
) -> WorkerResult<Vec<Image>> {
    let clip_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let frames_dir = project_path
        .join("assets")
        .join(".cond_clips")
        .join(Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&frames_dir).await?;
    let pattern = frames_dir.join("frame_%05d.png");
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let result = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            clip_path.display().to_string(),
            // Contain+pad (letterbox) to the output dims so a clip whose aspect differs from the
            // requested W×H is not stretched (sc-6229); reuses the `FRAME_PAD_COLOR` recipe. The
            // engine re-resizes to stage-1 half-res, so this only bounds the extracted footprint.
            "-vf".to_owned(),
            format!(
                "scale={width}:{height}:force_original_aspect_ratio=decrease,\
                 pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24"
            ),
            // First `count` decoded frames, sequential from the start at native cadence.
            "-frames:v".to_owned(),
            count.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    // Load the extracted PNGs (sorted by frame index) into `Image`s, off the async runtime.
    let load = async {
        result?;
        let dir = frames_dir.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
                .collect();
            paths.sort();
            let mut frames = Vec::with_capacity(paths.len());
            for path in paths {
                let decoded = crate::image_decode::decode_image_any(&path)
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "conditioning frame {}: {error}",
                            path.display()
                        ))
                    })?
                    .to_rgb8();
                frames.push(Image {
                    width: decoded.width(),
                    height: decoded.height(),
                    pixels: decoded.into_raw(),
                });
            }
            Ok(frames)
        })
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
    };
    let frames = load.await;
    // Best-effort cleanup of the scratch frame dir regardless of outcome.
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let frames = frames?;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip {asset_id} produced no decodable frames for conditioning"
        )));
    }
    Ok(frames)
}

/// Resolve extend_clip / video_bridge into the in-context [`Conditioning::VideoClip`] set (sc-3522).
/// Requires an installed IC-LoRA (the keyframe-append adapter) — mirrors the torch gate
/// (`_uses_ic_lora_pipeline` + the "requires at least one installed LTX-compatible LoRA" error),
/// since without it the appended clip tokens are inert. Then decodes each source clip's first
/// `num_frames` frames and builds the clips. `num_frames` is the generation's snapped frame count,
/// the same value [`generate_ltx`] passes to the engine.
#[cfg(target_os = "macos")]
async fn resolve_video_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    if !loras_contain_ic_lora(&request.loras) {
        return Err(WorkerError::InvalidPayload(format!(
            "{} requires an installed IC-LoRA (in-context conditioning adapter) — add an \
             LTX IC-LoRA to the selected preset; without it the source-clip conditioning is inert.",
            request.mode.replace('_', " ")
        )));
    }
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let num_frames = ltx_frame_count(request.raw_frame_count());
    let left_frames = extract_clip_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        num_frames,
    )
    .await?;
    let right_frames = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            extract_clip_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                num_frames,
            )
            .await?,
        )
    } else {
        None
    };
    build_video_clip_conditioning(request, left_frames, right_frames)
}

/// Raw-settings recorded on a real MLX LTX asset (`advanced` knobs + real-inference markers).
#[cfg(target_os = "macos")]
fn ltx_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("frameCount".to_owned(), json!(request.frame_count()));
    raw.insert("fps".to_owned(), json!(request.fps));
    Value::Object(raw)
}

/// Resolve an LTX request into a [`VideoGenInput`] and run it (sc-3035). Distilled 2-stage
/// → no negative prompt / guidance (CFG 1.0); quant is checkpoint-driven (`None`); frames
/// snap to `8k+1`; `advanced.noAudio` → `video_mode = "no_audio"` (full A/V denoise, audio
/// decode skipped); prompt-enhance + per-pass LoRA flow through.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_ltx(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let video_mode = advanced::bool(&request.advanced, "noAudio").then(|| "no_audio".to_owned());
    let enhance_max_tokens = request
        .advanced
        .get("enhanceMaxTokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let enhance_temperature = request
        .advanced
        .get("enhanceTemperature")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    // extend_clip / video_bridge build in-context VideoClip conditioning from decoded source
    // clips (async ffmpeg extraction); every other mode resolves keyframe/reference conditioning
    // synchronously from images.
    let conditioning = match request.mode.as_str() {
        "extend_clip" | "video_bridge" => {
            resolve_video_clip_conditioning(api, settings, job, request, project_path).await?
        }
        _ => resolve_ltx_conditioning(settings, request, project_path)?,
    };
    // The macOS default download is lean (q4 + gemma); a Q8 job fetches the bundle's q8/ on demand
    // before resolving (sc-5679). No-op unless Q8 is requested and q8/ is absent.
    ensure_ltx_q8_present(api, settings, job, request).await?;
    let model_dir = resolve_ltx_model_dir(settings, request)?;
    // When the resolved dir is the SceneWorks bundle subdir, its sibling `gemma/` is the text
    // encoder — point the engine at it (sc-5608) before load. No-op for legacy/local conversions.
    ensure_bundled_ltx_gemma_dir(&model_dir);
    let input = VideoGenInput {
        engine_id,
        model_dir,
        quant: None,
        adapters: resolve_ltx_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: ltx_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps: None,
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        control_scale: None,
        video_mode,
        enhance_prompt: advanced::bool(&request.advanced, "enhancePrompt"),
        use_uncensored_enhancer: advanced::bool(&request.advanced, "useUncensoredEnhancer"),
        enhance_max_tokens,
        enhance_temperature,
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

// ---------------------------------------------------------------------------
// Real MLX Stable Video Diffusion (SVD-XT) generation (macOS, via mlx-gen-svd, sc-3523):
// image→video ONLY — animates one source image into a fixed ~25-frame burst (no text prompt,
// no audio) via the `motion_bucket_id` / `noise_aug_strength` / conditioning-fps
// micro-conditioning. One engine model `svd_xt`. Source-of-truth = the torch
// `DiffusersVideoAdapter` `svd_video` path (`StableVideoDiffusionPipeline`, video_adapters.py).
// The engine loads the stock diffusers fp16 snapshot directly (vae/ + unet/ + image_encoder/),
// so there is no conversion step (unlike Wan/LTX).
//
// fps (sc-3764): the engine decouples the two cadences — the motion micro-conditioning fps
// (`added_time_ids` = fps − 1) rides `conditioning_fps` (manifest `condFps`, default 7 — the value
// the model was trained on, so MOTION stays correct), while the output/playback fps is the user's
// `request.fps` (mirroring the torch `export_to_video(fps=request.fps)`). So the burst now plays at
// the requested cadence with correct motion — full parity with the torch `svd_video` path.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX SVD asset — matches the torch `svd_video` adapter id so the
/// asset sidecar reads identically across the two backends.
#[cfg(target_os = "macos")]
const SVD_ADAPTER: &str = "svd_video";

/// The diffusers SVD-XT repo the engine loads directly (fp16 `vae/` + `unet/` + `image_encoder/`).
/// Shared by the MLX (macOS) lane and the candle (Windows/CUDA) lane (sc-5493).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SVD_REPO: &str = "stabilityai/stable-video-diffusion-img2vid-xt";

/// SceneWorks model id → mlx-gen registry id for the SVD family (only `svd` → `svd_xt`), or `None`.
#[cfg(target_os = "macos")]
fn svd_engine_id(model: &str) -> Option<&'static str> {
    (model == "svd").then_some("svd_xt")
}

/// Whether the linked SVD engine can serve this request now (image→video with resolvable weights).
/// SVD is image-conditioned only, so a request without a `sourceAssetId` can never run on it.
#[cfg(target_os = "macos")]
fn svd_available(request: &VideoRequest, settings: &Settings) -> bool {
    svd_engine_id(&request.model).is_some()
        && request.source_asset_id.is_some()
        && resolve_svd_model_dir(settings).is_ok()
}

/// Whether `dir` is a usable SVD-XT snapshot — each component subdir carries the safetensors the
/// engine reads (preferring the on-disk `.fp16` variant, else the full-precision file).
#[cfg(target_os = "macos")]
fn svd_dir_is_complete(dir: &Path) -> bool {
    let has = |sub: &str, stem: &str| {
        dir.join(sub)
            .join(format!("{stem}.fp16.safetensors"))
            .is_file()
            || dir.join(sub).join(format!("{stem}.safetensors")).is_file()
    };
    has("vae", "diffusion_pytorch_model")
        && has("unet", "diffusion_pytorch_model")
        && has("image_encoder", "model")
}

/// Resolve the SVD-XT snapshot dir: env override (`SCENEWORKS_MLX_SVD_DIR`) → the cached HF snapshot
/// of [`SVD_REPO`]. Only a dir carrying the three component subdirs ([`svd_dir_is_complete`]) counts.
#[cfg(target_os = "macos")]
fn resolve_svd_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_SVD_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if svd_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, SVD_REPO) {
        if svd_dir_is_complete(&dir) {
            return Ok(dir);
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "svd: no complete SVD-XT weights found (expected vae/ + unet/ + image_encoder/ under the \
         cached {SVD_REPO} snapshot, or set $SCENEWORKS_MLX_SVD_DIR)"
    )))
}

/// Read an SVD integer knob: `advanced[adv_key]` → `modelManifestEntry[manifest_key]` → `default`,
/// then clamp to `[min, max]`. Mirrors the torch `safe_int(advanced.get(adv_key),
/// target.get(manifest_key, default), min, max)` (advanced overrides the manifest, which overrides
/// the builtin default; the resolved value is clamped).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_i32(
    request: &VideoRequest,
    adv_key: &str,
    manifest_key: &str,
    default: i32,
    min: i32,
    max: i32,
) -> i32 {
    request
        .advanced
        .get(adv_key)
        .or_else(|| request.model_manifest_entry.get(manifest_key))
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as i32)
        .unwrap_or(default)
        .clamp(min, max)
}

/// Read an SVD float knob: `advanced[adv_key]` → `modelManifestEntry[manifest_key]` → `default`
/// (no clamp). Mirrors the torch `float(advanced.get(adv_key, target.get(manifest_key, default)))`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_f32(request: &VideoRequest, adv_key: &str, manifest_key: &str, default: f32) -> f32 {
    request
        .advanced
        .get(adv_key)
        .or_else(|| request.model_manifest_entry.get(manifest_key))
        .and_then(|v| v.as_f64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as f32)
        .unwrap_or(default)
}

/// Inference steps for an SVD request: `advanced.steps` → `modelManifestEntry.steps[quality]` (else
/// its `balanced`) → the builtin quality ladder (fast 15 / balanced 25 / best 30), clamped 1..=80.
/// Mirrors the torch `_num_inference_steps` for the `svd_video` adapter.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_steps(request: &VideoRequest) -> u32 {
    let builtin = match request.quality.as_str() {
        "fast" => 15,
        "best" => 30,
        _ => 25,
    };
    let manifest_default = request
        .model_manifest_entry
        .get("steps")
        .and_then(Value::as_object)
        .and_then(|steps| {
            steps
                .get(&request.quality)
                .or_else(|| steps.get("balanced"))
        })
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(builtin);
    request
        .advanced
        .get("steps")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as i32)
        .unwrap_or(manifest_default)
        .clamp(1, 80) as u32
}

/// The single `Reference` conditioning image (image→video source). SVD is image-conditioned only,
/// so a missing `sourceAssetId` is a hard error (the routing gate [`svd_available`] already
/// requires it; this guards the direct-call path).
#[cfg(target_os = "macos")]
fn resolve_svd_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let asset_id = request.source_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "svd image→video requires a source image (sourceAssetId).".to_owned(),
        )
    })?;
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(vec![Conditioning::Reference {
        image,
        strength: None,
    }])
}

/// Raw-settings recorded on a real SVD asset (the resolved knobs + real-inference markers). Shared by
/// the MLX (macOS) and candle (Windows/CUDA, sc-5493) lanes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert(
        "numFrames".to_owned(),
        json!(svd_i32(request, "numFrames", "numFrames", 25, 1, 25)),
    );
    raw.insert(
        "motionBucketId".to_owned(),
        json!(svd_i32(
            request,
            "motionBucketId",
            "motionBucketId",
            127,
            1,
            255
        )),
    );
    raw.insert(
        "conditioningFps".to_owned(),
        json!(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30)),
    );
    // The output/playback cadence (decoupled from conditioningFps; sc-3764).
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "noiseAugStrength".to_owned(),
        json!(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02
        )),
    );
    raw.insert(
        "decodeChunkSize".to_owned(),
        json!(svd_i32(
            request,
            "decodeChunkSize",
            "decodeChunkSize",
            8,
            1,
            64
        )),
    );
    raw.insert("steps".to_owned(), json!(svd_steps(request)));
    Value::Object(raw)
}

/// Resolve an SVD request into a [`VideoGenInput`] and run it (sc-3523). image→video only: no
/// prompt / negative / guidance (the engine uses its frame-wise CFG ramp); `frames` is the
/// model-fixed burst length (≤25); `fps` carries the motion-conditioning cadence (see the module
/// note); `motion_bucket_id` / `noise_aug_strength` / `decode_chunk_size` drive the SVD knobs.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_svd(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let input = VideoGenInput {
        engine_id,
        model_dir: resolve_svd_model_dir(settings)?,
        quant: None,
        adapters: Vec::new(),
        conditioning: resolve_svd_conditioning(settings, request, project_path)?,
        prompt: String::new(),
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: svd_i32(request, "numFrames", "numFrames", 25, 1, 25) as u32,
        // Output/playback cadence = the user's `fps` (mirrors the torch `export_to_video(fps=request.fps)`);
        // the motion cadence rides `conditioning_fps` below (sc-3764).
        fps: request.fps,
        steps: Some(svd_steps(request)),
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        motion_bucket_id: Some(
            svd_i32(request, "motionBucketId", "motionBucketId", 127, 1, 255) as f32,
        ),
        noise_aug_strength: Some(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02,
        )),
        decode_chunk_size: Some(
            svd_i32(request, "decodeChunkSize", "decodeChunkSize", 8, 1, 64) as u32,
        ),
        conditioning_fps: Some(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30) as u32),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

// ---------------------------------------------------------------------------
// Real MLX Wan-VACE replace_person generation (macOS, via mlx-gen-wan, sc-3521):
// route the `replace_person` mode / `PersonReplace` job to the native `wan_vace`
// provider — the equivalent of the torch `DiffusersVideoAdapter` `WanVACEPipeline`
// path. The worker builds the masked-control inputs (source clip frames + the
// onnx-track person mask + character refs) and the engine does the
// masking/neutralization + denoise. Person detect/track/segment stays upstream.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Wan-VACE replace_person asset.
#[cfg(target_os = "macos")]
const WAN_VACE_ADAPTER: &str = "mlx_wan_vace";

/// Letterbox pad colour for extracted source-clip frames — matches the Python `fit_frame`
/// background (`0x12110f` = RGB 18,17,15) so the box masks (rasterized from the same
/// normalized boxes at W×H) stay aligned with the control frames through the engine's
/// identity-resize preprocess.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const FRAME_PAD_COLOR: &str = "0x12110f";

/// Raw-settings recorded on a real Wan-VACE asset (`advanced` knobs + the real-inference
/// markers; the engine id is `wan_vace`, not the user-picked replace-capable model).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn wan_vace_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String("wan_vace".to_owned()));
    raw.insert(
        "frameCount".to_owned(),
        json!(wan_frame_count(request.raw_frame_count())),
    );
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "replacementMode".to_owned(),
        Value::String(request.replacement_mode.clone()),
    );
    Value::Object(raw)
}

/// SceneWorks `replacementMode` string → engine [`ReplacementMode`] (default FaceOnly).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn replacement_mode_from(value: &str) -> ReplacementMode {
    match value {
        "full_person_keep_outfit" => ReplacementMode::FullPersonKeepOutfit,
        "full_person_replace_outfit" => ReplacementMode::FullPersonReplaceOutfit,
        _ => ReplacementMode::FaceOnly,
    }
}

/// Whether `dir` is a load-ready assembled Wan-VACE snapshot — the diffusers VACE
/// `transformer/` plus the shared base-Wan UMT5/VAE/tokenizer that `gen_core::load("wan_vace")`
/// reads (sc-3467 `assemble_wan_vace_snapshot` layout).
#[cfg(target_os = "macos")]
fn wan_vace_dir_is_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file()
}

/// Resolve (assembling on first use) the converted Wan-VACE snapshot dir. Env override
/// (`SCENEWORKS_MLX_WAN_VACE_DIR`) → the app-managed `<data>/models/mlx/wan_vace` → assemble
/// it from the diffusers VACE transformer (HF `Wan-AI/Wan2.1-VACE-1.3B-diffusers`,
/// `transformer/`) + a converted base-Wan 14B snapshot's shared UMT5/z16-VAE/tokenizer
/// (sc-3467 `assemble_wan_vace_snapshot` — packaging, not conversion). Errors clearly when a
/// component is missing rather than degrading to the stub.
#[cfg(target_os = "macos")]
fn resolve_wan_vace_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if wan_vace_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let out_dir = settings
        .data_dir
        .join("models")
        .join("mlx")
        .join("wan_vace");
    if wan_vace_dir_is_complete(&out_dir) {
        return Ok(out_dir);
    }
    // Assemble on first use: the VACE transformer is diffusers-layout (no conversion); the
    // shared T5/VAE/tokenizer come from a converted base-Wan 14B snapshot (z16 VAE, shared
    // with VACE since both are Wan2.1-based).
    let vace_repo = "Wan-AI/Wan2.1-VACE-1.3B-diffusers";
    let transformer_dir = huggingface_snapshot_dir(&settings.data_dir, vace_repo)
        .map(|snapshot| snapshot.join("transformer"))
        .filter(|dir| dir.join("config.json").is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "replace_person: the Wan-VACE transformer ({vace_repo}) is not downloaded — \
                 fetch it via the model manager."
            ))
        })?;
    let base_wan = ["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]
        .into_iter()
        .find_map(|model| resolve_wan_model_dir(settings, model, model).ok())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "replace_person: Wan-VACE needs a converted base-Wan 14B snapshot (its shared \
                 UMT5 text encoder + z16 VAE + tokenizer). Convert/download wan_2_2_t2v_14b or \
                 wan_2_2_i2v_14b first."
                    .to_owned(),
            )
        })?;
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    mlx_gen_wan::convert::assemble_wan_vace_snapshot(&out_dir, &transformer_dir, &base_wan, true)
        .map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "replace_person: failed to assemble the Wan-VACE snapshot: {error}"
        ))
    })?;
    Ok(out_dir)
}

/// Decode the source clip into exactly `count` RGB frames at `width × height` (letterboxed,
/// `FRAME_PAD_COLOR`), evenly resampled across the clip — the new shared frame-extraction
/// helper (Python `load_source_video_frames`; also the seam extend/bridge will reuse). The
/// frames are the (un-neutralized) Wan-VACE control video; the engine masks them.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn load_source_video_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    count: usize,
) -> WorkerResult<Vec<Image>> {
    let asset_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a source clip (sourceClipAssetId).".to_owned(),
        )
    })?;
    let asset = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_asset(&request.project_id, asset_id)
        .map_err(|error| WorkerError::InvalidPayload(format!("source clip {asset_id}: {error}")))?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("source clip {asset_id} has no media path"))
        })?;
    let media_path = crate::safe_project_path(project_path, rel)?;
    if !tokio::fs::try_exists(&media_path).await? {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip file is missing: {}",
            media_path.display()
        )));
    }

    let work_dir = std::env::temp_dir().join(format!("sw-replace-frames-{}", job.id));
    tokio::fs::create_dir_all(&work_dir).await?;
    let pattern = work_dir.join("src_%05d.png");
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24",
        width = request.width,
        height = request.height,
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let extract = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    let frames = match extract {
        Ok(()) => select_extracted_frames(work_dir.clone(), count).await,
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    frames
}

/// Collect the extracted PNG frames in `work_dir`, resample them to `count` evenly-spaced
/// indices (Python `evenly_spaced_indices` — the same arithmetic as the mask resample), and
/// decode the selected frames to engine [`Image`]s. Blocking IO/decoding runs off the runtime.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn select_extracted_frames(work_dir: PathBuf, count: usize) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "source clip produced no decodable frames".to_owned(),
            ));
        }
        let indices = crate::person_replace::resample_indices(paths.len(), count);
        indices
            .into_iter()
            .map(|index| decode_png_image(&paths[index]))
            .collect()
    })
    .await
    .map_err(|error| task_join_error("frame decode task", error))?
}

/// The approved character reference images (≤4) for the replacement (Python
/// `character_reference_images`): the selected look's `approvedReferenceIds`, else the
/// character's approved `references`. Errors when none are readable (the torch
/// `_validate_inputs` parity). The engine cover-fits each to the output size.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_character_references(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let character_id = request.character_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload("replace_person requires a character (characterId).".to_owned())
    })?;
    let character = CharacterStore::new(&settings.data_dir, project_path.to_path_buf())
        .get_character(&request.project_id, character_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("character {character_id}: {error}"))
        })?;
    let mut ids: Vec<String> = Vec::new();
    if let Some(look_id) = request.character_look_id.as_deref() {
        if let Some(looks) = character.get("looks").and_then(Value::as_array) {
            for look in looks {
                if look.get("id").and_then(Value::as_str) == Some(look_id) {
                    if let Some(approved) =
                        look.get("approvedReferenceIds").and_then(Value::as_array)
                    {
                        ids.extend(approved.iter().filter_map(Value::as_str).map(str::to_owned));
                    }
                }
            }
        }
    }
    if ids.is_empty() {
        if let Some(references) = character.get("references").and_then(Value::as_array) {
            for reference in references {
                if reference
                    .get("approved")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Some(asset_id) = reference.get("assetId").and_then(Value::as_str) {
                        ids.push(asset_id.to_owned());
                    }
                }
            }
        }
    }
    let mut images = Vec::new();
    for asset_id in ids.into_iter().filter(|id| !id.is_empty()).take(4) {
        if let Ok(image) = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &asset_id,
            project_path,
        ) {
            images.push(image);
        }
    }
    if images.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        ));
    }
    Ok(images)
}

/// Convert an `image::RgbImage` (the rasterized mask) to an engine [`Image`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn rgb_image_to_engine(image: image::RgbImage) -> Image {
    Image {
        width: image.width(),
        height: image.height(),
        pixels: image.into_raw(),
    }
}

/// Build the Wan-VACE conditioning: one [`Conditioning::ControlClip`] (source frames + the
/// per-frame person mask; the engine neutralizes the masked region) plus one
/// [`Conditioning::Reference`] per character reference image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_vace_conditioning(
    frames: Vec<Image>,
    masks: Vec<image::RgbImage>,
    references: Vec<Image>,
    masking_strength: f32,
    mode: ReplacementMode,
) -> WorkerResult<Vec<Conditioning>> {
    if frames.len() != masks.len() {
        return Err(WorkerError::InvalidPayload(format!(
            "replace_person: control frames ({}) and masks ({}) length mismatch",
            frames.len(),
            masks.len()
        )));
    }
    let mask_images: Vec<Image> = masks.into_iter().map(rgb_image_to_engine).collect();
    let mut conditioning = Vec::with_capacity(1 + references.len());
    conditioning.push(Conditioning::ControlClip {
        frames,
        mask: mask_images,
        masking_strength,
        start_frame: 0,
        mode,
    });
    for image in references {
        conditioning.push(Conditioning::Reference {
            image,
            strength: None,
        });
    }
    Ok(conditioning)
}

/// The honest `replacementStatus` recorded on the asset fact (mirrors the torch
/// `replacement_status`); the API folds it into the video sidecar's normalizedSettings.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn replacement_status_value(
    track: &Value,
    track_id: &str,
    mask_mode: &str,
    masking_strength: f32,
    reference_count: usize,
    frame_count: usize,
    adapter: &str,
) -> Value {
    let status = track.get("status").and_then(Value::as_object);
    let person_tracking_active = status
        .and_then(|s| s.get("personTrackingActive"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mask_state = status
        .and_then(|s| s.get("maskState"))
        .and_then(Value::as_str)
        .unwrap_or("missing")
        .to_owned();
    let corrections = track.get("corrections").and_then(Value::as_array);
    let correction_count = corrections.map(|list| list.len()).unwrap_or(0);
    let resolved_track_id = track.get("id").and_then(Value::as_str).unwrap_or(track_id);
    json!({
        "personDetectionActive": true,
        "personTrackingActive": person_tracking_active,
        "replacementActive": true,
        "replacementAdapter": adapter,
        "maskMode": mask_mode,
        "maskState": mask_state,
        "maskingStrength": masking_strength,
        "personTrackId": resolved_track_id,
        "characterReferenceCount": reference_count,
        "controlFrameCount": frame_count,
        "usedCorrections": correction_count > 0,
        "correctionCount": correction_count,
    })
}

/// Resolve a replace_person request into a Wan-VACE generation: assemble/resolve the snapshot,
/// extract the source-clip control frames, build the per-frame person mask from the saved
/// track (corrections applied), load the character refs, run the engine, and return the decoded
/// video plus the honest `replacementStatus`. Person detect/track/segment stays upstream.
#[cfg(target_os = "macos")]
async fn generate_wan_vace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_wan_vace_model_dir(settings)?;
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Source frames + masks must match in count and be `1 + 4·k` (one z16 VAE temporal chunk),
    // which `wan_frame_count` guarantees — the engine `validate()` enforces it too.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let frames =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let (masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frames.len(),
    )?;
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let frame_total = frames.len();

    let masking_strength = advanced::f32(&request.advanced, "maskingStrength", 1.0);
    let conditioning = build_vace_conditioning(
        frames,
        masks,
        references,
        masking_strength,
        replacement_mode_from(&request.replacement_mode),
    )?;

    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let steps = request.advanced.get("steps").and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    });
    let guidance = request.advanced.get("guidanceScale").and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    });
    let input = VideoGenInput {
        engine_id: "wan_vace",
        model_dir,
        quant: resolve_wan_quant(request),
        adapters: resolve_wan_vace_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, input).await?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        WAN_VACE_ADAPTER,
    );
    Ok((decoded, status))
}

// ---------------------------------------------------------------------------
// Wan extend_clip / video_bridge — native Wan-VACE ControlClip (sc-3812, tier C).
//
// The TI2V-5B single-frame path (`build_wan_boundary_conditioning`, sc-3357) conditions on one
// boundary still, so it morphs *from* a frozen frame and cannot inherit the source clip's motion.
// Routing these modes to the `wan_vace` engine instead lets the model attend to *several real*
// source frames pinned at the kept positions (mask black = keep) while it generates the rest of the
// timeline freely (mask white = regenerate over a neutral-gray control video). That is the whole
// point of extend/bridge — genuine motion continuity — at the cost of the smaller VACE-1.3B base
// (vs TI2V-5B), so the single-frame path stays the baseline/fallback. No reference images: the
// content comes from the kept frames, not a character (the engine's reference path is optional).
// Raw-settings record `model = wan_vace` + `fidelityTier = vace_controlclip` so the engine
// substitution under the user's `wan_2_2` pick is an inspectable fact on the asset, not a black box.

/// Mid-gray (≈0 after the engine's `2·x/255 − 1` normalization) control frame for the
/// to-generate span: a neutral `reactive = video·mask` signal so the masked region is generated
/// freely from the kept frames + prompt, never biased toward a frozen filler image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn neutral_control_frame(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: vec![128u8; (width as usize) * (height as usize) * 3],
    }
}

/// A solid W×H mask (`0` = keep the control frame, `255` = regenerate; the engine binarizes at
/// 0.5), matching the `image::RgbImage` form `person_track_masks` produces for replace_person.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn solid_mask(width: u32, height: u32, value: u8) -> image::RgbImage {
    image::RgbImage::from_pixel(width, height, image::Rgb([value, value, value]))
}

/// How many real source frames to pin as the motion anchor per kept boundary (sc-3812). More =
/// truer continuity but fewer freely-generated frames. Overridable via advanced `motionAnchorFrames`
/// (per side); defaults to ~⅓ of the output budget (split across the two boundaries for bridge), and
/// is clamped so at least 5 frames (one z16 chunk) are left to generate.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn extend_anchor_frames(request: &VideoRequest, frame_count: usize) -> usize {
    let per_side = if request.mode == "video_bridge" { 2 } else { 1 };
    let max_total = frame_count.saturating_sub(5).max(1);
    let max_per_side = (max_total / per_side).max(1);
    let default = (frame_count / 3 / per_side).max(1);
    let requested = request
        .advanced
        .get("motionAnchorFrames")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as usize)
        .unwrap_or(default);
    requested.clamp(1, max_per_side)
}

/// Decode the `take`-end `count` frames of a source clip (its head or tail) to letterboxed W×H
/// engine [`Image`]s, in temporal order (sc-3812). Unlike [`load_source_video_frames`] — which
/// resamples the *whole* clip evenly — this keeps *consecutive* real frames so the model sees the
/// clip's actual motion velocity at the boundary. Decodes only the kept subset (`decode_png_image`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn load_clip_anchor_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    count: usize,
    take: ClipFramePosition,
) -> WorkerResult<Vec<Image>> {
    let media_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let work_dir = std::env::temp_dir().join(format!(
        "sw-anchor-frames-{}-{}",
        job.id,
        Uuid::new_v4().simple()
    ));
    tokio::fs::create_dir_all(&work_dir).await?;
    let pattern = work_dir.join("src_%05d.png");
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24",
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let extract = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    let frames = match extract {
        Ok(()) => select_anchor_frames(work_dir.clone(), count, take).await,
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    frames
}

/// Pick the head/tail `count` consecutive PNGs from `work_dir` (sorted) and decode them to engine
/// [`Image`]s, preserving temporal order. Fewer available than `count` ⇒ all of them (short clip).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn select_anchor_frames(
    work_dir: PathBuf,
    count: usize,
    take: ClipFramePosition,
) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "source clip produced no decodable frames".to_owned(),
            ));
        }
        let take_n = count.min(paths.len());
        let selected = match take {
            ClipFramePosition::Last => &paths[paths.len() - take_n..],
            ClipFramePosition::First => &paths[..take_n],
        };
        selected.iter().map(|path| decode_png_image(path)).collect()
    })
    .await
    .map_err(|error| task_join_error("frame decode task", error))?
}

/// Decode one RGB PNG into an engine [`Image`] (shared by the resample + anchor frame selectors).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn decode_png_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("source frame {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// Build the Wan-VACE extend/bridge ControlClip (sc-3812): real source frames pinned at the kept
/// positions (mask black) and a neutral-gray generated span (mask white). For `extend_clip` the
/// left-clip tail anchors the start and the continuation is generated; for `video_bridge` both
/// clips' boundary anchors are pinned at the two ends and the gap between them is generated. The
/// control clip is `frame_count` long (`1 + 4·k`, the engine's z16-chunk constraint) with no
/// reference images. `masking_strength`/`mode` are inert in the WanVACE mask math (carried for the
/// shared [`Conditioning::ControlClip`] contract), so they take the neutral defaults.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_extend_bridge_vace_conditioning(
    request: &VideoRequest,
    width: u32,
    height: u32,
    frame_count: usize,
    left_anchor: Vec<Image>,
    right_anchor: Option<Vec<Image>>,
) -> WorkerResult<Vec<Conditioning>> {
    let neutral = neutral_control_frame(width, height);
    let keep_mask = solid_mask(width, height, 0);
    let gen_mask = solid_mask(width, height, 255);
    let mut frames: Vec<Image> = Vec::with_capacity(frame_count);
    let mut masks: Vec<image::RgbImage> = Vec::with_capacity(frame_count);
    let left_n = left_anchor.len();
    match request.mode.as_str() {
        "extend_clip" => {
            if left_n + 1 > frame_count {
                return Err(WorkerError::InvalidPayload(format!(
                    "extend_clip: {left_n} anchor frames leave no room to generate in a \
                     {frame_count}-frame clip — reduce motionAnchorFrames."
                )));
            }
            for frame in left_anchor {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
            for _ in left_n..frame_count {
                frames.push(neutral.clone());
                masks.push(gen_mask.clone());
            }
        }
        "video_bridge" => {
            let right = right_anchor.ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
            let right_n = right.len();
            if left_n + right_n + 1 > frame_count {
                return Err(WorkerError::InvalidPayload(format!(
                    "video_bridge: {left_n}+{right_n} anchor frames leave no gap to generate in a \
                     {frame_count}-frame clip — reduce motionAnchorFrames."
                )));
            }
            for frame in left_anchor {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
            for _ in 0..(frame_count - left_n - right_n) {
                frames.push(neutral.clone());
                masks.push(gen_mask.clone());
            }
            for frame in right {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
        }
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "build_extend_bridge_vace_conditioning: unexpected mode {other}"
            )))
        }
    }
    build_vace_conditioning(frames, masks, Vec::new(), 1.0, ReplacementMode::default())
}

/// Raw-settings for a Wan-VACE extend/bridge asset: the request `advanced` knobs + the real-inference
/// markers, recording the actual engine (`wan_vace`) and `fidelityTier` so the substitution under the
/// user's `wan_2_2` pick is an inspectable fact (sc-3812). Unlike [`wan_vace_raw_settings`] there is
/// no `replacementMode` (these modes are not person-replacement).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn wan_vace_extend_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String("wan_vace".to_owned()));
    raw.insert(
        "frameCount".to_owned(),
        json!(wan_frame_count(request.raw_frame_count())),
    );
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "fidelityTier".to_owned(),
        Value::String("vace_controlclip".to_owned()),
    );
    Value::Object(raw)
}

/// Resolve an extend_clip / video_bridge request into a native Wan-VACE generation (sc-3812, tier C).
/// Loads the real source-clip anchor frames (the left clip's tail for extend; both clips' boundaries
/// for bridge), builds the source-at-kept-positions + generated-span ControlClip, and runs the
/// `wan_vace` engine. The TI2V-5B single-frame path ([`generate_wan`]) remains the baseline/fallback,
/// chosen by the dispatch seam when the VACE snapshot is unprovisioned.
#[cfg(target_os = "macos")]
async fn generate_wan_vace_extend_bridge(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    model_dir: PathBuf,
) -> WorkerResult<DecodedVideo> {
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let anchor = extend_anchor_frames(request, frame_count);
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_anchor = load_clip_anchor_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        anchor,
        ClipFramePosition::Last,
    )
    .await?;
    let right_anchor = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            load_clip_anchor_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                anchor,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    let conditioning = build_extend_bridge_vace_conditioning(
        request,
        request.width,
        request.height,
        frame_count,
        left_anchor,
        right_anchor,
    )?;
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let steps = request.advanced.get("steps").and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    });
    let guidance = request.advanced.get("guidanceScale").and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    });
    let input = VideoGenInput {
        engine_id: "wan_vace",
        model_dir,
        quant: resolve_wan_quant(request),
        adapters: resolve_wan_vace_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, input).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> VideoRequest {
        VideoRequest::from_payload(&value.as_object().cloned().unwrap())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stall_timeout_override_parses_or_falls_back() {
        // A valid positive override wins.
        assert_eq!(
            parse_stall_timeout(Some("120".to_owned())),
            Duration::from_secs(120)
        );
        assert_eq!(
            parse_stall_timeout(Some("  90 ".to_owned())),
            Duration::from_secs(90)
        );
        // Unset, blank, non-numeric, or zero all fall back to the default.
        assert_eq!(parse_stall_timeout(None), VIDEO_STALL_TIMEOUT);
        assert_eq!(
            parse_stall_timeout(Some(String::new())),
            VIDEO_STALL_TIMEOUT
        );
        assert_eq!(
            parse_stall_timeout(Some("nope".to_owned())),
            VIDEO_STALL_TIMEOUT
        );
        assert_eq!(
            parse_stall_timeout(Some("0".to_owned())),
            VIDEO_STALL_TIMEOUT
        );
    }

    #[test]
    fn plan_builds_nested_media_path() {
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox runs"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        assert!(plan
            .media_rel
            .starts_with(&format!("assets/videos/{}/", plan.genset_id)));
        assert!(plan.media_rel.ends_with(".mp4"));
        assert!(plan.media_rel.contains("_ltx_2_3_"));
        assert!(plan.asset_id.starts_with("asset_"));
        assert_eq!(plan.family, "ltx-video");
        assert_eq!(
            plan.media_path,
            Path::new("/tmp/project").join(&plan.media_rel)
        );
    }

    #[test]
    fn family_prefers_manifest_then_infers_from_model() {
        let manifest = request(json!({
            "projectId": "p", "model": "ltx_2_3",
            "modelManifestEntry": { "family": "ltx-custom" }
        }));
        assert_eq!(resolve_family(&manifest), "ltx-custom");
        let wan = request(json!({ "projectId": "p", "model": "wan_2_2_t2v_14b" }));
        assert_eq!(resolve_family(&wan), "wan-video");
        let other = request(json!({ "projectId": "p", "model": "mystery" }));
        assert_eq!(resolve_family(&other), "video");
    }

    #[test]
    fn resolve_seed_prefers_explicit_then_hashes_prompt() {
        let explicit = request(json!({ "projectId": "p", "seed": 123 }));
        assert_eq!(resolve_video_seed(&explicit), 123);
        // No seed → deterministic from the prompt (re-run reproduces).
        let a = request(json!({ "projectId": "p", "prompt": "sunset" }));
        let b = request(json!({ "projectId": "p", "prompt": "sunset" }));
        assert_eq!(resolve_video_seed(&a), resolve_video_seed(&b));
        let c = request(json!({ "projectId": "p", "prompt": "sunrise" }));
        assert_ne!(resolve_video_seed(&a), resolve_video_seed(&c));
    }

    #[test]
    fn stub_video_frames_have_correct_size_and_audio_split() {
        // LTX → audio present; the frame buffers are exactly width*height*3.
        let ltx = request(json!({
            "projectId": "p", "model": "ltx_2_3", "width": 256, "height": 256,
            "duration": 1.0, "fps": 9
        }));
        let decoded = generate_stub_video(&ltx, 7);
        assert_eq!(decoded.frames.len(), ltx.frame_count() as usize);
        assert_eq!(decoded.fps, 9);
        for frame in &decoded.frames {
            assert_eq!(frame.pixels.len(), 256 * 256 * 3);
        }
        let audio = decoded.audio.expect("LTX stub emits audio");
        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.channels, 1);
        assert!(!audio.samples.is_empty());

        // Wan → no audio track (mirrors the engine).
        let wan = request(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "duration": 1.0, "fps": 16
        }));
        assert!(generate_stub_video(&wan, 7).audio.is_none());
    }

    #[test]
    fn stub_frames_differ_across_time() {
        // The sweeping band makes frame 0 and a later frame differ (real motion).
        let request = request(json!({
            "projectId": "p", "model": "wan_2_2", "width": 256, "height": 64,
            "duration": 1.0, "fps": 16
        }));
        let decoded = generate_stub_video(&request, 3);
        assert!(decoded.frames.len() >= 2);
        assert_ne!(
            decoded.frames[0].pixels,
            decoded.frames[decoded.frames.len() - 1].pixels
        );
    }

    #[test]
    fn wav_header_is_canonical_and_peak_normalized() {
        let audio = AudioTrack {
            samples: vec![0.0, 0.5, -0.25, 0.5],
            sample_rate: 48_000,
            channels: 1,
        };
        let dir = std::env::temp_dir().join(format!("sw_wav_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.wav");
        write_wav_pcm16(&audio, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        // 4 mono 16-bit samples → 8 bytes of PCM, 44-byte header.
        assert_eq!(bytes.len(), 44 + 8);
        // Peak (0.5) maps to i16::MAX; the matching trough (-0.25) is half-scale negative.
        let first = i16::from_le_bytes([bytes[44], bytes[45]]);
        let peak = i16::from_le_bytes([bytes[46], bytes[47]]);
        let trough = i16::from_le_bytes([bytes[48], bytes[49]]);
        assert_eq!(first, 0);
        assert_eq!(peak, i16::MAX);
        assert_eq!(trough, -(i16::MAX / 2) - 1); // -0.25/0.5 * 32767, rounded
    }

    #[test]
    fn silent_audio_does_not_divide_by_zero() {
        let audio = AudioTrack {
            samples: vec![0.0; 16],
            sample_rate: 48_000,
            channels: 1,
        };
        let dir = std::env::temp_dir().join(format!("sw_wav_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("silent.wav");
        write_wav_pcm16(&audio, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes[44..].iter().all(|&b| b == 0));
    }

    #[test]
    fn asset_fact_and_streaming_result_shape() {
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox",
            "duration": 4.0, "fps": 24, "width": 768, "height": 512,
            "sourceAssetId": "asset_src", "personTrackId": "track_1"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        let fact = video_asset_fact(
            &plan,
            42,
            "procedural_video",
            stub_raw_settings(&request),
            None,
        );
        assert_eq!(fact["type"], json!("video"));
        assert_eq!(fact["mimeType"], json!("video/mp4"));
        assert_eq!(fact["mediaPath"], json!(plan.media_rel));
        assert_eq!(fact["adapter"], json!("procedural_video"));
        assert_eq!(fact["seed"], json!(42));
        assert_eq!(fact["duration"], json!(4.0));
        assert_eq!(fact["fps"], json!(24));
        assert_eq!(fact["sourceAssetId"], json!("asset_src"));
        assert_eq!(fact["personTrackId"], json!("track_1"));
        assert_eq!(fact["displayName"], json!("A red fox"));
        assert_eq!(fact["rawAdapterSettings"]["stub"], json!(true));

        let result = streaming_result(&plan, &fact, "procedural_video");
        assert_eq!(result["expectedCount"], json!(1));
        assert_eq!(result["adapter"], json!("procedural_video"));
        assert_eq!(result["assetWrites"].as_array().unwrap().len(), 1);
        assert_eq!(result["generationSet"]["count"], json!(1));
    }

    /// A replace_person asset fact carries the `replacementStatus` object the API folds into
    /// the video sidecar (sc-3521); a non-replace fact omits it.
    #[test]
    fn asset_fact_embeds_replacement_status_when_present() {
        let request = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "replace_person",
            "prompt": "swap the hero", "personTrackId": "track_9"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        let status = json!({ "replacementActive": true, "maskMode": "segmentation" });
        let fact = video_asset_fact(&plan, 7, "mlx_wan_vace", json!({}), Some(status));
        assert_eq!(fact["replacementStatus"]["replacementActive"], json!(true));
        assert_eq!(fact["replacementStatus"]["maskMode"], json!("segmentation"));
        // Without a status the key is absent (the non-replace paths).
        let bare = video_asset_fact(&plan, 7, "mlx_wan", json!({}), None);
        assert!(bare.get("replacementStatus").is_none());
    }

    /// SceneWorks `replacementMode` strings → engine `ReplacementMode` (default FaceOnly).
    #[cfg(target_os = "macos")]
    #[test]
    fn replacement_mode_maps_the_three_granularities() {
        assert_eq!(
            replacement_mode_from("face_only"),
            ReplacementMode::FaceOnly
        );
        assert_eq!(
            replacement_mode_from("full_person_keep_outfit"),
            ReplacementMode::FullPersonKeepOutfit
        );
        assert_eq!(
            replacement_mode_from("full_person_replace_outfit"),
            ReplacementMode::FullPersonReplaceOutfit
        );
        assert_eq!(replacement_mode_from("nonsense"), ReplacementMode::FaceOnly);
    }

    /// SCAIL-2 maps the SceneWorks video mode to the engine `video_mode` task: cross-identity
    /// `replace_person` → "replacement" (engine flips `replace_flag`), everything else (standalone
    /// `animate_character`) → "animation" (sc-5448 / sc-5452).
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_mode_maps_replacement_vs_animation() {
        assert_eq!(scail2_engine_video_mode("replace_person"), "replacement");
        assert_eq!(scail2_engine_video_mode("animate_character"), "animation");
        assert_eq!(scail2_engine_video_mode("text_to_video"), "animation");
    }

    /// The replacement status records the actual engine adapter, so a SCAIL-2-backed person-replace
    /// (sc-5452) reports `mlx_scail2` (not the Wan-VACE default).
    #[cfg(target_os = "macos")]
    #[test]
    fn replacement_status_records_scail2_adapter() {
        let track = json!({ "id": "trk_1", "status": { "maskState": "active" } });
        let status =
            replacement_status_value(&track, "trk_1", "segmentation", 1.0, 1, 81, SCAIL2_ADAPTER);
        assert_eq!(status["replacementAdapter"], json!("mlx_scail2"));
        assert_eq!(status["replacementActive"], json!(true));
        assert_eq!(status["controlFrameCount"], json!(81));
    }

    /// The Wan-VACE conditioning is one ControlClip (frames + per-frame mask) followed by one
    /// Reference per character image; mismatched frame/mask counts fail clearly.
    #[cfg(target_os = "macos")]
    #[test]
    fn vace_conditioning_builds_control_clip_plus_references() {
        let frame = |v: u8| Image {
            width: 2,
            height: 2,
            pixels: vec![v; 12],
        };
        let mask = || image::RgbImage::from_pixel(2, 2, image::Rgb([255, 255, 255]));
        let conditioning = build_vace_conditioning(
            vec![frame(10), frame(20)],
            vec![mask(), mask()],
            vec![frame(30)],
            0.75,
            ReplacementMode::FullPersonKeepOutfit,
        )
        .expect("conditioning builds");
        assert_eq!(conditioning.len(), 2); // 1 ControlClip + 1 Reference
        match &conditioning[0] {
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode,
            } => {
                assert_eq!(frames.len(), 2);
                assert_eq!(mask.len(), 2);
                assert_eq!(*masking_strength, 0.75);
                assert_eq!(*start_frame, 0);
                assert_eq!(*mode, ReplacementMode::FullPersonKeepOutfit);
            }
            other => panic!("expected ControlClip, got {other:?}"),
        }
        assert!(matches!(conditioning[1], Conditioning::Reference { .. }));
        // A frame/mask count mismatch is rejected.
        assert!(build_vace_conditioning(
            vec![frame(1)],
            vec![mask(), mask()],
            Vec::new(),
            1.0,
            ReplacementMode::FaceOnly,
        )
        .is_err());
    }

    /// sc-3812 extend: the ControlClip pins the real tail frames at the front (mask black = keep)
    /// and fills the rest of the budget with a neutral-gray generated span (mask white), no refs.
    #[cfg(target_os = "macos")]
    #[test]
    fn extend_vace_conditioning_pins_tail_and_generates_rest() {
        let anchor = |v: u8| Image {
            width: 2,
            height: 2,
            pixels: vec![v; 12],
        };
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip",
            "prompt": "keep walking", "sourceClipAssetId": "clip_a"
        }));
        let conditioning = build_extend_bridge_vace_conditioning(
            &req,
            2,
            2,
            5,
            vec![anchor(11), anchor(22)],
            None,
        )
        .expect("extend conditioning builds");
        assert_eq!(conditioning.len(), 1); // ControlClip only, no Reference
        match &conditioning[0] {
            Conditioning::ControlClip { frames, mask, .. } => {
                assert_eq!(frames.len(), 5);
                assert_eq!(mask.len(), 5);
                // First two are the real tail frames, kept (black mask).
                assert_eq!(frames[0].pixels[0], 11);
                assert_eq!(frames[1].pixels[0], 22);
                assert_eq!(mask[0].pixels[0], 0);
                assert_eq!(mask[1].pixels[0], 0);
                // The rest is the neutral-gray generated span (white mask).
                assert_eq!(frames[2].pixels[0], 128);
                assert_eq!(frames[4].pixels[0], 128);
                assert_eq!(mask[2].pixels[0], 255);
                assert_eq!(mask[4].pixels[0], 255);
            }
            other => panic!("expected ControlClip, got {other:?}"),
        }
    }

    /// sc-3812 bridge: both clips' boundary anchors are kept at the two ends; the gap between them
    /// is the generated span. A missing right clip / over-budget anchors fail clearly.
    #[cfg(target_os = "macos")]
    #[test]
    fn bridge_vace_conditioning_keeps_both_ends_generates_gap() {
        let anchor = |v: u8| Image {
            width: 1,
            height: 1,
            pixels: vec![v; 3],
        };
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "video_bridge",
            "prompt": "connect", "sourceClipAssetId": "left", "bridgeRightClipAssetId": "right"
        }));
        let conditioning = build_extend_bridge_vace_conditioning(
            &req,
            1,
            1,
            5,
            vec![anchor(10)],
            Some(vec![anchor(90)]),
        )
        .expect("bridge conditioning builds");
        match &conditioning[0] {
            Conditioning::ControlClip { frames, mask, .. } => {
                assert_eq!(frames.len(), 5);
                // Left end kept, gap generated, right end kept.
                assert_eq!((frames[0].pixels[0], mask[0].pixels[0]), (10, 0));
                assert_eq!((frames[1].pixels[0], mask[1].pixels[0]), (128, 255));
                assert_eq!((frames[3].pixels[0], mask[3].pixels[0]), (128, 255));
                assert_eq!((frames[4].pixels[0], mask[4].pixels[0]), (90, 0));
            }
            other => panic!("expected ControlClip, got {other:?}"),
        }
        // video_bridge without a right clip is rejected.
        assert!(
            build_extend_bridge_vace_conditioning(&req, 1, 1, 5, vec![anchor(10)], None).is_err()
        );
        // Anchors that leave no gap are rejected.
        assert!(build_extend_bridge_vace_conditioning(
            &req,
            1,
            1,
            5,
            vec![anchor(1), anchor(2), anchor(3)],
            Some(vec![anchor(4), anchor(5)]),
        )
        .is_err());
    }

    /// sc-3812 motion anchor: defaults to ~⅓ of the budget (halved per side for bridge), honors an
    /// explicit `motionAnchorFrames`, and always clamps so ≥5 frames stay generatable.
    #[cfg(target_os = "macos")]
    #[test]
    fn extend_anchor_frames_defaults_and_clamps() {
        let extend = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x"
        }));
        assert_eq!(extend_anchor_frames(&extend, 81), 27); // 81/3
        let bridge = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "video_bridge", "prompt": "x"
        }));
        assert_eq!(extend_anchor_frames(&bridge, 81), 13); // (81/3)/2

        let explicit = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x",
            "advanced": { "motionAnchorFrames": 4 }
        }));
        assert_eq!(extend_anchor_frames(&explicit, 81), 4);

        // Over-budget request clamps so 5 frames remain to generate (81 - 5 = 76).
        let greedy = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x",
            "advanced": { "motionAnchorFrames": 999 }
        }));
        assert_eq!(extend_anchor_frames(&greedy, 81), 76);
        // Minimum-length clip still yields a usable anchor.
        assert_eq!(extend_anchor_frames(&extend, 5), 1);
    }

    /// `replacement_status_value` reports the honest mask/track provenance the sidecar folds in.
    #[cfg(target_os = "macos")]
    #[test]
    fn replacement_status_reads_track_and_counts() {
        let track = json!({
            "id": "track_42",
            "status": { "maskState": "active", "personTrackingActive": true },
            "corrections": [ { "frameIndex": 0 }, { "frameIndex": 3 } ]
        });
        // 0.5 is exactly representable as f32 so the JSON widen to f64 is exact.
        let status = replacement_status_value(
            &track,
            "ignored",
            "segmentation",
            0.5,
            2,
            81,
            WAN_VACE_ADAPTER,
        );
        assert_eq!(status["personDetectionActive"], json!(true));
        assert_eq!(status["personTrackingActive"], json!(true));
        assert_eq!(status["replacementActive"], json!(true));
        assert_eq!(status["replacementAdapter"], json!("mlx_wan_vace"));
        assert_eq!(status["maskMode"], json!("segmentation"));
        assert_eq!(status["maskState"], json!("active"));
        assert_eq!(status["maskingStrength"], json!(0.5));
        assert_eq!(status["personTrackId"], json!("track_42"));
        assert_eq!(status["characterReferenceCount"], json!(2));
        assert_eq!(status["controlFrameCount"], json!(81));
        assert_eq!(status["usedCorrections"], json!(true));
        assert_eq!(status["correctionCount"], json!(2));
    }

    /// An assembled Wan-VACE snapshot dir if one is present (env override or the app-managed
    /// default), else `None` so the real-weight smoke skips.
    #[cfg(target_os = "macos")]
    fn wan_vace_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_DIR") {
            let path = PathBuf::from(dir.trim());
            if wan_vace_dir_is_complete(&path) {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join("Library/Application Support/SceneWorks/data/models/mlx/wan_vace");
        wan_vace_dir_is_complete(&path).then_some(path)
    }

    /// Real in-process Wan-VACE replace_person through the engine: load the assembled snapshot
    /// and denoise a tiny 5-frame clip from a synthetic control clip (gray frames + a centered
    /// box mask) + one reference, asserting frames come back RGB8-sized with streamed progress.
    /// `#[ignore]` — the weights live outside CI; run manually on a Mac where the snapshot is
    /// assembled (the real-Mac GPU parity gate, sc-3521).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan-VACE snapshot; run manually on a Mac with it assembled"]
    #[test]
    fn wan_vace_real_weights() {
        let Some(model_dir) = wan_vace_dir() else {
            eprintln!("skipping wan_vace_real_weights: no assembled wan_vace snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        let gray = || Image {
            width: w,
            height: h,
            pixels: vec![118u8; (w * h * 3) as usize],
        };
        let frames: Vec<Image> = (0..5).map(|_| gray()).collect();
        let masks: Vec<image::RgbImage> = (0..5)
            .map(|_| {
                crate::person_replace::box_mask(
                    Some(&json!({ "x": 0.3, "y": 0.2, "width": 0.4, "height": 0.6 })),
                    w,
                    h,
                )
            })
            .collect();
        let conditioning =
            build_vace_conditioning(frames, masks, vec![gray()], 1.0, ReplacementMode::FaceOnly)
                .expect("conditioning builds");
        let input = VideoGenInput {
            engine_id: "wan_vace",
            model_dir,
            conditioning,
            prompt: "a person walking, cinematic".to_owned(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 7,
            control_scale: Some(1.0),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("VACE generation");
        assert!(decoded.fps >= 1);
        assert!(decoded.audio.is_none(), "Wan-VACE emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// A Bernini MLX snapshot dir if present: env override (`SCENEWORKS_MLX_BERNINI_DIR`), then the
    /// bring-up staging / cache dirs, then the app-managed default. `None` ⇒ the smoke skips.
    #[cfg(target_os = "macos")]
    fn bernini_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_BERNINI_DIR") {
            let path = PathBuf::from(dir.trim());
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        for rel in [
            ".cache/mlx-gen-models/bernini-mlx-upload",
            ".cache/mlx-gen-models/bernini_full_mlx_bf16",
            "Library/Application Support/SceneWorks/data/models/mlx/bernini",
        ] {
            let path = PathBuf::from(&home).join(rel);
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        None
    }

    /// Test-only override for the quant tier a Bernini smoke loads (sc-4709): `SCENEWORKS_BERNINI_
    /// SMOKE_QUANT` = `8` → Q8, `0` → bf16 (no quant), anything else / unset → Q4 (the committed
    /// default). Lets the manual smokes profile peak memory at either tier without changing the
    /// product default (e.g. capturing the Q8 video peak this story tracks).
    #[cfg(target_os = "macos")]
    fn bernini_smoke_quant() -> Option<Quant> {
        match std::env::var("SCENEWORKS_BERNINI_SMOKE_QUANT")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
        {
            Some(bits) if bits <= 0 => None,
            Some(bits) if bits > 4 => Some(Quant::Q8),
            _ => Some(Quant::Q4),
        }
    }

    /// A test-only `usize`/`u32` env override, falling back to `default` when unset/unparseable.
    #[cfg(target_os = "macos")]
    fn bernini_smoke_u32(key: &str, default: u32) -> u32 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(default)
    }

    /// Real in-process Bernini text-to-video through the WORKER registry path (epic 4699 / sc-4707):
    /// `gen_core::load("bernini")` (proves the `mlx_gen_bernini` force-link survived in the worker
    /// binary — the "no generator registered" trap) → Q4 → a tiny t2v clip, asserting RGB8 frames
    /// stream back with denoise progress. Also confirms the lean published `SceneWorks/bernini-mlx`
    /// snapshot loads. `#[ignore]` — weights live outside CI; run on a Mac with the snapshot present.
    /// The quant tier (`SCENEWORKS_BERNINI_SMOKE_QUANT`) and dims (`..._W`/`_H`/`_FRAMES`/`_STEPS`)
    /// are env-overridable for memory profiling (sc-4709 Q8 peak); defaults reproduce the Q4 run.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
    #[test]
    fn bernini_t2v_real_weights() {
        let Some(model_dir) = bernini_dir() else {
            eprintln!("skipping bernini_t2v_real_weights: no Bernini MLX snapshot found");
            return;
        };
        let w = bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_W", 832);
        let h = bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_H", 480);
        let input = VideoGenInput {
            engine_id: "bernini",
            model_dir,
            quant: bernini_smoke_quant(),
            prompt: "a golden retriever puppy running across a sunlit meadow, cinematic".to_owned(),
            width: w,
            height: h,
            frames: bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_FRAMES", 17),
            fps: 16,
            steps: Some(bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_STEPS", 20)),
            seed: 7,
            video_mode: Some("t2v".to_owned()),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("bernini t2v generation");
        assert!(decoded.fps >= 1);
        assert!(steps > 0, "denoise progress streamed");
        assert!(!decoded.frames.is_empty(), "frames returned");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// The SceneWorks mode → engine guidance-task mapping (sc-4703). Pure; runs in CI on Mac.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_engine_video_mode_maps_each_sceneworks_mode() {
        assert_eq!(bernini_engine_video_mode("text_to_video"), "t2v");
        assert_eq!(bernini_engine_video_mode("video_to_video"), "v2v");
        assert_eq!(bernini_engine_video_mode("reference_to_video"), "r2v");
        assert_eq!(
            bernini_engine_video_mode("reference_video_to_video"),
            "rv2v"
        );
        // Multi-source modes (sc-5425).
        assert_eq!(bernini_engine_video_mode("multi_video_to_video"), "mv2v");
        assert_eq!(bernini_engine_video_mode("ads2v"), "ads2v");
        // Unknown / unset falls back to plain text-to-video.
        assert_eq!(bernini_engine_video_mode("image_to_video"), "t2v");
        assert_eq!(bernini_engine_video_mode(""), "t2v");
    }

    /// Only the `bernini` catalog id routes to the Bernini engine (sc-4707). Other video
    /// families (Wan/LTX/SVD/image-typed ids) are `None` so they keep their own routing.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_engine_id_maps_only_the_bernini_family() {
        assert_eq!(bernini_engine_id("bernini"), Some("bernini"));
        assert_eq!(bernini_engine_id("wan_2_2"), None);
        assert_eq!(bernini_engine_id("wan_2_2_t2v_14b"), None);
        assert_eq!(bernini_engine_id("ltx_2_3"), None);
        assert_eq!(bernini_engine_id("z_image_turbo"), None);
        assert_eq!(bernini_engine_id(""), None);
    }

    /// Bernini load quantization (sc-4709): Q4 is the default (the validated 64 GB-fitting tier),
    /// `mlxQuantize` opts up to Q8 or down to bf16 (`<= 0`), and the control parses from a JSON
    /// number or a string. The snapshot is ~93 GB at bf16 so a missing control NEVER means bf16.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_resolve_quant_defaults_q4_and_honors_override() {
        let quant = |advanced: Value| {
            resolve_bernini_quant(&request(json!({ "projectId": "p", "advanced": advanced })))
        };
        // No control → Q4 default (never bf16).
        assert!(matches!(quant(json!({})), Some(Quant::Q4)));
        // Explicit tiers (number).
        assert!(matches!(
            quant(json!({ "mlxQuantize": 4 })),
            Some(Quant::Q4)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": 8 })),
            Some(Quant::Q8)
        ));
        // `<= 0` opts into bf16 (no quantization) for power users with ample RAM.
        assert!(quant(json!({ "mlxQuantize": 0 })).is_none());
        assert!(quant(json!({ "mlxQuantize": -1 })).is_none());
        // String forms parse the same (the advanced map can carry stringly-typed values).
        assert!(matches!(
            quant(json!({ "mlxQuantize": "8" })),
            Some(Quant::Q8)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": " 4 " })),
            Some(Quant::Q4)
        ));
        assert!(quant(json!({ "mlxQuantize": "0" })).is_none());
        // A bits value between 4 and 8 rounds to the nearest supported tier (> 4 ⇒ Q8).
        assert!(matches!(
            quant(json!({ "mlxQuantize": 6 })),
            Some(Quant::Q8)
        ));
    }

    /// Raw-settings lineage captured on a real Bernini asset (sc-4709): the real-inference marker,
    /// the catalog model id, the produced frame count / fps, the resolved engine guidance task
    /// (`berniniTask`), and a pass-through of the user's advanced controls.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_raw_settings_capture_lineage_and_task() {
        let req = request(json!({
            "projectId": "p",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "duration": 5,
            "fps": 16,
            "advanced": { "mlxQuantize": 8, "userKnob": "keep-me" }
        }));
        let raw = bernini_raw_settings(&req);
        let raw = raw.as_object().expect("raw settings is an object");
        assert_eq!(raw["realModelInference"], json!(true));
        assert_eq!(raw["model"], json!("bernini"));
        assert_eq!(raw["fps"], json!(req.fps));
        assert_eq!(raw["frameCount"], json!(req.frame_count()));
        // The SceneWorks mode resolved to its engine guidance task for observability/lineage.
        assert_eq!(raw["berniniTask"], json!("rv2v"));
        // The user's advanced controls survive verbatim (provenance).
        assert_eq!(raw["userKnob"], json!("keep-me"));
        assert_eq!(raw["mlxQuantize"], json!(8));
    }

    /// Bernini conditioning resolution enforces each editing/reference mode's required media
    /// (sc-4703 / sc-4709), failing loudly BEFORE any IO when it is missing — defense in depth
    /// behind the API-side validation. `text_to_video` needs none and resolves to empty
    /// conditioning. The guards fire before touching the api / job / disk, so a minimal snapshot
    /// suffices (mirrors `video_clip_conditioning_requires_ic_lora`).
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn bernini_conditioning_enforces_required_media() {
        let settings = Settings::from_env();
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-bernini-1",
            "type": "video_generate",
            "status": "preparing",
            "projectId": "p",
            "projectName": "P",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-14T00:00:00Z",
            "updatedAt": "2026-06-14T00:00:00Z"
        }))
        .expect("job snapshot");
        let resolve = |mode: &str, extra: Value| {
            let mut payload = json!({ "projectId": "p", "model": "bernini", "prompt": "go" });
            payload
                .as_object_mut()
                .unwrap()
                .insert("mode".to_owned(), json!(mode));
            for (k, v) in extra.as_object().cloned().unwrap_or_default() {
                payload.as_object_mut().unwrap().insert(k, v);
            }
            let req = request(payload);
            let api = &api;
            let settings = &settings;
            let job = &job;
            async move {
                resolve_bernini_conditioning(api, settings, job, &req, Path::new("/tmp/p")).await
            }
        };

        // text_to_video needs no source media → empty conditioning, no IO.
        let t2v = resolve("text_to_video", json!({}))
            .await
            .expect("t2v resolves");
        assert!(t2v.is_empty(), "t2v needs no conditioning");

        // video_to_video / reference_video_to_video require a source clip.
        let v2v_err = resolve("video_to_video", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            v2v_err.contains("source clip"),
            "v2v missing-clip error: {v2v_err}"
        );
        let rv2v_err = resolve("reference_video_to_video", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            rv2v_err.contains("source clip"),
            "rv2v missing-clip error: {rv2v_err}"
        );

        // reference_to_video requires at least one reference image.
        let r2v_err = resolve("reference_to_video", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            r2v_err.contains("reference image"),
            "r2v missing-refs error: {r2v_err}"
        );
    }

    /// Real in-process Bernini editing/reference/multi-source video modes through the engine
    /// (sc-4703 / sc-4709 / sc-5425): drive v2v (synthetic source clip → `VideoClip`), r2v
    /// (synthetic reference → `MultiReference`), rv2v (both), mv2v (two source clips), and ads2v
    /// (source clip + reference clip + reference), asserting each mode loads, consumes its
    /// conditioning, and streams RGB8 frames. `#[ignore]` — weights live outside CI; run on a Mac
    /// with the `SceneWorks/bernini-mlx` snapshot.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
    #[test]
    fn bernini_editing_reference_modes_real_weights() {
        let Some(model_dir) = bernini_dir() else {
            eprintln!("skipping bernini_editing_reference_modes_real_weights: no snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        let frame = |shade: u8| Image {
            width: w,
            height: h,
            pixels: vec![shade; (w * h * 3) as usize],
        };
        // Two 5-frame synthetic source clips and one synthetic subject reference.
        let clip: Vec<Image> = (0..5).map(|i| frame(60 + i * 12)).collect();
        let clip_b: Vec<Image> = (0..5).map(|i| frame(40 + i * 16)).collect();
        let reference = frame(150);
        let video_clip = |frames: Vec<Image>| Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        };

        let cases: &[(&str, Vec<Conditioning>)] = &[
            ("v2v", vec![video_clip(clip.clone())]),
            (
                "r2v",
                vec![Conditioning::MultiReference {
                    images: vec![reference.clone()],
                }],
            ),
            (
                "rv2v",
                vec![
                    video_clip(clip.clone()),
                    Conditioning::MultiReference {
                        images: vec![reference.clone()],
                    },
                ],
            ),
            // mv2v: two source clips, no references.
            (
                "mv2v",
                vec![video_clip(clip.clone()), video_clip(clip_b.clone())],
            ),
            // ads2v: source clip + reference clip + reference image (videos first, then images).
            (
                "ads2v",
                vec![
                    video_clip(clip.clone()),
                    video_clip(clip_b.clone()),
                    Conditioning::MultiReference {
                        images: vec![reference.clone()],
                    },
                ],
            ),
        ];

        for (task, conditioning) in cases {
            let input = VideoGenInput {
                engine_id: "bernini",
                model_dir: model_dir.clone(),
                quant: Some(Quant::Q4),
                conditioning: conditioning.clone(),
                prompt: "the subject walks through a neon-lit street, cinematic".to_owned(),
                width: w,
                height: h,
                frames: 5,
                fps: 16,
                steps: Some(8),
                seed: 11,
                video_mode: Some((*task).to_owned()),
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|error| panic!("bernini {task} generation: {error}"));
            assert!(steps > 0, "{task}: denoise progress streamed");
            assert!(!decoded.frames.is_empty(), "{task}: frames returned");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{task}: frames are RGB8-sized"
            );
        }
    }

    // ===================== SCAIL-2 validation (epic 5439 / sc-5450) =====================

    /// Only the `scail2_14b` catalog id routes to the SCAIL-2 engine (sc-5448). Every other video
    /// family (Wan/LTX/SVD/Bernini/image ids) is `None`, so they keep their own routing.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_engine_id_maps_only_the_scail2_family() {
        assert_eq!(scail2_engine_id("scail2_14b"), Some("scail2_14b"));
        assert_eq!(scail2_engine_id("bernini"), None);
        assert_eq!(scail2_engine_id("wan_2_2"), None);
        assert_eq!(scail2_engine_id("ltx_2_3"), None);
        assert_eq!(scail2_engine_id("scail2"), None);
        assert_eq!(scail2_engine_id(""), None);
    }

    /// SCAIL-2 load quantization (sc-5450): Q4 is the default (the validated ~16 GB tier),
    /// `mlxQuantize` opts up to Q8 or down to bf16 (`<= 0`), parsing a JSON number or string. The
    /// bf16 snapshot is ~47 GB so a missing control NEVER means bf16. Mirrors the Bernini quant test.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_resolve_quant_defaults_q4_and_honors_override() {
        let quant = |advanced: Value| {
            resolve_scail2_quant(&request(json!({ "projectId": "p", "advanced": advanced })))
        };
        assert!(matches!(quant(json!({})), Some(Quant::Q4)));
        assert!(matches!(
            quant(json!({ "mlxQuantize": 4 })),
            Some(Quant::Q4)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": 8 })),
            Some(Quant::Q8)
        ));
        assert!(quant(json!({ "mlxQuantize": 0 })).is_none());
        assert!(quant(json!({ "mlxQuantize": -1 })).is_none());
        // String forms parse the same; a tier between 4 and 8 rounds up to Q8.
        assert!(matches!(
            quant(json!({ "mlxQuantize": "8" })),
            Some(Quant::Q8)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": " 4 " })),
            Some(Quant::Q4)
        ));
        assert!(quant(json!({ "mlxQuantize": "0" })).is_none());
        assert!(matches!(
            quant(json!({ "mlxQuantize": 6 })),
            Some(Quant::Q8)
        ));
    }

    /// Raw-settings lineage on a real SCAIL-2 asset (sc-5450): the real-inference marker, the catalog
    /// model id, the produced frame count / fps, the resolved engine task (`scail2Task`), and a
    /// pass-through of the user's advanced controls. `replace_person` → "replacement",
    /// `animate_character` → "animation".
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_raw_settings_capture_lineage_and_task() {
        let raw_for = |mode: &str| {
            let req = request(json!({
                "projectId": "p",
                "model": "scail2_14b",
                "mode": mode,
                "duration": 5,
                "fps": 16,
                "advanced": { "mlxQuantize": 4, "userKnob": "keep-me" }
            }));
            let raw = scail2_raw_settings(&req, false);
            (raw, req)
        };
        let (raw, req) = raw_for("animate_character");
        let raw = raw.as_object().expect("raw settings is an object");
        assert_eq!(raw["realModelInference"], json!(true));
        assert_eq!(raw["model"], json!("scail2_14b"));
        assert_eq!(raw["fps"], json!(req.fps));
        assert_eq!(raw["frameCount"], json!(req.frame_count()));
        assert_eq!(raw["scail2Task"], json!("animation"));
        assert_eq!(raw["userKnob"], json!("keep-me"));
        // No lightning LoRA ⇒ no effective-recipe override recorded (engine quality defaults stand).
        assert!(raw.get("scail2Lightning").is_none());
        assert!(raw.get("effectiveSteps").is_none());
        // replace_person resolves to the replacement task (flips the engine replace_flag).
        let (raw_replace, _) = raw_for("replace_person");
        assert_eq!(raw_replace["scail2Task"], json!("replacement"));
    }

    /// The lightx2v lightning toggle (sc-5700): selecting a diff-patch LoRA applies the step-distill
    /// recipe — 8 steps, CFG off (guidance 1.0), scheduler shift 1.0 — with an explicit user step
    /// count still honored, and the effective recipe is recorded on the asset. Without lightning the
    /// sampling is all-`None` (the engine's quality defaults stand, unchanged).
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_lightning_recipe_overrides_sampling_and_records() {
        let req = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "animate_character",
            "duration": 5, "fps": 16, "advanced": {}
        }));
        // Non-lightning: untouched (engine defaults).
        assert_eq!(scail2_sampling(&req, false), (None, None, None));
        // Lightning: CFG off + shift 1.0 forced, steps default 8.
        assert_eq!(
            scail2_sampling(&req, true),
            (Some(8), Some(1.0), Some(1.0)),
            "lightning applies the 8-step CFG-off recipe"
        );
        // An explicit user step count is honored; CFG/shift stay forced.
        let req_steps = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "animate_character",
            "duration": 5, "fps": 16, "advanced": { "steps": 4 }
        }));
        assert_eq!(
            scail2_sampling(&req_steps, true),
            (Some(4), Some(1.0), Some(1.0))
        );
        // The effective recipe is recorded for observability when lightning is active.
        let raw = scail2_raw_settings(&req, true);
        let raw = raw.as_object().unwrap();
        assert_eq!(raw["scail2Lightning"], json!(true));
        assert_eq!(raw["effectiveSteps"], json!(8));
        assert_eq!(raw["effectiveGuidanceScale"], json!(1.0));
        assert_eq!(raw["effectiveSchedulerShift"], json!(1.0));
    }

    /// SCAIL-2 conditioning resolution fails loudly BEFORE any IO when its required media is missing
    /// (sc-5450 — defense in depth behind the API-side validation): `animate_character` needs a
    /// reference character image, and the integrated `replace_person` backend needs a person track.
    /// Both guards fire before touching the api / job / disk, so a minimal job suffices.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn scail2_conditioning_guards_fire_before_io() {
        let settings = Settings::from_env();
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-scail2-1",
            "type": "video_generate",
            "status": "preparing",
            "projectId": "p",
            "projectName": "P",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-14T00:00:00Z",
            "updatedAt": "2026-06-14T00:00:00Z"
        }))
        .expect("job snapshot");

        // animate_character with no reference image / source asset → reference guard.
        let animate_req = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "animate_character", "prompt": "go"
        }));
        let animate_err =
            resolve_scail2_conditioning(&api, &settings, &job, &animate_req, Path::new("/tmp/p"))
                .await
                .unwrap_err()
                .to_string();
        assert!(
            animate_err.contains("reference character image"),
            "animate missing-reference error: {animate_err}"
        );

        // replace_person with no person track → person-track guard.
        let replace_req = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "replace_person", "prompt": "go"
        }));
        let replace_err = resolve_scail2_replace_conditioning(
            &api,
            &settings,
            &job,
            &replace_req,
            Path::new("/tmp/p"),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(
            replace_err.contains("person track"),
            "replace missing-track error: {replace_err}"
        );
    }

    /// A SCAIL-2 MLX snapshot dir if present: env override (`SCENEWORKS_MLX_SCAIL2_DIR`), then the
    /// bring-up convert/staging dirs, then the app-managed default. `None` ⇒ the smoke skips.
    #[cfg(target_os = "macos")]
    fn scail2_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_SCAIL2_DIR") {
            let path = PathBuf::from(dir.trim());
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        for rel in [
            ".cache/scail2-mlx-convert",
            ".cache/mlx-gen-models/scail2-mlx-upload",
            "Library/Application Support/SceneWorks/data/models/mlx/scail2",
        ] {
            let path = PathBuf::from(&home).join(rel);
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        None
    }

    /// A synthetic SCAIL-2 color-coded mask: a centered blue rectangle (person 0) on a solid `bg`,
    /// matching the engine's 7-class color scheme (only the chromatic channels matter, not the exact
    /// silhouette — the smoke proves the engine consumes the shape and streams frames, not parity).
    #[cfg(target_os = "macos")]
    fn scail2_smoke_color_mask(w: u32, h: u32, bg: [u8; 3]) -> Image {
        let (wu, hu) = (w as usize, h as usize);
        let mut px = vec![0u8; wu * hu * 3];
        for chunk in px.chunks_exact_mut(3) {
            chunk.copy_from_slice(&bg);
        }
        for y in (hu / 4)..(3 * hu / 4) {
            for x in (wu / 4)..(3 * wu / 4) {
                let o = (y * wu + x) * 3;
                px[o..o + 3].copy_from_slice(&crate::scail2_masks::PALETTE[0]);
            }
        }
        Image {
            width: w,
            height: h,
            pixels: px,
        }
    }

    /// Real in-process SCAIL-2 through the WORKER registry path (epic 5439 / sc-5450): drive both
    /// engine tasks — `animation` (standalone `animate_character`) and `replacement` (cross-identity
    /// `replace_person`) — with a synthetic reference + color mask + driving clip + per-frame color
    /// masks, asserting each `video_mode` loads via `gen_core::load("scail2_14b")` (proving the
    /// `mlx_gen_scail2` force-link survived in the worker binary — the "no generator registered"
    /// trap), consumes the conditioning, and streams RGB8 frames with denoise progress. The driving
    /// background follows the replacement convention (animation → black, replacement → white).
    /// `#[ignore]` — the ~47 GB snapshot lives outside CI; run manually on a Mac where it is present
    /// (Q4 ≈ 16 GB peak). The bg conventions mirror `scail2_masks` (sc-5448 / sc-5452).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real SCAIL-2 snapshot (~47 GB); run manually on a Mac where the scail2 weights are present"]
    #[test]
    fn scail2_animation_and_replacement_real_weights() {
        let Some(model_dir) = scail2_dir() else {
            eprintln!("skipping scail2_animation_and_replacement_real_weights: no snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        let frame = |shade: u8| Image {
            width: w,
            height: h,
            pixels: vec![shade; (w * h * 3) as usize],
        };
        // 5 driving frames → one clean VAE-aligned segment (the engine keeps ((T-1)/4)*4 + 1).
        let driving: Vec<Image> = (0..5).map(|i| frame(50 + i * 20)).collect();
        let reference = frame(160);

        // animation keeps the reference's world (driving bg black, ref bg white); cross-identity
        // replacement keeps the driving's world (driving bg white, ref bg black) — sc-5448 / sc-5452.
        for (task, driving_bg, reference_bg) in [
            (
                "animation",
                crate::scail2_masks::BG_BLACK,
                crate::scail2_masks::BG_WHITE,
            ),
            (
                "replacement",
                crate::scail2_masks::BG_WHITE,
                crate::scail2_masks::BG_BLACK,
            ),
        ] {
            let reference_mask = scail2_smoke_color_mask(w, h, reference_bg);
            let driving_masks: Vec<Image> = driving
                .iter()
                .map(|_| scail2_smoke_color_mask(w, h, driving_bg))
                .collect();
            let conditioning = vec![
                Conditioning::Reference {
                    image: reference.clone(),
                    strength: None,
                },
                Conditioning::Mask {
                    image: reference_mask,
                },
                Conditioning::ControlClip {
                    frames: driving.clone(),
                    mask: driving_masks,
                    masking_strength: 1.0,
                    start_frame: 0,
                    mode: ReplacementMode::default(),
                },
            ];
            let input = VideoGenInput {
                engine_id: "scail2_14b",
                model_dir: model_dir.clone(),
                quant: Some(Quant::Q4),
                conditioning,
                prompt: "the character walks forward, cinematic".to_owned(),
                width: w,
                height: h,
                frames: 5,
                fps: 16,
                steps: Some(8),
                seed: 11,
                video_mode: Some(task.to_owned()),
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|error| panic!("scail2 {task} generation: {error}"));
            assert!(steps > 0, "{task}: denoise progress streamed");
            assert!(!decoded.frames.is_empty(), "{task}: frames returned");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{task}: frames are RGB8-sized"
            );
        }
    }

    /// Real in-process Wan-VACE extend/bridge through the engine (sc-3812): build the tier-C
    /// control clip (real anchor frames pinned + a neutral generated span, no references) and run
    /// the assembled snapshot, asserting RGB8 frames stream back. `#[ignore]` — the weights live
    /// outside CI; run manually on a Mac with the snapshot assembled (the real-Mac gate; the A/B
    /// vs the TI2V-5B single-frame path is the practical fidelity judge, sc-3800).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan-VACE snapshot; run manually on a Mac with it assembled"]
    #[test]
    fn wan_vace_extend_bridge_real_weights() {
        let Some(model_dir) = wan_vace_dir() else {
            eprintln!("skipping wan_vace_extend_bridge_real_weights: no assembled snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        // Two distinct real "source" frames as the extend motion anchor; the engine generates the
        // remaining 3 frames of the 5-frame budget over the neutral span.
        let anchor = |v: u8| Image {
            width: w,
            height: h,
            pixels: vec![v; (w * h * 3) as usize],
        };
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip",
            "prompt": "the camera keeps gliding forward, cinematic",
            "sourceClipAssetId": "clip_a"
        }));
        let conditioning = build_extend_bridge_vace_conditioning(
            &req,
            w,
            h,
            5,
            vec![anchor(90), anchor(110)],
            None,
        )
        .expect("extend conditioning builds");
        let input = VideoGenInput {
            engine_id: "wan_vace",
            model_dir,
            conditioning,
            prompt: req.prompt.clone(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 11,
            control_scale: Some(1.0),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("VACE extend generation");
        assert_eq!(decoded.frames.len(), 5);
        assert!(decoded.audio.is_none(), "Wan-VACE emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// Wan model-id → engine-id mapping + the family predicates that drive routing.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_engine_id_maps_the_three_models() {
        assert_eq!(wan_engine_id("wan_2_2"), Some("wan2_2_ti2v_5b"));
        assert_eq!(wan_engine_id("wan_2_2_t2v_14b"), Some("wan2_2_t2v_14b"));
        assert_eq!(wan_engine_id("wan_2_2_i2v_14b"), Some("wan2_2_i2v_14b"));
        assert_eq!(wan_engine_id("ltx_2_3"), None);
        assert_eq!(wan_engine_id("z_image_turbo"), None);
    }

    /// Per-model sampling (sc-4997): both A14B MoE models (T2V + I2V) force the 4-step
    /// Lightning preset (CFG off); the dense 5B honors an explicit user `steps`/`guidanceScale`
    /// and otherwise applies the interim default with CFG retained.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_sampling_overrides_both_14b_and_5b_interim() {
        let req = request(json!({ "projectId": "p" }));
        // Both A14B MoE models: forced 4-step / guide 1.0 (Lightning baked).
        assert_eq!(wan_sampling("wan2_2_t2v_14b", &req), (Some(4), Some(1.0)));
        assert_eq!(wan_sampling("wan2_2_i2v_14b", &req), (Some(4), Some(1.0)));
        // 5B, no override → interim default steps, CFG left to the engine (None ⇒ guide 5.0).
        assert_eq!(
            wan_sampling("wan2_2_ti2v_5b", &req),
            (Some(WAN5B_INTERIM_STEPS), None)
        );
        // 5B honors an explicit user steps + guidanceScale (e.g. the ComfyUI fast settings).
        let over = request(json!({
            "projectId": "p", "advanced": { "steps": 6, "guidanceScale": 1.0 }
        }));
        assert_eq!(wan_sampling("wan2_2_ti2v_5b", &over), (Some(6), Some(1.0)));
        // The 14B Lightning preset ignores user overrides — the distill is mandatory.
        assert_eq!(wan_sampling("wan2_2_t2v_14b", &over), (Some(4), Some(1.0)));
        assert_eq!(wan_sampling("wan2_2_i2v_14b", &over), (Some(4), Some(1.0)));
    }

    /// `advanced.mlxQuantize` maps to a quant level; absent → dense / engine-resolved.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_quant_maps_mlx_quantize() {
        let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(resolve_wan_quant(&q4), Some(Quant::Q4));
        let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(resolve_wan_quant(&q8), Some(Quant::Q8));
        let dense = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(resolve_wan_quant(&dense), None);
        let absent = request(json!({ "projectId": "p" }));
        assert_eq!(resolve_wan_quant(&absent), None);
    }

    /// The `.high_noise.safetensors` → `.low_noise.safetensors` sibling convention
    /// (case-insensitive; only fires when the sibling file exists).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_moe_sibling_pairs_high_and_low() {
        let dir = std::env::temp_dir().join(format!("sw_moe_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let high = dir.join("char.high_noise.safetensors");
        let low = dir.join("char.low_noise.safetensors");
        std::fs::write(&high, b"x").unwrap();
        // No sibling yet → None.
        assert_eq!(wan_moe_low_noise_sibling(&high), None);
        std::fs::write(&low, b"x").unwrap();
        assert_eq!(wan_moe_low_noise_sibling(&high), Some(low));
        // A single-file (non-high-noise) LoRA never pairs.
        let single = dir.join("plain.safetensors");
        std::fs::write(&single, b"x").unwrap();
        assert_eq!(wan_moe_low_noise_sibling(&single), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Minimal valid safetensors (8-byte LE header length + JSON header), optionally stamping
    /// `__metadata__.networkType` so `classify_adapter` can distinguish peft LoKr from plain LoRA.
    #[cfg(target_os = "macos")]
    fn write_lora_fixture(path: &Path, network_type: Option<&str>) {
        let mut meta = serde_json::Map::new();
        meta.insert("format".to_owned(), json!("pt"));
        if let Some(nt) = network_type {
            meta.insert("networkType".to_owned(), json!(nt));
        }
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), Value::Object(meta));
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        std::fs::write(path, buffer).unwrap();
    }

    /// Wan-VACE is single-dense: each user LoRA/LoKr resolves to one shared spec with
    /// `moe_expert: None` (no Lightning, no high/low split), the kind set by the file's metadata,
    /// and the scale taken from the request `weight` (sc-3893).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_vace_adapters_are_single_dense() {
        let dir = std::env::temp_dir().join(format!("sw_vace_lora_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("style.safetensors");
        let lokr = dir.join("char.safetensors");
        write_lora_fixture(&plain, None);
        write_lora_fixture(&lokr, Some("lokr"));

        let req = request(json!({
            "projectId": "p",
            "loras": [
                { "path": plain.to_string_lossy(), "weight": 0.5 },
                { "path": lokr.to_string_lossy(), "weight": 0.9 },
            ],
        }));
        // sc-5723: LoRA paths are confined to the app data dir, so point data_dir at
        // the fixture dir the temp LoRAs live in.
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };
        let specs = resolve_wan_vace_adapters(&settings, &req).expect("resolve vace adapters");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].path, plain.canonicalize().unwrap());
        assert_eq!(specs[0].kind, AdapterKind::Lora);
        assert!((specs[0].scale - 0.5).abs() < 1e-6);
        assert!(specs[0].moe_expert.is_none(), "VACE is single-dense");
        assert!(specs[0].pass_scales.is_none());
        assert_eq!(specs[1].kind, AdapterKind::Lokr);
        assert!((specs[1].scale - 0.9).abs() < 1e-6);
        assert!(specs[1].moe_expert.is_none());

        // Over the per-job cap → a clear payload error (mirrors the base Wan path).
        let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
            .map(|_| json!({ "path": plain.to_string_lossy() }))
            .collect();
        let over = request(json!({ "projectId": "p", "loras": many }));
        assert!(matches!(
            resolve_wan_vace_adapters(&settings, &over),
            Err(WorkerError::InvalidPayload(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-5723 (WKA-002): a LoRA path from the (attacker-controllable) payload that
    /// resolves outside every app-managed root is rejected before the file is opened —
    /// the worker must not be pointed at an arbitrary host `.safetensors`. The fixture
    /// lives in a sibling temp dir that is NOT under the configured `data_dir`.
    #[cfg(target_os = "macos")]
    #[test]
    fn lora_path_outside_app_managed_root_is_rejected() {
        let data_dir =
            std::env::temp_dir().join(format!("sw_lora_data_{}", Uuid::new_v4().simple()));
        let outside =
            std::env::temp_dir().join(format!("sw_lora_evil_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let evil = outside.join("evil.safetensors");
        write_lora_fixture(&evil, None);

        let settings = Settings {
            data_dir: data_dir.clone(),
            ..Settings::from_env()
        };
        let req = request(json!({
            "projectId": "p",
            "loras": [ { "path": evil.to_string_lossy(), "weight": 0.5 } ],
        }));
        // The HF cache roots are env-derived; this temp path is under neither, so it
        // must be refused. (Guard against the host's real HF cache happening to be a
        // parent — vanishingly unlikely for a fresh uuid temp dir.)
        let result = resolve_wan_vace_adapters(&settings, &req);
        assert!(
            matches!(result, Err(WorkerError::InvalidPayload(_))),
            "expected out-of-root LoRA to be rejected, got {result:?}"
        );

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// SCAIL-2 is single-dense like Wan-VACE (sc-5451/sc-5686): each user LoRA/LoKr (and the bundled
    /// Bias-Aware DPO LoRA, which arrives the same way) resolves to one shared spec with
    /// `moe_expert: None` and no `pass_scales`, the kind set by the file's metadata, scale from
    /// `weight`; over the per-job cap is a clear payload error.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_adapters_are_single_dense() {
        let dir = std::env::temp_dir().join(format!("sw_scail2_lora_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("dpo.safetensors");
        let lokr = dir.join("char.safetensors");
        write_lora_fixture(&plain, None);
        write_lora_fixture(&lokr, Some("lokr"));

        let req = request(json!({
            "projectId": "p",
            "loras": [
                { "path": plain.to_string_lossy(), "weight": 1.0 },
                { "path": lokr.to_string_lossy(), "weight": 0.7 },
            ],
        }));
        // sc-5723: LoRA paths are confined to the app data dir, so point data_dir at
        // the fixture dir the temp LoRAs live in.
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };
        let specs = resolve_scail2_adapters(&settings, &req).expect("resolve scail2 adapters");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].path, plain.canonicalize().unwrap());
        assert_eq!(specs[0].kind, AdapterKind::Lora);
        assert!((specs[0].scale - 1.0).abs() < 1e-6);
        assert!(specs[0].moe_expert.is_none(), "SCAIL-2 is single-dense");
        assert!(specs[0].pass_scales.is_none());
        assert_eq!(specs[1].kind, AdapterKind::Lokr);
        assert!((specs[1].scale - 0.7).abs() < 1e-6);
        assert!(specs[1].moe_expert.is_none());

        let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
            .map(|_| json!({ "path": plain.to_string_lossy() }))
            .collect();
        let over = request(json!({ "projectId": "p", "loras": many }));
        assert!(matches!(
            resolve_scail2_adapters(&settings, &over),
            Err(WorkerError::InvalidPayload(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A locally-converted Wan2.2 TI2V-5B dir if one is present (env override or the
    /// app-managed default), else `None` so the real-weight smoke skips.
    #[cfg(target_os = "macos")]
    fn wan_5b_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_WAN5B_DIR") {
            let path = PathBuf::from(dir.trim());
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2");
        path.join("config.json").is_file().then_some(path)
    }

    /// Real in-process Wan2.2 TI2V-5B T2V through the engine (the lightest Wan model —
    /// ~10 GB, safe on a 128 GB Mac). Loads the converted 5B snapshot and denoises a
    /// tiny 5-frame clip, asserting frames come back RGB8-sized with streamed progress.
    /// `#[ignore]` — the weights live outside CI; run manually where they are present.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan2.2 TI2V-5B weights; run manually on a Mac with them present"]
    #[test]
    fn wan_5b_real_weights() {
        let Some(model_dir) = wan_5b_dir() else {
            eprintln!("skipping wan_5b_real_weights: no converted TI2V-5B dir found");
            return;
        };
        let input = VideoGenInput {
            engine_id: "wan2_2_ti2v_5b",
            model_dir,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width: 256,
            height: 256,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("5B T2V generation");
        assert_eq!(decoded.frames.len(), 5, "5 frames (1 + 4·1)");
        assert!(decoded.fps >= 1);
        assert!(decoded.audio.is_none(), "Wan emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// Real-weight image→video fit smoke (sc-6139): proves the Crop/Pad pre-fit reaches the
    /// engine and the engine renders the chosen off-aspect output without re-stretching. A
    /// solid-color SQUARE source is fit to a 448×256 landscape via the SAME `fit_engine_image`
    /// the I2V resolve paths call — `pad` letterboxes (black side bars), `crop` fills+trims —
    /// asserted deterministically; then the pad-fit reference drives a tiny TI2V-5B I2V clip and
    /// the decoded frames are asserted to be exactly 448×256 (a re-stretch would change the
    /// aspect). Frame 0 is dumped to `$TMPDIR/sc6139_i2v_pad_frame0.png` for a visual bar check.
    /// `#[ignore]` — needs the converted TI2V-5B weights; run manually on a Mac where present.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan2.2 TI2V-5B weights; run manually on a Mac with them present"]
    #[test]
    fn image_to_video_fit_real_weights() {
        // Off-aspect output: a square source must letterbox (pad) or fill+trim (crop).
        const OUT_W: u32 = 448; // 14·32
        const OUT_H: u32 = 256; // 8·32
                                // A solid non-black square so pad's black bars are unambiguous against the content.
        let mk_square = || Image {
            width: 256,
            height: 256,
            pixels: [200u8, 80, 40].repeat((256 * 256) as usize),
        };
        let is_black_col = |img: &Image, x: u32| -> bool {
            (0..img.height).all(|y| {
                let i = ((y * img.width + x) * 3) as usize;
                img.pixels[i] == 0 && img.pixels[i + 1] == 0 && img.pixels[i + 2] == 0
            })
        };

        // PAD: contained (fit height), centered → black bars on the left/right edges.
        let padded =
            crate::image_jobs::fit_engine_image(mk_square(), OUT_W, OUT_H, "pad").expect("pad fit");
        assert_eq!((padded.width, padded.height), (OUT_W, OUT_H));
        assert!(is_black_col(&padded, 0), "pad: left edge is a black bar");
        assert!(
            is_black_col(&padded, OUT_W - 1),
            "pad: right edge is a black bar"
        );
        let ci = ((OUT_H / 2 * OUT_W + OUT_W / 2) * 3) as usize;
        assert_eq!(
            &padded.pixels[ci..ci + 3],
            &[200, 80, 40],
            "pad: center keeps the source color"
        );

        // CROP: covered (fit width), center-cropped → fully filled, no black border.
        let cropped = crate::image_jobs::fit_engine_image(mk_square(), OUT_W, OUT_H, "crop")
            .expect("crop fit");
        assert_eq!((cropped.width, cropped.height), (OUT_W, OUT_H));
        assert!(
            !is_black_col(&cropped, 0),
            "crop: left edge is filled, not a bar"
        );
        assert!(
            !is_black_col(&cropped, OUT_W - 1),
            "crop: right edge is filled, not a bar"
        );

        // Real-weight run: the pad-fit reference drives a tiny TI2V-5B I2V clip. If the engine
        // re-stretched the reference it could not land at the output aspect; asserting the decoded
        // frames are exactly OUT_W×OUT_H confirms the pre-fit reference is consumed as-is.
        let Some(model_dir) = wan_5b_dir() else {
            eprintln!("skipping image_to_video_fit_real_weights: no converted TI2V-5B dir found");
            return;
        };
        let input = VideoGenInput {
            engine_id: "wan2_2_ti2v_5b",
            model_dir,
            conditioning: vec![Conditioning::Reference {
                image: padded,
                strength: None,
            }],
            prompt: "a slow gentle camera move, cinematic".to_owned(),
            width: OUT_W,
            height: OUT_H,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .expect("TI2V-5B I2V generation from a pad-fit square reference");
        assert_eq!(decoded.frames.len(), 5, "5 frames (1 + 4·1)");
        assert!(steps > 0, "denoise progress streamed");
        for f in &decoded.frames {
            assert_eq!(
                (f.width, f.height),
                (OUT_W, OUT_H),
                "engine renders the requested off-aspect output (no re-stretch)"
            );
            assert_eq!(f.pixels.len(), (f.width * f.height * 3) as usize);
        }
        // Dump frame 0 so the pad letterbox is visually verifiable.
        let frame0 = &decoded.frames[0];
        if let Some(buf) =
            image::RgbImage::from_raw(frame0.width, frame0.height, frame0.pixels.clone())
        {
            let out = std::env::temp_dir().join("sc6139_i2v_pad_frame0.png");
            let _ = buf.save(&out);
            eprintln!("wrote pad-fit I2V frame 0 to {}", out.display());
        }
    }

    /// Real-weight perf probe for the TI2V-5B (sc-4997): measures the load / DiT-denoise /
    /// VAE-decode wall-clock split at a configurable resolution, frame count, step count, and
    /// CFG — to ground the "under 10 min" target on real numbers instead of estimates. Env-driven
    /// so configs run without recompiling. MUST be `--release` (debug MLX timing is meaningless):
    ///   SCENEWORKS_MLX_WAN5B_DIR=~/.cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16 \
    ///   WAN_TIMING_W=1280 WAN_TIMING_H=720 WAN_TIMING_FRAMES=121 WAN_TIMING_STEPS=20 WAN_TIMING_CFG=on \
    ///   cargo test -p sceneworks-worker --release --lib wan_5b_timing -- --ignored --nocapture
    /// CFG: `off`/`1` ⇒ guide 1.0 (no CFG); a number ⇒ that scale; `on`/unset ⇒ engine default (5.0).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight perf probe; needs the converted TI2V-5B snapshot + a Metal device"]
    fn wan_5b_timing() {
        use std::time::Instant;
        let Some(model_dir) = wan_5b_dir() else {
            eprintln!("skipping wan_5b_timing: no converted TI2V-5B dir found");
            return;
        };
        let env_u32 = |k: &str, d: u32| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(d)
        };
        let width = env_u32("WAN_TIMING_W", 1280);
        let height = env_u32("WAN_TIMING_H", 720);
        let frames = env_u32("WAN_TIMING_FRAMES", 121);
        let steps = env_u32("WAN_TIMING_STEPS", 20);
        let guidance = match std::env::var("WAN_TIMING_CFG")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("off") | Some("1") | Some("1.0") => Some(1.0_f32),
            Some("on") | Some("") | None => None,
            Some(other) => other.parse().ok(),
        };
        // Optional MLX memory-limit override (GB): lowers `get_memory_limit()` so the z48 vae22
        // decode tile planner (`auto_tiling_budgeted`) picks SMALLER tiles — to test whether a
        // smaller decode working-set is faster than the default "largest tile under budget" (sc-5089).
        let mem_limit_gb = env_u32("WAN_TIMING_MEM_LIMIT_GB", 0);
        if mem_limit_gb > 0 {
            let prev =
                mlx_rs::memory::set_memory_limit((mem_limit_gb as usize) * 1024 * 1024 * 1024);
            eprintln!(
                "WAN5B_TIMING mem_limit set to {mem_limit_gb} GB (was {:.0} GB)",
                prev as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        mlx_rs::memory::reset_peak_memory();
        let input = VideoGenInput {
            engine_id: "wan2_2_ti2v_5b",
            model_dir,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width,
            height,
            frames,
            fps: 24,
            steps: Some(steps),
            guidance,
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let start = Instant::now();
        let mut first_step: Option<Instant> = None;
        let mut last_step: Option<Instant> = None;
        let mut decode_at: Option<Instant> = None;
        let mut dit_peak_bytes: Option<usize> = None;
        let mut on_progress = |progress: Progress| match progress {
            Progress::Step { .. } => {
                let now = Instant::now();
                first_step.get_or_insert(now);
                last_step = Some(now);
            }
            Progress::Decoding => {
                decode_at.get_or_insert(Instant::now());
                // Peak so far = the DiT-denoise peak (the VAE decode is the *next* stage).
                dit_peak_bytes.get_or_insert(mlx_rs::memory::get_peak_memory());
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("5B generation");
        let end = Instant::now();
        let total_peak_bytes = mlx_rs::memory::get_peak_memory();
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let secs = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64();
        let fs = first_step.unwrap_or(start);
        let dec = decode_at.or(last_step).unwrap_or(end);
        eprintln!(
            "WAN5B_TIMING {width}x{height} frames={frames} steps={steps} cfg={} => \
             total={:.1}s load={:.1}s dit={:.1}s vae={:.1}s | peak_dit={:.0}GB peak_total={:.0}GB \
             out_frames={}",
            guidance.map_or_else(|| "5.0(default)".to_owned(), |g| format!("{g}")),
            secs(start, end),
            secs(start, fs),
            secs(fs, dec),
            secs(dec, end),
            gib(dit_peak_bytes.unwrap_or(0)),
            gib(total_peak_bytes),
            decoded.frames.len(),
        );
    }

    /// LTX model-id → engine-id mapping (both base + eros load the one engine model).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_engine_id_maps_base_and_eros() {
        assert_eq!(ltx_engine_id("ltx_2_3"), Some("ltx_2_3"));
        assert_eq!(ltx_engine_id("ltx_2_3_eros"), Some("ltx_2_3"));
        assert_eq!(ltx_engine_id("wan_2_2"), None);
        assert_eq!(ltx_engine_id("z_image_turbo"), None);
    }

    /// SceneWorks bundle resolution (sc-5608): `ltx_bundle_subdir` picks the requested quant subdir,
    /// prefers a complete one over an incomplete sibling, and `bundled_ltx_gemma_dir` finds `gemma/`.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_bundle_subdir_picks_quant_and_finds_gemma() {
        fn write_complete_ltx_dir(dir: &Path) {
            std::fs::create_dir_all(dir).unwrap();
            for file in [
                "connector.safetensors",
                "transformer.safetensors",
                "upsampler.safetensors",
                "vae_decoder.safetensors",
                "vae_encoder.safetensors",
                "audio_vae.safetensors",
                "vocoder.safetensors",
            ] {
                std::fs::write(dir.join(file), b"x").unwrap();
            }
        }
        let root = std::env::temp_dir().join(format!("sw_ltx_bundle_{}", Uuid::new_v4().simple()));
        let (q4, q8) = (root.join("q4"), root.join("q8"));
        write_complete_ltx_dir(&q4);
        write_complete_ltx_dir(&q8);
        std::fs::create_dir_all(root.join("gemma")).unwrap();

        // Default prefers q4; mlxQuantize: 8 prefers q8.
        assert_eq!(
            ltx_bundle_subdir(&root, false).as_deref(),
            Some(q4.as_path())
        );
        assert_eq!(
            ltx_bundle_subdir(&root, true).as_deref(),
            Some(q8.as_path())
        );

        // The gemma encoder is found as a sibling of the loaded quant dir.
        assert_eq!(
            bundled_ltx_gemma_dir(&q4).as_deref(),
            Some(root.join("gemma").as_path())
        );

        // An incomplete preferred subdir falls back to the complete sibling.
        std::fs::remove_file(q8.join("vocoder.safetensors")).unwrap();
        assert_eq!(
            ltx_bundle_subdir(&root, true).as_deref(),
            Some(q4.as_path())
        );

        // No complete subdir → None; no gemma sibling → None.
        let bare = std::env::temp_dir().join(format!("sw_ltx_bare_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(bare.join("q4")).unwrap();
        assert!(ltx_bundle_subdir(&bare, false).is_none());
        assert!(bundled_ltx_gemma_dir(&bare.join("q4")).is_none());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&bare);
    }

    /// Q8 opt-in detection (sc-5679): `advanced.mlxQuantize: 8` (int or string) → true; absent / Q4
    /// → false. Drives both the resolve quant preference and the on-demand q8 fetch.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_wants_q8_reads_mlx_quantize() {
        let with = |adv: Value| {
            request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x", "advanced": adv }))
        };
        assert!(ltx_wants_q8(&with(json!({ "mlxQuantize": 8 }))));
        assert!(ltx_wants_q8(&with(json!({ "mlxQuantize": "8" }))));
        assert!(!ltx_wants_q8(&with(json!({ "mlxQuantize": 4 }))));
        assert!(!ltx_wants_q8(&request(
            json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x" })
        )));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn svd_engine_id_maps_only_svd() {
        assert_eq!(svd_engine_id("svd"), Some("svd_xt"));
        assert_eq!(svd_engine_id("ltx_2_3"), None);
        assert_eq!(svd_engine_id("wan_2_2"), None);
        assert_eq!(svd_engine_id("svd_xt"), None);
    }

    /// SVD knobs resolve advanced → manifest entry → builtin default, then clamp; `conditioningFps`
    /// reads the `condFps` manifest key. Mirrors the torch `svd_video` `_pipeline_kwargs` mapping.
    #[cfg(target_os = "macos")]
    #[test]
    fn svd_knobs_resolve_advanced_over_manifest_over_default() {
        // Bare request → builtin defaults.
        let bare = request(json!({ "projectId": "p", "model": "svd", "sourceAssetId": "a" }));
        assert_eq!(svd_i32(&bare, "numFrames", "numFrames", 25, 1, 25), 25);
        assert_eq!(
            svd_i32(&bare, "motionBucketId", "motionBucketId", 127, 1, 255),
            127
        );
        assert_eq!(svd_i32(&bare, "conditioningFps", "condFps", 7, 1, 30), 7);
        assert_eq!(
            svd_i32(&bare, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
            8
        );
        assert_eq!(
            svd_f32(&bare, "noiseAugStrength", "noiseAugStrength", 0.02),
            0.02
        );
        assert_eq!(svd_steps(&bare), 25); // balanced

        // Manifest entry overrides the builtin default; advanced overrides the manifest.
        let layered = request(json!({
            "projectId": "p", "model": "svd", "sourceAssetId": "a", "quality": "fast",
            "modelManifestEntry": {
                "motionBucketId": 180, "condFps": 6, "noiseAugStrength": 0.1,
                "steps": { "fast": 12, "balanced": 25, "best": 30 }
            },
            "advanced": { "motionBucketId": 200, "decodeChunkSize": "16" }
        }));
        // advanced wins for motionBucketId; manifest wins for condFps + noiseAug.
        assert_eq!(
            svd_i32(&layered, "motionBucketId", "motionBucketId", 127, 1, 255),
            200
        );
        assert_eq!(svd_i32(&layered, "conditioningFps", "condFps", 7, 1, 30), 6);
        assert_eq!(
            svd_f32(&layered, "noiseAugStrength", "noiseAugStrength", 0.02),
            0.1
        );
        // numeric string parses.
        assert_eq!(
            svd_i32(&layered, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
            16
        );
        // steps from manifest's quality ladder (fast).
        assert_eq!(svd_steps(&layered), 12);

        // Out-of-range values clamp to the engine-safe bounds.
        let extreme = request(json!({
            "projectId": "p", "model": "svd", "sourceAssetId": "a",
            "advanced": { "motionBucketId": 999, "numFrames": 99, "decodeChunkSize": 0 }
        }));
        assert_eq!(
            svd_i32(&extreme, "motionBucketId", "motionBucketId", 127, 1, 255),
            255
        );
        assert_eq!(svd_i32(&extreme, "numFrames", "numFrames", 25, 1, 25), 25);
        assert_eq!(
            svd_i32(&extreme, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
            1
        );
    }

    /// Locate the cached SVD-XT diffusers snapshot (the stock HF repo the engine loads directly), or
    /// `None` if absent — `$SCENEWORKS_MLX_SVD_DIR` else the default HF hub cache.
    #[cfg(target_os = "macos")]
    fn svd_real_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_SVD_DIR") {
            let path = PathBuf::from(dir.trim());
            if svd_dir_is_complete(&path) {
                return Some(path);
            }
        }
        let snaps = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/huggingface/hub")
            .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
            .join("snapshots");
        std::fs::read_dir(&snaps)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| svd_dir_is_complete(path))
    }

    /// Real in-process SVD-XT image→video through the engine: load the stock diffusers snapshot,
    /// animate a synthetic reference image into a tiny clip, and assert the decode seam returns the
    /// requested RGB8 frames at the OUTPUT/playback fps (decoupled from the conditioning fps; sc-3764)
    /// with NO audio and streamed denoise progress. Exercises the worker's `run_video_generation`
    /// path (with the sc-3523 motion knobs) end to end. `#[ignore]` — the weights live outside CI.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real SVD-XT weights; run manually on a Mac with the checkpoint cached"]
    #[test]
    fn svd_real_weights_image_to_video() {
        let Some(model_dir) = svd_real_dir() else {
            eprintln!("skipping svd_real_weights_image_to_video: no SVD-XT checkpoint found");
            return;
        };
        let (w, h) = (64u32, 64u32);
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                pixels[i] = (x * 255 / w) as u8;
                pixels[i + 1] = (y * 255 / h) as u8;
                pixels[i + 2] = 128;
            }
        }
        let input = VideoGenInput {
            engine_id: "svd_xt",
            model_dir,
            width: 256,
            height: 256,
            frames: 4,
            // Playback fps (24) distinct from the conditioning fps (7) to prove the decouple (sc-3764).
            fps: 24,
            steps: Some(2),
            seed: 7,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: w,
                    height: h,
                    pixels,
                },
                strength: None,
            }],
            motion_bucket_id: Some(127.0),
            noise_aug_strength: Some(0.02),
            decode_chunk_size: Some(2),
            conditioning_fps: Some(7),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("svd i2v generation");
        assert_eq!(decoded.frames.len(), 4, "4 frames");
        assert_eq!(
            decoded.fps, 24,
            "output fps follows the playback fps, not the conditioning fps (sc-3764)"
        );
        assert!(decoded.audio.is_none(), "SVD emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// SeedVR2 video upscale (epic 4811 / sc-4816) end-to-end against the real 3B weights:
    /// drives the same `run_loaded_video_generation` path the `video_upscale` handler uses
    /// (a `VideoClip` of LR frames + a target size + `softness`), asserting an upscaled,
    /// frame-count-preserving clip comes back. `#[ignore]` — the ~7 GB checkpoint lives outside
    /// CI; run manually with `SCENEWORKS_SEEDVR2_DIR` pointed at a `numz/SeedVR2_comfyUI` snapshot
    /// (e.g. the HF cache dir), on a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real SeedVR2 3B weights (~7GB); run manually with SCENEWORKS_SEEDVR2_DIR set"]
    fn seedvr2_video_upscale_real_weights() {
        let dir = std::env::var("SCENEWORKS_SEEDVR2_DIR")
            .expect("set SCENEWORKS_SEEDVR2_DIR to a numz/SeedVR2_comfyUI checkpoint snapshot dir");
        // 8 low-res frames (a multiple of 4 ≥ 8 so the 4:1 causal-VAE compression preserves the
        // count) fed as the LR `VideoClip`; target = 2× (both ÷16).
        let frames: Vec<Image> = (0..8)
            .map(|i| Image {
                width: 64,
                height: 48,
                pixels: stub_video_rgb8(64, 48, 7, i, 8),
            })
            .collect();
        let input = VideoGenInput {
            engine_id: "seedvr2_3b",
            model_dir: PathBuf::from(dir),
            conditioning: vec![Conditioning::VideoClip {
                frames,
                frame_idx: 0,
                strength: 1.0,
            }],
            width: 128,
            height: 96,
            frames: 8,
            fps: 16,
            seed: 0,
            softness: Some(0.3),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let decoded = run_video_generation(input, &cancel, &mut |_| {})
            .expect("seedvr2 video upscale generation");
        assert_eq!(
            decoded.frames.len(),
            8,
            "frame count preserved (chunk multiple-of-4, ≥8)"
        );
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.width == 128 && f.height == 96),
            "every frame upscaled to the target 128x96"
        );
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "RGB8 buffers are well-formed"
        );
        assert!(
            decoded.audio.is_none(),
            "the engine emits no audio (worker muxes the source)"
        );
    }

    /// `advanced.noAudio` maps to the engine's `video_mode = "no_audio"`; enhance flags
    /// flow through. Asserts the LTX request build (the VideoGenInput, pre-load).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_advanced_flags_map_to_video_gen_input() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "advanced": { "noAudio": true, "enhancePrompt": true }
        }));
        assert!(advanced::bool(&req.advanced, "noAudio"));
        assert!(advanced::bool(&req.advanced, "enhancePrompt"));
        assert!(!advanced::bool(&req.advanced, "useUncensoredEnhancer"));
        // LTX adapters: a plain user LoRA is uniform (no per-pass schedule, no moe tag).
        let settings = Settings::from_env();
        let none = resolve_ltx_adapters(&settings, &req).unwrap();
        assert!(none.is_empty());
    }

    /// The FLF keyframe knobs (sc-3055) parse from JSON numbers + numeric strings and fall back
    /// to their defaults — these drive the two `Keyframe` strengths + the first keyframe's index.
    #[cfg(target_os = "macos")]
    #[test]
    fn advanced_numeric_helpers_parse_flf_keyframe_knobs() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "mode": "first_last_frame",
            "advanced": {
                "imageConditioningStrength": 0.8,        // JSON number
                "lastFrameConditioningStrength": "0.65",  // numeric string
                "imageFrameIndex": 2
            }
        }));
        assert_eq!(
            advanced::f32(&req.advanced, "imageConditioningStrength", 1.0),
            0.8
        );
        assert_eq!(
            advanced::f32(&req.advanced, "lastFrameConditioningStrength", 1.0),
            0.65
        );
        assert_eq!(advanced::i32(&req.advanced, "imageFrameIndex", 0), 2);
        // Absent keys → the fully-pinned defaults (strength 1.0, first index 0).
        let bare = request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "a fox" }));
        assert_eq!(
            advanced::f32(&bare.advanced, "imageConditioningStrength", 1.0),
            1.0
        );
        assert_eq!(
            advanced::f32(&bare.advanced, "lastFrameConditioningStrength", 1.0),
            1.0
        );
        assert_eq!(advanced::i32(&bare.advanced, "imageFrameIndex", 0), 0);
    }

    /// `resolve_keyframe_conditioning` fails clearly when an FLF source/last-frame asset id is
    /// missing (the guards run before any project/image IO, so no fixture is needed).
    #[cfg(target_os = "macos")]
    #[test]
    fn keyframe_conditioning_requires_both_frame_assets() {
        let settings = Settings::from_env();
        // No sourceAssetId.
        let no_first = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "mode": "first_last_frame", "lastFrameAssetId": "asset_last"
        }));
        let err = resolve_keyframe_conditioning(&settings, &no_first, Path::new("/tmp/p"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("source image"), "got: {err}");
        // sourceAssetId but no lastFrameAssetId.
        let no_last = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "mode": "first_last_frame", "sourceAssetId": "asset_first"
        }));
        let err = resolve_keyframe_conditioning(&settings, &no_last, Path::new("/tmp/p"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("last-frame image"), "got: {err}");
    }

    /// FLF on a 14B Wan MoE engine is rejected at the conditioning resolver (defence-in-depth
    /// behind the routing gate, which already restricts FLF to `wan_2_2`/TI2V-5B).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_flf_rejected_on_non_ti2v_engine() {
        let settings = Settings::from_env();
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "prompt": "a fox",
            "mode": "first_last_frame",
            "sourceAssetId": "a", "lastFrameAssetId": "b"
        }));
        let err = resolve_wan_conditioning(&settings, &req, Path::new("/tmp/p"), "wan2_2_t2v_14b")
            .unwrap_err()
            .to_string();
        assert!(err.contains("TI2V-5B"), "got: {err}");
    }

    /// A 1×1 RGB [`Image`] for clip-conditioning construction tests (the engine resizes; the
    /// content is irrelevant — only the variant / frame_idx / strength mapping is under test).
    #[cfg(target_os = "macos")]
    fn pixel(n: u8) -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![n, n, n],
        }
    }

    /// extend_clip → one `VideoClip` pinned at latent frame 0, strength `videoConditioningStrength`
    /// (default 1.0); the bridge-only right knob is ignored.
    #[cfg(target_os = "macos")]
    #[test]
    fn video_clip_conditioning_extend_maps_single_clip_at_zero() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "extend it",
            "mode": "extend_clip",
            "advanced": { "videoConditioningStrength": 0.7 }
        }));
        let cond = build_video_clip_conditioning(&req, vec![pixel(1), pixel(2)], None).unwrap();
        assert_eq!(cond.len(), 1);
        match &cond[0] {
            Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            } => {
                assert_eq!(frames.len(), 2);
                assert_eq!(*frame_idx, 0);
                assert_eq!(*strength, 0.7);
            }
            other => panic!("expected VideoClip, got {other:?}"),
        }
    }

    /// video_bridge → left clip at 0 (`videoConditioningStrength`) + right clip at -1
    /// (`bridgeRightVideoConditioningStrength`); both default to 1.0 when absent.
    #[cfg(target_os = "macos")]
    #[test]
    fn video_clip_conditioning_bridge_maps_left_zero_right_tail() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "bridge them",
            "mode": "video_bridge",
            "advanced": { "bridgeRightVideoConditioningStrength": "0.5" }
        }));
        let cond =
            build_video_clip_conditioning(&req, vec![pixel(1)], Some(vec![pixel(2), pixel(3)]))
                .unwrap();
        assert_eq!(cond.len(), 2);
        match (&cond[0], &cond[1]) {
            (
                Conditioning::VideoClip {
                    frames: left,
                    frame_idx: left_idx,
                    strength: left_strength,
                },
                Conditioning::VideoClip {
                    frames: right,
                    frame_idx: right_idx,
                    strength: right_strength,
                },
            ) => {
                assert_eq!(left.len(), 1);
                assert_eq!(*left_idx, 0);
                assert_eq!(*left_strength, 1.0); // default
                assert_eq!(right.len(), 2);
                assert_eq!(*right_idx, -1); // engine negative-from-end (lf + idx)
                assert_eq!(*right_strength, 0.5); // numeric-string advanced knob
            }
            other => panic!("expected two VideoClips, got {other:?}"),
        }
    }

    /// video_bridge without the right clip frames is a construction error (defence behind the
    /// resolver's `bridgeRightClipAssetId` guard).
    #[cfg(target_os = "macos")]
    #[test]
    fn video_clip_conditioning_bridge_requires_right_clip() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "bridge them",
            "mode": "video_bridge"
        }));
        let err = build_video_clip_conditioning(&req, vec![pixel(1)], None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("right-side source clip"), "got: {err}");
    }

    /// Wan extend_clip → one boundary `Keyframe` at latent frame 0 (the source clip's last frame),
    /// strength `videoConditioningStrength` (default 1.0); the right frame is ignored (sc-3357).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_boundary_conditioning_extend_pins_last_frame_at_zero() {
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "prompt": "extend it",
            "mode": "extend_clip",
            "advanced": { "videoConditioningStrength": 0.7 }
        }));
        let cond = build_wan_boundary_conditioning(&req, pixel(1), Some(pixel(2))).unwrap();
        assert_eq!(cond.len(), 1);
        match &cond[0] {
            Conditioning::Keyframe {
                frame_idx,
                strength,
                ..
            } => {
                assert_eq!(*frame_idx, 0);
                assert_eq!(*strength, 0.7);
            }
            other => panic!("expected Keyframe, got {other:?}"),
        }
    }

    /// Wan video_bridge → left clip's last frame at 0 (`videoConditioningStrength`) + right clip's
    /// first frame at -1 (`bridgeRightVideoConditioningStrength`); mechanically FLF (sc-3357).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_boundary_conditioning_bridge_pins_both_boundaries() {
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "prompt": "bridge them",
            "mode": "video_bridge",
            "advanced": { "bridgeRightVideoConditioningStrength": "0.5" }
        }));
        let cond = build_wan_boundary_conditioning(&req, pixel(1), Some(pixel(2))).unwrap();
        assert_eq!(cond.len(), 2);
        match (&cond[0], &cond[1]) {
            (
                Conditioning::Keyframe {
                    frame_idx: left_idx,
                    strength: left_strength,
                    ..
                },
                Conditioning::Keyframe {
                    frame_idx: right_idx,
                    strength: right_strength,
                    ..
                },
            ) => {
                assert_eq!(*left_idx, 0);
                assert_eq!(*left_strength, 1.0); // default
                assert_eq!(*right_idx, -1); // engine negative-from-end
                assert_eq!(*right_strength, 0.5); // numeric-string advanced knob
            }
            other => panic!("expected two Keyframes, got {other:?}"),
        }
    }

    /// Wan video_bridge without the right boundary frame is a construction error (defence behind
    /// the resolver's `bridgeRightClipAssetId` guard).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_boundary_conditioning_bridge_requires_right_frame() {
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "prompt": "bridge them",
            "mode": "video_bridge"
        }));
        let err = build_wan_boundary_conditioning(&req, pixel(1), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("right-side source clip"), "got: {err}");
    }

    /// The IC-LoRA detector matches the torch `lora_looks_like_ic_lora` markers (flags, role,
    /// and "ic-lora" / "ltx-2-3-ic-" in id/name/path/files) and rejects an ordinary LoRA.
    #[cfg(target_os = "macos")]
    #[test]
    fn ic_lora_detection_matches_torch_markers() {
        // Explicit flags.
        assert!(loras_contain_ic_lora(&[json!({ "icLora": true })]));
        assert!(loras_contain_ic_lora(&[json!({ "isIcLora": true })]));
        // conditioningRole (with the `-`/`_` normalisation).
        assert!(loras_contain_ic_lora(&[
            json!({ "conditioningRole": "IC-Lora" })
        ]));
        // Name / id / path markers.
        assert!(loras_contain_ic_lora(&[
            json!({ "name": "LTX-2.3-22b-IC-LoRA-Union-Control" })
        ]));
        assert!(loras_contain_ic_lora(&[
            json!({ "id": "ltx_2_3_ic_union" })
        ]));
        assert!(loras_contain_ic_lora(&[
            json!({ "source": { "files": ["my-ic-lora.safetensors"] } })
        ]));
        // A bare string id.
        assert!(loras_contain_ic_lora(&[json!("some-ic-lora-v2")]));
        // An ordinary LoRA is not an IC-LoRA.
        assert!(!loras_contain_ic_lora(&[
            json!({ "name": "cinematic-style", "path": "/loras/cinematic.safetensors" })
        ]));
        assert!(!loras_contain_ic_lora(&[]));
    }

    /// extend/bridge conditioning fails clearly (before any IO) when no IC-LoRA is installed —
    /// mirrors the torch gate; without the adapter the appended clip tokens are inert.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn video_clip_conditioning_requires_ic_lora() {
        let settings = Settings::from_env();
        let api = ApiClient::new(&settings);
        // The IC-LoRA gate is the resolver's first check, so it returns before touching `job`
        // / the api / disk — a minimal snapshot suffices.
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-extend-1",
            "type": "video_extend",
            "status": "preparing",
            "projectId": "p",
            "projectName": "P",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-09T00:00:00Z",
            "updatedAt": "2026-06-09T00:00:00Z"
        }))
        .expect("job snapshot");
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "extend it",
            "mode": "extend_clip", "sourceClipAssetId": "clip_a",
            "loras": [{ "name": "cinematic-style" }]
        }));
        let err = resolve_video_clip_conditioning(&api, &settings, &job, &req, Path::new("/tmp/p"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("IC-LoRA"), "got: {err}");
    }

    /// An LTX-2.3 snapshot **complete for the current engine** ([`ltx_dir_is_complete`]),
    /// else `None` so the smoke skips. Checks `$SCENEWORKS_MLX_LTX_DIR`, the turnkey SceneWorks
    /// bundle's `q4/`/`q8/` subdirs in the HF cache ([`LTX_BUNDLE_REPO`], `SceneWorks/ltx-2.3-mlx`),
    /// and the app-managed `<data>/models/mlx/*` dirs (which predate the audio+i2v layout, so skip).
    #[cfg(target_os = "macos")]
    fn ltx_complete_dir() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_LTX_DIR") {
            candidates.push(PathBuf::from(dir.trim()));
        }
        if let Ok(home) = std::env::var("HOME") {
            let hub = PathBuf::from(&home).join(".cache/huggingface/hub");
            // The turnkey SceneWorks bundle: each snapshot carries `q4/` + `q8/` checkpoint subdirs.
            let snapshots = hub
                .join("models--SceneWorks--ltx-2.3-mlx")
                .join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots) {
                for snapshot in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
                    candidates.push(snapshot.join("q4"));
                    candidates.push(snapshot.join("q8"));
                }
            }
            let base =
                PathBuf::from(home).join("Library/Application Support/SceneWorks/data/models/mlx");
            for id in [
                "ltx_2_3_base_q4",
                "ltx_2_3_base_q8",
                "ltx_2_3_eros",
                "ltx_2_3",
            ] {
                candidates.push(base.join(id));
            }
        }
        candidates.into_iter().find(|dir| ltx_dir_is_complete(dir))
    }

    /// Real in-process LTX-2.3 T2V **with synchronized audio** through the engine. Loads a
    /// complete converted snapshot + the cached Gemma TE and denoises a tiny 9-frame clip,
    /// asserting frames come back RGB8-sized **and an audio track is produced** with streamed
    /// progress. `#[ignore]` + skips unless a complete snapshot is present (the cached
    /// `ltx_2_3_base_*` dirs predate the engine's vocoder/vae_encoder layout).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real LTX-2.3 weights + Gemma TE; needs a snapshot complete for the current engine"]
    #[test]
    fn ltx_real_weights_with_audio() {
        let Some(model_dir) = ltx_complete_dir() else {
            eprintln!("skipping ltx_real_weights_with_audio: no complete LTX snapshot found");
            return;
        };
        // The bundle ships gemma beside the q4/q8 dir; point the engine at it (matches the worker
        // path) so the smoke needs no separate mlx-community/gemma snapshot in the HF cache.
        ensure_bundled_ltx_gemma_dir(&model_dir);
        let input = VideoGenInput {
            engine_id: "ltx_2_3",
            model_dir,
            prompt: "a calm ocean wave at sunset, gentle surf".to_owned(),
            width: 256,
            height: 256,
            frames: 9,
            fps: 24,
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("LTX A/V generation");
        assert_eq!(decoded.frames.len(), 9, "9 frames (1 + 8·1)");
        let audio = decoded
            .audio
            .expect("LTX produces a synchronized audio track");
        assert!(audio.sample_rate >= 1 && !audio.samples.is_empty());
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// Full encode → mp4 + poster, exercised against a real ffmpeg. Skips when no
    /// ffmpeg is reachable (SCENEWORKS_FFMPEG or `ffmpeg` on PATH) so it never fails a
    /// host without the binary; CI with ffmpeg runs it for real.
    #[tokio::test]
    async fn encode_stub_to_mp4_with_audio_and_poster() {
        if !ffmpeg_reachable() {
            eprintln!("skipping encode_stub_to_mp4_with_audio_and_poster: ffmpeg not found");
            return;
        }
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "fox",
            "duration": 1.0, "fps": 9, "width": 128, "height": 128
        }));
        let decoded = generate_stub_video(&request, 11);
        assert!(decoded.audio.is_some());
        let dir = std::env::temp_dir().join(format!("sw_vid_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let media_path = dir.join("clip.mp4");
        encode_media(&media_path, decoded, None).await.unwrap();
        assert!(media_path.exists(), "mp4 must be written");
        assert!(media_path.metadata().unwrap().len() > 0);
        assert!(
            media_path.with_extension("poster.jpg").exists(),
            "poster must be extracted"
        );
        // Intermediates are cleaned up.
        assert!(!media_path.with_extension("frames").exists());
        assert!(!media_path.with_extension("enc.mp4").exists());
        assert!(!media_path.with_extension("audio.wav").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ffmpeg_reachable() -> bool {
        if let Ok(path) = std::env::var("SCENEWORKS_FFMPEG") {
            if !path.trim().is_empty() && Path::new(&path).exists() {
                return true;
            }
        }
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
}

// Candle video lane labeling + engine-mapping unit tests (sc-5099). Windows/candle-gated; pure maps.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_video_label_tests {
    use super::*;

    #[test]
    fn candle_video_engine_ids_map_5b_ltx_and_14b() {
        assert_eq!(candle_video_engine_id("wan_2_2"), Some("wan2_2_ti2v_5b"));
        // The Wan2.2 14B dual-expert MoE pair (sc-5175): T2V (text→video) + I2V (image→video).
        assert_eq!(
            candle_video_engine_id("wan_2_2_t2v_14b"),
            Some("wan2_2_t2v_14b")
        );
        assert_eq!(
            candle_video_engine_id("wan_2_2_i2v_14b"),
            Some("wan2_2_i2v_14b")
        );
        // ltx maps to the candle distilled id, not the MLX `ltx_2_3`.
        assert_eq!(candle_video_engine_id("ltx_2_3"), Some("ltx_2_3_distilled"));
        // SVD maps to the candle `svd_xt` engine (sc-5493).
        assert_eq!(candle_video_engine_id("svd"), Some("svd_xt"));
        // The eros LTX still has no candle provider.
        assert_eq!(candle_video_engine_id("ltx_2_3_eros"), None);
        assert!(!is_candle_video_engine("ltx_2_3_eros"));
        for model in [
            "wan_2_2",
            "wan_2_2_t2v_14b",
            "wan_2_2_i2v_14b",
            "ltx_2_3",
            "svd",
        ] {
            assert!(is_candle_video_engine(model), "{model}");
        }
    }

    #[test]
    fn candle_video_adapter_labels_are_per_family() {
        // Every wan engine (5B + 14B T2V/I2V) reports the shared `candle_wan` adapter.
        for engine_id in ["wan2_2_ti2v_5b", "wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            assert_eq!(
                candle_video_adapter_label(engine_id),
                "candle_wan",
                "{engine_id}"
            );
        }
        assert_eq!(
            candle_video_adapter_label("ltx_2_3_distilled"),
            "candle_ltx"
        );
        assert_eq!(candle_video_adapter_label("svd_xt"), "candle_svd");
    }

    #[test]
    fn candle_video_default_repos_are_per_engine() {
        assert_eq!(
            candle_video_default_repo("wan2_2_ti2v_5b"),
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("wan2_2_t2v_14b"),
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("wan2_2_i2v_14b"),
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("ltx_2_3_distilled"),
            "Lightricks/LTX-2.3"
        );
        // SVD-XT loads the stock diffusers img2vid-xt snapshot directly (sc-5493).
        assert_eq!(
            candle_video_default_repo("svd_xt"),
            "stabilityai/stable-video-diffusion-img2vid-xt"
        );
    }

    #[test]
    fn candle_video_engine_ids_map_svd() {
        assert_eq!(candle_video_engine_id("svd"), Some("svd_xt"));
        assert!(is_candle_video_engine("svd"));
    }

    #[test]
    fn candle_video_conditioning_only_for_i2v() {
        let settings = crate::Settings::from_env();
        let project_path = std::path::Path::new("");
        // txt2video engines never build conditioning (even if a stray source asset is present).
        for engine_id in ["wan2_2_ti2v_5b", "wan2_2_t2v_14b", "ltx_2_3_distilled"] {
            let payload = json!({ "sourceAssetId": "asset_1" });
            let request = VideoRequest::from_payload(payload.as_object().expect("object"));
            let conditioning =
                resolve_candle_video_conditioning(&settings, &request, project_path, engine_id)
                    .expect("txt2video conditioning resolves");
            assert!(
                conditioning.is_empty(),
                "{engine_id} must be txt2video-only"
            );
        }
        // The 14B I2V + SVD-XT require a source image — a request without one errors before touching
        // disk (sc-5175 / sc-5493).
        for engine_id in ["wan2_2_i2v_14b", "svd_xt"] {
            let payload = json!({ "mode": "image_to_video" });
            let no_source = VideoRequest::from_payload(payload.as_object().expect("object"));
            assert!(
                resolve_candle_video_conditioning(&settings, &no_source, project_path, engine_id)
                    .is_err(),
                "{engine_id} without a source image must error"
            );
        }
    }
}
