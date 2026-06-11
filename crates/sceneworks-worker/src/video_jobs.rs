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
//! `mlx_gen::GenerationOutput::Video { frames, fps, audio }` into the same
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
#[cfg(target_os = "macos")]
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image,
    LoadSpec, MoeExpert, Precision, Progress, Quant, ReplacementMode, WeightsSource,
};
#[cfg(target_os = "macos")]
use mlx_gen_ltx as _;
#[cfg(target_os = "macos")]
use mlx_gen_svd as _;
#[cfg(target_os = "macos")]
use mlx_gen_wan as _;
#[cfg(target_os = "macos")]
use sceneworks_core::character_store::CharacterStore;
#[cfg(target_os = "macos")]
use sceneworks_core::video_request::{ltx_frame_count, wan_frame_count};
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

/// Stub adapter id recorded on generated assets — matches the Python
/// `ProceduralVideoAdapter.id` so the asset sidecar reads identically.
const STUB_ADAPTER: &str = "procedural_video";
const CANCEL_MESSAGE: &str = "Video generation canceled by user.";

/// Decoded video ready for muxing — the worker-side shape both the procedural stub
/// (sc-3033) and the real engine output (`mlx_gen::GenerationOutput::Video`,
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
    // Generate: real MLX on macOS for Wan (sc-3034) / LTX+audio (sc-3035) models with
    // resolvable weights, else the procedural stub (non-macOS or missing weights = stub).
    // replace_person (sc-3521) always routes to the native Wan-VACE provider regardless of the
    // user-picked (replace-capable) model — the native equivalent of the torch `WanVACEPipeline`
    // path. It errors clearly if the VACE snapshot is unprovisioned rather than degrading to the
    // procedural stub (a stubbed person-replace would be meaningless). It also reports the honest
    // `replacementStatus` the asset sidecar folds in (project_store::build_video_sidecar_parts).
    #[cfg(target_os = "macos")]
    let (decoded, adapter, raw_settings, replacement_status) = if request.mode == "replace_person" {
        let (decoded, status) =
            generate_wan_vace(api, settings, job, &request, &project_path, backend).await?;
        (
            decoded,
            WAN_VACE_ADAPTER,
            wan_vace_raw_settings(&request),
            Some(status),
        )
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
                let engine_id =
                    wan_engine_id(&request.model).expect("checked wan2_2_ti2v_5b above");
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
                    wan_raw_settings(&request),
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
            wan_raw_settings(&request),
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
    #[cfg(not(target_os = "macos"))]
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
/// the real-inference markers (mirrors the image `mlx_raw_settings`).
#[cfg(target_os = "macos")]
fn wan_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("frameCount".to_owned(), json!(request.frame_count()));
    raw.insert("fps".to_owned(), json!(request.fps));
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
        "wan_2_2" => ("SCENEWORKS_MLX_WAN5B_DIR", "wan_2_2", None),
        "wan_2_2_t2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_T2V_DIR",
            "wan_2_2_t2v_14b",
            Some("AITRADER/Wan2.2-T2V-A14B-mlx-bf16"),
        ),
        "wan_2_2_i2v_14b" => ("SCENEWORKS_MLX_WAN14B_I2V_DIR", "wan_2_2_i2v_14b", None),
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

/// The 4-step Lightning distill LoRA pair (high/low) for the T2V-A14B
/// (`lightx2v/Wan2.2-Lightning`, the rank-64 Seko V1.1 distill). Errors if not downloaded.
#[cfg(target_os = "macos")]
fn resolve_lightning_loras(settings: &Settings) -> WorkerResult<(PathBuf, PathBuf)> {
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, "lightx2v/Wan2.2-Lightning")
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "wan_2_2_t2v_14b: the Lightning distill LoRA (lightx2v/Wan2.2-Lightning) is not \
                 downloaded — fetch it via the model manager"
                    .to_owned(),
            )
        })?;
    let base = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1";
    let high = snapshot.join(base).join("high_noise_model.safetensors");
    let low = snapshot.join(base).join("low_noise_model.safetensors");
    for file in [&high, &low] {
        if !file.is_file() {
            return Err(WorkerError::InvalidPayload(format!(
                "wan_2_2_t2v_14b: Lightning LoRA file missing: {}",
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
#[cfg(target_os = "macos")]
fn resolve_lora_file(path: PathBuf) -> WorkerResult<PathBuf> {
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
/// (T2V-14B only, tagged high/low) followed by the user LoRAs. On the MoE models a user
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

    // Lightning distill (T2V-14B only): 4-step, applied per-expert at strength 1.0.
    if engine_id == "wan2_2_t2v_14b" {
        let (high, low) = resolve_lightning_loras(settings)?;
        specs.push(moe_adapter(high, 1.0, AdapterKind::Lora, MoeExpert::High));
        specs.push(moe_adapter(low, 1.0, AdapterKind::Lora, MoeExpert::Low));
    }

    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(path)?;
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
fn resolve_wan_vace_adapters(request: &VideoRequest) -> WorkerResult<Vec<AdapterSpec>> {
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
        let file = resolve_lora_file(path)?;
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
#[cfg(target_os = "macos")]
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
        strength: advanced_f32(request, "videoConditioningStrength", 1.0),
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
            strength: advanced_f32(request, "bridgeRightVideoConditioningStrength", 1.0),
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

/// Decode a single boundary frame (first or last) of a source clip into an [`Image`], scaled to the
/// output `width`×`height` (sc-3357, the Wan boundary-keyframe conditioning input). The last frame
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
    args.push(format!("scale={width}:{height}"));
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
            let decoded = image::open(&path)
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

/// Per-model sampling overrides: the T2V-14B runs the 4-step Lightning distill at guide
/// 1.0; the 5B / I2V-14B use the engine's config defaults (`None` → engine fills in).
#[cfg(target_os = "macos")]
fn wan_sampling(engine_id: &str) -> (Option<u32>, Option<f32>) {
    if engine_id == "wan2_2_t2v_14b" {
        (Some(4), Some(1.0))
    } else {
        (None, None)
    }
}

/// The resolved inputs for one video generation (engine load + request build), shared by
/// Wan (sc-3034) and LTX (sc-3035) — split out so the engine call is unit-testable on real
/// weights without the API/job plumbing. The LTX-only knobs (`video_mode` no_audio,
/// prompt-enhance) default off for Wan; the Wan-only `moe_expert` rides on `adapters`.
#[cfg(target_os = "macos")]
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
}

#[cfg(target_os = "macos")]
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
        }
    }
}

/// Load a video model and run one generation to a [`DecodedVideo`] (RGB8 frames + fps +
/// optional audio), streaming denoise progress via `on_progress` and honoring `cancel`.
/// Synchronous + blocking (the `Box<dyn Generator>` is `!Send`); the caller runs it on a
/// blocking thread. The engine fills the audio track (LTX) or leaves it `None` (Wan).
#[cfg(target_os = "macos")]
fn run_video_generation(
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let spec = LoadSpec {
        weights: WeightsSource::Dir(input.model_dir),
        quantize: input.quant,
        precision: Precision::Bf16,
        control: None,
        // MultiControlNet (sc-3378) is image-only; video providers ignore it.
        extra_controls: Vec::new(),
        ip_adapter: None,
        adapters: input.adapters,
    };
    let generator = mlx_gen::load(input.engine_id, &spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("video load failed: {error}")))?;
    let req = GenerationRequest {
        prompt: input.prompt,
        negative_prompt: input.negative_prompt,
        width: input.width,
        height: input.height,
        frames: Some(input.frames),
        fps: Some(input.fps),
        steps: input.steps,
        guidance: input.guidance,
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
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&req, on_progress).map_err(|error| {
        WorkerError::InvalidPayload(format!("video generation failed: {error}"))
    })?;
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
        GenerationOutput::Images(_) => Err(WorkerError::InvalidPayload(
            "video model returned images, expected video frames".to_owned(),
        )),
    }
}

/// Drive a `run_video_generation` on a blocking thread, forwarding its streamed denoise
/// progress to the async worker (Generating stage ~0.25..0.58) + polling cancel ~every 2s.
/// The shared blocking + mpsc + cancel plumbing for Wan and LTX.
#[cfg(target_os = "macos")]
async fn generate_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    input: VideoGenInput,
) -> WorkerResult<DecodedVideo> {
    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Progress>(64);
    let blocking = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<DecodedVideo> {
            let mut on_progress = |progress: Progress| {
                let _ = tx.blocking_send(progress);
            };
            run_video_generation(input, &cancel, &mut on_progress)
        })
    };

    let mut canceled = false;
    let mut last_cancel = Instant::now();
    // Interval arm so the cold model-load phase (mlx_gen::load emits no progress)
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
                if canceled {
                    continue; // drain so the blocking sender never blocks.
                }
        if last_cancel.elapsed() >= Duration::from_secs(2) {
            last_cancel = Instant::now();
            if cancel_requested(api, &job.id, CANCEL_MESSAGE).await {
                cancel.cancel();
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
                    if cancel_requested(api, &job.id, CANCEL_MESSAGE).await {
                        cancel.cancel();
                        canceled = true;
                    }
                }
            }
        }
    }

    let result = blocking
        .await
        .map_err(|error| WorkerError::InvalidPayload(format!("video task join: {error}")))?;
    if canceled {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    result
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
    let (steps, guidance) = wan_sampling(engine_id);
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

/// The converted A/V repo the engine's LTX path consumes (the one it replaces); the Q4
/// base checkpoint with the full audio+I2V component set.
#[cfg(target_os = "macos")]
const LTX_TURNKEY_REPO: &str = "notapalindrome/ltx23-mlx-av-q4";

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

/// Resolve the converted LTX MLX snapshot dir. Env override (`SCENEWORKS_MLX_LTX_DIR` /
/// `…_EROS_DIR`) → `<data>/models/mlx/<candidate>` → (base only) the turnkey HF snapshot
/// [`LTX_TURNKEY_REPO`]. Only a dir **complete for the current engine**
/// ([`ltx_dir_is_complete`]) counts, so a stale local conversion is skipped. For the base
/// model the Q4 checkpoint is the default (`mlxQuantize: 8` prefers the Q8 one); the engine
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
    let wants_q8 = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|bits| bits >= 8)
        .unwrap_or(false);
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
    // Turnkey converted A/V snapshot for the base model (the repo the engine replaces).
    if !eros {
        if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, LTX_TURNKEY_REPO) {
            if ltx_dir_is_complete(&dir) {
                return Ok(dir);
            }
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}: no complete converted LTX MLX weights found under {} (expected one of {candidates:?} \
         with the audio vocoder + i2v vae_encoder; or the turnkey {LTX_TURNKEY_REPO}; or set ${env})",
        request.model,
        settings.data_dir.join("models").join("mlx").display(),
    )))
}

/// User LoRAs for an LTX generation (sc-3035): each at a uniform per-pass strength
/// (`pass_scales` left `None` → the engine applies `scale` on every distilled stage; a
/// per-stage schedule is parity-plus). No distill/Lightning prepend — the 2-stage distill
/// is baked into the checkpoint. peft LoKr allowed (engine residual), LyCORIS rejected.
#[cfg(target_os = "macos")]
fn resolve_ltx_adapters(request: &VideoRequest) -> WorkerResult<Vec<AdapterSpec>> {
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
        let file = resolve_lora_file(path)?;
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
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None => Ok(Vec::new()),
    }
}

/// Read an `advanced` boolean flag (JSON bool), default `false` (Python `bool(.get(k))`).
#[cfg(target_os = "macos")]
fn advanced_bool(request: &VideoRequest, key: &str) -> bool {
    request
        .advanced
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Read an `advanced` float (JSON number or numeric string), default `fallback` — mirrors the
/// Python `_advanced_float`.
#[cfg(target_os = "macos")]
fn advanced_f32(request: &VideoRequest, key: &str, fallback: f32) -> f32 {
    request
        .advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(fallback)
}

/// Read an `advanced` integer (JSON int or numeric string), default `fallback`.
#[cfg(target_os = "macos")]
fn advanced_i32(request: &VideoRequest, key: &str, fallback: i32) -> i32 {
    request
        .advanced
        .get(key)
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as i32)
        .unwrap_or(fallback)
}

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
    Ok(vec![
        Conditioning::Keyframe {
            image: first,
            frame_idx: advanced_i32(request, "imageFrameIndex", 0),
            strength: advanced_f32(request, "imageConditioningStrength", 1.0),
        },
        Conditioning::Keyframe {
            image: last,
            frame_idx: -1,
            strength: advanced_f32(request, "lastFrameConditioningStrength", 1.0),
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
        strength: advanced_f32(request, "videoConditioningStrength", 1.0),
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
            strength: advanced_f32(request, "bridgeRightVideoConditioningStrength", 1.0),
        });
    }
    Ok(conditioning)
}

/// Resolve an asset id to its on-disk media file path (the source clip mp4), mirroring the asset
/// lookup in [`load_reference_image`] but returning the path for ffmpeg frame extraction (the
/// Rust equivalent of the torch `source_asset_media_path`).
#[cfg(target_os = "macos")]
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
/// cadence (no fps resample), scaled to the output `width`×`height` (the engine `build_clips`
/// LANCZOS-downsizes each frame to stage-1 half-res, so this only bounds memory). `count` is the
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
            // Plain scale (stretch) to the output dims — matches the engine's LANCZOS resize
            // model (it re-resizes to stage-1 half-res); bounds the extracted frame footprint.
            "-vf".to_owned(),
            format!("scale={width}:{height}"),
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
                let decoded = image::open(&path)
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
    let video_mode = advanced_bool(request, "noAudio").then(|| "no_audio".to_owned());
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
    let input = VideoGenInput {
        engine_id,
        model_dir: resolve_ltx_model_dir(settings, request)?,
        quant: None,
        adapters: resolve_ltx_adapters(request)?,
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
        enhance_prompt: advanced_bool(request, "enhancePrompt"),
        use_uncensored_enhancer: advanced_bool(request, "useUncensoredEnhancer"),
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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

/// Raw-settings recorded on a real MLX SVD asset (the resolved knobs + real-inference markers).
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
const FRAME_PAD_COLOR: &str = "0x12110f";

/// Raw-settings recorded on a real Wan-VACE asset (`advanced` knobs + the real-inference
/// markers; the engine id is `wan_vace`, not the user-picked replace-capable model).
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
fn replacement_mode_from(value: &str) -> ReplacementMode {
    match value {
        "full_person_keep_outfit" => ReplacementMode::FullPersonKeepOutfit,
        "full_person_replace_outfit" => ReplacementMode::FullPersonReplaceOutfit,
        _ => ReplacementMode::FaceOnly,
    }
}

/// Whether `dir` is a load-ready assembled Wan-VACE snapshot — the diffusers VACE
/// `transformer/` plus the shared base-Wan UMT5/VAE/tokenizer that `mlx_gen::load("wan_vace")`
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
#[cfg(target_os = "macos")]
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
    let media_path = project_path.join(rel);
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
#[cfg(target_os = "macos")]
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
    .map_err(|error| WorkerError::InvalidPayload(format!("frame decode task: {error}")))?
}

/// The approved character reference images (≤4) for the replacement (Python
/// `character_reference_images`): the selected look's `approvedReferenceIds`, else the
/// character's approved `references`. Errors when none are readable (the torch
/// `_validate_inputs` parity). The engine cover-fits each to the output size.
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
fn replacement_status_value(
    track: &Value,
    track_id: &str,
    mask_mode: &str,
    masking_strength: f32,
    reference_count: usize,
    frame_count: usize,
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
        "replacementAdapter": WAN_VACE_ADAPTER,
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

    let masking_strength = advanced_f32(request, "maskingStrength", 1.0);
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
        adapters: resolve_wan_vace_adapters(request)?,
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
        control_scale: Some(advanced_f32(request, "conditioningScale", 1.0)),
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
#[cfg(target_os = "macos")]
fn neutral_control_frame(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: vec![128u8; (width as usize) * (height as usize) * 3],
    }
}

/// A solid W×H mask (`0` = keep the control frame, `255` = regenerate; the engine binarizes at
/// 0.5), matching the `image::RgbImage` form `person_track_masks` produces for replace_person.
#[cfg(target_os = "macos")]
fn solid_mask(width: u32, height: u32, value: u8) -> image::RgbImage {
    image::RgbImage::from_pixel(width, height, image::Rgb([value, value, value]))
}

/// How many real source frames to pin as the motion anchor per kept boundary (sc-3812). More =
/// truer continuity but fewer freely-generated frames. Overridable via advanced `motionAnchorFrames`
/// (per side); defaults to ~⅓ of the output budget (split across the two boundaries for bridge), and
/// is clamped so at least 5 frames (one z16 chunk) are left to generate.
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
    .map_err(|error| WorkerError::InvalidPayload(format!("frame decode task: {error}")))?
}

/// Decode one RGB PNG into an engine [`Image`] (shared by the resample + anchor frame selectors).
#[cfg(target_os = "macos")]
fn decode_png_image(path: &Path) -> WorkerResult<Image> {
    let decoded = image::open(path)
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
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
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
        adapters: resolve_wan_vace_adapters(request)?,
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
        control_scale: Some(advanced_f32(request, "conditioningScale", 1.0)),
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
        let status = replacement_status_value(&track, "ignored", "segmentation", 0.5, 2, 81);
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

    /// Per-model sampling: only the T2V-14B forces the 4-step Lightning preset; the
    /// others defer to the engine config defaults.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_sampling_only_overrides_t2v_14b() {
        assert_eq!(wan_sampling("wan2_2_t2v_14b"), (Some(4), Some(1.0)));
        assert_eq!(wan_sampling("wan2_2_ti2v_5b"), (None, None));
        assert_eq!(wan_sampling("wan2_2_i2v_14b"), (None, None));
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
        let specs = resolve_wan_vace_adapters(&req).expect("resolve vace adapters");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].path, plain);
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
            resolve_wan_vace_adapters(&over),
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

    /// LTX model-id → engine-id mapping (both base + eros load the one engine model).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_engine_id_maps_base_and_eros() {
        assert_eq!(ltx_engine_id("ltx_2_3"), Some("ltx_2_3"));
        assert_eq!(ltx_engine_id("ltx_2_3_eros"), Some("ltx_2_3"));
        assert_eq!(ltx_engine_id("wan_2_2"), None);
        assert_eq!(ltx_engine_id("z_image_turbo"), None);
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
                image: mlx_gen::Image {
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

    /// `advanced.noAudio` maps to the engine's `video_mode = "no_audio"`; enhance flags
    /// flow through. Asserts the LTX request build (the VideoGenInput, pre-load).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_advanced_flags_map_to_video_gen_input() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "advanced": { "noAudio": true, "enhancePrompt": true }
        }));
        assert!(advanced_bool(&req, "noAudio"));
        assert!(advanced_bool(&req, "enhancePrompt"));
        assert!(!advanced_bool(&req, "useUncensoredEnhancer"));
        // LTX adapters: a plain user LoRA is uniform (no per-pass schedule, no moe tag).
        let none = resolve_ltx_adapters(&req).unwrap();
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
        assert_eq!(advanced_f32(&req, "imageConditioningStrength", 1.0), 0.8);
        assert_eq!(
            advanced_f32(&req, "lastFrameConditioningStrength", 1.0),
            0.65
        );
        assert_eq!(advanced_i32(&req, "imageFrameIndex", 0), 2);
        // Absent keys → the fully-pinned defaults (strength 1.0, first index 0).
        let bare = request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "a fox" }));
        assert_eq!(advanced_f32(&bare, "imageConditioningStrength", 1.0), 1.0);
        assert_eq!(
            advanced_f32(&bare, "lastFrameConditioningStrength", 1.0),
            1.0
        );
        assert_eq!(advanced_i32(&bare, "imageFrameIndex", 0), 0);
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
    /// else `None` so the smoke skips. Checks `$SCENEWORKS_MLX_LTX_DIR`, the app-managed
    /// `<data>/models/mlx/*` dirs (which predate the audio+i2v layout, so usually skip), and
    /// the turnkey HF-cache snapshot ([`LTX_TURNKEY_REPO`], `notapalindrome/ltx23-mlx-av-q4`).
    #[cfg(target_os = "macos")]
    fn ltx_complete_dir() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_LTX_DIR") {
            candidates.push(PathBuf::from(dir.trim()));
        }
        if let Ok(home) = std::env::var("HOME") {
            let hub = PathBuf::from(&home).join(".cache/huggingface/hub");
            let snapshots = hub
                .join("models--notapalindrome--ltx23-mlx-av-q4")
                .join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots) {
                candidates.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
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
