use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use reqwest::header;
use reqwest::StatusCode;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, JobSnapshot, JobStatus, JobType, JsonObject,
    ProgressRequest, ProgressStage, WorkerCapability, WorkerHeartbeatRequest,
    WorkerRegisterRequest, WorkerSnapshot, WorkerStatus, WorkerUtilizationSnapshot,
};
use sceneworks_core::hf_home::{huggingface_hub_cache_dir, huggingface_repo_cache_path};
use sceneworks_core::jsonc::strip_jsonc_comments;
use sceneworks_core::lora_family::{
    apply_model_manifest_defaults, detect_lora_family, detect_model_family, first_safetensors_path,
    read_safetensors_header, reconcile_detected_family, FamilyMismatch, SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{
    lora_source_url_file_name, lora_source_url_file_stem, parse_lora_source_url_with_private,
    validate_public_ip,
};
use sceneworks_core::project_store::{ProjectStore, ProjectStoreError};
use sceneworks_core::slug::slugify;
use sceneworks_core::time::{format_unix_seconds, now_unix_seconds};
use serde::Deserialize;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time::MissedTickBehavior;
use tracing::Level;
use uuid::Uuid;

// Shared `advanced` knob accessors (sc-4281). The MLX image/video job paths are macOS-gated; the
// candle InstantID lane (sc-5491) is the first off-Mac caller, so the module also compiles on the
// Windows candle build. The candle lane calls only a subset (`flag`/`str`/`f32_clamped`), so allow
// dead_code there (the rest are MLX-only) — same pattern as `openpose_skeleton`. On a non-candle
// Windows/Linux build it stays excluded, so its accessors are never uncalled-dead there.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod advanced;
mod api_client;
// Lazy, on-demand download-credential pull from the macOS desktop credential socket
// (sc-5891). Compiles on all targets; the socket I/O is `cfg(unix)` and inert unless
// the desktop injects `SCENEWORKS_CRED_IPC_*`, so server/Docker/Windows are unaffected.
mod credentials_ipc;
// Backend-neutral generator load/run cache (epic 3720, sc-3724). Typed entirely against
// `gen_core::*` (no tensor types leak), so it links on ALL targets — the production load seam
// (`with_cached_generator`) is reached only from the macOS image/video paths, but the all-targets
// stub test exercises the load→progress→cancel→output contract with no backend linked. Off macOS
// the production caller is cfg'd out, so allow dead_code there (the engines.rs precedent).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod generator_cache;
use api_client::*;
// Backend-neutral engine dispatch table + registry-derived capability advertisement
// (sc-3723). All-targets: the table is pure data and the derivation runs off-macOS off an
// (empty) registry, so a future candle backend lights up with zero worker changes. Off
// macOS the only consumers are the (all-targets) registry-derivation tests — the production
// caller (`mlx_gpu`) is macOS-gated — so allow dead_code on the non-macOS lib build (the
// person_replace pattern); the stub test still exercises it on every target.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod engines;
mod gpu;
use gpu::*;
mod supervisor;
use supervisor::*;
mod model_jobs;
use model_jobs::*;
mod media_jobs;
use media_jobs::*;
// Image-decode backstop (sc-6143): transcodes a valid-but-unsupported image (AVIF/HEIC/HEIF/TIFF/
// BMP/GIF) to PNG at decode time. Compiles on all targets; the transcoder is the shared
// `sceneworks_core::media_convert` routine (sips on macOS, ffmpeg elsewhere).
mod image_decode;
mod image_jobs;
use image_jobs::*;
// Ideogram 4 mandatory JSON-caption conditioning + placeholder detect-and-recover (epic 4725,
// sc-6501). Pure prompt-guard + post-render heuristic, compiled cross-platform so its unit tests run
// on the Linux parity lane. sc-6610: its functions are called only from the macOS MLX generate path
// (`image_jobs/base.rs` `generate_stream`, `#[cfg(target_os = "macos")]`) — off-Mac, Ideogram routes
// to candle (txt2img) or the torch worker, neither of which applies the caption guard, so they read
// as dead code on EVERY non-macOS build (the candle `backend-candle` lane included; the prior
// `not(feature = "backend-candle")` carve-out wrongly assumed the candle path called them).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod ideogram_caption;
// SenseNova-U1 understanding + interleave jobs (epic 3180, sc-3905 — Path B). VQA + Document
// Studio (interleave) consume the concrete `T2iModel` directly (the `Generator` contract emits
// Images/Video only). The handlers are compiled cross-platform (with non-macOS error stubs); the
// real in-process MLX work is macOS-gated inside the module.
mod sensenova_jobs;
use sensenova_jobs::*;
mod video_jobs;
use video_jobs::*;
// Replace-person mask pipeline (epic 3040, sc-3521): cross-platform mask rasterization /
// resample / stored-seg-mask load, so the mask-port-vs-Python parity test runs on the
// Linux CI lane. Its masks are consumed only by the macOS Wan-VACE path in `video_jobs`,
// so off macOS the items are otherwise unused (the parity tests still build + run).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod person_replace;
mod training_jobs;
use training_jobs::*;
mod caption_jobs;
use caption_jobs::*;
mod dataset_analysis_jobs;
use dataset_analysis_jobs::*;
mod face_analysis_jobs;
use face_analysis_jobs::*;
// sc-4407 — the shared, generator-agnostic face-likeness scorer (epic 4406): the backbone identity-
// likeness component the Angles (sc-4409) / Poses (sc-4410) / With-Character (sc-4411) surfaces call as
// a post-pass over a finished generation. Its public seam (`FaceLikenessScorer`) has no production
// caller YET — the consuming surfaces are separate stories — so allow the unused seam here; the pure
// scoring core is exercised by the module's unit tests and the seam by the ignored real-weight test.
#[allow(dead_code)]
mod face_likeness;
// sc-4415 — on-demand "compare image to another" likeness tool (epic 4406): scores a CANDIDATE asset
// against a SOURCE identity reference asset through the shared `face_likeness` scorer. Lives in
// Character Studio Assets; routed as the `face_likeness_compare` job type.
mod face_likeness_compare_jobs;
use face_likeness_compare_jobs::*;
mod prompt_refine_jobs;
use prompt_refine_jobs::*;
mod downloads;
// sc-6541 closed-loop study: test-only LoRA output-quality eval harness (research instrument) —
// see the module doc + docs/sc-6541/closed-loop-protocol.md.
#[cfg(all(test, target_os = "macos"))]
mod lora_eval_harness;
// sc-6541 closed-loop study: native-Rust LoRA train→generate driver (research instrument) —
// see the module doc + docs/sc-6541/closed-loop-protocol.md.
#[cfg(all(test, target_os = "macos"))]
mod lora_train_driver;
// Real-weight GPU smoke for the candle SCAIL-2 lane (sc-7078). Test-only + candle-only; never built
// in normal compiles. Drives the shipped worker conditioning + `gen_core::load("scail2_14b")`.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod scail2_gpu_smoke;
// Real-weight GPU smoke for the candle RealVisXL Lightning lane (sc-7176). Test-only + candle-only;
// drives `gen_core::load("sdxl")` with the forced `lightning` sampler against the distilled checkpoint.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod realvisxl_lightning_gpu_smoke;
// Real-weight GPU smoke for the candle FLUX.2-dev lane (epic 6564 sc-7458). Test-only + candle-only;
// drives `gen_core::load("flux2_dev")` with a Q4 LoadSpec (CPU-stage → quantize-onto-GPU) against the
// dense diffusers snapshot — the worker-lane validation backing the off-Mac candle routing wire.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_dev_gpu_smoke;
// Real-weight MLX smoke for the Krea 2 Turbo worker lane (epic 7565 sc-7575). Test-only + macOS-only;
// drives `gen_core::load("krea_2_turbo")` with a Q8 LoadSpec against the packed `q8/` turnkey subdir —
// the worker-lane validation (the crate links + drives the engine), not just the mlx-gen-krea crate.
#[cfg(all(test, target_os = "macos"))]
mod krea_turbo_mlx_smoke;
// Real-weight MLX smoke for the FLUX.1-dev strict-control worker lane (sc-8244; engine E2 sc-8239).
// Test-only + macOS-only; drives `gen_core::load("flux1_dev_control")` (Dir base + Shakker control
// overlay) per control mode (pose/canny/depth) and asserts a control-vs-control-free steer — the
// worker-lane validation that the crate links + drives the registered control generator end-to-end.
#[cfg(all(test, target_os = "macos"))]
mod flux1_control_mlx_smoke;
// Real-weight MLX smokes for the SD3.5 worker lane (epic 7841 S6 sc-7875 — the MLX-path validation
// boundary). Test-only + macOS-only; drive `gen_core::load("sd3_5_large" | "sd3_5_large_turbo" |
// "sd3_5_medium")` against the gated stabilityai/* diffusers snapshots (the worker crate links + drives
// all three registered generators + the LoRA `with_adapters` apply seam), not just mlx-gen-sd3 in
// isolation.
#[cfg(all(test, target_os = "macos"))]
mod sd3_5_mlx_smoke;
// Real-weight MLX smoke for the SDXL base 1.0 Q8 worker lane (sc-8746, epic 8506 Group-B). Test-only +
// macOS-only; drives `gen_core::load("sdxl")` with a Q8 LoadSpec against the packed `q8/` turnkey subdir.
// Closes the stale sc-1975 Q8-on-SDXL loop on-device: asserts the fixed mlx-gen Q8 path (sc-2641) renders
// non-degenerate AND specifically NOT all-zero (the retired Apple recipe's exact failure signature).
#[cfg(all(test, target_os = "macos"))]
mod sdxl_base_q8_mlx_smoke;
// Real-weight MLX smoke for the Lens-Turbo Q4 worker lane (sc-8763, epic 8506 Group-B). Test-only +
// macOS-only; drives `gen_core::load("lens_turbo")` with a Q4 LoadSpec against the packed `q4/` turnkey
// subdir. On-device evidence that the SceneWorks/lens-turbo-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate (both transformer + gpt-oss MoE TE are packed per-tier; NOT a dense-TE model).
#[cfg(all(test, target_os = "macos"))]
mod lens_turbo_q4_mlx_smoke;
// Real-weight MLX smoke for the recovered base Lens Q4 worker lane (sc-8767, epic 8506 Group-B).
// Test-only + macOS-only; drives `gen_core::load("lens")` with a Q4 LoadSpec against the packed `q4/`
// turnkey subdir. On-device evidence that the SceneWorks/lens-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate (both transformer + gpt-oss MoE TE are packed per-tier; NOT a dense-TE model).
#[cfg(all(test, target_os = "macos"))]
mod lens_base_q4_mlx_smoke;
// Real-weight MLX smoke for the Chroma1-Base Q4 worker lane (sc-8777, epic 8506 Group-B). Test-only +
// macOS-only; drives `gen_core::load("chroma1_base")` with a Q4 LoadSpec against the packed `q4/` turnkey
// subdir. On-device evidence that the SceneWorks/chroma1-base-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate. Chroma packs ONLY the transformer per-tier (the T5-XXL TE + VAE stay dense — chroma
// never quantizes its T5, so no denseTextEncoderTier). hd/flash share this crate + layout.
#[cfg(all(test, target_os = "macos"))]
mod chroma1_base_q4_mlx_smoke;
// On-device per-tier memory-footprint measurement harness (sc-8516, epic 8506). Test-only + macOS-only;
// #[ignore]d real-weight smokes that drive `gen_core::load(id)` + ONE generation while sampling the MLX
// process-global memory counters (mlx_rs::memory::{reset_peak_memory, get_active_memory, get_peak_memory})
// generator_cache.rs already publishes — producing measured resident + peak footprint per (model, tier)
// to calibrate the sc-8509 RAM→tier suggestion (apps/web/src/tierSuggestion.js) and backfill the sc-8508
// manifest footprint fields.
#[cfg(all(test, target_os = "macos"))]
mod footprint_measure;
// The DWPose skeleton rasterizer is consumed only by the macOS Z-Image strict-pose
// control path; on Mac AND the off-Mac candle DWPose lane (sc-5496) it backs the
// `pose_jobs` skeleton render; on a candle-disabled box off Mac it still builds +
// unit-tests (cross-platform raster) but its items are otherwise unused — so allow
// dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod openpose_skeleton;
// Native canny edge-map preprocessor for the Fun-Controlnet-Union canny head
// (epic 8236, sc-8240). Pure CPU raster (cross-platform + testable everywhere),
// sibling of `openpose_skeleton`: arbitrary image → `ControlKind::Canny` control
// image. Consumed by the shared strict-control driver (sc-8243) on macOS AND the
// off-Mac candle strict-control trio (sc-8304); on a candle-disabled box off Mac
// it still builds + unit-tests but its items are otherwise unused — so allow
// dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod canny;
// Depth-map preprocessor for the Fun-Controlnet-Union depth head (epic 8236): arbitrary image →
// `ControlKind::Depth` control image via a Depth Anything V2 port. Sibling of `canny` /
// `openpose_skeleton`, but — unlike those pure raster preprocessors — depth needs neural
// inference, so it is backend-gated: macOS = `mlx-gen-depth` (sc-8242), off-Mac + `backend-candle`
// = `candle-gen-depth` (sc-8413, the Windows/CUDA sibling). Consumed by the shared strict-control
// driver (sc-8243 mac) AND the off-Mac candle strict-control trio (sc-8304, which wires the candle
// estimator into `preprocess_control_entry`); on a candle-disabled box off Mac the estimator stays
// unused — so allow dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod depth;
// DWPose pose detection via onnxruntime (epic 3482, sc-3487). On Mac the CoreML EP +
// on the off-Mac candle GPU-worker lane the CUDA EP (sc-5496, epic 5482) run the same
// RTMW detector in-process; on a candle-disabled box the Python rtmlib path stays the
// Windows/Linux backend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod pose_jobs;
// CUDA execution-provider dependency preloading for the off-Mac candle `ort` paths
// (sc-6209, epic 5482): `ort::ep::cuda::preload_dylibs` dlopens the CUDA-12 runtime +
// cuDNN-9 DLLs the onnxruntime CUDA EP needs, so it engages the GPU regardless of PATH
// (the Mac CoreML path needs no equivalent). Shared by pose_jobs (DWPose, sc-5496) +
// person_jobs (YOLO, sc-5498), and Real-ESRGAN (sc-5499) next — gated to the candle GPU
// lane only.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod ort_cuda;
// SCRFD 5-point face-landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac, plus the
// candle SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482) — the same
// InstantID face-stack detector reused in-process for the Key Point Library "extract kps from this
// image" capability. So the module compiles on Mac AND the candle lane; on a candle-disabled box the
// Python InsightFace path stays the Windows/Linux backend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod kps_jobs;
// Image upscaling: Real-ESRGAN (epic 3482, sc-3489) RRDBNet x2/x4 via `ort`/CoreML on Mac, plus the
// SeedVR2 one-step diffusion upscaler — native MLX on Mac (sc-4815) and the candle CUDA backend on
// Windows (sc-5928). So the module compiles on Mac AND the Windows/CUDA candle lane; the ort/CoreML
// Real-ESRGAN path inside stays Mac-gated (the Python torch Real-ESRGAN / AuraSR path is the
// Windows/Linux backend), while the SeedVR2 path is backend-neutral (`gen_core::load("seedvr2")`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod upscale_jobs;
// YOLO11 person detection + selected-person ByteTrack tracking (epic 3482, sc-3488/sc-3633;
// off-Mac candle lane sc-5498, epic 5482). Native-MLX YOLO11m on Mac, `ort`/CUDA on the off-Mac
// candle GPU-worker lane (the pure-Rust ByteTrack in `person_track` is backend-neutral). So both
// modules compile on Mac AND the candle lane; on a candle-disabled box the Python Ultralytics
// path stays the Windows/Linux backend. Person *segmentation* (SAM masks) stays Mac-only
// (`person_segment*` below) — off-Mac tracks are box-only; a candle SAM backport is epic 3792.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_jobs;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_track;
// Native-MLX SAM2 person segmentation (epic 3704, sc-3709): the `mlx-gen-sam2`
// box-prompt segmenter generates per-frame masks in `run_person_track`. macOS-only
// like person_jobs (mlx-gen builds MLX from source); the Python SAM2 path stays the
// Windows/Linux backend.
#[cfg(target_os = "macos")]
mod person_segment;
// SAM3 text-concept (PCS) person segmenter — the box-prompt-free upgrade of `person_segment`
// (epic 4910, sc-4926). macOS-only (native MLX `mlx-gen-sam3`); the off-Mac Windows/CUDA candle
// sibling is `person_segment_sam3_candle` below.
#[cfg(target_os = "macos")]
mod person_segment_sam3;
// Smart-select image segmentation (epic 6087, sc-6105): the `image_segment` job runs SAM3
// box-prompt segmentation in-process to produce an inpaint mask asset for the Image Editor.
// macOS-only like its `person_segment_sam3` (SAM3) dependency; no torch/candle image-segment path.
#[cfg(target_os = "macos")]
mod segment_jobs;
// Off-Mac candle SAM3 text-concept person segmenter (sc-6247, epic 5482 under sc-5062) — the
// Windows/CUDA sibling of `person_segment_sam3`, driving `candle-gen-sam3`'s `Sam3VideoModel` to
// replace the SAM2 box-prompt STUB in the off-Mac person-track (`media_jobs` `maskState = "missing"`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod person_segment_sam3_candle;
// SCAIL-2 color-coded segmentation-mask painting (epic 5439, sc-5448): turns native SAM3
// per-person masks into the palette-painted RGB masks the SCAIL-2 engine consumes. Backend-neutral
// (pure pixel painting over `AllPersonMasks`); available on both the macOS MLX lane (sc-5448) and the
// off-Mac candle lane (sc-6837, the candle SCAIL-2 sibling), each over its own SAM3 module's
// structurally-identical `AllPersonMasks`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod scail2_masks;
use downloads::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use kps_jobs::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use pose_jobs::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use upscale_jobs::*;

mod credentials;
pub use credentials::*;
mod error;
pub use error::*;
mod manifest;
pub(crate) use manifest::*;
mod paths;
pub use paths::*;
mod payload;
pub(crate) use payload::*;
mod settings;
pub use settings::*;

mod imports;
pub use imports::*;
mod progress;
pub(crate) use progress::*;
mod util;
pub use util::*;
mod preflight;
pub use preflight::*;

const INSTALL_MARKER: &str = ".sceneworks-download-complete.json";
const DEFAULT_API_URL: &str = "http://localhost:8000";
const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const DEFAULT_MAX_LORA_URL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_MODEL_URL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const DEFAULT_TRANSITION_DURATION_SECONDS: f64 = 0.5;
const PERSON_TRACK_SAMPLE_RATE_FPS: f64 = 2.0;
const PERSON_TRACK_MAX_SAMPLES: usize = 24;
const PERSON_TRACK_X_DRIFT: f64 = 0.018;

#[derive(Debug, Clone, PartialEq)]
struct DiscoveredGpu {
    id: String,
    name: String,
    capabilities: Vec<WorkerCapability>,
    utilization: Option<WorkerUtilizationSnapshot>,
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Emit a pre-built structured-event object (already carrying its `event` key) at a
/// **declared** level through the `tracing` backbone. The format-adaptive subscriber
/// renders the `{ event, level, reportedAt, ... }` line on stdout (captured into the
/// per-process log file + the in-app Logs buffer); `reportedAt` is stamped at render
/// time. Replaces the old `println!` of the same JSON so the level is now authoritative
/// rather than inferred from the line text downstream.
fn emit_event_value(level: Level, payload: Value) {
    sceneworks_core::observability::emit_event(level, payload);
}

/// Emit a structured worker event at **info** level (the per-generation lifecycle
/// events — pipeline load / inference start+complete — that the Rust MLX path mirrors
/// from the torch worker, sc-3450). `event` is injected into `payload`.
// Only the macOS image-generation path emits these today; on other targets the
// generation code is cfg'd out, so the helper would be dead code.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn emit_event(event: &str, payload: Value) {
    let mut value = payload;
    if let Some(object) = value.as_object_mut() {
        object.insert("event".to_owned(), Value::String(event.to_owned()));
    }
    emit_event_value(Level::INFO, value);
}

pub async fn run() -> WorkerResult<()> {
    // Install the tracing backbone before anything emits (covers both the
    // standalone `sceneworks-rust-worker` binary and the API's GPU-worker path,
    // which both funnel here). Idempotent — a second call is a no-op.
    sceneworks_core::observability::init_logging();
    // Host mode (no HF cache env set): default HF_HOME to the shared ~/.cache/
    // huggingface so downloads land in the OS cache rather than the private data
    // dir (sc-1904 follow-up). Set before spawning child workers so they inherit
    // it; desktop/Compose already inject HF_HOME, making this a no-op there.
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        tracing::info!(
            event = "hf_home_defaulted",
            home = %home.display(),
            "rust_worker defaulting HF_HOME"
        );
    }
    let settings = Settings::from_env();
    if !settings.is_child_worker {
        if settings.gpu_id == "auto" {
            return supervise_auto_workers(settings).await;
        }
        if settings.gpu_id == "cpu" && settings.utility_workers > 1 {
            let specs = utility_worker_specs(&settings.worker_id, settings.utility_workers);
            return supervise_children(settings, specs).await;
        }
    }
    run_worker_loop(settings).await
}

pub async fn run_worker_loop(settings: Settings) -> WorkerResult<()> {
    // sc-4482 (epic 3720): log the resolved backend-neutral gen-core contract version at startup
    // so a pin skew that slips past the CI guard (`scripts/check-gen-core-skew.sh`) is
    // diagnosable from one log line. One shared contract version backs every linked backend.
    tracing::info!(
        event = "gen_core_contract_version",
        version = %gen_core::VERSION,
        gpuId = %settings.gpu_id,
        "rust_worker gen-core contract version"
    );
    // sc-7820 (epic 7819): apply the user's GPU memory ceiling to the MLX runtime once at startup,
    // before any model load. The MLX limit is process-global, so this single call covers
    // generations, upscales, AND LoRA training. No-op when unset (0) and on non-macOS/candle builds.
    generator_cache::apply_gpu_memory_limit(settings.gpu_memory_limit_bytes);
    // sc-7825 (epic 7819): on the MLX GPU worker only, publish live MLX memory telemetry to the
    // shared config dir for the Settings readout. Gated to `mlx` so the CPU utility workers (which
    // do no MLX work) don't clobber the file with zeros.
    if settings.gpu_id == "mlx" {
        generator_cache::spawn_gpu_telemetry(settings.config_dir.clone());
    }
    let gpu = discover_gpu(&settings).await;
    let api = ApiClient::new(&settings);
    let http_client = reqwest::Client::new();
    register_worker_with_retry(&api, &settings, &gpu).await?;
    let mut lock_failures = 0_u32;
    let mut idle_heartbeat = IdleHeartbeat::new(progress_report_interval(&settings));
    loop {
        tokio::select! {
            result = poll_once(&api, &settings, &http_client, &mut idle_heartbeat) => {
                match result {
                    Ok(()) => lock_failures = 0,
                    Err(error) if is_database_locked(&error) => {
                        // SQLite claim contention. With busy_timeout + BEGIN IMMEDIATE in the
                        // store this should be rare, but back off (instead of hammering at the
                        // flat poll interval) and make it visible so an MLX-eligible job lost to
                        // lock contention is explained rather than silently retried into torch.
                        lock_failures = lock_failures.saturating_add(1);
                        let delay = retry_delay(settings.poll_seconds, lock_failures);
                        emit_event_value(
                            Level::WARN,
                            json!({
                                "event": "claim_lock_contention",
                                "workerId": settings.worker_id,
                                "gpuId": settings.gpu_id,
                                "consecutiveFailures": lock_failures,
                                "retryInSeconds": delay,
                                "error": error.to_string(),
                            }),
                        );
                        tokio::time::sleep(Duration::from_secs(delay)).await;
                    }
                    Err(error) => {
                        lock_failures = 0;
                        tracing::error!(
                            event = "rust_worker_poll_failed",
                            error = %error,
                            "worker claim poll failed"
                        );
                        tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))).await;
                    }
                }
            }
            _ = shutdown_signal() => {
                let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                return Ok(());
            }
        }
    }
}

/// True when an error ultimately stems from SQLite reporting the jobs database as locked.
/// The claim travels worker→API→store, so a lock surfaces as an `Api { detail }` whose
/// message embeds the SQLite text; match on the rendered string rather than a typed variant.
fn is_database_locked(error: &WorkerError) -> bool {
    error
        .to_string()
        .to_ascii_lowercase()
        .contains("database is locked")
}

async fn register_worker_with_retry(
    api: &ApiClient,
    settings: &Settings,
    gpu: &DiscoveredGpu,
) -> WorkerResult<()> {
    let mut attempt = 0_u32;
    loop {
        match register_worker(api, settings, gpu).await {
            Ok(_) => return Ok(()),
            Err(error) => {
                attempt = attempt.saturating_add(1);
                let delay = retry_delay(settings.poll_seconds, attempt);
                tracing::warn!(
                    event = "rust_worker_register_failed",
                    attempt,
                    retryInSeconds = delay,
                    error = %error,
                    "worker registration failed; will retry"
                );
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                    _ = shutdown_signal() => return Err(WorkerError::Canceled(
                        "Worker shutdown requested before registration completed.".to_owned(),
                    )),
                }
            }
        }
    }
}

async fn poll_once(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    idle_heartbeat: &mut IdleHeartbeat,
) -> WorkerResult<()> {
    // sc-7824 (epic 7819): pick up a live GPU-memory-limit change here, before claiming the next
    // job, so a Settings slider move applies between jobs (not mid-flight) with no worker restart.
    // No-op unless this is the MLX worker and the desktop has written the live-handoff file.
    if settings.gpu_id == "mlx" {
        generator_cache::sync_gpu_memory_limit(&settings.config_dir);
    }
    if idle_heartbeat.should_send() {
        heartbeat(api, settings, WorkerStatus::Idle, None).await?;
        idle_heartbeat.mark_sent();
    }
    let claim: ClaimResponse = api
        .post_json(
            "/api/v1/jobs/claim",
            &ClaimRequest {
                worker_id: settings.worker_id.clone(),
                extra: BTreeMap::new(),
            },
        )
        .await?;
    let Some(job) = claim.job else {
        tokio::time::sleep(Duration::from_secs(settings.poll_seconds)).await;
        return Ok(());
    };
    run_utility_job(api, settings, http_client, job).await;
    idle_heartbeat.mark_due();
    Ok(())
}

struct IdleHeartbeat {
    interval: Duration,
    next_due: Instant,
}

impl IdleHeartbeat {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            next_due: Instant::now(),
        }
    }

    fn should_send(&self) -> bool {
        Instant::now() >= self.next_due
    }

    fn mark_sent(&mut self) {
        self.next_due = Instant::now() + self.interval;
    }

    fn mark_due(&mut self) {
        self.next_due = Instant::now();
    }
}

async fn register_worker(
    api: &ApiClient,
    settings: &Settings,
    gpu: &DiscoveredGpu,
) -> WorkerResult<WorkerSnapshot> {
    api.post_json(
        "/api/v1/workers/register",
        &WorkerRegisterRequest {
            worker_id: settings.worker_id.clone(),
            gpu_id: gpu.id.clone(),
            gpu_name: Some(gpu.name.clone()),
            capabilities: worker_capabilities(gpu),
            loaded_models: Vec::new(),
            utilization: gpu.utilization.clone(),
            extra: BTreeMap::new(),
        },
    )
    .await
}

/// Post a worker heartbeat. A transport-level failure (`WorkerError::Http`: the API
/// is briefly unreachable — a restart, a transient network blip) is logged and
/// swallowed rather than propagated: a running job must not be torn down for
/// telemetry we can simply resend. The next heartbeat (≤15s) refreshes the worker's
/// `last_seen` well inside the API's stale-sweep window (default 90s), so a brief
/// outage no longer false-positives a live job to `interrupted`; a sustained outage
/// (> the timeout) still lets the sweep fire — the API stays the authority on
/// declaring a worker gone. A non-transport error (the API answered and rejected
/// the heartbeat, e.g. the worker is no longer registered) is a real signal and is
/// still propagated. (sc-6320)
async fn heartbeat(
    api: &ApiClient,
    settings: &Settings,
    status: WorkerStatus,
    current_job_id: Option<&str>,
) -> WorkerResult<()> {
    // Capture the label before `status` is moved into the request, for the log line.
    let status_label = status.as_str().to_owned();
    let outcome: WorkerResult<WorkerSnapshot> = api
        .post_json(
            &format!("/api/v1/workers/{}/heartbeat", settings.worker_id),
            &WorkerHeartbeatRequest {
                status,
                current_job_id: current_job_id.map(str::to_owned),
                loaded_models: Vec::new(),
                utilization: gpu_utilization(&settings.gpu_id).await,
                extra: BTreeMap::new(),
            },
        )
        .await;
    match outcome {
        Ok(_) => Ok(()),
        Err(WorkerError::Http(error)) => {
            emit_event_value(
                Level::ERROR,
                json!({
                    "event": "worker_heartbeat_transport_failed",
                    "workerId": settings.worker_id,
                    "jobId": current_job_id,
                    "status": status_label,
                    "error": error.to_string(),
                }),
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

async fn run_utility_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: JobSnapshot,
) {
    let result = match job.job_type {
        JobType::Placeholder => run_placeholder_job(api, settings, &job)
            .await
            .map_err(|error| ("Placeholder job failed.", error)),
        // Native MLX image generation, served in-process by the linked mlx-gen
        // engine on the macOS Apple-Silicon GPU worker (epic 3018). Off macOS the
        // capability is never advertised, so this arm is unreachable there.
        JobType::ImageGenerate => run_image_generate_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Image generation failed.", error)),
        // Plain Image Edit (sc-3513): the distinct `image_edit` job type (`mode=edit_image`
        // + `sourceAssetId`, epic 2427) shares the generate handler — it dispatches on
        // payload model+mode (qwen/flux2/sdxl edit streams), not job type. The API only
        // routes MLX-eligible edit models here (jobs_store::image_job_is_mlx_eligible); off
        // macOS the `image_edit` capability is never advertised, so this arm is unreachable.
        JobType::ImageEdit => run_image_generate_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Image edit failed.", error)),
        // Native MLX tile-ControlNet detail refine (epic 3041, sc-3060), served in-process
        // by the engine on the macOS Apple-Silicon GPU worker. Off macOS the capability is
        // never advertised, so this arm is unreachable there (image_detail runs on torch).
        JobType::ImageDetail => run_image_detail_job(api, settings, &job)
            .await
            .map_err(|error| ("Image detail enhancement failed.", error)),
        // SenseNova-U1 visual question answering + Document Studio interleave (epic 3180,
        // sc-3905). These bypass the `Generator` registry and call the concrete `T2iModel`
        // directly (text / text+images output the `GenerationOutput` contract can't express).
        // The API routes them here only on Mac (`understanding_job_is_mlx_eligible`); off macOS
        // the `image_vqa`/`image_interleave` capabilities are never advertised, so these arms
        // are unreachable there (the Python torch worker serves them on Windows/Linux).
        JobType::ImageVqa => run_vqa_job(api, settings, &job)
            .await
            .map_err(|error| ("Visual question answering failed.", error)),
        JobType::ImageInterleave => run_interleave_job(api, settings, &job)
            .await
            .map_err(|error| ("Interleaved generation failed.", error)),
        // Native MLX video generation, served in-process by the linked mlx-gen engine
        // on the macOS Apple-Silicon GPU worker (epic 3018). sc-3033 ships the runtime
        // + procedural stub; the real Wan (sc-3034) / LTX+audio (sc-3035) models link
        // their provider crates. Off macOS the capability is never advertised, so this
        // arm is unreachable there.
        // The clip-conditioning advanced video modes (epic 3040, sc-3522) share the video
        // generation handler — `run_video_generate_job` dispatches `extend_clip` /
        // `video_bridge` by the request `mode` into the LTX IC-LoRA `VideoClip` path. The API
        // only routes the LTX-eligible jobs here (`video_job_is_mlx_eligible`); off macOS the
        // VideoExtend/VideoBridge capabilities are never advertised, so these arms are
        // unreachable there (the procedural stub would otherwise ignore the conditioning).
        JobType::VideoGenerate | JobType::VideoExtend | JobType::VideoBridge => {
            run_video_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Video generation failed.", error))
        }
        // replace_person → native Wan-VACE (epic 3040, sc-3521): the `PersonReplace` job
        // type (and `video_generate` mode=`replace_person`) shares the video handler, which
        // dispatches on `mode == "replace_person"` to the engine `wan_vace` provider — the
        // native equivalent of the torch `WanVACEPipeline` path. The API routes only
        // MLX-eligible replace_person jobs here (`jobs_store::video_job_is_mlx_eligible`);
        // off macOS the `person_replace` capability is never advertised, so this arm only
        // produces a real video on the macOS MLX worker (and the Python torch path serves
        // Windows/Linux + non-VACE replacement).
        JobType::PersonReplace => run_video_generate_job(api, settings, &job)
            .await
            .map_err(|error| ("Person replacement failed.", error)),
        // Native MLX LoRA/LoKr training (epic 3039, sc-3043/3049), served in-process
        // by the linked mlx-gen engine on the macOS Apple-Silicon GPU worker. The API
        // routes only MLX-native families here (jobs_store::training_job_is_mlx_eligible);
        // kolors/lens + LoKr-on-Wan stay on the Python torch worker, which is also the
        // Windows/Linux path. Off macOS the execute capability is never advertised.
        JobType::LoraTrain => run_lora_train_job(api, settings, &job)
            .await
            .map_err(|error| ("LoRA training failed.", error)),
        // Native MLX JoyCaption dataset captioning (epic 3550, sc-3556). The API
        // routes only `captioner=joy_caption` jobs here; Windows/Linux and
        // explicit non-MLX GPU choices keep the Python torch captioner fallback.
        JobType::TrainingCaption => run_training_caption_job(api, settings, &job)
            .await
            .map_err(|error| ("Training captioning failed.", error)),
        // Dataset Doctor CLIP-embedding analysis (sc-6535): the macOS MLX worker embeds every dataset
        // image (clip_vit_l14) and POSTs the content-hash sidecar; off-Mac the handler returns a
        // precise unsupported error (no candle CLIP embedder yet).
        JobType::DatasetAnalysis => run_dataset_analysis_job(api, settings, &job)
            .await
            .map_err(|error| ("Dataset analysis failed.", error)),
        // Dataset Doctor face pass (sc-6538): the native SCRFD+ArcFace stack embeds the largest face of
        // each Person-dataset image and POSTs the face sidecar. MLX on Mac (`mlx-gen-face`), candle on
        // the candle lane; off both the handler returns a precise unsupported error.
        JobType::DatasetFaceAnalysis => run_dataset_face_analysis_job(api, settings, &job)
            .await
            .map_err(|error| ("Dataset face analysis failed.", error)),
        // On-demand "compare image to another" likeness tool (sc-4415): scores a CANDIDATE asset
        // against a SOURCE identity reference asset through the shared SCRFD+ArcFace scorer. MLX on Mac,
        // candle off-Mac; off both the handler returns a precise unsupported error. Like the
        // dataset-face pass, the job-type capability is gpu.rs-hardcoded (the face stack has no gen-core
        // registry), so a job stays queued rather than mis-claimed where the stack isn't linked.
        JobType::FaceLikenessCompare => run_face_likeness_compare_job(api, settings, &job)
            .await
            .map_err(|error| ("Face likeness compare failed.", error)),
        // Native candle prompt refinement (epic 5095, sc-5525; consolidated onto candle-llm in sc-7404):
        // routes `prompt_refine` to the candle `core_llm::TextLlm` provider (candle-llama, resolved
        // model-first). The candle worker advertises `prompt_refine` only when `backend_candle_enabled`
        // (engines::registry_capabilities from the registered core_llm provider); off the Windows candle
        // build the capability is never advertised, so this arm is unreachable there and the Python torch
        // refiner serves the job (sc-5525 keeps it as the Mac + default-installer fallback).
        JobType::PromptRefine => run_prompt_refine_job(api, settings, &job)
            .await
            .map_err(|error| ("Prompt refinement failed.", error)),
        JobType::ModelDownload => run_model_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model download failed.", error)),
        JobType::LoraImport => run_lora_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA import failed.", error)),
        JobType::LoraDownload => run_lora_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA download failed.", error)),
        JobType::ModelImport => run_model_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model import failed.", error)),
        JobType::ModelConvert => run_model_convert_job(api, settings, &job)
            .await
            .map_err(|error| ("Model conversion failed.", error)),
        JobType::FrameExtract => run_frame_extract_job(api, settings, &job)
            .await
            .map_err(|error| ("Frame extraction failed.", error)),
        JobType::TimelineExport => run_timeline_export_job(api, settings, &job)
            .await
            .map_err(|error| ("Timeline export failed.", error)),
        JobType::PersonDetect => run_person_detect_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Person detection failed.", error)),
        // DWPose whole-body pose detection (epic 3482, sc-3487 Mac / sc-5496 off-Mac):
        // RTMW via onnxruntime, replacing the Python rtmlib path — CoreML EP on the
        // macOS MLX worker, CUDA EP on the off-Mac candle GPU worker. Available on Mac
        // AND the candle lane; on a candle-disabled box `PoseDetect` is never advertised
        // by the Rust worker (the Python worker handles it), so this falls to the `_`
        // arm there.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::PoseDetect => run_pose_detect_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Pose detection failed.", error)),
        // SCRFD 5-point landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac + the candle
        // SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482), served in-process
        // for the Key Point Library. Available on Mac AND the candle lane; on a candle-disabled box
        // `KpsExtract` is never advertised by the Rust worker (the Python InsightFace path handles it),
        // so this falls to the `_` arm there.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::KpsExtract => run_kps_extract_job(api, settings, &job)
            .await
            .map_err(|error| ("Keypoint extraction failed.", error)),
        // Image upscaling, served in-process by `upscale_jobs::run_image_upscale_job`: Real-ESRGAN
        // RRDBNet x2/x4 via onnxruntime/CoreML (epic 3482, sc-3489, Mac) + SeedVR2 one-step diffusion
        // (native MLX on Mac sc-4815 / candle CUDA on Windows sc-5928). Available on Mac AND the
        // Windows/CUDA candle lane; on a plain Windows/Linux box `ImageUpscale` is never advertised by
        // the Rust worker, so it falls to the `_` arm (Python Real-ESRGAN/AuraSR). The routing oracle
        // refuses `engine=seedvr2` on torch and `engine=real-esrgan`/`aura-sr` on the candle worker.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::ImageUpscale => run_image_upscale_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Image upscale failed.", error)),
        // Dataset Doctor one-tap upscale (sc-6539): Real-ESRGAN over flagged low-res items, then
        // re-point each via the API. Same engine + worker lanes as image_upscale.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::DatasetUpscale => run_dataset_upscale_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Dataset upscale failed.", error)),
        // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt segmentation,
        // served in-process by `segment_jobs::run_image_segment_job` — a box prompt → a binary
        // inpaint mask asset for the Image Editor. macOS-only (the capability is advertised only by
        // `mlx_gpu`), so off-Mac this arm is absent and a segment job is never claimed there.
        #[cfg(target_os = "macos")]
        JobType::ImageSegment => {
            segment_jobs::run_image_segment_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Smart-select segmentation failed.", error))
        }
        // SeedVR2 video upscaling (epic 4811): one-step super-resolution — native MLX on Mac (sc-4816)
        // / candle CUDA on Windows (sc-5928). SceneWorks' first video upscaler: decodes the source
        // clip, runs the temporal-chunked 5D upscale, re-encodes, and passes the source audio through.
        // Available on Mac + the Windows/CUDA candle lane; elsewhere `VideoUpscale` is never advertised
        // (no torch path), so it falls to the `_` arm and the routing oracle reports it unsupported.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::VideoUpscale => run_video_upscale_job(api, settings, &job)
            .await
            .map_err(|error| ("Video upscale failed.", error)),
        JobType::PersonTrack => run_person_track_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Person tracking failed.", error)),
        _ => {
            let result = fail_job(
                api,
                &job.id,
                "No Rust utility exists for this job type.",
                Some(format!(
                    "Unsupported utility job type: {}",
                    job.job_type.as_str()
                )),
            )
            .await;
            result.map_err(|error| ("Utility job failed.", error))
        }
    };
    if matches!(job.job_type, JobType::LoraImport | JobType::ModelImport) {
        let _ = cleanup_uploaded_import_source(settings, &job.payload).await;
    }
    if let Err((message, error)) = result {
        match error {
            WorkerError::Canceled(_) => {}
            error => {
                let _ = fail_job(api, &job.id, message, Some(error.to_string())).await;
                tracing::error!(
                    event = "utility_job_failed",
                    jobId = %job.id,
                    error = %error,
                    "{message}"
                );
            }
        }
    }
    let _ = heartbeat(api, settings, WorkerStatus::Idle, None).await;
}

async fn run_placeholder_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let stages = [
        (
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Preparing placeholder job.",
        ),
        (
            JobStatus::Running,
            ProgressStage::Running,
            0.35,
            "Running placeholder step 1.",
        ),
        (
            JobStatus::Running,
            ProgressStage::Running,
            0.65,
            "Running placeholder step 2.",
        ),
        (
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving placeholder result.",
        ),
    ];

    for (status, stage, progress, message) in stages {
        let snapshot: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{}", job.id)).await?;
        if snapshot.cancel_requested {
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Canceled,
                    ProgressStage::Canceled,
                    progress,
                    "Worker canceled the job before completion.",
                    None,
                    None,
                    None,
                ),
            )
            .await?;
            return Err(WorkerError::Canceled(
                "Worker canceled the job before completion.".to_owned(),
            ));
        }

        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
        update_job(
            api,
            &job.id,
            progress_payload(status, stage, progress, message, None, None, None),
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }

    let mut result = JsonObject::new();
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    result.insert("output".to_owned(), Value::String("placeholder".to_owned()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Placeholder job completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

fn progress_report_interval(settings: &Settings) -> Duration {
    Duration::from_secs(settings.heartbeat_seconds.clamp(5, 15))
}

fn retry_delay(poll_seconds: u64, attempt: u32) -> u64 {
    let multiplier = 2_u64.saturating_pow(attempt.saturating_sub(1).min(4));
    poll_seconds.max(1).saturating_mul(multiplier).clamp(1, 30)
}

#[cfg(test)]
mod tests;
