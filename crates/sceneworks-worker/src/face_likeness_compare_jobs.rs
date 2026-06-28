//! On-demand identity-likeness compare of two existing assets (epic 4406, sc-4415).
//!
//! The "compare image to another" tool behind Character Studio Assets: given a SOURCE identity
//! reference asset (the approved Reference Asset that carries the character's face) and a CANDIDATE
//! asset (any image the user picked from the character's assets), score the candidate's face against
//! the source identity and return the standard face-likeness result. This is NOT a generation
//! post-pass — both images already exist on disk; the job loads them, runs the SHARED
//! [`crate::face_likeness::FaceLikenessScorer`] (the same SCRFD + ArcFace scorer every other epic-4406
//! surface uses — NO new scoring algorithm), and returns the result.
//!
//! Generator-agnostic + cross-platform: MLX on macOS (`mlx-gen-face`), candle off-Mac with
//! `--features backend-candle` (`candle-gen-face`); on a platform with neither, a precise unsupported
//! error. GPU-routed like `dataset_face_analysis` (the native face stack lives on the GPU worker).
//!
//! ## Result honesty (carried intact from sc-4407)
//! ArcFace is a frontal-identity signal only. A non-frontal / no-face CANDIDATE returns an explicit
//! `detected: false`, `score: null` result carrying a `reason` — an honest N/A, NOT a misleading low
//! number. A SOURCE reference with no detectable face yields the same N/A (`no_source_face`).
//!
//! ## Non-fatal
//! Scoring errors never crash the worker: a backend forward failure becomes a logged N/A result via
//! [`crate::face_likeness::FaceLikenessScorer::score_or_null`], and a missing/garbled asset is a clean
//! `InvalidPayload` the dispatcher turns into a failed job (never a panic).

use std::path::PathBuf;

// `Image` resolves to `mlx_gen::Image` on macOS and `gen_core::Image` under the candle backend — the
// same cfg-gated split the scorer (face_likeness.rs) and sibling job modules use. The whole module
// only exists under these two configs; on the plain Linux parity build (no `backend-candle`) `Image`
// is intentionally absent and only the unsupported stub at the bottom is compiled.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use gen_core::Image;
#[cfg(target_os = "macos")]
use mlx_gen::Image;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::face_likeness::FaceLikenessScorer;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::image_jobs::load_reference_image;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::{
    heartbeat, progress_payload, run_blocking_with_heartbeat, update_job, ApiClient, Settings,
    WorkerError, WorkerResult,
};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::project_store::ProjectStore;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use serde_json::Value;

/// Resolve `sourceAssetId` + `candidateAssetId` (both in the same project) to decoded RGB images.
/// Mirrors the `kps_extract` / reference-asset resolution: read each asset record, confine its
/// `file.path` to the project directory ([`load_reference_image`] routes through `safe_project_path`),
/// and decode. A missing/garbled asset id is a clean `InvalidPayload`, never a panic.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn load_compare_images(settings: &Settings, job: &JobSnapshot) -> WorkerResult<(Image, Image)> {
    let payload = &job.payload;
    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .or(job.project_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "face likeness compare needs a projectId to resolve its assets".to_owned(),
            )
        })?;
    let source_asset_id = required_asset_id(payload, "sourceAssetId")?;
    let candidate_asset_id = required_asset_id(payload, "candidateAssetId")?;

    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project_path = store
        .get_project(project_id)
        .map(|p| PathBuf::from(p.path))
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "face likeness compare project {project_id}: {error}"
            ))
        })?;

    let source = load_reference_image(
        &settings.data_dir,
        project_id,
        source_asset_id,
        &project_path,
    )?;
    let candidate = load_reference_image(
        &settings.data_dir,
        project_id,
        candidate_asset_id,
        &project_path,
    )?;
    Ok((source, candidate))
}

/// Read a required, non-empty asset id from the payload, or a precise `InvalidPayload`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn required_asset_id<'a>(payload: &'a JsonObject, field: &str) -> WorkerResult<&'a str> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("face likeness compare needs a {field}"))
        })
}

/// Build the scorer from the SOURCE face, score the CANDIDATE, and return the standard result object
/// (`{ score, detected, method, sourceRef, reason? }`). Reuses the SHARED scorer — no new algorithm.
/// `source_ref` is stamped on the result so the UI can attribute the score to the chosen reference.
/// Runs the `!Send` MLX/candle work; call inside `spawn_blocking`. Non-fatal: a backend forward
/// failure becomes a logged N/A result (`score_or_null`), never an `Err` that fails the job.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn compare(
    weights_dir: &std::path::Path,
    source: &Image,
    candidate: &Image,
    source_ref: &str,
) -> WorkerResult<JsonObject> {
    let scorer = FaceLikenessScorer::load_for_compare(weights_dir, source)?;
    Ok(compare_with_scorer(&scorer, candidate, source_ref))
}

/// Score the CANDIDATE against an already-constructed scorer and shape the standard result object. The
/// JSON-producing half of [`compare`], factored out so the wiring tests exercise the exact same path
/// over the weight-free stub scorer (no antelopev2 weights in CI). Non-fatal: a backend forward
/// failure becomes a logged N/A via `score_or_null`, never an `Err`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn compare_with_scorer(
    scorer: &FaceLikenessScorer,
    candidate: &Image,
    source_ref: &str,
) -> JsonObject {
    scorer.score_or_null(candidate).to_json(Some(source_ref))
}

/// Run a `face_likeness_compare` job: load the SOURCE reference + CANDIDATE assets, score the
/// candidate's face against the source identity through the shared scorer, and return the result.
/// MLX on Mac, candle off-Mac — the result shape is backend-identical.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_face_likeness_compare_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Preparing face scorer.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let (source, candidate) = load_compare_images(settings, job)?;
    let source_ref = required_asset_id(&job.payload, "sourceAssetId")?.to_owned();

    // Stage the SCRFD + ArcFace bundle (download-on-first-use; a prior InstantID / kps / dataset-face
    // run leaves it cached). The same dir the candle stack loads and the MLX path joins file names in.
    let weights_dir = crate::image_jobs::ensure_face_stack_dir(api, settings, job).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            "Comparing faces.",
            None,
            None,
            None,
        ),
    )
    .await?;

    // Keep the worker heartbeat alive across the blocking face stack load + two forwards (a cold weight
    // load can run long) so a slow compare never trips the API's 90s stale-sweep (sc-8390). Not
    // cancelable. A hard error here (e.g. missing weights) propagates and fails the job — but the
    // *scoring* itself is non-fatal (an N/A result, never an error).
    let result = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        None,
        "",
        "face likeness compare task",
        tokio::task::spawn_blocking(move || {
            compare(&weights_dir, &source, &candidate, &source_ref)
        }),
    )
    .await?;

    let detected = result
        .get("detected")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let message = if detected {
        "Face likeness computed."
    } else {
        "No comparable frontal face detected."
    };
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            message,
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod tests {
    use super::*;
    use crate::face_likeness::FaceLikenessScorer;
    use serde_json::json;

    /// A synthetic image whose first pixel byte selects the weight-free stub detection (see the scorer's
    /// `StubFaceBackend`): `0` ⇒ no face; `>=77` ⇒ a reliable face; `1..=76` ⇒ a low-confidence face.
    fn image(pixel0: u8) -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![pixel0, 0, 0],
        }
    }

    #[test]
    fn same_identity_candidate_scores_high_and_attributes_the_source() {
        // The compare's happy path: a candidate matching the SOURCE identity (same stub pixel) scores a
        // high cosine, detected:true, with the result attributed to the chosen reference asset id.
        let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(200));
        let result = compare_with_scorer(&scorer, &image(200), "ref_src_1");
        assert_eq!(result.get("detected"), Some(&Value::Bool(true)));
        assert_eq!(result.get("sourceRef"), Some(&json!("ref_src_1")));
        assert_eq!(
            result.get("method"),
            Some(&json!("arcface_antelopev2")),
            "reuses the shared scorer's method (no new algorithm)"
        );
        let score = result
            .get("score")
            .and_then(Value::as_f64)
            .expect("a detected candidate carries a numeric score");
        assert!(score > 0.9, "same-identity cosine should be high: {score}");
    }

    #[test]
    fn no_face_candidate_is_na_not_a_low_number() {
        // The honesty linchpin: a non-frontal / no-face CANDIDATE (stub pixel 0) is an explicit
        // detected:false / score:null N/A carrying a reason — NEVER a misleading low number.
        let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(200));
        let result = compare_with_scorer(&scorer, &image(0), "ref_src_1");
        assert_eq!(
            Value::Object(result),
            json!({
                "score": Value::Null,
                "detected": false,
                "method": "arcface_antelopev2",
                "sourceRef": "ref_src_1",
                "reason": "no_face",
            }),
            "a no-face candidate is an honest N/A, not a low score"
        );
    }

    #[test]
    fn low_confidence_candidate_is_na() {
        // An extreme-but-detected candidate below the detection floor (stub pixel 40 ⇒ det_score ≈ 0.578
        // < 0.65) records an honest low_confidence N/A, not a noisy number.
        let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(200));
        let result = compare_with_scorer(&scorer, &image(40), "ref_src_1");
        assert_eq!(result.get("detected"), Some(&Value::Bool(false)));
        assert_eq!(result.get("score"), Some(&Value::Null));
        assert_eq!(result.get("reason"), Some(&json!("low_confidence")));
    }

    #[test]
    fn source_with_no_face_makes_every_compare_na_never_a_crash() {
        // Non-fatal end-to-end: a SOURCE reference with no detectable face yields a scorer (NOT an
        // error) whose every candidate compare is the honest NoSourceFace N/A — the compare never
        // crashes the worker, it just reports there is no comparable source identity.
        let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(0));
        let result = compare_with_scorer(&scorer, &image(200), "ref_src_1");
        assert_eq!(result.get("detected"), Some(&Value::Bool(false)));
        assert_eq!(result.get("reason"), Some(&json!("no_source_face")));
    }

    #[test]
    fn required_asset_id_rejects_missing_and_blank() {
        // The payload-validation guard: a missing or blank asset id is a clean InvalidPayload (which the
        // dispatcher turns into a failed job), never a panic.
        let mut payload = JsonObject::new();
        assert!(required_asset_id(&payload, "sourceAssetId").is_err());
        payload.insert("sourceAssetId".to_owned(), json!("   "));
        assert!(required_asset_id(&payload, "sourceAssetId").is_err());
        payload.insert("sourceAssetId".to_owned(), json!("asset_42"));
        assert_eq!(
            required_asset_id(&payload, "sourceAssetId").unwrap(),
            "asset_42"
        );
    }

    /// Real-weight worker integration (sc-4415): proves the worker binary links + loads the MLX face
    /// stack and that the on-demand compare (a) scores a same-identity pair high and (b) returns an
    /// honest N/A for a no-face candidate — through the SAME shared scorer the rest of epic 4406 uses.
    /// `#[ignore]` per convention (weights live outside CI). Run on a Mac with the bundle staged + Metal:
    ///   SCENEWORKS_INSTANTID_WEIGHTS=/path/to/instantid-mlx \
    ///   SCENEWORKS_TEST_FACE=/path/to/personA.png \
    ///     cargo test -p sceneworks-worker --lib -- --ignored face_likeness_compare_real_weights --nocapture
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight: needs the SceneWorks/instantid-mlx SCRFD+ArcFace bundle + Metal + a face photo"]
    fn face_likeness_compare_real_weights_scores_and_reports_na() {
        use std::path::PathBuf;
        let home = std::env::var("HOME").expect("HOME");
        let bundle = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(&home)
                    .join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
            });
        let face_path = std::env::var("SCENEWORKS_TEST_FACE")
            .expect("set SCENEWORKS_TEST_FACE to a face photo");
        let decoded = image::open(&face_path)
            .unwrap_or_else(|e| panic!("face {face_path}: {e}"))
            .to_rgb8();
        let source = Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        };

        // (a) same-identity compare (source vs itself) scores high + detected.
        let same = compare(&bundle, &source, &source, "ref_src_1").expect("compare runs");
        assert_eq!(same.get("detected"), Some(&Value::Bool(true)));
        let score = same.get("score").and_then(Value::as_f64).expect("scored");
        assert!(
            score > 0.8,
            "same-identity cosine ≈ baseline (>0.8): {score}"
        );

        // (b) a no-face candidate (synthetic gradient) is an honest detected:false N/A.
        let mut gradient = image::RgbImage::new(128, 128);
        for (x, y, px) in gradient.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 2) as u8, (y * 2) as u8, 96]);
        }
        let gradient = Image {
            width: gradient.width(),
            height: gradient.height(),
            pixels: gradient.into_raw(),
        };
        let none = compare(&bundle, &source, &gradient, "ref_src_1").expect("compare runs");
        assert_eq!(none.get("detected"), Some(&Value::Bool(false)));
        assert_eq!(none.get("score"), Some(&Value::Null));
        println!("face-likeness compare ok: same={score:.4}");
    }
}

/// Plain-Linux parity build (no MLX, no candle): the native SCRFD + ArcFace face stack is unavailable,
/// so the compare cannot run. Return a precise unsupported error rather than failing to compile or
/// silently mis-handling the job — mirrors `run_dataset_face_analysis_job`.
#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub(crate) async fn run_face_likeness_compare_job(
    _api: &crate::ApiClient,
    _settings: &crate::Settings,
    _job: &sceneworks_core::contracts::JobSnapshot,
) -> crate::WorkerResult<()> {
    Err(crate::WorkerError::InvalidPayload(
        "Face likeness compare (SCRFD + ArcFace) needs the macOS MLX backend or the candle backend \
         (build with --features backend-candle)."
            .to_owned(),
    ))
}
