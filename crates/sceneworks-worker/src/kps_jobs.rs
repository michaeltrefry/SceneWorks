//! SCRFD 5-point face-landmark extraction on the Rust worker (epic 4422, sc-4433).
//!
//! The reusable "extract kps from this image" capability behind the Key Point Library:
//! given one photo, detect the largest face and return its 5 landmarks
//! (`[left_eye, right_eye, nose, mouth_left, mouth_right]`) normalized to a square
//! `[0,1]` canvas — directly consumable as an InstantID angle/framing preset by the
//! engine pass-in path (`generate_with_kps`, sc-4425). The reference supplies *identity*;
//! these kps supply the *pose/framing*, so a user can capture the framing of any photo and
//! reuse it on any character.
//!
//! Native-MLX SCRFD in-process (the same `scrfd_10g` detector the InstantID face stack
//! already runs), so the capability is Python-free on Mac (epic 3482); the Python worker
//! keeps the InsightFace path for Windows/Linux. macOS-only here.
//!
//! Normalization mirrors the engine's centered square letterbox (`kps::letterbox`): a
//! detected pixel `(px, py)` in a `w×h` image maps to the square-normalized point
//! `((1 − w/M)/2 + px/M, (1 − h/M)/2 + py/M)` with `M = max(w, h)` — the side cancels, so
//! the result is scale-free and round-trips through `generate_with_kps` (which draws
//! `kps_norm · side` on a square canvas after letterboxing the reference the same way).

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_face::{detector_blob, Scrfd};
use serde_json::{json, Value};

use crate::image_jobs::{ensure_scrfd_weights, load_reference_image};
use crate::{
    heartbeat, progress_payload, update_job, ApiClient, Settings, WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::ProjectStore;

/// SCRFD detection confidence floor (InsightFace / `FaceAnalysis` default).
const DET_THRESH: f32 = 0.5;
/// Non-max-suppression IoU threshold (InsightFace / `FaceAnalysis` default).
const NMS_THRESH: f32 = 0.4;
/// Below this detection score the landmarks are returned but flagged `lowConfidence`, so the
/// caller can warn that the captured angle may be unreliable (extreme profile / poor framing).
const LOW_CONF_THRESH: f32 = 0.65;
/// The landmark order, surfaced on the result so a consumer never has to assume it.
const KPS_ORDER: [&str; 5] = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"];
const DETECTOR_ID: &str = "scrfd_10g";

/// Map a detected pixel coordinate into the square-normalized `[0,1]` preset space, applying
/// the engine's centered-letterbox geometry analytically (no image resize needed). Pure +
/// unit-tested. `M = max(w, h)`; for a square image this is just `px/w, py/h`.
fn normalize_to_square(px: f32, py: f32, w: u32, h: u32) -> [f32; 2] {
    let m = w.max(h) as f32;
    [
        (1.0 - w as f32 / m) / 2.0 + px / m,
        (1.0 - h as f32 / m) / 2.0 + py / m,
    ]
}

/// The outcome of a single-image extraction. `detected = false` is an explicit, well-formed
/// result (the acceptance "clear failure, not silent bad data"), not a job error.
struct KpsExtraction {
    detected: bool,
    /// Normalized 5-point landmarks (square `[0,1]`), largest face. `None` when no face.
    kps: Option<[[f32; 2]; 5]>,
    /// Normalized `[x1, y1, x2, y2]` face box (square `[0,1]`). `None` when no face.
    bbox: Option<[f32; 4]>,
    det_score: f32,
    low_confidence: bool,
    source_width: u32,
    source_height: u32,
}

impl KpsExtraction {
    fn none(source_width: u32, source_height: u32) -> Self {
        Self {
            detected: false,
            kps: None,
            bbox: None,
            det_score: 0.0,
            low_confidence: false,
            source_width,
            source_height,
        }
    }

    fn into_result(self) -> JsonObject {
        let mut result = JsonObject::new();
        result.insert("detected".to_owned(), Value::Bool(self.detected));
        result.insert("sourceWidth".to_owned(), json!(self.source_width));
        result.insert("sourceHeight".to_owned(), json!(self.source_height));
        result.insert(
            "detector".to_owned(),
            json!({ "id": DETECTOR_ID, "backend": "mlx" }),
        );
        match self.kps {
            Some(kps) => {
                result.insert(
                    "kps".to_owned(),
                    json!(kps.iter().map(|p| json!([p[0], p[1]])).collect::<Vec<_>>()),
                );
                result.insert(
                    "kpsOrder".to_owned(),
                    json!(KPS_ORDER.iter().collect::<Vec<_>>()),
                );
                if let Some(bbox) = self.bbox {
                    result.insert("bbox".to_owned(), json!(bbox));
                }
                result.insert("detScore".to_owned(), json!(self.det_score));
                result.insert("lowConfidence".to_owned(), Value::Bool(self.low_confidence));
            }
            None => {
                result.insert("reason".to_owned(), json!("no_face"));
            }
        }
        result
    }
}

/// Load SCRFD + detect the largest face's landmarks in `image`, normalized to the square
/// preset space. Runs the `!Send` MLX work; call inside `spawn_blocking`.
fn detect_largest_kps(scrfd_path: &Path, image: &Image) -> WorkerResult<KpsExtraction> {
    let weights = Weights::from_file(scrfd_path).map_err(|error| {
        WorkerError::InvalidPayload(format!("SCRFD weights {scrfd_path:?}: {error}"))
    })?;
    let scrfd = Scrfd::from_weights(&weights)
        .map_err(|error| WorkerError::InvalidPayload(format!("SCRFD load: {error}")))?;
    let (blob, det_scale) =
        detector_blob(&image.pixels, image.height as usize, image.width as usize);
    let dets = scrfd
        .detect(&blob, det_scale, DET_THRESH, NMS_THRESH)
        .map_err(|error| WorkerError::InvalidPayload(format!("SCRFD detect: {error}")))?;

    let (w, h) = (image.width, image.height);
    let largest = dets.into_iter().max_by(|a, b| {
        let area = |d: &mlx_gen_face::Detection| {
            (d.bbox[2] - d.bbox[0]).max(0.0) * (d.bbox[3] - d.bbox[1]).max(0.0)
        };
        area(a).total_cmp(&area(b))
    });
    let Some(det) = largest else {
        return Ok(KpsExtraction::none(w, h));
    };

    let mut kps = [[0.0f32; 2]; 5];
    for (i, point) in det.kps.iter().enumerate() {
        kps[i] = normalize_to_square(point[0], point[1], w, h);
    }
    let tl = normalize_to_square(det.bbox[0], det.bbox[1], w, h);
    let br = normalize_to_square(det.bbox[2], det.bbox[3], w, h);
    Ok(KpsExtraction {
        detected: true,
        kps: Some(kps),
        bbox: Some([tl[0], tl[1], br[0], br[1]]),
        det_score: det.score,
        low_confidence: det.score < LOW_CONF_THRESH,
        source_width: w,
        source_height: h,
    })
}

/// Resolve the source image from the payload: a project `sourceAssetId` (+ `projectId`) or a
/// staged `sourcePath`. Mirrors the dual source contract pose detection accepts.
fn load_source_image(settings: &Settings, job: &JobSnapshot) -> WorkerResult<Image> {
    let payload = &job.payload;
    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref());

    if let Some(asset_id) = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        let project_id = project_id.ok_or_else(|| {
            WorkerError::InvalidPayload("kps extraction sourceAssetId needs a projectId".to_owned())
        })?;
        let store = ProjectStore::new(settings.data_dir.clone(), "worker");
        let project_path = store
            .get_project(project_id)
            .map(|p| PathBuf::from(p.path))
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("kps extraction project {project_id}: {error}"))
            })?;
        return load_reference_image(&settings.data_dir, project_id, asset_id, &project_path);
    }

    if let Some(path) = payload
        .get("sourcePath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        // Decode by CONTENT, not file extension: staged uploads land as `upload-<uuid>.tmp`
        // (the API's generic temp-file writer keeps no extension), so `image::open` — which
        // picks the codec from the extension — would reject a perfectly valid JPEG/PNG. Read
        // the bytes and let `load_from_memory` sniff the format from the magic bytes.
        let bytes = std::fs::read(path).map_err(|error| {
            WorkerError::InvalidPayload(format!("kps extraction source {path}: {error}"))
        })?;
        let decoded = image::load_from_memory(&bytes)
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("kps extraction source {path}: {error}"))
            })?
            .to_rgb8();
        return Ok(Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        });
    }

    Err(WorkerError::InvalidPayload(
        "kps extraction needs a sourceAssetId or sourcePath".to_owned(),
    ))
}

/// Run a `kps_extract` job: SCRFD landmark extraction from one image → normalized 5-point
/// preset (or an explicit `detected: false` result on no face).
#[cfg(target_os = "macos")]
pub(crate) async fn run_kps_extract_job(
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
            "Preparing face detector.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let image = load_source_image(settings, job)?;
    let scrfd_path = ensure_scrfd_weights(settings).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            "Detecting face landmarks.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let extraction = tokio::task::spawn_blocking(move || detect_largest_kps(&scrfd_path, &image))
        .await
        .map_err(|e| WorkerError::InvalidPayload(format!("kps extraction task: {e}")))??;

    let message = if extraction.detected {
        "Face landmarks extracted."
    } else {
        "No face detected in the image."
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
            Some(extraction.into_result()),
            None,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_square_image_is_plain_fraction() {
        // A square image: side cancels, normalized = px/side, py/side.
        assert_eq!(normalize_to_square(512.0, 256.0, 1024, 1024), [0.5, 0.25]);
        assert_eq!(normalize_to_square(0.0, 0.0, 800, 800), [0.0, 0.0]);
    }

    #[test]
    fn normalize_landscape_centers_vertically() {
        // 1000x500 landscape (M=1000): x is px/1000; y is letterboxed into the centered band
        // [0.25, 0.75], so the vertical midpoint (py=250) maps to 0.5.
        let [x, y] = normalize_to_square(500.0, 250.0, 1000, 500);
        assert!((x - 0.5).abs() < 1e-6, "x={x}");
        assert!((y - 0.5).abs() < 1e-6, "y={y}");
        // Top edge of the source sits at 0.25 in the square (the top pad band).
        assert!((normalize_to_square(0.0, 0.0, 1000, 500)[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn normalize_portrait_centers_horizontally() {
        // 500x1000 portrait (M=1000): y is py/1000; x is centered into [0.25, 0.75].
        let [x, y] = normalize_to_square(250.0, 500.0, 500, 1000);
        assert!((x - 0.5).abs() < 1e-6, "x={x}");
        assert!((y - 0.5).abs() < 1e-6, "y={y}");
        assert!((normalize_to_square(0.0, 0.0, 500, 1000)[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn no_face_result_is_explicit() {
        let result = KpsExtraction::none(640, 480).into_result();
        assert_eq!(result.get("detected"), Some(&Value::Bool(false)));
        assert_eq!(
            result.get("reason").and_then(Value::as_str),
            Some("no_face")
        );
        assert!(result.get("kps").is_none());
    }

    #[test]
    fn detected_result_carries_normalized_kps_and_order() {
        let extraction = KpsExtraction {
            detected: true,
            kps: Some([
                [0.4, 0.34],
                [0.6, 0.34],
                [0.5, 0.43],
                [0.43, 0.53],
                [0.57, 0.53],
            ]),
            bbox: Some([0.3, 0.2, 0.7, 0.6]),
            det_score: 0.92,
            low_confidence: false,
            source_width: 1024,
            source_height: 1024,
        };
        let result = extraction.into_result();
        assert_eq!(result.get("detected"), Some(&Value::Bool(true)));
        let kps = result.get("kps").and_then(Value::as_array).unwrap();
        assert_eq!(kps.len(), 5);
        let order = result.get("kpsOrder").and_then(Value::as_array).unwrap();
        assert_eq!(order[2].as_str(), Some("nose"));
        assert_eq!(result.get("lowConfidence"), Some(&Value::Bool(false)));
    }

    /// Real-weights smoke (sc-4433): run the actual native-MLX SCRFD path on a real face photo
    /// and assert the extracted landmarks are well-formed and in valid frontal geometry —
    /// eyes above nose above mouth, left-eye left of right-eye, all normalized into `[0,1]`.
    /// This exercises `detect_largest_kps` end to end (Weights load → blob → detect → largest
    /// → normalize). Needs the `scrfd_10g.safetensors` bundle (`SCENEWORKS_INSTANTID_WEIGHTS`
    /// overrides) + a face image (`SCENEWORKS_TEST_FACE` overrides) + a Metal device. On demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored kps_extract_real_weights --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SCRFD weights + Metal device"]
    fn kps_extract_real_weights_extracts_frontal_landmarks() {
        let home = std::env::var("HOME").expect("HOME");
        let scrfd_path = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(&home)
                    .join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
            })
            .join("scrfd_10g.safetensors");
        assert!(scrfd_path.exists(), "missing SCRFD weights: {scrfd_path:?}");
        let face_path = std::env::var("SCENEWORKS_TEST_FACE").unwrap_or_else(|_| {
            "/Users/michael/Library/Application Support/SceneWorks/data/projects/ab.sceneworks/assets/images/genset_e6b07eb5b5374627af1bf47083bac305/2026-06-10_qwen_image_edit_2511_lightning_22-year-old-woman-with-fair-complexion-a-p_0001.png".to_owned()
        });
        let decoded = image::open(&face_path)
            .unwrap_or_else(|e| panic!("face {face_path}: {e}"))
            .to_rgb8();
        let image = Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        };

        let extraction = detect_largest_kps(&scrfd_path, &image).expect("detect");
        assert!(extraction.detected, "expected a face in the test image");
        let kps = extraction.kps.expect("kps on a detected face");
        let order = KPS_ORDER;
        for (i, p) in kps.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]),
                "{} out of [0,1]: {p:?}",
                order[i]
            );
        }
        let [le, re, nose, ml, mr] = kps;
        assert!(le[0] < re[0], "left eye should be left of right eye");
        assert!(ml[0] < mr[0], "left mouth should be left of right mouth");
        let eye_y = (le[1] + re[1]) / 2.0;
        let mouth_y = (ml[1] + mr[1]) / 2.0;
        assert!(eye_y < nose[1], "eyes above nose ({eye_y} !< {})", nose[1]);
        assert!(
            nose[1] < mouth_y,
            "nose above mouth ({}, {mouth_y})",
            nose[1]
        );
        println!(
            "  extracted (score {:.3}, low_conf {}): le={le:?} re={re:?} nose={nose:?} ml={ml:?} mr={mr:?}",
            extraction.det_score, extraction.low_confidence
        );
    }
}
