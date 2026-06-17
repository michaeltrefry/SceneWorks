//! DWPose whole-body pose detection on the Rust worker (epic 3482, sc-3487).
//!
//! Ports the Python `scene_worker/pose_adapters.py` `pose_detect` job to Rust so the
//! Pose Library "create from photo" flow + InstantID pose conditioning keep working
//! on a Python-free Mac. The detector is rtmlib's performance preset — `yolox_m`
//! (person boxes) + `rtmw-dw-x-l` (COCO-WholeBody-133 SimCC) — run via `ort`
//! (onnxruntime) with the CoreML execution provider. The path-selection spike
//! (`docs/sc-3487/spike-findings.md`) validated sub-pixel keypoint parity + matched
//! latency vs the shipped rtmlib detector.
//!
//! Available on macOS (CoreML EP) AND the off-Mac candle GPU-worker lane (sc-5496,
//! epic 5482): the Windows/Linux/CUDA sibling runs the SAME RTMW detector + the SAME
//! pure pre/post math — only the `ort` execution provider differs (CoreML on Mac, CUDA
//! with a CPU fallback off-Mac). On a candle-disabled box the Python rtmlib path stays
//! the Windows/Linux backend; the candle worker advertises `pose_detect` while the
//! Python path stays a co-resident fallback (retired wholesale in Phase 7, epic 5483).
//! The keypoint conversion + geometry (`wholebody_to_openpose`, `squareify`,
//! facing/bbox) are pure and unit-tested without the onnx weights; only the
//! onnxruntime inference is gated.
//!
//! Pipeline parity (matched to rtmlib source — see the spike doc):
//!  - YOLOX: letterbox to 640 (ratio=min(640/h,640/w), pad 114, top-left), NO
//!    BGR→RGB swap, NO mean/std; this export bakes in NMS → `dets`(1,N,5) f32 +
//!    `labels`(1,N) i64, so use the `shape[-1]==5` branch (boxes/ratio, keep score>0.3).
//!  - RTMPose: xyxy→center/scale (padding 1.25) → aspect-fix to 288/384 → top-down
//!    affine crop (border 0) → `(px−mean)/std` on **BGR** channels → SimCC argmax
//!    decode (split 2.0) → rescale into original px.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::downloads::{ensure_cached_file, DownloadContext};
use image::RgbImage;
#[cfg(not(target_os = "macos"))]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Value};

use crate::openpose_skeleton::{body_stickwidth, draw_wholebody, Keypoint};
use crate::{
    heartbeat, normalize_app_managed_cache_path, optional_payload_string, progress_payload,
    task_join_error, update_job, ApiClient, Settings, WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::ProjectStore;

// ---------------------------------------------------------------------------
// detector constants (rtmlib performance preset)
// ---------------------------------------------------------------------------

const DET: usize = 640;
const PW: usize = 288; // pose model input width
const PH: usize = 384; // pose model input height
const MEAN: [f32; 3] = [123.675, 116.28, 103.53]; // BGR-channel order (rtmlib feeds BGR)
const STD: [f32; 3] = [58.395, 57.12, 57.375];
const PAD: f32 = 1.25;
const SPLIT: f32 = 2.0;
const DET_FILE: &str = "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx";
const POSE_FILE: &str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx";
const DET_URL: &str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip";
const POSE_URL: &str = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip";
const DETECTOR_ID: &str = "rtmw-dw-x-l/ort";

/// The hardware execution provider this build registers before the CPU fallback:
/// CoreML on macOS (sc-3487), CUDA off-Mac on the candle GPU-worker lane (sc-5496).
/// Reported as the result `detector.device` for observability.
#[cfg(target_os = "macos")]
const ACCEL_DEVICE: &str = "coreml";
#[cfg(not(target_os = "macos"))]
const ACCEL_DEVICE: &str = "cuda";

/// Default per-keypoint confidence floor for rendering/thresholding. rtmlib's RTMW
/// SimCC scores are NOT in `[0,1]` (good keypoints observed ~4–8 in the spike), so
/// this is a render/threshold knob, not a probability; raw scores are preserved in
/// the result. Mirrors Python `DEFAULT_POSE_MIN_CONF`.
const DEFAULT_POSE_MIN_CONF: f64 = 0.3;

// COCO-WholeBody-133 (rtmlib raw) → SceneWorks OpenPose-18 body. (op_idx, coco_idx);
// OpenPose-18 inserts neck(1) = midpoint of shoulders (computed separately).
const COCO_TO_OPENPOSE: [(usize, usize); 17] = [
    (0, 0),
    (2, 6),
    (3, 8),
    (4, 10),
    (5, 5),
    (6, 7),
    (7, 9),
    (8, 12),
    (9, 14),
    (10, 16),
    (11, 11),
    (12, 13),
    (13, 15),
    (14, 2),
    (15, 1),
    (16, 4),
    (17, 3),
];

// ---------------------------------------------------------------------------
// pure detector math (ported from the spike crate; unit-tested without weights)
// ---------------------------------------------------------------------------

/// Bilinear sample of a BGR channel from an RgbImage; out-of-bounds → `border`.
/// `c` is the BGR channel index (0=B,1=G,2=R); RgbImage stores RGB so we remap.
#[inline]
fn sample_bgr(img: &RgbImage, x: f32, y: f32, c: usize, border: f32) -> f32 {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let rgb_c = [2usize, 1, 0][c];
    let get = |xi: i64, yi: i64| -> f32 {
        if xi < 0 || yi < 0 || xi >= w || yi >= h {
            border
        } else {
            img.get_pixel(xi as u32, yi as u32)[rgb_c] as f32
        }
    };
    let v00 = get(x0, y0);
    let v10 = get(x0 + 1, y0);
    let v01 = get(x0, y0 + 1);
    let v11 = get(x0 + 1, y0 + 1);
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bot = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bot * fy
}

#[derive(Clone, Copy)]
struct Box4 {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn yolox_preprocess(img: &RgbImage) -> (Vec<f32>, f32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let ratio = (DET as f32 / h).min(DET as f32 / w);
    let new_w = (w * ratio) as usize;
    let new_h = (h * ratio) as usize;
    let sx = w / new_w.max(1) as f32;
    let sy = h / new_h.max(1) as f32;
    let mut chw = vec![114.0f32; 3 * DET * DET];
    for c in 0..3 {
        let plane = c * DET * DET;
        for dy in 0..new_h {
            let src_y = (dy as f32 + 0.5) * sy - 0.5; // cv2 INTER_LINEAR half-pixel
            for dx in 0..new_w {
                let src_x = (dx as f32 + 0.5) * sx - 0.5;
                let v = sample_bgr(
                    img,
                    src_x.clamp(0.0, w - 1.0),
                    src_y.clamp(0.0, h - 1.0),
                    c,
                    0.0,
                );
                chw[plane + dy * DET + dx] = v;
            }
        }
    }
    (chw, ratio)
}

/// This YOLOX export bakes in NMS: `dets`(1,N,5)=[xyxy,score]. rtmlib's
/// `shape[-1]==5` branch: boxes /= ratio, keep score > 0.3.
fn yolox_decode(dets: &[f32], shape: &[i64], ratio: f32) -> Vec<Box4> {
    let mut out = Vec::new();
    if shape.last().copied() != Some(5) {
        return out;
    }
    let n = shape[1] as usize;
    for i in 0..n {
        let base = i * 5;
        if dets[base + 4] > 0.3 {
            out.push(Box4 {
                x1: dets[base] / ratio,
                y1: dets[base + 1] / ratio,
                x2: dets[base + 2] / ratio,
                y2: dets[base + 3] / ratio,
            });
        }
    }
    out
}

/// Build the (1,3,PH,PW) BGR-normalized pose input for one bbox + return the
/// aspect-fixed crop geometry (center, scale) needed to rescale the output.
fn pose_preprocess(img: &RgbImage, b: &Box4) -> (Vec<f32>, f32, f32, f32, f32) {
    let cx = (b.x1 + b.x2) / 2.0;
    let cy = (b.y1 + b.y2) / 2.0;
    let sw0 = (b.x2 - b.x1) * PAD;
    let sh0 = (b.y2 - b.y1) * PAD;
    let aspect = PW as f32 / PH as f32; // 0.75
    let (sw, sh) = if sw0 > sh0 * aspect {
        (sw0, sw0 / aspect)
    } else {
        (sh0 * aspect, sh0)
    };
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    let mut chw = vec![0.0f32; 3 * PH * PW];
    for c in 0..3 {
        let plane = c * PH * PW;
        let (m, sd) = (MEAN[c], STD[c]);
        for dy in 0..PH {
            let src_y = y0 + dy as f32 * sh / PH as f32; // affine inverse (no half-pixel)
            for dx in 0..PW {
                let src_x = x0 + dx as f32 * sw / PW as f32;
                chw[plane + dy * PW + dx] = (sample_bgr(img, src_x, src_y, c, 0.0) - m) / sd;
            }
        }
    }
    (chw, cx, cy, sw, sh)
}

#[inline]
fn argmax(slice: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    (bi, bv)
}

/// Decode SimCC outputs → 133 `[x, y, score]` in original px.
fn pose_decode(
    simcc_x: &[f32],
    sx_shape: &[i64],
    simcc_y: &[f32],
    cx: f32,
    cy: f32,
    sw: f32,
    sh: f32,
) -> Vec<[f32; 3]> {
    let k = sx_shape[1] as usize;
    let wx = sx_shape[2] as usize;
    let wy = simcc_y.len() / k;
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    (0..k)
        .map(|j| {
            let (xloc, mvx) = argmax(&simcc_x[j * wx..(j + 1) * wx]);
            let (yloc, mvy) = argmax(&simcc_y[j * wy..(j + 1) * wy]);
            let val = 0.5 * (mvx + mvy);
            let (lx, ly) = if val <= 0.0 {
                (-1.0, -1.0)
            } else {
                (xloc as f32, yloc as f32)
            };
            let px = (lx / SPLIT) / PW as f32 * sw + x0;
            let py = (ly / SPLIT) / PH as f32 * sh + y0;
            [px, py, val]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// keypoint conversion (pure; ported from pose_adapters.py)
// ---------------------------------------------------------------------------

/// One person's whole-body record: body18 + hands[left21,right21] + face68, each
/// `[x, y, conf]` normalized to `[0,1]`.
struct PoseRecord {
    keypoints: Vec<[f32; 3]>,
    hands: [Vec<[f32; 3]>; 2],
    face: Vec<[f32; 3]>,
}

/// Convert one person's raw 133 `[x,y,score]` (original px) into the SceneWorks
/// OpenPose record, normalized by (w,h). Mirrors `wholebody_to_openpose`.
fn wholebody_to_openpose(kps: &[[f32; 3]], w: f32, h: f32) -> PoseRecord {
    let pt = |i: usize| [kps[i][0] / w, kps[i][1] / h, kps[i][2]];
    let seq = |lo: usize, hi: usize| (lo..hi).map(pt).collect::<Vec<_>>();
    let mut body = vec![[0.0f32; 3]; 18];
    for (op, coco) in COCO_TO_OPENPOSE {
        body[op] = pt(coco);
    }
    let (ls, rs) = (kps[5], kps[6]); // shoulders → neck
    body[1] = [
        (ls[0] + rs[0]) / 2.0 / w,
        (ls[1] + rs[1]) / 2.0 / h,
        ls[2].min(rs[2]),
    ];
    PoseRecord {
        keypoints: body,
        hands: [seq(91, 112), seq(112, 133)],
        face: seq(23, 91),
    }
}

/// Re-normalize a source-aspect pose into a centered `max(w,h)` SQUARE (pad short
/// axis, never crop) so the stored pose is aspect-canonical. Mirrors `squareify`.
fn squareify(rec: &PoseRecord, w: f32, h: f32) -> PoseRecord {
    let side = w.max(h);
    let ox = (side - w) / 2.0;
    let oy = (side - h) / 2.0;
    let sq = |p: &[f32; 3]| [(p[0] * w + ox) / side, (p[1] * h + oy) / side, p[2]];
    PoseRecord {
        keypoints: rec.keypoints.iter().map(sq).collect(),
        hands: [
            rec.hands[0].iter().map(sq).collect(),
            rec.hands[1].iter().map(sq).collect(),
        ],
        face: rec.face.iter().map(sq).collect(),
    }
}

/// Render-ready keypoints: drop points below the confidence floor (→ `None`).
fn thresholded(group: &[[f32; 3]], min_conf: f64) -> Vec<Keypoint> {
    group
        .iter()
        .map(|p| {
            if (p[2] as f64) < min_conf {
                None
            } else {
                Some((p[0], p[1]))
            }
        })
        .collect()
}

/// Bbox over the thresholded points: `[minx, miny, maxx, maxy]` or `None`.
fn bbox(groups: &[&[[f32; 3]]], min_conf: f64) -> Option<[f32; 4]> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for g in groups {
        for p in *g {
            if (p[2] as f64) >= min_conf {
                xs.push(p[0]);
                ys.push(p[1]);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let (mut x0, mut x1, mut y0, mut y1) = (xs[0], xs[0], ys[0], ys[0]);
    for &x in &xs {
        x0 = x0.min(x);
        x1 = x1.max(x);
    }
    for &y in &ys {
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    Some([x0, y0, x1, y1])
}

fn mean_conf(group: &[[f32; 3]]) -> f64 {
    if group.is_empty() {
        return 0.0;
    }
    let sum: f64 = group.iter().map(|p| p[2] as f64).sum();
    (sum / group.len() as f64 * 1000.0).round() / 1000.0
}

/// front / back / profile from the OpenPose-18 head keypoints. Mirrors `_facing`.
fn facing(body: &[[f32; 3]], min_conf: f64) -> &'static str {
    let ok = |i: usize| (body[i][2] as f64) >= min_conf;
    let (nose, r_eye, l_eye, r_ear, l_ear) = (ok(0), ok(14), ok(15), ok(16), ok(17));
    if !nose && !r_eye && !l_eye {
        return "back";
    }
    if r_ear && l_ear {
        return "front";
    }
    if r_ear != l_ear {
        return "profile";
    }
    "front"
}

fn record_to_json(group: &[[f32; 3]]) -> Value {
    Value::Array(
        group
            .iter()
            .map(|p| json!([p[0], p[1], p[2]]))
            .collect::<Vec<_>>(),
    )
}

// ---------------------------------------------------------------------------
// onnxruntime detector (cached process-wide like Python `_DETECTOR_CACHE`)
// ---------------------------------------------------------------------------

struct Detector {
    det: Session,
    pose: Session,
    device: &'static str,
}

static DETECTOR: OnceLock<Mutex<Option<Detector>>> = OnceLock::new();

/// Build a session, optionally registering the platform hardware EP (CoreML on macOS,
/// CUDA off-Mac). `accel == false` builds a plain CPU session (the fallback).
fn build_session(path: &Path, accel: bool) -> WorkerResult<Session> {
    let mut b = Session::builder().map_err(ort_err)?;
    if accel {
        #[cfg(target_os = "macos")]
        {
            b = b
                .with_execution_providers([CoreMLExecutionProvider::default().build()])
                .map_err(ort_err)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            // `error_on_failure` so a CUDA provider that can't initialise (no GPU, or a
            // cuDNN/CUDA mismatch in the loaded onnxruntime) surfaces as an error here —
            // `Detector::load` then falls back to a CPU session and reports `device =
            // "cpu"` honestly, instead of onnxruntime silently CPU-executing while we
            // still claim CUDA.
            b = b
                .with_execution_providers([CUDAExecutionProvider::default()
                    .build()
                    .error_on_failure()])
                .map_err(ort_err)?;
        }
    }
    b.commit_from_file(path).map_err(ort_err)
}

fn ort_err<R>(e: ort::Error<R>) -> WorkerError {
    WorkerError::Engine(format!("onnxruntime: {e}"))
}

/// Run a single CHW float input; returns each f32 output as (shape, data), skipping
/// non-f32 outputs (YOLOX's int64 `labels`).
fn run(
    session: &mut Session,
    shape: [i64; 4],
    data: Vec<f32>,
) -> WorkerResult<Vec<(Vec<i64>, Vec<f32>)>> {
    let input = Tensor::from_array((shape.to_vec(), data)).map_err(ort_err)?;
    let outputs = session.run(ort::inputs![input]).map_err(ort_err)?;
    let mut out = Vec::new();
    for i in 0..outputs.len() {
        if let Ok((shp, d)) = outputs[i].try_extract_tensor::<f32>() {
            out.push((shp.to_vec(), d.to_vec()));
        }
    }
    Ok(out)
}

impl Detector {
    /// Build the cached detector, preferring the hardware EP (CoreML on Mac / CUDA
    /// off-Mac) and falling back to CPU if the provider can't initialise (mirrors
    /// Python `load_pose_detector`).
    fn load(det_path: &Path, pose_path: &Path) -> WorkerResult<Self> {
        match (
            build_session(det_path, true),
            build_session(pose_path, true),
        ) {
            (Ok(det), Ok(pose)) => Ok(Self {
                det,
                pose,
                device: ACCEL_DEVICE,
            }),
            _ => Ok(Self {
                det: build_session(det_path, false)?,
                pose: build_session(pose_path, false)?,
                device: "cpu",
            }),
        }
    }

    /// Detect every person in one image → per-person raw 133 `[x,y,score]` (original
    /// px), largest-person-area first.
    fn detect(&mut self, img: &RgbImage) -> WorkerResult<Vec<Vec<[f32; 3]>>> {
        let (din, ratio) = yolox_preprocess(img);
        let dout = run(&mut self.det, [1, 3, DET as i64, DET as i64], din)?;
        let boxes = dout
            .first()
            .map(|(shape, data)| yolox_decode(data, shape, ratio))
            .unwrap_or_default();
        let mut people = Vec::new();
        for b in &boxes {
            let (pin, cx, cy, sw, sh) = pose_preprocess(img, b);
            let pout = run(&mut self.pose, [1, 3, PH as i64, PW as i64], pin)?;
            if pout.len() < 2 {
                continue;
            }
            // disambiguate simcc_x (last dim 576) vs simcc_y (768) by shape
            let (xi, yi) = if pout[0].0.last() == Some(&((PW * 2) as i64)) {
                (0, 1)
            } else {
                (1, 0)
            };
            people.push(pose_decode(
                &pout[xi].1,
                &pout[xi].0,
                &pout[yi].1,
                cx,
                cy,
                sw,
                sh,
            ));
        }
        Ok(people)
    }
}

/// One source image's raw detection (post-inference, pre-SceneWorks-conversion).
struct RawSource {
    width: u32,
    height: u32,
    people: Vec<Vec<[f32; 3]>>,
}

/// Blocking detection over a batch of resolved image paths. All `ort` objects live
/// inside this closure (built once, then cached process-wide) and never cross an
/// await. `None` entries are unreadable images.
fn detect_batch(
    det_path: PathBuf,
    pose_path: PathBuf,
    images: Vec<Option<PathBuf>>,
) -> WorkerResult<(Vec<Option<RawSource>>, &'static str)> {
    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        *guard = Some(Detector::load(&det_path, &pose_path)?);
    }
    let detector = guard.as_mut().expect("detector loaded");
    let device = detector.device;
    let mut out = Vec::with_capacity(images.len());
    for path in images {
        let Some(path) = path else {
            out.push(None);
            continue;
        };
        let img = match image::open(&path) {
            Ok(img) => img.to_rgb8(),
            Err(_) => {
                out.push(None);
                continue;
            }
        };
        let (width, height) = (img.width(), img.height());
        let people = detector.detect(&img)?;
        out.push(Some(RawSource {
            width,
            height,
            people,
        }));
    }
    Ok((out, device))
}

// ---------------------------------------------------------------------------
// weights provisioning (download-on-first-use parity with rtmlib)
// ---------------------------------------------------------------------------

/// Resolve the two onnx weights, downloading them on first use. Order: explicit env
/// pin (`SCENEWORKS_DWPOSE_DET`/`POSE`), then the app cache
/// `<data_dir>/cache/dwpose/`, then rtmlib's own cache (dev machines), else download
/// + unzip the openmmlab bundle into the app cache.
async fn ensure_weights(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<(PathBuf, PathBuf)> {
    let cache = settings.data_dir.join("cache").join("dwpose");
    let download_context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Pose detection canceled while fetching DWPose weights.",
        fresh_download: false,
    };
    let det = ensure_one(
        "SCENEWORKS_DWPOSE_DET",
        DET_FILE,
        DET_URL,
        &cache,
        &download_context,
    )
    .await?;
    let pose = ensure_one(
        "SCENEWORKS_DWPOSE_POSE",
        POSE_FILE,
        POSE_URL,
        &cache,
        &download_context,
    )
    .await?;
    Ok((det, pose))
}

async fn ensure_one(
    env_key: &str,
    file: &str,
    url: &str,
    cache: &Path,
    context: &DownloadContext<'_>,
) -> WorkerResult<PathBuf> {
    if let Ok(pinned) = std::env::var(env_key) {
        let path = PathBuf::from(pinned);
        if path.exists() {
            return Ok(path);
        }
    }
    let target = cache.join(file);
    if target.exists() {
        return Ok(target);
    }
    // rtmlib's own checkpoint cache, present on dev machines / prior Python installs.
    if let Some(home) = std::env::var_os("HOME") {
        let rtmlib = PathBuf::from(home)
            .join(".cache/rtmlib/hub/checkpoints")
            .join(file);
        if rtmlib.exists() {
            return Ok(rtmlib);
        }
    }
    tokio::fs::create_dir_all(cache).await?;
    let zip_path = target.with_extension("zip");
    ensure_cached_file(
        context,
        url,
        &zip_path,
        &format!("DWPose {file} bundle"),
        None,
    )
    .await?;
    // The openmmlab bundle is a .zip containing a single .onnx; extract it.
    let target_clone = target.clone();
    let file_owned = file.to_owned();
    tokio::task::spawn_blocking(move || extract_onnx(&zip_path, &file_owned, &target_clone))
        .await
        .map_err(|error| task_join_error("weight extract task", error))??;
    Ok(target)
}

fn extract_onnx(zip_path: &Path, file: &str, target: &Path) -> WorkerResult<()> {
    let reader = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| WorkerError::InvalidPayload(format!("dwpose zip: {e}")))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| WorkerError::InvalidPayload(format!("dwpose zip entry: {e}")))?;
        let name = entry.name().to_owned();
        if name.ends_with(".onnx") {
            let tmp = target.with_extension("onnx.tmp");
            let mut sink = std::fs::File::create(&tmp)?;
            std::io::copy(&mut entry, &mut sink)?;
            std::fs::rename(&tmp, target)?;
            return Ok(());
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "no .onnx in DWPose bundle for {file}"
    )))
}

// ---------------------------------------------------------------------------
// source resolution (mirrors pose_adapters `_normalize_sources` / `_resolve_source_path`)
// ---------------------------------------------------------------------------

struct PoseSource {
    asset_id: Option<String>,
    display_name: Option<String>,
    temp: bool,
    path: Option<PathBuf>,
}

fn normalize_sources(payload: &JsonObject) -> Vec<JsonObject> {
    if let Some(arr) = payload.get("sources").and_then(Value::as_array) {
        return arr.iter().filter_map(|s| s.as_object().cloned()).collect();
    }
    if let Some(path) = optional_payload_string(payload, "path") {
        let mut single = JsonObject::new();
        single.insert("path".to_owned(), Value::String(path.to_owned()));
        if let Some(id) = optional_payload_string(payload, "sourceAssetId") {
            single.insert("assetId".to_owned(), Value::String(id.to_owned()));
        }
        return vec![single];
    }
    Vec::new()
}

fn resolve_source(
    src: &JsonObject,
    store: &ProjectStore,
    settings: &Settings,
    project_id: Option<&str>,
    project_path: Option<&Path>,
) -> WorkerResult<PoseSource> {
    let asset_id = src
        .get("assetId")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let display_name = src
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let temp = src.get("temp").and_then(Value::as_bool).unwrap_or(false);
    let raw = src.get("path").and_then(Value::as_str);

    // Staged upload path, confined to the same pose-uploads cache cleaned up below.
    if let Some(raw) = raw {
        let abs = Path::new(raw);
        if abs.is_absolute() {
            let path =
                normalize_app_managed_cache_path(settings, raw, "pose-uploads", "pose sourcePath")?;
            if path.exists() {
                return Ok(PoseSource {
                    asset_id,
                    display_name,
                    temp,
                    path: Some(path),
                });
            }
            return Ok(PoseSource {
                asset_id,
                display_name,
                temp,
                path: None,
            });
        }
    }
    // asset id resolved against the originating project (Create tab sends ids)
    if let (Some(id), Some(pid), Some(proj)) = (&asset_id, project_id, project_path) {
        if let Some(path) = resolve_asset_path(store, pid, id, proj) {
            return Ok(PoseSource {
                asset_id,
                display_name,
                temp,
                path: Some(path),
            });
        }
    }
    // project-relative path
    if let (Some(raw), Some(proj)) = (raw, project_path) {
        let candidate = proj.join(raw);
        if candidate.exists() {
            return Ok(PoseSource {
                asset_id,
                display_name,
                temp,
                path: Some(candidate),
            });
        }
    }
    Ok(PoseSource {
        asset_id,
        display_name,
        temp,
        path: raw.map(PathBuf::from),
    })
}

fn resolve_asset_path(
    store: &ProjectStore,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> Option<PathBuf> {
    let asset = store.get_asset(project_id, asset_id).ok()?;
    let rel = asset.get("file")?.get("path")?.as_str()?;
    let mut path = project_path.to_path_buf();
    for component in Path::new(rel).components() {
        if let std::path::Component::Normal(value) = component {
            path.push(value);
        } else {
            return None;
        }
    }
    path.exists().then_some(path)
}

// ---------------------------------------------------------------------------
// job handler
// ---------------------------------------------------------------------------

pub(crate) async fn run_pose_detect_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing DWPose detector.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let payload = &job.payload;
    let sources = normalize_sources(payload);
    if sources.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "No source images supplied for pose detection.".to_owned(),
        ));
    }
    let min_conf = payload
        .get("minConf")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(DEFAULT_POSE_MIN_CONF);

    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref());
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project_path = project_id
        .and_then(|pid| store.get_project(pid).ok())
        .map(|p| PathBuf::from(p.path));

    let resolved: Vec<PoseSource> = sources
        .iter()
        .map(|s| resolve_source(s, &store, settings, project_id, project_path.as_deref()))
        .collect::<WorkerResult<_>>()?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Downloading,
            0.1,
            "Loading DWPose weights.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let (det_path, pose_path) = ensure_weights(api, settings, http_client, job).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.3,
            "Detecting poses.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let image_paths: Vec<Option<PathBuf>> = resolved.iter().map(|s| s.path.clone()).collect();
    let (raw, device) =
        tokio::task::spawn_blocking(move || detect_batch(det_path, pose_path, image_paths))
            .await
            .map_err(|error| task_join_error("pose detection task", error))??;

    let out_dir = settings
        .data_dir
        .join("cache")
        .join("pose_detect")
        .join(&job.id);
    tokio::fs::create_dir_all(&out_dir).await?;

    let mut out_sources: Vec<Value> = Vec::new();
    for (src, raw_src) in resolved.iter().zip(raw) {
        let stem = src
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "source".to_owned());
        let source_path_str = src.path.as_ref().map(|p| p.to_string_lossy().into_owned());
        let Some(raw_src) = raw_src else {
            out_sources.push(json!({
                "sourceAssetId": src.asset_id,
                "sourcePath": source_path_str,
                "error": "unreadable image",
                "poses": [],
            }));
            continue;
        };
        let (w, h) = (raw_src.width as f32, raw_src.height as f32);

        // convert + squareify + order largest-person-area first
        let mut ordered: Vec<(f32, PoseRecord, Option<[f32; 4]>)> = raw_src
            .people
            .iter()
            .map(|kps| {
                let rec = squareify(&wholebody_to_openpose(kps, w, h), w, h);
                let bb = bbox(
                    &[&rec.keypoints, &rec.hands[0], &rec.hands[1], &rec.face],
                    min_conf,
                );
                let area = bb.map_or(0.0, |b| (b[2] - b[0]) * (b[3] - b[1]));
                (area, rec, bb)
            })
            .collect();
        ordered.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let side = raw_src.width.max(raw_src.height);
        let stick = body_stickwidth(side, side);
        let mut poses: Vec<Value> = Vec::new();
        for (person_index, (_area, rec, bb)) in ordered.iter().enumerate() {
            let body_t = thresholded(&rec.keypoints, min_conf);
            let hands_t = [
                thresholded(&rec.hands[0], min_conf),
                thresholded(&rec.hands[1], min_conf),
            ];
            let face_t = thresholded(&rec.face, min_conf);
            let skeleton =
                draw_wholebody(side, side, &body_t, Some(&hands_t), Some(&face_t), stick);
            let preview = out_dir.join(format!("{stem}_p{person_index}_skel.png"));
            skeleton
                .save(&preview)
                .map_err(|e| WorkerError::InvalidPayload(format!("pose preview write: {e}")))?;

            poses.push(json!({
                "personIndex": person_index,
                "bbox": bb,
                "facing": facing(&rec.keypoints, min_conf),
                "meanConf": {
                    "body": mean_conf(&rec.keypoints),
                    "hands": ((mean_conf(&rec.hands[0]) + mean_conf(&rec.hands[1])) / 2.0 * 1000.0).round() / 1000.0,
                    "face": mean_conf(&rec.face),
                },
                "keypoints": record_to_json(&rec.keypoints),
                "hands": [record_to_json(&rec.hands[0]), record_to_json(&rec.hands[1])],
                "face": record_to_json(&rec.face),
                "skeletonPreview": preview.to_string_lossy(),
            }));
        }

        out_sources.push(json!({
            "sourceAssetId": src.asset_id,
            "sourcePath": source_path_str,
            "displayName": src.display_name.clone().unwrap_or_else(|| stem.clone()),
            "sourceWidth": raw_src.width,
            "sourceHeight": raw_src.height,
            "sourceAspect": (w / h * 10000.0).round() / 10000.0,
            "poses": poses,
        }));
    }

    // Delete File-Upload temp sources now detection is done (guarded to the
    // pose-uploads cache so a project asset resolved by id is never removed).
    cleanup_temp_sources(settings, &resolved).await;

    let mut result = JsonObject::new();
    result.insert("sources".to_owned(), Value::Array(out_sources));
    result.insert(
        "detector".to_owned(),
        json!({"id": DETECTOR_ID, "device": device, "minConf": min_conf}),
    );
    result.insert("poseDetectionActive".to_owned(), Value::Bool(true));

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Pose detection complete.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn cleanup_temp_sources(settings: &Settings, sources: &[PoseSource]) {
    let uploads_root = settings.data_dir.join("cache").join("pose-uploads");
    let Ok(uploads_root) = uploads_root.canonicalize() else {
        return;
    };
    for src in sources {
        if !src.temp {
            continue;
        }
        let Some(path) = &src.path else { continue };
        if let Ok(resolved) = path.canonicalize() {
            if resolved.starts_with(&uploads_root) {
                let _ = tokio::fs::remove_file(&resolved).await;
            }
        }
    }
}

#[cfg(test)]
mod tests;
