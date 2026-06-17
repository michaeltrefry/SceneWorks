//! Unit tests for the pure DWPose conversion + decode math (sc-3487). No onnx
//! weights required — the onnxruntime inference is validated end-to-end against the
//! Python reference in the spike (docs/sc-3487/spike-findings.md).

use super::*;

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn wholebody_to_openpose_maps_indices_and_computes_neck() {
    // 133 keypoints; set shoulders (coco 5,6) and nose (0) to known values.
    let mut kps = vec![[0.0f32; 3]; 133];
    kps[0] = [100.0, 10.0, 5.0]; // nose -> op 0
    kps[5] = [80.0, 40.0, 4.0]; // l_sho -> op 5
    kps[6] = [120.0, 50.0, 6.0]; // r_sho -> op 6
    let rec = wholebody_to_openpose(&kps, 200.0, 100.0);
    // nose normalized
    assert!(approx(rec.keypoints[0][0], 0.5, 1e-6));
    assert!(approx(rec.keypoints[0][1], 0.1, 1e-6));
    // neck (op 1) = midpoint of shoulders, conf = min(4,6) = 4
    assert!(approx(
        rec.keypoints[1][0],
        (80.0 + 120.0) / 2.0 / 200.0,
        1e-6
    ));
    assert!(approx(
        rec.keypoints[1][1],
        (40.0 + 50.0) / 2.0 / 100.0,
        1e-6
    ));
    assert!(approx(rec.keypoints[1][2], 4.0, 1e-6));
    // shapes
    assert_eq!(rec.keypoints.len(), 18);
    assert_eq!(rec.hands[0].len(), 21);
    assert_eq!(rec.hands[1].len(), 21);
    assert_eq!(rec.face.len(), 68);
}

#[test]
fn squareify_pads_short_axis_centered() {
    // landscape 200x100 -> square side 200; y padded by (200-100)/2 = 50.
    let rec = PoseRecord {
        keypoints: vec![[0.5, 0.5, 1.0]; 18],
        hands: [vec![[0.5, 0.5, 1.0]; 21], vec![[0.5, 0.5, 1.0]; 21]],
        face: vec![[0.5, 0.5, 1.0]; 68],
    };
    let sq = squareify(&rec, 200.0, 100.0);
    // x: (0.5*200 + 0)/200 = 0.5 ; y: (0.5*100 + 50)/200 = 0.5  (center stays center)
    assert!(approx(sq.keypoints[0][0], 0.5, 1e-6));
    assert!(approx(sq.keypoints[0][1], 0.5, 1e-6));
    // a top point y=0 -> (0 + 50)/200 = 0.25 (letterboxed inward)
    let rec2 = PoseRecord {
        keypoints: vec![[0.5, 0.0, 1.0]; 18],
        hands: [vec![[0.0, 0.0, 0.0]; 21], vec![[0.0, 0.0, 0.0]; 21]],
        face: vec![[0.0, 0.0, 0.0]; 68],
    };
    let sq2 = squareify(&rec2, 200.0, 100.0);
    assert!(approx(sq2.keypoints[0][1], 0.25, 1e-6));
}

#[test]
fn facing_classifies_head_keypoints() {
    let mut body = vec![[0.0f32; 3]; 18];
    // no nose/eyes -> back
    assert_eq!(facing(&body, 0.3), "back");
    // both ears present -> front
    body[0] = [0.0, 0.0, 1.0]; // nose
    body[16] = [0.0, 0.0, 1.0]; // r_ear
    body[17] = [0.0, 0.0, 1.0]; // l_ear
    assert_eq!(facing(&body, 0.3), "front");
    // only one ear -> profile
    body[17] = [0.0, 0.0, 0.0];
    assert_eq!(facing(&body, 0.3), "profile");
}

#[test]
fn bbox_and_mean_conf_respect_threshold() {
    let group = [[0.1, 0.2, 1.0], [0.8, 0.9, 0.05], [0.4, 0.5, 0.5]];
    // 0.05 point dropped at min_conf 0.3
    let bb = bbox(&[&group], 0.3).expect("bbox");
    assert!(approx(bb[0], 0.1, 1e-6));
    assert!(approx(bb[1], 0.2, 1e-6));
    assert!(approx(bb[2], 0.4, 1e-6));
    assert!(approx(bb[3], 0.5, 1e-6));
    // mean conf uses ALL points (not thresholded)
    let m = mean_conf(&group);
    assert!((m - (1.0 + 0.05 + 0.5) / 3.0).abs() < 1e-3);
    // no points above threshold -> None
    assert!(bbox(&[&[[0.0, 0.0, 0.0]]], 0.3).is_none());
}

#[test]
fn thresholded_drops_low_conf_points() {
    let group = [[0.1, 0.2, 1.0], [0.3, 0.4, 0.1]];
    let t = thresholded(&group, 0.3);
    assert_eq!(t[0], Some((0.1, 0.2)));
    assert_eq!(t[1], None);
}

#[test]
fn yolox_decode_embedded_nms_branch() {
    // dets (1, 2, 5): one above 0.3, one below; ratio 0.5 -> boxes *2.
    let dets = vec![
        10.0, 20.0, 30.0, 40.0, 0.9, // keep
        1.0, 2.0, 3.0, 4.0, 0.1, // drop (score <= 0.3)
    ];
    let boxes = yolox_decode(&dets, &[1, 2, 5], 0.5);
    assert_eq!(boxes.len(), 1);
    assert!(approx(boxes[0].x1, 20.0, 1e-6));
    assert!(approx(boxes[0].y2, 80.0, 1e-6));
    // non-NMS shape (last dim != 5) -> empty (we only ship the embedded-NMS export)
    assert!(yolox_decode(&[0.0; 85], &[1, 1, 85], 1.0).is_empty());
}

#[test]
fn pose_decode_argmax_and_rescale() {
    // 1 keypoint, Wx=4, Wy=4 ; peak at x-bin 2, y-bin 2.
    let k = 1usize;
    let wx = 4usize;
    let wy = 4usize;
    let mut sx = vec![0.0f32; k * wx];
    let mut sy = vec![0.0f32; k * wy];
    sx[2] = 9.0;
    sy[2] = 7.0;
    // crop geometry: center (50,60), scale (PW, PH) so px = (loc/2)/PW*PW + x0.
    let (cx, cy, sw, sh) = (50.0f32, 60.0f32, PW as f32, PH as f32);
    let kp = pose_decode(&sx, &[1, k as i64, wx as i64], &sy, cx, cy, sw, sh);
    assert_eq!(kp.len(), 1);
    // x: loc 2 -> 2/2=1 ; (1)/PW*sw + (cx - sw/2) = 1 + (50 - PW/2)
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    assert!(approx(kp[0][0], 1.0 + x0, 1e-3));
    assert!(approx(kp[0][1], 1.0 + y0, 1e-3));
    // score = 0.5*(9+7) = 8
    assert!(approx(kp[0][2], 8.0, 1e-6));
}

#[test]
fn pose_decode_negative_value_marks_missing() {
    let sx = vec![-1.0f32; 4];
    let sy = vec![-1.0f32; 4];
    let kp = pose_decode(&sx, &[1, 1, 4], &sy, 50.0, 60.0, PW as f32, PH as f32);
    // val <= 0 -> loc (-1,-1) -> negative rescaled coords, score <= 0
    assert!(kp[0][2] <= 0.0);
}

/// sc-5496 off-Mac GPU smoke (ignored — needs the real RTMW onnx weights + CUDA). Validates the
/// Windows/Linux candle DWPose lane end-to-end: load the detector through `Detector::load`, run it on a
/// person photo, and confirm it engages the hardware EP (or CPU fallback), finds ≥1 person, and decodes
/// a sane normalized OpenPose-18 body (neck between the shoulders). The numeric parity vs the Python
/// rtmlib reference was already established in the spike (docs/sc-3487/spike-findings.md); this just
/// proves the off-Mac `ort` CUDA path produces faithful output through the same code. Stage the weights
/// + image via env, then:
///   cargo test -p sceneworks-worker --features backend-candle --lib -- --ignored pose_detect_candle
#[cfg(not(target_os = "macos"))]
#[test]
#[ignore]
fn pose_detect_candle_real_weights_finds_person() {
    let det = std::env::var("SCENEWORKS_DWPOSE_DET")
        .expect("set SCENEWORKS_DWPOSE_DET to the yolox onnx");
    let pose = std::env::var("SCENEWORKS_DWPOSE_POSE")
        .expect("set SCENEWORKS_DWPOSE_POSE to the rtmw onnx");
    let img = std::env::var("SCENEWORKS_TEST_POSE_IMAGE")
        .expect("set SCENEWORKS_TEST_POSE_IMAGE to a person photo");

    let (out, device) = detect_batch(
        PathBuf::from(det),
        PathBuf::from(pose),
        vec![Some(PathBuf::from(img))],
    )
    .expect("detect_batch");
    eprintln!("DWPose execution device = {device}");

    let src = out
        .into_iter()
        .next()
        .flatten()
        .expect("readable test image");
    assert!(
        !src.people.is_empty(),
        "expected at least one detected person"
    );
    let (w, h) = (src.width as f32, src.height as f32);
    let rec = squareify(&wholebody_to_openpose(&src.people[0], w, h), w, h);
    assert_eq!(rec.keypoints.len(), 18);

    // Confident body points are normalized (not raw pixels) and land near the [0,1]
    // square. DWPose legitimately extrapolates a keypoint a little past the frame edge
    // (a body part the crop cuts off), so allow a small overshoot rather than a hard
    // clamp — the assertion's job is to catch raw-pixel leakage (coords in the 100s),
    // not to reject normal edge points.
    for p in &rec.keypoints {
        if p[2] as f64 >= DEFAULT_POSE_MIN_CONF {
            assert!(
                (-0.25..=1.25).contains(&p[0]) && (-0.25..=1.25).contains(&p[1]),
                "confident keypoint not normalized near [0,1]: {p:?}"
            );
        }
    }

    // Neck (op 1) is the shoulder midpoint, so its x sits between the two shoulders (op 2 / op 5).
    let (neck, r_sho, l_sho) = (rec.keypoints[1], rec.keypoints[2], rec.keypoints[5]);
    eprintln!("neck={neck:?} r_sho={r_sho:?} l_sho={l_sho:?}");
    if neck[2] as f64 >= DEFAULT_POSE_MIN_CONF {
        let (lo, hi) = (r_sho[0].min(l_sho[0]), r_sho[0].max(l_sho[0]));
        assert!(
            neck[0] >= lo - 1e-3 && neck[0] <= hi + 1e-3,
            "neck x {} should sit between shoulders [{lo}, {hi}]",
            neck[0]
        );
    }
}
