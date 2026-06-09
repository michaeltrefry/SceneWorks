//! Unit tests for the YOLO11 person detector (sc-3633). The pure detector math
//! (letterbox / decode / NMS / normalization) is covered without weights; the
//! `#[ignore]` parity tests run the native MLX forward against a captured per-block
//! reference oracle (`refs.safetensors`) and reproduce `ultralytics.predict`'s boxes,
//! when the fused MLX weights + fixtures are staged in the app cache.

use super::*;
use mlx_gen::weights::Weights;
use mlx_rs::Array;
use std::path::PathBuf;

#[test]
fn letterbox_centers_pad_on_the_short_axis() {
    // bus.jpg is 810×1080 (w×h): fit-by-height → 480×640, 80px pad each side in x.
    let lb = Letterbox::compute(810, 1080);
    assert!((lb.ratio - 0.59259).abs() < 1e-4, "ratio {}", lb.ratio);
    assert_eq!(lb.pad_x, 80.0);
    assert_eq!(lb.pad_y, 0.0);
}

#[test]
fn un_letterbox_inverts_the_forward_mapping() {
    let lb = Letterbox::compute(810, 1080);
    // A point at original (300, 540) maps forward then back to itself.
    let fx = 300.0 * lb.ratio + lb.pad_x;
    let fy = 540.0 * lb.ratio + lb.pad_y;
    assert!((lb.un_x(fx) - 300.0).abs() < 1e-3);
    assert!((lb.un_y(fy) - 540.0).abs() < 1e-3);
}

#[test]
fn decode_keeps_person_anchors_above_conf_and_builds_xyxy() {
    // (1,6,2): 4 box channels + 2 class channels, 2 anchors, channel-major.
    let shape = [1_i64, 6, 2];
    // anchor0: strong person; anchor1: below conf.
    let data = vec![
        100.0, 200.0, // cx
        100.0, 200.0, // cy
        40.0, 10.0, // w
        80.0, 10.0, // h
        0.9, 0.1, // class0 = person
        0.1, 0.05, // class1 = other
    ];
    let lb = Letterbox {
        ratio: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    let dets = decode(&data, &shape, &lb, 0.25, 640, 640);
    assert_eq!(dets.len(), 1, "only the above-conf person anchor survives");
    let d = dets[0];
    assert!((d.x1 - 80.0).abs() < 1e-3);
    assert!((d.y1 - 60.0).abs() < 1e-3);
    assert!((d.x2 - 120.0).abs() < 1e-3);
    assert!((d.y2 - 140.0).abs() < 1e-3);
    assert!((d.score - 0.9).abs() < 1e-6);
}

#[test]
fn decode_clamps_boxes_to_the_frame() {
    let shape = [1_i64, 5, 1];
    // A box hanging off the top-left corner.
    let data = vec![10.0, 10.0, 60.0, 60.0, 0.9];
    let lb = Letterbox {
        ratio: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    let dets = decode(&data, &shape, &lb, 0.25, 640, 640);
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].x1, 0.0);
    assert_eq!(dets[0].y1, 0.0);
}

#[test]
fn nms_drops_high_overlap_keeps_disjoint() {
    let strong = Detection {
        x1: 0.0,
        y1: 0.0,
        x2: 100.0,
        y2: 100.0,
        score: 0.9,
    };
    let overlap = Detection {
        x1: 5.0,
        y1: 5.0,
        x2: 105.0,
        y2: 105.0,
        score: 0.8,
    };
    let disjoint = Detection {
        x1: 400.0,
        y1: 400.0,
        x2: 500.0,
        y2: 500.0,
        score: 0.7,
    };
    let kept = nms(vec![overlap, disjoint, strong], NMS_IOU);
    assert_eq!(kept.len(), 2, "overlapping duplicate suppressed");
    assert!((kept[0].score - 0.9).abs() < 1e-6, "highest score first");
    assert!(kept.iter().any(|d| (d.score - 0.7).abs() < 1e-6));
}

#[test]
fn detections_to_json_normalizes_orders_and_drops_degenerate() {
    let dets = vec![
        Detection {
            x1: 64.0,
            y1: 72.0,
            x2: 320.0,
            y2: 360.0,
            score: 0.5,
        },
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 128.0,
            y2: 72.0,
            score: 0.95,
        },
        // degenerate (zero width) — dropped.
        Detection {
            x1: 10.0,
            y1: 10.0,
            x2: 10.0,
            y2: 50.0,
            score: 0.99,
        },
    ];
    let json = detections_to_json(&dets, 640, 360);
    assert_eq!(json.len(), 2, "degenerate box dropped");
    // Confidence-descending: the 0.95 box becomes person_1.
    assert_eq!(json[0]["id"], "person_1");
    assert_eq!(json[0]["label"], "Person 1");
    assert!((json[0]["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-9);
    let b = &json[0]["box"];
    assert!((b["x"].as_f64().unwrap() - 0.0).abs() < 1e-9);
    assert!((b["width"].as_f64().unwrap() - 0.2).abs() < 1e-9); // 128/640
    assert!((b["height"].as_f64().unwrap() - 0.2).abs() < 1e-9); // 72/360
    assert_eq!(json[1]["id"], "person_2");
    assert_eq!(json[0]["frameWidth"], 640);
    assert_eq!(json[0]["maskState"], "missing");
}

/// Cache fixtures staged during development (sc-3633): the exported detector,
/// the bus.jpg test image, and the `ultralytics.predict` reference detections.
fn cache_fixture(name: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/cache/person-detect")
        .join(name);
    path.exists().then_some(path)
}

/// Flatten an array to a contiguous f32 vec in logical row-major order (forces a copy
/// of a transposed view), so two arrays of equal shape compare element-by-element.
fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .expect("flatten")
        .as_slice::<f32>()
        .to_vec()
}

/// Max absolute difference between two equally-shaped arrays.
fn max_abs_diff(got: &Array, want: &Array) -> f32 {
    assert_eq!(got.shape(), want.shape(), "shape mismatch");
    flat(got)
        .iter()
        .zip(flat(want))
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

/// Per-block + final parity: the native MLX forward, fed the oracle's exact letterboxed
/// input, must match the captured reference (`refs.safetensors`) block-for-block. This is
/// the make-or-break correctness gate for the port — it isolates the forward pass from
/// the (separately-verified) letterbox/decode/NMS math. Ignored by default — run with the
/// fused weights + oracle staged in the app cache:
///   cargo test -p sceneworks-worker person_jobs -- --ignored --nocapture
#[test]
#[ignore = "requires staged yolo11m_fused_mlx.safetensors + refs.safetensors in the app cache"]
fn yolo11_mlx_forward_matches_reference_oracle() {
    let (Some(weights), Some(refs)) = (
        cache_fixture("yolo11m_fused_mlx.safetensors"),
        cache_fixture("refs.safetensors"),
    ) else {
        eprintln!("skipping: fixtures not staged");
        return;
    };

    let model = Yolo::load(&weights).expect("weights load");
    let oracle = Weights::from_file(&refs).expect("refs load");
    // The oracle input is NCHW (1,3,640,640); the forward runs NHWC.
    let input = oracle
        .require("input")
        .expect("input tensor")
        .transpose_axes(&[0, 2, 3, 1])
        .expect("nhwc");
    let fwd = model.run(&input).expect("forward runs");

    // Block tensors are NHWC; the oracle is NCHW — transpose back before comparing.
    // block4 is near-bit-exact (proves the whole backbone CSP assembly). The deeper
    // blocks accumulate fp32 backend drift (MLX/Metal vs the torch oracle, ~1e-2 by the
    // neck — first jumping at the C2PSA softmax/matmul); that drift is benign and proven
    // not to move results by the box/class assertions below + the e2e box parity test.
    let checks = [
        ("block4", &fwd.block4, 1e-3f32),
        ("block10", &fwd.block10, 2e-2),
        ("block16", &fwd.block16, 2e-2),
        ("block19", &fwd.block19, 2e-2),
        ("block22", &fwd.block22, 2e-2),
    ];
    for (name, got_nhwc, tol) in checks {
        let got = got_nhwc.transpose_axes(&[0, 3, 1, 2]).expect("nchw");
        let diff = max_abs_diff(&got, oracle.require(name).expect(name));
        eprintln!("{name}: max|Δ| = {diff:.3e} (tol {tol:.0e})");
        assert!(diff < tol, "{name} max|Δ| {diff:.3e} exceeds {tol:.0e}");
    }

    // The head output `(1,84,8400)`: rows 0..4 are box geometry in letterbox px, rows
    // 4..84 are class probabilities. Assert them separately — box error is a sub-pixel
    // tolerance, class error a tight probability tolerance.
    let want_final = oracle.require("final").expect("final");
    let gf = flat(&fwd.output);
    let wf = flat(want_final);
    let (mut box_d, mut cls_d) = (0.0f32, 0.0f32);
    for r in 0..84usize {
        for a in 0..8400usize {
            let d = (gf[r * 8400 + a] - wf[r * 8400 + a]).abs();
            if r < 4 {
                box_d = box_d.max(d);
            } else {
                cls_d = cls_d.max(d);
            }
        }
    }
    eprintln!("final box-rows(0..4) max|Δ| = {box_d:.3e} px");
    eprintln!("final cls-rows(4..84) max|Δ| = {cls_d:.3e}");
    assert!(
        box_d < 2.0,
        "final box rows max|Δ| {box_d:.3e} px exceeds 2px"
    );
    assert!(
        cls_d < 5e-3,
        "final class rows max|Δ| {cls_d:.3e} exceeds 5e-3"
    );
}

/// End-to-end parity: the full detector (letterbox → MLX forward → decode → NMS) must
/// reproduce `ultralytics.predict`'s 4 people on the staged photo. Ignored by default —
/// run with the model + fixtures staged in the app cache:
///   cargo test -p sceneworks-worker person_jobs -- --ignored --nocapture
#[test]
#[ignore = "requires staged yolo11m_fused_mlx.safetensors + people.jpg fixtures in the app cache"]
fn yolo11_matches_ultralytics_reference_on_photo() {
    let (Some(weights), Some(image), Some(reference)) = (
        cache_fixture("yolo11m_fused_mlx.safetensors"),
        cache_fixture("people.jpg"),
        cache_fixture("ref_people.json"),
    ) else {
        eprintln!("skipping: fixtures not staged");
        return;
    };

    let result = detect_people_blocking(weights, image, 0.25).expect("detection runs");
    eprintln!(
        "device={} detections={}",
        result.device,
        result.detections.len()
    );
    assert_eq!(
        (result.width, result.height),
        (810, 1080),
        "people.jpg dims"
    );

    let ref_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let ref_dets = ref_json["dets"].as_array().unwrap();
    assert_eq!(
        result.detections.len(),
        ref_dets.len(),
        "person count must match ultralytics ({} ref)",
        ref_dets.len()
    );

    // Each reference box must have a Rust box within ~2px corners + ~0.02 conf.
    let mut rust = result.detections.clone();
    rust.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    for r in ref_dets {
        let rb = r["xyxy"].as_array().unwrap();
        let (rx1, ry1, rx2, ry2) = (
            rb[0].as_f64().unwrap() as f32,
            rb[1].as_f64().unwrap() as f32,
            rb[2].as_f64().unwrap() as f32,
            rb[3].as_f64().unwrap() as f32,
        );
        let rconf = r["conf"].as_f64().unwrap() as f32;
        let matched = rust.iter().find(|d| {
            (d.x1 - rx1).abs() < 2.0
                && (d.y1 - ry1).abs() < 2.0
                && (d.x2 - rx2).abs() < 2.0
                && (d.y2 - ry2).abs() < 2.0
                && (d.score - rconf).abs() < 0.02
        });
        assert!(
            matched.is_some(),
            "no Rust box matches ref [{rx1},{ry1},{rx2},{ry2}] conf {rconf}"
        );
    }
}

/// Provisioning parity: download the fused weights from the public HuggingFace URL
/// (`DET_URL`) into a throwaway dir, then prove a fresh download loads and detects the 4
/// reference people. Validates the URL + that the hosted artifact is the right weights.
/// Ignored by default (network); run with the people.jpg fixture staged:
///   cargo test -p sceneworks-worker person_jobs -- --ignored --nocapture
#[test]
#[ignore = "network: downloads the fused weights from HuggingFace"]
fn yolo11_downloads_and_detects_from_huggingface() {
    let Some(image) = cache_fixture("people.jpg") else {
        eprintln!("skipping: people.jpg not staged");
        return;
    };
    let dir = std::env::temp_dir().join("sceneworks-person-detect-dl-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("yolo11m_fused_mlx.safetensors");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = reqwest::Client::new();
        let bytes = client
            .get(DET_URL)
            .send()
            .await
            .expect("GET weights")
            .error_for_status()
            .expect("200 OK")
            .bytes()
            .await
            .expect("body");
        tokio::fs::write(&target, &bytes).await.unwrap();
    });
    eprintln!(
        "downloaded {} bytes → {}",
        std::fs::metadata(&target).unwrap().len(),
        target.display()
    );

    let result =
        detect_people_blocking(target, image, 0.25).expect("detection runs on downloaded weights");
    assert_eq!(
        result.detections.len(),
        4,
        "4 people from downloaded weights"
    );
}
