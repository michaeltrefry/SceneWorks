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

    // Block tensors are NHWC; the oracle is NCHW — transpose back before comparing. Every
    // block is now true-fp32 faithful (max|Δ| ≤ ~1.5e-4). sc-3734 traced the previous
    // ~1e-2 divergence to MLX's Metal `matmul` reduced-precision simdgroup path (NOT fp32
    // backend drift, which is ~1e-5) in the C2PSA attention, and fixed it by running the
    // two SDPA matmuls on the CPU stream (see `attention()` + `yolo11_mlx_per_block_isolation`
    // + docs/sc-3734). Thresholds re-tightened from the loosened 2e-2 accordingly.
    let checks = [
        ("block4", &fwd.block4, 1e-4f32),
        ("block10", &fwd.block10, 1e-3),
        ("block16", &fwd.block16, 1e-3),
        ("block19", &fwd.block19, 1e-3),
        ("block22", &fwd.block22, 1e-3),
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
    // Box rows are DFL-expected-value × stride (up to 32), so a ~1e-5 logit difference is
    // amplified to ~0.5 sub-pixel; classes are now ~2e-5 (was 1.98e-3 before the sc-3734 fix).
    assert!(
        box_d < 1.0,
        "final box rows max|Δ| {box_d:.3e} px exceeds 1px"
    );
    assert!(
        cls_d < 1e-3,
        "final class rows max|Δ| {cls_d:.3e} exceeds 1e-3"
    );
}

/// sc-3734 per-block isolation. Each backbone block 5..10 is fed the *previous* block's
/// clean torch ground-truth output (from `refs_ext.safetensors`, generated by
/// `docs/sc-3734/torch_ref.py`) and run through ONLY that block's Rust op, so its
/// intrinsic Rust-MLX-vs-torch error is measured with ZERO upstream accumulation. This
/// splits the 6-block (block4→block10) gap that the accumulated oracle could not isolate,
/// and pins whether the ~1e-2 divergence originates in SPPF(9) or C2PSA(10) or is just
/// per-block fp32 drift compounding through the depth. Run single-threaded with the
/// extended oracle staged:
///   cargo test -p sceneworks-worker person_jobs::tests::yolo11_mlx_per_block_isolation \
///     -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "requires staged yolo11m_fused_mlx.safetensors + refs_ext.safetensors in the app cache"]
fn yolo11_mlx_per_block_isolation() {
    let (Some(weights), Some(refs)) = (
        cache_fixture("yolo11m_fused_mlx.safetensors"),
        cache_fixture("refs_ext.safetensors"),
    ) else {
        eprintln!("skipping: fixtures not staged (run docs/sc-3734/torch_ref.py first)");
        return;
    };
    let model = Yolo::load(&weights).expect("weights load");
    let oracle = Weights::from_file(&refs).expect("refs_ext load");

    // Load an NCHW oracle tensor as an NHWC Array (the layout the forward runs in).
    let nhwc = |name: &str| {
        oracle
            .require(name)
            .unwrap_or_else(|_| panic!("oracle {name}"))
            .transpose_axes(&[0, 2, 3, 1])
            .expect("nhwc")
    };
    // Compare an NHWC result against an NCHW oracle tensor (transpose result back).
    let diff_nchw = |got_nhwc: &Array, name: &str| -> f32 {
        let got = got_nhwc.transpose_axes(&[0, 3, 1, 2]).expect("nchw");
        max_abs_diff(
            &got,
            oracle.require(name).unwrap_or_else(|_| panic!("{name}")),
        )
    };

    eprintln!("--- per-block isolation (clean input per block, no accumulation) ---");

    // block5: Conv k3 s2 p1 on clean block4.
    let b5 = model
        .conv(&nhwc("block4"), "model.5", 2, 1, true)
        .expect("conv5");
    eprintln!(
        "block5  (Conv)  intrinsic max|Δ| = {:.3e}",
        diff_nchw(&b5, "block5")
    );

    // block6: C3k2 on clean block5.
    let b6 = model.c3k2(&nhwc("block5"), 6, true).expect("c3k2.6");
    eprintln!(
        "block6  (C3k2)  intrinsic max|Δ| = {:.3e}",
        diff_nchw(&b6, "block6")
    );

    // block7: Conv k3 s2 p1 on clean block6.
    let b7 = model
        .conv(&nhwc("block6"), "model.7", 2, 1, true)
        .expect("conv7");
    eprintln!(
        "block7  (Conv)  intrinsic max|Δ| = {:.3e}",
        diff_nchw(&b7, "block7")
    );

    // block8: C3k2 on clean block7.
    let b8 = model.c3k2(&nhwc("block7"), 8, true).expect("c3k2.8");
    eprintln!(
        "block8  (C3k2)  intrinsic max|Δ| = {:.3e}",
        diff_nchw(&b8, "block8")
    );

    // block9: SPPF on clean block8 — the first novel hand-rolled primitive.
    let b9 = model.sppf(&nhwc("block8"), 9).expect("sppf.9");
    let d9 = diff_nchw(&b9, "block9");
    eprintln!("block9  (SPPF)  intrinsic max|Δ| = {d9:.3e}  <-- novel: 25-tap max-pool");

    // block10: C2PSA on clean block9 — the second novel hand-rolled primitive, and (before
    // the sc-3734 fix) the sole carrier of the divergence. With `attention()` running its
    // two SDPA matmuls on the CPU stream, this is now true-fp32.
    let b10 = model.c2psa(&nhwc("block9"), 10).expect("c2psa.10");
    let d10 = diff_nchw(&b10, "block10");
    eprintln!("block10 (C2PSA) intrinsic max|Δ| = {d10:.3e}  <-- novel: attention + PE");

    // Every backbone block, fed clean input, is faithful fp32 (this is what made the
    // accumulated-oracle 400× jump a *localization* signal, not noise — sc-3734).
    for (name, d) in [
        ("block5", diff_nchw(&b5, "block5")),
        ("block6", diff_nchw(&b6, "block6")),
        ("block7", diff_nchw(&b7, "block7")),
        ("block8", diff_nchw(&b8, "block8")),
        ("block9", d9),
        ("block10", d10),
    ] {
        assert!(d < 1e-3, "{name} intrinsic max|Δ| {d:.3e} exceeds 1e-3");
    }

    // C2PSA attention sub-step decomposition. The PE (depthwise conv) and the two SDPA
    // matmuls are isolated against the torch ground truth. The matmuls are computed BOTH on
    // the GPU (raw `matmul`, the reduced-precision Metal simdgroup path sc-3734 diagnosed)
    // and on the CPU stream (true fp32, the production fix) to lock in the contrast.
    use mlx_rs::ops::{matmul, multiply, softmax_axis, split, split_sections};
    use mlx_rs::StreamOrDevice;
    let cv1 = model
        .conv(&nhwc("block9"), "model.10.cv1", 1, 0, true)
        .expect("cv1");
    let parts = split(&cv1, 2, 3).expect("split");
    let b_half = parts[1].clone();

    // Mirror Yolo::attention() projection so x_attn (GPU vs CPU) and pe can be compared.
    let sh = b_half.shape();
    let (h, wd, c) = (sh[1], sh[2], sh[3]);
    let n = h * wd;
    let (nh, kd, hd) = (4i32, 32i32, 64i32);
    let qkv = model
        .conv(&b_half, "model.10.m.0.attn.qkv", 1, 0, false)
        .expect("qkv");
    let qkv = qkv.reshape(&[n, nh, 2 * kd + hd]).expect("reshape qkv");
    let qkvp = split_sections(&qkv, &[kd, 2 * kd], 2).expect("qkv split");
    let (q, k, v) = (&qkvp[0], &qkvp[1], &qkvp[2]);
    let qh = q.transpose_axes(&[1, 0, 2]).expect("qh");
    let kh = k.transpose_axes(&[1, 2, 0]).expect("kh");
    let scale = Array::from_f32((kd as f32).powf(-0.5));
    let vh = v.transpose_axes(&[1, 0, 2]).expect("vh");

    let x_attn = |s: &dyn Fn(&Array, &Array) -> Array| -> f32 {
        let attn = multiply(s(&qh, &kh), &scale).expect("scale");
        let attn = softmax_axis(&attn, -1, true).expect("softmax");
        let xa = s(&attn, &vh)
            .transpose_axes(&[1, 0, 2])
            .expect("t")
            .reshape(&[1, h, wd, c])
            .expect("x_attn");
        diff_nchw(&xa, "attn_xattn")
    };
    let gpu_xattn = x_attn(&|a, b| matmul(a, b).expect("gpu matmul"));
    let cpu = StreamOrDevice::cpu();
    let cpu_xattn = x_attn(&|a, b| a.matmul_device(b, &cpu).expect("cpu matmul"));

    // Optional: dump MLX's own GPU attn/vh/x_attn so docs/sc-3734/mlx_matmul_check.py can
    // recompute attn·v in fp64 from MLX's OWN inputs and confirm the loss is inside MLX's
    // matmul (≈4.8e-3), not our algorithm. Gated — diagnostic only.
    if std::env::var_os("SC3734_DUMP").is_some() {
        let attn = softmax_axis(
            multiply(matmul(&qh, &kh).unwrap(), &scale).unwrap(),
            -1,
            true,
        )
        .expect("attn");
        let x_attn_gpu = matmul(&attn, &vh)
            .unwrap()
            .transpose_axes(&[1, 0, 2])
            .unwrap()
            .reshape(&[1, h, wd, c])
            .unwrap()
            .transpose_axes(&[0, 3, 1, 2])
            .expect("nchw");
        for (nm, a) in [("attn", &attn), ("vh", &vh), ("xattn_chw", &x_attn_gpu)] {
            let bytes: Vec<u8> = flat(a).iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(format!("/tmp/sc3734_{nm}.f32"), bytes).expect("dump");
        }
        eprintln!("  [SC3734_DUMP: wrote /tmp/sc3734_{{attn,vh,xattn_chw}}.f32]");
    }
    let v_nhwc = v.reshape(&[1, h, wd, c]).expect("v_nhwc");
    let pe = model
        .depthwise3x3(&v_nhwc, "model.10.m.0.attn.pe", false)
        .expect("pe");
    let pe_d = diff_nchw(&pe, "attn_pe");
    eprintln!(
        "  attn.x_attn [GPU matmul] max|Δ| = {gpu_xattn:.3e}  <-- reduced-precision Metal path"
    );
    eprintln!(
        "  attn.x_attn [CPU stream] max|Δ| = {cpu_xattn:.3e}  <-- production fix (true fp32)"
    );
    eprintln!("  attn.pe     (depthwise)  max|Δ| = {pe_d:.3e}");

    // The fix's whole point: the CPU-stream SDPA matmul is faithful fp32; the depthwise PE
    // and all convs always were. (The GPU raw matmul stays ~7e-3 — left unasserted so this
    // guard doesn't break if a future MLX makes the Metal matmul precise, which would just
    // mean the CPU detour could be dropped.)
    assert!(
        cpu_xattn < 1e-3,
        "CPU-stream x_attn {cpu_xattn:.3e} exceeds 1e-3"
    );
    assert!(pe_d < 1e-3, "pe {pe_d:.3e} exceeds 1e-3");
    eprintln!(
        "sc-3734: SPPF + depthwise PE + all convs faithful; the entire ~1e-2 divergence was \
         MLX's Metal matmul, fixed by the CPU-stream SDPA."
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
