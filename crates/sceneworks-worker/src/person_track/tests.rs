use super::*;

fn b(x: f64, y: f64, w: f64, h: f64) -> NormalizedBox {
    NormalizedBox::new(x, y, w, h)
}

#[test]
fn iou_matches_known_values() {
    let a = b(0.0, 0.0, 0.4, 0.4);
    assert!((box_iou(a, a) - 1.0).abs() < 1e-9);
    // disjoint
    assert_eq!(box_iou(b(0.0, 0.0, 0.1, 0.1), b(0.5, 0.5, 0.1, 0.1)), 0.0);
    // half-overlap along x: A=[0,0.2]x[0,0.2], B=[0.1,0.3]x[0,0.2]
    let iou = box_iou(b(0.0, 0.0, 0.2, 0.2), b(0.1, 0.0, 0.2, 0.2));
    // inter = 0.1*0.2=0.02, union = 0.04+0.04-0.02=0.06 → 1/3
    assert!((iou - (1.0 / 3.0)).abs() < 1e-9);
}

#[test]
fn xyxy_normalizes_and_orders_corners() {
    // reversed corners still produce a positive box
    let nb = xyxy_to_normalized(300.0, 200.0, 100.0, 50.0, 1000, 1000);
    assert!((nb.x - 0.1).abs() < 1e-9);
    assert!((nb.y - 0.05).abs() < 1e-9);
    assert!((nb.width - 0.2).abs() < 1e-9);
    assert!((nb.height - 0.15).abs() < 1e-9);
    // zero frame is safe
    assert_eq!(
        xyxy_to_normalized(0.0, 0.0, 1.0, 1.0, 0, 0),
        b(0.0, 0.0, 0.0, 0.0)
    );
}

#[test]
fn sample_cadence_clamps_and_spans_ends() {
    assert_eq!(sample_count_for_duration(0.0), MIN_SAMPLES); // floor
    assert_eq!(sample_count_for_duration(1000.0), MAX_SAMPLES); // ceil
    assert_eq!(sample_count_for_duration(6.0), 12); // 6 * 2fps
    let ts = sample_timestamps(6.0);
    assert_eq!(ts.len(), 12);
    assert_eq!(ts.first().copied(), Some(0.0));
    assert!((ts.last().copied().unwrap() - 6.0).abs() < 1e-6);
    // zero duration → single zero stamp
    assert_eq!(sample_timestamps(0.0), vec![0.0]);
}

#[test]
fn tracker_keeps_one_id_for_a_moving_person() {
    let mut t = PersonTracker::default();
    let mut ids = Vec::new();
    // a person drifting right across frames, high confidence
    for k in 0..5 {
        let x = 0.10 + 0.03 * k as f64;
        let out = t.update(&[(b(x, 0.2, 0.2, 0.6), 0.9)]);
        assert_eq!(out.len(), 1);
        ids.push(out[0].0);
    }
    assert!(
        ids.iter().all(|&id| id == ids[0]),
        "id changed across frames: {ids:?}"
    );
}

#[test]
fn tracker_separates_two_people() {
    let mut t = PersonTracker::default();
    let left = b(0.05, 0.2, 0.2, 0.6);
    let right = b(0.70, 0.2, 0.2, 0.6);
    let f0 = t.update(&[(left, 0.9), (right, 0.88)]);
    assert_eq!(f0.len(), 2);
    let id_left = f0.iter().find(|(_, bx, _)| bx.x < 0.5).unwrap().0;
    let id_right = f0.iter().find(|(_, bx, _)| bx.x >= 0.5).unwrap().0;
    assert_ne!(id_left, id_right);
    // next frame, slight drift — ids stick to the right person
    let f1 = t.update(&[
        (b(0.08, 0.2, 0.2, 0.6), 0.9),
        (b(0.73, 0.2, 0.2, 0.6), 0.88),
    ]);
    assert_eq!(f1.iter().find(|(_, bx, _)| bx.x < 0.5).unwrap().0, id_left);
    assert_eq!(
        f1.iter().find(|(_, bx, _)| bx.x >= 0.5).unwrap().0,
        id_right
    );
}

#[test]
fn tracker_drops_track_after_max_age() {
    let mut t = PersonTracker::default();
    let id0 = t.update(&[(b(0.1, 0.2, 0.2, 0.6), 0.9)])[0].0;
    // disappears for max_age+1 frames
    for _ in 0..4 {
        let out = t.update(&[]);
        assert!(out.is_empty());
    }
    // a fresh detection elsewhere starts a NEW id (old track aged out)
    let id1 = t.update(&[(b(0.7, 0.2, 0.2, 0.6), 0.9)])[0].0;
    assert_ne!(id0, id1);
}

#[test]
fn choose_target_matches_overlapping_id() {
    let obs = observe(vec![(
        0.0,
        vec![(b(0.05, 0.2, 0.2, 0.6), 0.9), (b(0.70, 0.2, 0.2, 0.6), 0.9)],
    )]);
    // selected box overlapping the right person
    let target = choose_target_track_id(&obs, b(0.72, 0.21, 0.2, 0.6), 0.0);
    let right_id = obs[0]
        .boxes
        .iter()
        .find(|(_, bx, _)| bx.x >= 0.5)
        .unwrap()
        .0;
    assert_eq!(target, Some(right_id));
    // a non-overlapping selection finds nothing
    assert_eq!(
        choose_target_track_id(&obs, b(0.4, 0.0, 0.05, 0.05), 0.0),
        None
    );
}

#[test]
fn assemble_marks_detected_and_lost_frames() {
    // 3 sampled frames; the target (a person near x=0.1) is missing in the middle frame.
    let person = |x: f64, c: f64| (b(x, 0.2, 0.2, 0.6), c);
    let obs = observe(vec![
        (0.0, vec![person(0.10, 0.9)]),
        (1.0, vec![]),                  // lost
        (2.0, vec![person(0.16, 0.3)]), // low confidence → flagged
    ]);
    let timestamps = vec![0.0, 1.0, 2.0];
    let assembly = assemble_track(&obs, b(0.10, 0.2, 0.2, 0.6), 0.0, &timestamps);
    assert!(assembly.target_track_id.is_some());
    assert_eq!(assembly.frames.len(), 3);
    assert!(assembly.frames[0].detected);
    assert!(!assembly.frames[1].detected);
    assert_eq!(assembly.frames[1].flags, vec!["lost_target"]);
    // lost frame holds the last known box, not a fabricated one
    assert_eq!(assembly.frames[1].box_, assembly.frames[0].box_);
    assert!(assembly.frames[2].detected);
    assert_eq!(assembly.frames[2].flags, vec!["low_confidence"]);
    assert_eq!(assembly.detected_frames, 2);
    assert_eq!(assembly.quality["detectedFrames"], serde_json::json!(2));
    assert_eq!(assembly.quality["lostFrames"], serde_json::json!([1]));
}

#[test]
fn frames_serialize_to_sidecar_shape() {
    let frames = vec![
        TrackFrame {
            timestamp: 1.23456,
            box_: b(0.1, 0.2, 0.3, 0.4),
            confidence: 0.876_54,
            detected: true,
            flags: vec![],
        },
        TrackFrame {
            timestamp: 2.0,
            box_: b(0.1, 0.2, 0.3, 0.4),
            confidence: 0.0,
            detected: false,
            flags: vec!["lost_target"],
        },
    ];
    let json = frames_to_json(&frames);
    assert_eq!(json[0]["timestamp"], serde_json::json!(1.2346));
    assert_eq!(json[0]["confidence"], serde_json::json!(0.8765));
    assert_eq!(json[0]["detected"], serde_json::json!(true));
    assert!(json[0].get("flags").is_none()); // empty flags omitted
    assert_eq!(json[0]["mask"], Value::Null);
    assert_eq!(json[1]["flags"], serde_json::json!(["lost_target"]));
    assert!((average_confidence(&frames) - 0.8765).abs() < 1e-9);
}
