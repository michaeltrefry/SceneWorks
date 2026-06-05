//! DWPose / OpenPose skeleton rendering for the Z-Image strict-pose ControlNet
//! (epic 3018, sc-3028). A faithful CPU-raster port of the Python worker's
//! `openpose_skeleton.py` (`draw_wholebody`): the COCO-18 body skeleton (colored
//! rotated-ellipse limbs + joint dots) plus optional hand (21×2) and face (68)
//! landmarks, in the format the `Z-Image-Turbo-Fun-Controlnet-Union-2.1` pose head
//! was trained on. Pure raster (cross-platform + testable everywhere); only the MLX
//! control generation that consumes the skeleton is macOS-gated.
//!
//! Keypoints arrive square-canonical (normalized `[0,1]`, short axis padded at
//! capture — epic 2282), so they render into the largest centered square of the
//! (possibly non-square) canvas with black letterbox margins.

use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_polygon_mut};
use imageproc::point::Point;
use serde_json::Value;

/// A normalized keypoint in `[0,1]` square-canonical space, or `None` (dropped /
/// low-confidence).
pub type Keypoint = Option<(f32, f32)>;

/// One hand's 21 keypoints.
pub type Hand = Vec<Keypoint>;

/// COCO-18 limb connections (joint index pairs), in render order.
const LIMB_SEQ: [(usize, usize); 17] = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
];

/// Per-limb / per-joint colors (RGB), matching controlnet_aux's `draw_bodypose`.
const COLORS: [(u8, u8, u8); 18] = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
];

/// DWPose hand skeleton (21 keypoints): wrist (0) + 4 joints per finger. Edge order
/// matches controlnet_aux `draw_handpose` so the per-edge HSV coloring lands the same
/// way the Fun-Controlnet-Union pose head was trained on.
const HAND_EDGES: [(usize, usize); 20] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

/// Coerce a job-payload keypoint list into exactly `count` normalized points.
/// Accepts `[x, y]`, `[x, y, conf]` (conf ≤ 0 → dropped), or `null` per entry.
/// Mirrors Python `normalize_points`.
pub fn normalize_points(raw: &Value, count: usize) -> Vec<Keypoint> {
    let mut points: Vec<Keypoint> = Vec::with_capacity(count);
    if let Some(items) = raw.as_array() {
        for entry in items {
            points.push(parse_keypoint(entry));
        }
    }
    points.truncate(count);
    points.resize(count, None);
    points
}

fn parse_keypoint(entry: &Value) -> Keypoint {
    let array = entry.as_array()?;
    if array.len() < 2 {
        return None;
    }
    // A present, non-positive confidence drops the point (Python `conf <= 0`).
    if let Some(conf) = array.get(2).and_then(Value::as_f64) {
        if conf <= 0.0 {
            return None;
        }
    }
    let x = array.first().and_then(Value::as_f64)?;
    let y = array.get(1).and_then(Value::as_f64)?;
    Some((x as f32, y as f32))
}

/// 18 normalized body keypoints (COCO-18).
pub fn normalize_keypoints(raw: &Value) -> Vec<Keypoint> {
    normalize_points(raw, 18)
}

/// Optional hand keypoints as `[left21, right21]`. Accepts a `[left, right]` pair or a
/// flat 42-point list; `None`/empty → `None`. Mirrors Python `normalize_hands`.
pub fn normalize_hands(raw: &Value) -> Option<Vec<Vec<Keypoint>>> {
    let items = raw.as_array().filter(|items| !items.is_empty())?;
    // A [left, right] pair (each a list of points) vs a flat 42-point list: the pair
    // form has an array whose first element is itself an array.
    let is_pair = items
        .first()
        .and_then(Value::as_array)
        .and_then(|first| first.first())
        .map(Value::is_array)
        .unwrap_or(false);
    let mut hands: Vec<Vec<Keypoint>> = if is_pair {
        items
            .iter()
            .take(2)
            .map(|hand| normalize_points(hand, 21))
            .collect()
    } else {
        let flat = normalize_points(raw, 42);
        vec![flat[..21].to_vec(), flat[21..].to_vec()]
    };
    while hands.len() < 2 {
        hands.push(vec![None; 21]);
    }
    Some(hands)
}

/// Optional face keypoints as 68 normalized points; `None`/empty → `None`.
pub fn normalize_face(raw: &Value) -> Option<Vec<Keypoint>> {
    let has_points = raw.as_array().is_some_and(|items| !items.is_empty());
    has_points.then(|| normalize_points(raw, 68))
}

/// Centered-square placement for normalized `[0,1]` keypoints: `(side, off_x, off_y)`.
fn square_fit(canvas_w: u32, canvas_h: u32) -> (f32, f32, f32) {
    let side = canvas_w.min(canvas_h);
    let off_x = (canvas_w - side) / 2;
    let off_y = (canvas_h - side) / 2;
    (side as f32, off_x as f32, off_y as f32)
}

/// Place a normalized keypoint into canvas pixel space via the centered square.
fn placed(point: Keypoint, side: f32, ox: f32, oy: f32) -> Option<(f32, f32)> {
    point.map(|(x, y)| (ox + x * side, oy + y * side))
}

/// HSV→RGB at full saturation/value (matplotlib `hsv_to_rgb([h, 1, 1])`), for the
/// per-edge hand-bone coloring. Mirrors Python `_hsv_to_rgb`.
fn hsv_to_rgb(hue: f32) -> Rgb<u8> {
    let i = (hue * 6.0) as i32 % 6;
    let f = hue * 6.0 - (hue * 6.0).floor();
    let (q, t) = (1.0 - f, f);
    let (r, g, b) = match i {
        0 => (1.0, t, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, t),
        3 => (0.0, q, 1.0),
        4 => (t, 0.0, 1.0),
        _ => (1.0, 0.0, q),
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

/// `cv2.ellipse2Poly`: sample an ellipse (`axes` half-extents, rotated by `angle_deg`)
/// at 1° steps into a closed polygon. 360 distinct vertices (first ≠ last).
fn ellipse2poly(cx: f32, cy: f32, ax: f32, ay: f32, angle_deg: f32) -> Vec<Point<i32>> {
    let alpha = angle_deg.to_radians();
    let (sa, ca) = (alpha.sin(), alpha.cos());
    (0..360)
        .map(|theta| {
            let rad = (theta as f32).to_radians();
            let (x, y) = (ax * rad.cos(), ay * rad.sin());
            let px = cx + x * ca - y * sa;
            let py = cy + x * sa + y * ca;
            Point::new(px.round() as i32, py.round() as i32)
        })
        .collect()
}

/// Fill a convex polygon (`cv2.fillConvexPoly`). No-ops on a degenerate polygon
/// (imageproc panics if the first and last points coincide / it is empty).
fn fill_poly(canvas: &mut RgbImage, poly: &[Point<i32>], color: Rgb<u8>) {
    if poly.len() < 3 || poly.first() == poly.last() {
        return;
    }
    draw_polygon_mut(canvas, poly, color);
}

/// A thick line as a filled rotated rectangle (`cv2.line` with `thickness`).
fn draw_thick_line(
    canvas: &mut RgbImage,
    a: (f32, f32),
    b: (f32, f32),
    width: f32,
    color: Rgb<u8>,
) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return;
    }
    // Unit normal, scaled to half the line width.
    let (nx, ny) = (-dy / len * width / 2.0, dx / len * width / 2.0);
    let quad = [
        Point::new((a.0 + nx).round() as i32, (a.1 + ny).round() as i32),
        Point::new((b.0 + nx).round() as i32, (b.1 + ny).round() as i32),
        Point::new((b.0 - nx).round() as i32, (b.1 - ny).round() as i32),
        Point::new((a.0 - nx).round() as i32, (a.1 - ny).round() as i32),
    ];
    fill_poly(canvas, &quad, color);
}

/// Render the COCO-18 body skeleton onto a fresh black canvas.
fn draw_bodypose(
    canvas_w: u32,
    canvas_h: u32,
    keypoints: &[Keypoint],
    stickwidth: u32,
) -> RgbImage {
    let mut canvas = RgbImage::new(canvas_w, canvas_h);
    let (side, ox, oy) = square_fit(canvas_w, canvas_h);
    let pts: Vec<Option<(f32, f32)>> = keypoints.iter().map(|p| placed(*p, side, ox, oy)).collect();
    let stick = stickwidth as f32;

    for (i, (a, b)) in LIMB_SEQ.iter().enumerate() {
        let (Some((xa, ya)), Some((xb, yb))) = (
            pts.get(*a).copied().flatten(),
            pts.get(*b).copied().flatten(),
        ) else {
            continue;
        };
        let (mx, my) = ((xa + xb) / 2.0, (ya + yb) / 2.0);
        let length = (xa - xb).hypot(ya - yb);
        let angle = (ya - yb).atan2(xa - xb).to_degrees();
        let poly = ellipse2poly(mx, my, length / 2.0, stick, angle);
        fill_poly(&mut canvas, &poly, rgb(COLORS[i]));
    }

    for (i, point) in pts.iter().enumerate().take(18) {
        if let Some((x, y)) = point {
            draw_filled_circle_mut(
                &mut canvas,
                (x.round() as i32, y.round() as i32),
                stickwidth as i32,
                rgb(COLORS[i]),
            );
        }
    }
    canvas
}

/// Draw one or both 21-point hand skeletons in place: per-edge HSV finger bones + blue
/// joint dots.
fn draw_handpose(
    canvas: &mut RgbImage,
    hands: &[Vec<Keypoint>],
    line_width: f32,
    point_radius: i32,
) {
    let (side, ox, oy) = square_fit(canvas.width(), canvas.height());
    for hand in hands {
        let pts: Vec<Option<(f32, f32)>> = hand.iter().map(|p| placed(*p, side, ox, oy)).collect();
        for (index, (a, b)) in HAND_EDGES.iter().enumerate() {
            let (Some(pa), Some(pb)) = (
                pts.get(*a).copied().flatten(),
                pts.get(*b).copied().flatten(),
            ) else {
                continue;
            };
            let color = hsv_to_rgb(index as f32 / HAND_EDGES.len() as f32);
            draw_thick_line(canvas, pa, pb, line_width, color);
        }
        for point in pts.iter().flatten() {
            draw_filled_circle_mut(
                canvas,
                (point.0.round() as i32, point.1.round() as i32),
                point_radius,
                Rgb([0, 0, 255]),
            );
        }
    }
}

/// Draw face landmark constellations (white dots) in place.
fn draw_facepose(canvas: &mut RgbImage, face: &[Keypoint], point_radius: i32) {
    let (side, ox, oy) = square_fit(canvas.width(), canvas.height());
    for point in face.iter().flatten() {
        let (x, y) = (ox + point.0 * side, oy + point.1 * side);
        draw_filled_circle_mut(
            canvas,
            (x.round() as i32, y.round() as i32),
            point_radius,
            Rgb([255, 255, 255]),
        );
    }
}

/// Render a DWPose whole-body control image: the COCO-18 body skeleton plus optional
/// hand (21×2) + face (68) landmarks. `stickwidth` is the body stick/joint size (the
/// caller scales it to the output resolution); hand bones / face dots scale here.
pub fn draw_wholebody(
    canvas_w: u32,
    canvas_h: u32,
    keypoints: &[Keypoint],
    hands: Option<&[Vec<Keypoint>]>,
    face: Option<&[Keypoint]>,
    stickwidth: u32,
) -> RgbImage {
    let mut canvas = draw_bodypose(canvas_w, canvas_h, keypoints, stickwidth);
    let scale = canvas_w.min(canvas_h) as f32;
    if let Some(hands) = hands.filter(|hands| !hands.is_empty()) {
        let line_width = (scale * 0.004).round().max(2.0);
        let point_radius = (scale * 0.004).round().max(2.0) as i32;
        draw_handpose(&mut canvas, hands, line_width, point_radius);
    }
    if let Some(face) = face.filter(|face| !face.is_empty()) {
        let point_radius = (scale * 0.003).round().max(2.0) as i32;
        draw_facepose(&mut canvas, face, point_radius);
    }
    canvas
}

/// The body stick/joint width for an output resolution: `max(6, min(w,h) * 0.012)`
/// (~12px @1024²). Mirrors the Python adapter's `stick`.
pub fn body_stickwidth(width: u32, height: u32) -> u32 {
    ((width.min(height) as f32 * 0.012).round() as u32).max(6)
}

fn rgb(color: (u8, u8, u8)) -> Rgb<u8> {
    Rgb([color.0, color.1, color.2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_points_pads_truncates_and_drops_low_confidence() {
        // [x,y], [x,y,conf>0], conf<=0 → None, null → None, short → None.
        let raw = json!([[0.1, 0.2], [0.3, 0.4, 0.9], [0.5, 0.6, 0.0], null, [0.7]]);
        let pts = normalize_points(&raw, 6);
        assert_eq!(pts.len(), 6);
        assert_eq!(pts[0], Some((0.1, 0.2)));
        assert_eq!(pts[1], Some((0.3, 0.4)));
        assert_eq!(pts[2], None); // conf == 0 dropped
        assert_eq!(pts[3], None); // null
        assert_eq!(pts[4], None); // length < 2
        assert_eq!(pts[5], None); // padded
    }

    #[test]
    fn normalize_keypoints_is_always_18() {
        assert_eq!(normalize_keypoints(&json!([])).len(), 18);
        assert_eq!(normalize_keypoints(&json!([[0.5, 0.5]])).len(), 18);
    }

    #[test]
    fn normalize_hands_accepts_pair_and_flat() {
        let pair = json!([[[0.1, 0.1]], [[0.2, 0.2]]]);
        let hands = normalize_hands(&pair).unwrap();
        assert_eq!(hands.len(), 2);
        assert_eq!(hands[0].len(), 21);
        assert_eq!(hands[0][0], Some((0.1, 0.1)));

        let flat: Vec<Value> = (0..42).map(|i| json!([i as f32 / 42.0, 0.5])).collect();
        let hands = normalize_hands(&json!(flat)).unwrap();
        assert_eq!(hands.len(), 2);
        assert_eq!(hands[0].len(), 21);
        assert_eq!(hands[1].len(), 21);

        assert!(normalize_hands(&json!([])).is_none());
    }

    #[test]
    fn square_fit_letterboxes_non_square() {
        assert_eq!(square_fit(100, 100), (100.0, 0.0, 0.0));
        // Wide canvas: square is the height, centered horizontally.
        assert_eq!(square_fit(200, 100), (100.0, 50.0, 0.0));
        // Tall canvas: square is the width, centered vertically.
        assert_eq!(square_fit(100, 200), (100.0, 0.0, 50.0));
    }

    #[test]
    fn body_stickwidth_scales_and_floors() {
        assert_eq!(body_stickwidth(1024, 1024), 12);
        assert_eq!(body_stickwidth(512, 512), 6); // 512*0.012=6.14 → 6
        assert_eq!(body_stickwidth(256, 256), 6); // floor
        assert_eq!(body_stickwidth(2048, 2048), 25); // 24.576 → 25
    }

    #[test]
    fn draw_wholebody_paints_a_skeleton_on_black() {
        // A minimal torso (neck→hips) renders colored pixels on a black canvas.
        let kp = normalize_keypoints(&json!([
            [0.5, 0.2],
            [0.5, 0.35],
            [0.42, 0.35],
            [0.40, 0.5],
            [0.40, 0.65],
            [0.58, 0.35],
            [0.60, 0.5],
            [0.60, 0.65],
            [0.45, 0.6],
            [0.45, 0.8],
            [0.45, 0.95],
            [0.55, 0.6],
            [0.55, 0.8],
            [0.55, 0.95],
            [0.48, 0.18],
            [0.52, 0.18],
            [0.46, 0.2],
            [0.54, 0.2]
        ]));
        let img = draw_wholebody(128, 128, &kp, None, None, body_stickwidth(128, 128));
        assert_eq!((img.width(), img.height()), (128, 128));
        let non_black = img.pixels().filter(|p| p.0 != [0, 0, 0]).count();
        assert!(non_black > 0, "expected a rendered skeleton");
        // The top joint (nose, COLORS[0] = red) should appear somewhere.
        assert!(
            img.pixels().any(|p| p.0 == [255, 0, 0]),
            "expected red nose joint"
        );
    }

    #[test]
    fn draw_wholebody_empty_pose_is_all_black() {
        let kp = normalize_keypoints(&json!([]));
        let img = draw_wholebody(64, 64, &kp, None, None, body_stickwidth(64, 64));
        assert!(img.pixels().all(|p| p.0 == [0, 0, 0]));
    }

    #[test]
    fn hsv_to_rgb_spans_the_wheel() {
        assert_eq!(hsv_to_rgb(0.0), Rgb([255, 0, 0])); // red
                                                       // Mid-wheel is cyan-ish (green+blue present, red absent).
        let mid = hsv_to_rgb(0.5);
        assert_eq!(mid.0[0], 0);
    }
}
