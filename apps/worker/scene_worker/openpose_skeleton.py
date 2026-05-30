"""OpenPose / DWPose skeleton helpers for the pose library (sc-2064, sc-2257).

The pose library (apps/web/public/poses/) ships normalized 18-point skeletons; the web
sends the selected poses' keypoints in the job. This module renders an OpenPose control
image from those keypoints (matching the controlnet_aux draw_bodypose format the xinsir
SDXL OpenPose ControlNet was trained on — no controlnet_aux dependency) and derives a
small face box from the head keypoints so InstantID can anchor the face before the
face-restoration pass.

`draw_wholebody` (sc-2257) extends this to the full DWPose format — body + optional
hand (21x2) + face (68) landmarks — which the Z-Image Fun-Controlnet-Union pose head
was trained on. Body-only callers (InstantID / Qwen) keep using `draw_bodypose`; the
strict Z-Image tier uses `draw_wholebody` and renders hands/face when a pose carries
them (the bundled poses are body-only today; richer poses come from a DWPose detector).

Keypoint order (COCO-18):
 0 nose 1 neck 2 r_sho 3 r_elb 4 r_wri 5 l_sho 6 l_elb 7 l_wri
 8 r_hip 9 r_kne 10 r_ank 11 l_hip 12 l_kne 13 l_ank 14 r_eye 15 l_eye 16 r_ear 17 l_ear
"""
from __future__ import annotations

import math

import numpy as np

LIMB_SEQ: tuple[tuple[int, int], ...] = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)

# DWPose / OpenPose hand skeleton (21 keypoints): wrist (0) + 4 joints per finger.
# Edge order matches controlnet_aux draw_handpose so the per-edge HSV colouring
# lands the same way the Fun-Controlnet-Union pose head was trained on (sc-2257).
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
)

Keypoint = tuple[float, float] | None


def normalize_points(raw: object, count: int) -> list[Keypoint]:
    """Coerce a job-payload keypoint list into exactly ``count`` normalized
    (x, y) | None points. Accepts [x, y], [x, y, conf] (conf<=0 -> dropped), or
    None per entry. Shared by body (18) / hand (21) / face (68) keypoints."""
    points: list[Keypoint] = []
    items = raw if isinstance(raw, (list, tuple)) else []
    for entry in items:
        if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 2:
            points.append(None)
            continue
        if len(entry) >= 3 and entry[2] is not None and float(entry[2]) <= 0:
            points.append(None)
            continue
        try:
            points.append((float(entry[0]), float(entry[1])))
        except (TypeError, ValueError):
            points.append(None)
    points = points[:count] + [None] * max(0, count - len(points))
    return points


def normalize_keypoints(raw: object) -> list[Keypoint]:
    """Coerce a job-payload keypoint list into exactly 18 normalized body points."""
    return normalize_points(raw, 18)


def normalize_hands(raw: object) -> list[list[Keypoint]] | None:
    """Coerce optional hand keypoints into ``[left21, right21]`` (each a 21-point
    list). Accepts a [left, right] pair or a flat 42-point list; None / empty -> None."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    # A [left, right] pair (each itself a list of points) vs a flat 42-point list.
    if isinstance(raw[0], (list, tuple)) and raw[0] and isinstance(raw[0][0], (list, tuple)):
        hands = [normalize_points(hand, 21) for hand in raw[:2]]
    else:
        flat = normalize_points(raw, 42)
        hands = [flat[:21], flat[21:]]
    while len(hands) < 2:
        hands.append([None] * 21)
    return hands


def normalize_face(raw: object) -> list[Keypoint] | None:
    """Coerce optional face keypoints into 68 normalized points; None/empty -> None."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    return normalize_points(raw, 68)


def square_fit(canvas_w: int, canvas_h: int) -> tuple[int, int, int]:
    """Centered-square placement for normalized [0,1] keypoints, returning
    ``(side, offset_x, offset_y)``. Poses are stored square-canonical (proportions
    preserved by padding the short axis at capture — epic 2282), so render them into
    the largest centered square of a (possibly non-square) canvas and letterbox the
    margins (black = no control signal). A square canvas (w==h) maps 1:1, so square
    generations are byte-identical to the old full-canvas mapping (no regression)."""
    side = min(canvas_w, canvas_h)
    return side, (canvas_w - side) // 2, (canvas_h - side) // 2


def draw_bodypose(canvas_w: int, canvas_h: int, keypoints: list[Keypoint], stickwidth: int = 4) -> np.ndarray:
    """Render an OpenPose (COCO-18) skeleton (black background, colored sticks + joints)
    matching the controlnet_aux format. Returns an RGB uint8 array."""
    import cv2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    side, ox, oy = square_fit(canvas_w, canvas_h)
    pts = [None if p is None else (ox + float(p[0]) * side, oy + float(p[1]) * side) for p in keypoints]

    for i, (a, b) in enumerate(LIMB_SEQ):
        if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
            continue
        xa, ya = pts[a]
        xb, yb = pts[b]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        length = math.hypot(xa - xb, ya - yb)
        angle = math.degrees(math.atan2(ya - yb, xa - xb))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, COLORS[i])

    for i in range(min(18, len(pts))):
        if pts[i] is None:
            continue
        x, y = pts[i]
        cv2.circle(canvas, (int(x), int(y)), stickwidth, COLORS[i], thickness=-1)
    return canvas


def _hsv_to_rgb(hue: float) -> tuple[int, int, int]:
    """HSV->RGB at full saturation/value (mirrors controlnet_aux's per-edge hand
    colouring, which uses matplotlib hsv_to_rgb([h, 1, 1])). No matplotlib dep."""
    i = int(hue * 6) % 6
    f = hue * 6 - int(hue * 6)
    q, t = 1 - f, f
    r, g, b = ((1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q))[i]
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_handpose(canvas: np.ndarray, hands: list[list[Keypoint]], line_width: int, point_radius: int) -> np.ndarray:
    """Draw one or both 21-point hand skeletons onto an existing canvas (in place),
    matching the controlnet_aux/DWPose format: per-edge HSV-coloured finger bones +
    blue joint dots. ``hands`` is a list of 21-point lists (normalized x,y | None)."""
    import cv2

    h, w = canvas.shape[:2]
    side, ox, oy = square_fit(w, h)
    for hand in hands:
        if not hand:
            continue
        pts = [None if p is None else (int(ox + p[0] * side), int(oy + p[1] * side)) for p in hand]
        for index, (a, b) in enumerate(HAND_EDGES):
            if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
                continue
            cv2.line(canvas, pts[a], pts[b], _hsv_to_rgb(index / len(HAND_EDGES)), thickness=line_width)
        for p in pts:
            if p is None:
                continue
            cv2.circle(canvas, p, point_radius, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, faces: list[list[Keypoint]], point_radius: int) -> np.ndarray:
    """Draw face landmark constellations (white dots) onto an existing canvas (in
    place), matching the DWPose face render. ``faces`` is a list of point lists."""
    import cv2

    h, w = canvas.shape[:2]
    side, ox, oy = square_fit(w, h)
    for face in faces:
        if not face:
            continue
        for p in face:
            if p is None:
                continue
            cv2.circle(canvas, (int(ox + p[0] * side), int(oy + p[1] * side)), point_radius, (255, 255, 255), thickness=-1)
    return canvas


def draw_wholebody(
    canvas_w: int,
    canvas_h: int,
    keypoints: list[Keypoint],
    hands: list[list[Keypoint]] | None = None,
    face: list[Keypoint] | None = None,
    stickwidth: int = 4,
) -> np.ndarray:
    """Render a DWPose whole-body control image: the COCO-18 body skeleton plus
    optional hand (21x2) + face (68) landmarks, in the format the Z-Image
    Fun-Controlnet-Union pose head was trained on (sc-2257). Body-only callers can
    keep using ``draw_bodypose``; this is the strict Z-Image tier's renderer. Hand
    bones / face dots scale with resolution so they stay visible at 1024²+."""
    canvas = draw_bodypose(canvas_w, canvas_h, keypoints, stickwidth=stickwidth)
    scale = min(canvas_w, canvas_h)
    if hands:
        line_width = max(2, round(scale * 0.004))
        point_radius = max(2, round(scale * 0.004))
        draw_handpose(canvas, hands, line_width=line_width, point_radius=point_radius)
    if face:
        draw_facepose(canvas, [face], point_radius=max(2, round(scale * 0.003)))
    return canvas


def face_box_from_keypoints(keypoints: list[Keypoint]) -> tuple[float, float, float] | None:
    """(cx, cy, height_frac) for placing the InstantID face kps, derived from the head
    keypoints (nose / eyes / neck). Returns None when the head is not visible (e.g. a
    back view or a pose where the face is occluded), so the adapter disables IdentityNet
    + the face-restoration pass and lets the shared seed carry continuity there."""
    nose = keypoints[0] if len(keypoints) > 0 else None
    r_eye = keypoints[14] if len(keypoints) > 14 else None
    l_eye = keypoints[15] if len(keypoints) > 15 else None
    neck = keypoints[1] if len(keypoints) > 1 else None
    eyes = [e for e in (r_eye, l_eye) if e is not None]
    if nose is None and not eyes:
        return None  # no usable face landmarks

    cx = nose[0] if nose is not None else sum(e[0] for e in eyes) / len(eyes)
    head_ys = [p[1] for p in (nose, r_eye, l_eye) if p is not None]
    top_y = min(head_ys)
    # Estimate face height from the neck->nose span when available (head is ~1.4x that
    # vertical run), else a sensible default; clamp to a small full-body face fraction.
    if neck is not None and nose is not None:
        face_h = abs(neck[1] - nose[1]) * 1.4
    else:
        face_h = 0.09
    face_h = max(0.045, min(0.20, face_h))
    cy = top_y + face_h * 0.45
    return (cx, cy, face_h)
