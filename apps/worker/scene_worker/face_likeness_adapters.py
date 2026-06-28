"""Shared, generator-agnostic face-likeness scorer (epic 4406, sc-4407).

The Windows/Linux InsightFace counterpart of the macOS/CUDA native scorer
(``crates/sceneworks-worker/src/face_likeness.rs``). Given a **source face image** (embedded
ONCE per job) and any number of **generated images**, it returns an identity-likeness result
by cosine-comparing antelopev2 ArcFace embeddings of the largest detected face in each.

Model-independent on purpose: NOT coupled to the InstantID adapter. InstantID, Z-Image
identity, and Flux IP-adapter generations all produce a finished image, and this scorer runs
as a post-pass over that image — so it serves every identity generator through one path. It
reuses the antelopev2 FaceAnalysis app InstantID already provisions
(``instantid_adapter._ensure_antelopev2``), so no new weights.

Result honesty (drives the design): ArcFace is a frontal-identity signal only. A profile /
extreme-angle / no-face generation legitimately has no detectable frontal face — that is an
explicit ``detected: false`` / ``score: None`` result carrying a ``reason``, NOT a misleading
low number. Kept byte-identical in shape to the Rust scorer so both platforms produce the same
``{ score, detected, method, sourceRef, reason? }`` result.

Source-embed caching (an explicit acceptance criterion): ``FaceLikenessScorer`` embeds the
source face exactly once at construction and stores the L2-normalized vector. Every ``score``
call re-embeds only the *generated* image — never the source again.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any, Optional

# The recognition method surfaced on every result. Stable string the asset sidecar (sc-4408)
# and the UI bands (sc-4414) key off — the same antelopev2 ArcFace stack on both platforms.
LIKENESS_METHOD = "arcface_antelopev2"

# SCRFD detection-confidence floor below which a detection is treated as "no reliable frontal
# face". Mirrors the Rust ``MIN_DET_SCORE`` (and the kps-extract ``LOW_CONF_THRESH``): below it
# the face is an unreliable extreme profile / poor framing, so the scorer returns an explicit
# N/A rather than a noisy number.
MIN_DET_SCORE = 0.65

# N/A reason strings — must match the Rust ``NoScoreReason::as_str`` values exactly.
REASON_NO_FACE = "no_face"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_NO_SOURCE_FACE = "no_source_face"
REASON_EMBEDDING_ERROR = "embedding_error"

# SCRFD detector input (square), matches the macOS native path + InstantID.
DET_SIZE = (640, 640)


def face_likeness_backend_available() -> bool:
    """True when InsightFace + onnxruntime + OpenCV are importable (the SCRFD/ArcFace path)."""
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("insightface", "onnxruntime", "cv2")
    )


def _l2_normalized(vec: Any) -> Optional[Any]:
    """L2-normalize a raw ArcFace embedding. ``None`` for an empty / zero-norm vector."""
    import numpy as np

    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        return None
    return arr / norm


def _cosine(a: Any, b: Any) -> Optional[float]:
    """Cosine of two already-L2-normalized vectors (a plain dot). ``None`` on a length mismatch."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or a.size != b.size:
        return None
    return float(np.dot(a, b))


def _largest_face(app: Any, pil_image: Any) -> Optional[Any]:
    """Detect the largest face in a PIL RGB image via the antelopev2 app. ``None`` if no face."""
    import cv2
    import numpy as np

    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None
    return sorted(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )[-1]


def _face_app(settings: Any) -> Any:
    """Load the antelopev2 FaceAnalysis app (CPU EP), reusing InstantID's provisioning."""
    from insightface.app import FaceAnalysis

    from .instantid_adapter import _ensure_antelopev2

    root = _ensure_antelopev2()
    app = FaceAnalysis(
        name="antelopev2", root=str(root), providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    return app


def score_result(
    source_normalized: Optional[Any],
    generated_face: Optional[Any],
    source_ref: Optional[str],
) -> dict[str, Any]:
    """The pure scoring decision (no IO) — the N/A policy, shared by every caller + unit-tested.

    ``source_normalized`` is the cached L2-normalized source embedding (``None`` ⇒ no source
    face). ``generated_face`` is the largest detected face of the generated image (an object
    with ``.det_score`` + ``.normed_embedding`` / ``.embedding``), or ``None`` when SCRFD found
    no face. Returns the acceptance result shape.
    """
    if source_normalized is None:
        return _na(REASON_NO_SOURCE_FACE, source_ref)
    if generated_face is None:
        return _na(REASON_NO_FACE, source_ref)

    det_score = float(getattr(generated_face, "det_score", 0.0))
    if det_score < MIN_DET_SCORE:
        return _na(REASON_LOW_CONFIDENCE, source_ref)

    # Prefer insightface's own normed_embedding; fall back to normalizing the raw embedding.
    normed = getattr(generated_face, "normed_embedding", None)
    if normed is not None:
        gen_normalized = _l2_normalized(normed)
    else:
        gen_normalized = _l2_normalized(getattr(generated_face, "embedding", []))
    if gen_normalized is None:
        return _na(REASON_EMBEDDING_ERROR, source_ref)

    score = _cosine(source_normalized, gen_normalized)
    if score is None:
        return _na(REASON_EMBEDDING_ERROR, source_ref)
    return {
        "score": score,
        "detected": True,
        "method": LIKENESS_METHOD,
        "sourceRef": source_ref,
    }


def _na(reason: str, source_ref: Optional[str]) -> dict[str, Any]:
    """An N/A result — ``detected: False``, ``score: None``, carrying the reason."""
    return {
        "score": None,
        "detected": False,
        "method": LIKENESS_METHOD,
        "sourceRef": source_ref,
        "reason": reason,
    }


class FaceLikenessScorer:
    """Generator-agnostic identity-likeness scorer (sc-4407). Construct once per job from the
    source face image — the source embedding is computed ONCE here and cached — then call
    :meth:`score` for each generated image. The source is never re-embedded.
    """

    def __init__(self, app: Any, source_image: Any, source_ref: Optional[str] = None) -> None:
        self._app = app
        self._source_ref = source_ref
        self._source_embed_count = 0
        face = _largest_face(app, source_image)
        self._source_embed_count = 1
        if face is None:
            self._source_normalized = None
        else:
            normed = getattr(face, "normed_embedding", None)
            self._source_normalized = (
                _l2_normalized(normed)
                if normed is not None
                else _l2_normalized(getattr(face, "embedding", []))
            )

    @classmethod
    def load(
        cls, settings: Any, source_image: Any, source_ref: Optional[str] = None
    ) -> "FaceLikenessScorer":
        """Load the antelopev2 app + embed the source face once."""
        return cls(_face_app(settings), source_image, source_ref)

    @property
    def has_source_face(self) -> bool:
        """``False`` ⇒ the whole job is N/A (no detectable source face) — never an error."""
        return self._source_normalized is not None

    @property
    def source_embed_count(self) -> int:
        """How many times the source was embedded (always <= 1). For the caching test."""
        return self._source_embed_count

    def score(self, generated_image: Any) -> dict[str, Any]:
        """Score one generated image against the CACHED source embedding. Re-embeds only the
        generated image — never the source. Returns the acceptance result shape (an N/A dict for
        the no-face / low-confidence / no-source-face cases, never raising for those)."""
        face = _largest_face(self._app, generated_image)
        return score_result(self._source_normalized, face, self._source_ref)

    def score_or_null(self, generated_image: Any) -> dict[str, Any]:
        """Score one generated image, turning any backend error into a logged ``null`` result —
        the "scoring errors are non-fatal; never block a generation" acceptance criterion."""
        try:
            return self.score(generated_image)
        except Exception as error:  # noqa: BLE001 — non-fatal by contract
            import logging

            logging.getLogger(__name__).warning(
                "face-likeness scoring failed; recording null: %s", error
            )
            return _na(REASON_EMBEDDING_ERROR, self._source_ref)
