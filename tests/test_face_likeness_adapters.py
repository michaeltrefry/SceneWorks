"""Unit tests for the shared face-likeness scorer adapter (epic 4406, sc-4407).

Cover the pure cosine / N-A scoring policy (kept in parity with the Rust
``face_likeness::score_against_source``), the backend-availability gate, and the
``FaceLikenessScorer`` caching contract with a fake antelopev2 app so coverage needs neither
the onnx weights nor a GPU. Only numpy is required (light-dep, runs in the CI parity lane).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scene_worker import face_likeness_adapters as fa

np = pytest.importorskip("numpy")


def _face(det_score: float, embedding):
    """A fake insightface ``Face``: a det_score + a raw embedding (no normed_embedding, so the
    scorer normalizes the raw vector — exercising the same path as the Rust core)."""
    return SimpleNamespace(
        det_score=det_score,
        embedding=np.asarray(embedding, dtype=np.float32),
        bbox=[0.0, 0.0, 100.0, 100.0],
    )


def test_backend_available_reflects_optional_deps(monkeypatch):
    monkeypatch.setattr(fa.importlib.util, "find_spec", lambda _name: object())
    assert fa.face_likeness_backend_available() is True
    monkeypatch.setattr(
        fa.importlib.util,
        "find_spec",
        lambda name: None if name == "insightface" else object(),
    )
    assert fa.face_likeness_backend_available() is False


def test_same_identity_scores_high():
    source = fa._l2_normalized([1.0, 0.2, 0.1, 0.05])
    gen = _face(0.95, [0.98, 0.21, 0.11, 0.04])
    result = fa.score_result(source, gen, "asset_a")
    assert result["detected"] is True
    assert result["score"] > 0.99
    assert result["method"] == "arcface_antelopev2"
    assert result["sourceRef"] == "asset_a"
    assert "reason" not in result


def test_different_identity_scores_low():
    # A nearly-orthogonal embedding ⇒ a real low score — still detected (a face WAS found).
    source = fa._l2_normalized([1.0, 0.0, 0.0, 0.0])
    gen = _face(0.92, [0.02, 1.0, 0.0, 0.0])
    result = fa.score_result(source, gen, "asset_a")
    assert result["detected"] is True
    assert result["score"] < 0.1


def test_no_generated_face_is_na_not_a_low_score():
    # The honesty linchpin: no detectable face ⇒ detected:False / score:None / reason no_face,
    # NEVER a misleading low number.
    source = fa._l2_normalized([1.0, 0.0])
    result = fa.score_result(source, None, "asset_a")
    assert result["detected"] is False
    assert result["score"] is None
    assert result["reason"] == "no_face"


def test_low_confidence_detection_is_na():
    source = fa._l2_normalized([1.0, 0.0])
    gen = _face(fa.MIN_DET_SCORE - 0.01, [1.0, 0.0])
    result = fa.score_result(source, gen, "asset_a")
    assert result["detected"] is False
    assert result["score"] is None
    assert result["reason"] == "low_confidence"


def test_at_threshold_is_scored():
    source = fa._l2_normalized([1.0, 0.0])
    gen = _face(fa.MIN_DET_SCORE, [1.0, 0.0])
    result = fa.score_result(source, gen, "asset_a")
    assert result["detected"] is True
    assert result["score"] == pytest.approx(1.0, abs=1e-5)


def test_no_source_face_is_na():
    gen = _face(0.95, [1.0, 0.0])
    result = fa.score_result(None, gen, "asset_a")
    assert result["detected"] is False
    assert result["score"] is None
    assert result["reason"] == "no_source_face"


def test_normed_embedding_is_preferred_when_present():
    source = fa._l2_normalized([1.0, 0.0])
    # normed_embedding already unit-length and pointing the same way ⇒ cosine 1.0.
    gen = SimpleNamespace(
        det_score=0.9,
        normed_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        embedding=np.asarray([5.0, 5.0], dtype=np.float32),  # ignored when normed present
    )
    result = fa.score_result(source, gen, None)
    assert result["score"] == pytest.approx(1.0, abs=1e-5)
    assert result["sourceRef"] is None


def test_scorer_embeds_source_once_across_n_scores(monkeypatch):
    """The explicit caching AC: the SOURCE is embedded exactly once, then reused across N
    generated-image scores. A fake app counts how many largest-face detections it runs."""
    calls = {"n": 0}
    source_face = _face(0.99, [1.0, 0.0, 0.0])
    gen_face = _face(0.9, [0.9, 0.1, 0.0])

    def fake_largest(_app, image):
        calls["n"] += 1
        # First call (construction) embeds the source; subsequent calls are the generated images.
        return source_face if image == "SOURCE" else gen_face

    monkeypatch.setattr(fa, "_largest_face", fake_largest)

    scorer = fa.FaceLikenessScorer(app=object(), source_image="SOURCE", source_ref="asset_a")
    assert scorer.has_source_face is True
    assert scorer.source_embed_count == 1
    assert calls["n"] == 1  # source embedded once at construction

    for _ in range(3):
        result = scorer.score("GEN")
        assert result["detected"] is True

    # The source was embedded exactly once; the 3 extra detections are the generated images only.
    assert scorer.source_embed_count == 1
    assert calls["n"] == 4  # 1 source + 3 generated, NOT 1 + 3*2 (no source re-embed)


def test_scorer_no_source_face_is_na_for_every_score(monkeypatch):
    monkeypatch.setattr(fa, "_largest_face", lambda _app, _img: None)
    scorer = fa.FaceLikenessScorer(app=object(), source_image="SOURCE", source_ref="asset_a")
    assert scorer.has_source_face is False
    assert scorer.source_embed_count == 1
    result = scorer.score("GEN")
    assert result["detected"] is False
    assert result["reason"] == "no_source_face"


def test_score_or_null_swallows_backend_error(monkeypatch):
    source_face = _face(0.99, [1.0, 0.0])
    seq = iter([source_face])

    def fake_largest(_app, image):
        # Source embeds fine; the generated-image detection raises (a backend failure).
        nxt = next(seq, None)
        if nxt is not None:
            return nxt
        raise RuntimeError("simulated onnxruntime failure")

    monkeypatch.setattr(fa, "_largest_face", fake_largest)
    scorer = fa.FaceLikenessScorer(app=object(), source_image="SOURCE")
    result = scorer.score_or_null("GEN")
    assert result["detected"] is False
    assert result["score"] is None
    assert result["reason"] == "embedding_error"
