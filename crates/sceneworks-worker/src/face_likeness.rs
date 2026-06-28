//! Shared, generator-agnostic face-likeness scorer (epic 4406, sc-4407).
//!
//! The backbone identity-likeness component every surface in epic 4406 calls — Character Studio
//! Angles (sc-4409) / Poses (sc-4410) and Image Studio "With Character" (sc-4411) — plus the
//! stretch compare-two-images tool (sc-4415). Given a **source face image** (embedded ONCE per job)
//! and any number of **generated images**, it returns an identity-likeness result by cosine-comparing
//! ArcFace embeddings of the largest detected face in each.
//!
//! It is **model-independent on purpose**: it is NOT coupled to the InstantID adapter. InstantID,
//! Z-Image identity, and Flux IP-adapter generations all produce a finished image, and this scorer
//! runs as a post-pass over that image — so it serves every identity generator through one path. It
//! reuses the same SCRFD + ArcFace (antelopev2 `glintr100`) stack the InstantID / kps / dataset-face
//! paths already provision (`mlx-gen-face` on macOS, `candle-gen-face` off-Mac), so no new weights.
//!
//! ## Result honesty (drives the whole design)
//! ArcFace is a **frontal-identity** signal only. A profile / extreme-angle / no-face generation
//! legitimately has no detectable frontal face — that is an explicit `detected: false`, `score: null`
//! result carrying a `reason`, NOT a misleading low number. The downstream UI (sc-4413) frames the
//! score as frontal-identity confidence, so the N/A state must be a first-class outcome, not an error.
//!
//! ## Source-embed caching (an explicit acceptance criterion)
//! [`FaceLikenessScorer`] embeds the source face exactly once at construction and stores the L2-
//! normalized vector. Every subsequent [`score`](FaceLikenessScorer::score) call re-embeds only the
//! *generated* image and dots it against the cached source — never the source again. A job that scores
//! N generated images runs the source recognition forward once, not N times.
//!
//! ## Non-fatal
//! Scoring never fails a generation. The backend embedding legs surface errors, but the intended
//! caller wraps each [`score`](FaceLikenessScorer::score) so a hard error becomes a logged `null`
//! result (see [`score_or_null`]); a *no-face* generation is already a clean `detected: false`.

use serde_json::{json, Value};

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::Image;

use sceneworks_core::contracts::JsonObject;

// `WorkerResult` / `WorkerError` are re-exported from the crate root (`error::*`); the backend legs
// surface model failures through them, the same as every other worker face path.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::{WorkerError, WorkerResult};

/// The recognition method surfaced on every result. Stable string the asset sidecar (sc-4408) and the
/// UI bands (sc-4414) key off — the same antelopev2 ArcFace stack on both platforms.
pub(crate) const LIKENESS_METHOD: &str = "arcface_antelopev2";

/// SCRFD detection-confidence floor below which a detection is treated as "no reliable frontal face".
/// Mirrors the kps-extract `LOW_CONF_THRESH` (0.65): below it, the captured face is an unreliable
/// extreme profile / poor framing, so the scorer returns an explicit N/A rather than a noisy number.
pub(crate) const MIN_DET_SCORE: f32 = 0.65;

// ---------------------------------------------------------------------------
// Pure scoring core — no GPU, no IO, no cfg gating. Always compiled + unit-tested
// (so the cosine / caching / N-A contract is verified on every platform incl. CI).
// ---------------------------------------------------------------------------

/// Why a generated image produced no likeness score. Serialized in `camelCase` snake string on the
/// result `reason` field so the UI can phrase the N/A precisely (vs. a bare `null`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoScoreReason {
    /// SCRFD found no face at all in the generated image.
    NoFace,
    /// A face was found but below [`MIN_DET_SCORE`] — an unreliable extreme-angle / profile detection.
    LowConfidence,
    /// The source face could not be embedded (no detectable source face). The whole job is N/A.
    NoSourceFace,
    /// The embedding forward errored (non-fatal: logged + surfaced as `null`, never blocks a gen).
    EmbeddingError,
}

impl NoScoreReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            NoScoreReason::NoFace => "no_face",
            NoScoreReason::LowConfidence => "low_confidence",
            NoScoreReason::NoSourceFace => "no_source_face",
            NoScoreReason::EmbeddingError => "embedding_error",
        }
    }
}

/// One identity-likeness outcome for a single generated image. The acceptance result shape:
/// `{ score: f32|null, detected: bool, method, sourceRef }` (+ a `reason` on the N/A branch).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LikenessResult {
    /// Cosine similarity in `[-1, 1]` (ArcFace cosines are effectively `[0, 1]` for same/other faces).
    /// `None` ⇒ N/A (see `reason`); never a misleading low number.
    pub score: Option<f64>,
    /// Whether a reliable frontal face was detected in the generated image AND scored.
    pub detected: bool,
    /// Why there is no score, when `score` is `None`. `None` on a scored result.
    pub reason: Option<NoScoreReason>,
}

impl LikenessResult {
    /// A scored result (a reliable face was detected + cosine-compared).
    fn scored(score: f64) -> Self {
        Self {
            score: Some(score),
            detected: true,
            reason: None,
        }
    }

    /// An N/A result — `detected: false`, `score: null`, carrying the reason.
    fn na(reason: NoScoreReason) -> Self {
        Self {
            score: None,
            detected: false,
            reason: Some(reason),
        }
    }

    /// Serialize to the acceptance result object. `source_ref` is the caller's stable id for the
    /// source face (e.g. the reference asset id) so a consumer can attribute the score.
    pub(crate) fn to_json(&self, source_ref: Option<&str>) -> JsonObject {
        let mut object = JsonObject::new();
        object.insert(
            "score".to_owned(),
            match self.score {
                Some(score) => json!(score),
                None => Value::Null,
            },
        );
        object.insert("detected".to_owned(), Value::Bool(self.detected));
        object.insert("method".to_owned(), json!(LIKENESS_METHOD));
        object.insert(
            "sourceRef".to_owned(),
            match source_ref {
                Some(id) => json!(id),
                None => Value::Null,
            },
        );
        if let Some(reason) = self.reason {
            object.insert("reason".to_owned(), json!(reason.as_str()));
        }
        object
    }
}

/// L2-normalize a raw ArcFace embedding. `None` for an empty or zero-norm vector (so callers can treat
/// "no usable embedding" explicitly). Identical contract to the dataset-face / lora-eval normalizers.
fn l2_normalized(v: &[f32]) -> Option<Vec<f32>> {
    if v.is_empty() {
        return None;
    }
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return None;
    }
    Some(v.iter().map(|x| (f64::from(*x) / norm) as f32).collect())
}

/// Cosine of two already-L2-normalized vectors (a plain dot product). `None` on a length mismatch
/// (defensive — a length mismatch means the two embeddings are not comparable).
fn cosine_normalized(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum(),
    )
}

/// The pure scoring decision, shared by both backends + fully unit-tested without weights. Given the
/// cached (already-normalized) source embedding and the generated image's largest detection — its
/// detection score + raw embedding, or `None` when SCRFD found no face — produce the [`LikenessResult`].
///
/// This is where the N/A policy lives: no generated face ⇒ `NoFace`; a face below [`MIN_DET_SCORE`] ⇒
/// `LowConfidence`; an un-normalizable embedding ⇒ `EmbeddingError`; otherwise the cosine score.
fn score_against_source(
    source_normalized: &[f32],
    generated: Option<(f32, &[f32])>,
) -> LikenessResult {
    let Some((det_score, raw_embedding)) = generated else {
        return LikenessResult::na(NoScoreReason::NoFace);
    };
    if det_score < MIN_DET_SCORE {
        return LikenessResult::na(NoScoreReason::LowConfidence);
    }
    let Some(generated_normalized) = l2_normalized(raw_embedding) else {
        return LikenessResult::na(NoScoreReason::EmbeddingError);
    };
    match cosine_normalized(source_normalized, &generated_normalized) {
        Some(score) => LikenessResult::scored(score),
        None => LikenessResult::na(NoScoreReason::EmbeddingError),
    }
}

// ---------------------------------------------------------------------------
// The shared scorer — caches the source embedding once, scores N generated images.
// Backend legs (MLX on macOS, candle off-Mac) feed the same pure core above.
// ---------------------------------------------------------------------------

/// The loaded SCRFD + ArcFace backend, abstracted so the [`FaceLikenessScorer`] is backend-agnostic.
/// MLX on macOS (`mlx-gen-face`), candle off-Mac (`candle-gen-face`); both produce the same
/// `(det_score, raw 512-d embedding)` for the largest detected face — or `None` when there is no face.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
enum FaceBackend {
    #[cfg(target_os = "macos")]
    Mlx(mlx_gen_face::FaceAnalysis),
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    Candle(Box<dyn gen_core::FaceEmbedder>),
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl FaceBackend {
    /// Detect + embed the largest face of `image`. `Ok(None)` ⇒ no detectable face (a clean N/A,
    /// not an error); `Err` ⇒ a real failure (weights / forward). MLX embeds only the largest
    /// detection (`detect` then `embed`, one recognition forward); candle's `largest_face` errors on
    /// no face, so it is mapped to `Ok(None)`.
    fn largest_face(&self, image: &Image) -> WorkerResult<Option<(f32, Vec<f32>)>> {
        match self {
            #[cfg(target_os = "macos")]
            FaceBackend::Mlx(analysis) => {
                let (h, w) = (image.height as usize, image.width as usize);
                let dets = analysis
                    .detect(&image.pixels, h, w)
                    .map_err(|error| WorkerError::Engine(format!("face detect: {error}")))?;
                // `detect` returns largest-first; embed just `[0]` (one recognition forward).
                let Some(det) = dets.first() else {
                    return Ok(None);
                };
                let face = analysis
                    .embed(&image.pixels, h, w, det)
                    .map_err(|error| WorkerError::Engine(format!("face embed: {error}")))?;
                Ok(Some((face.det_score, face.embedding)))
            }
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            FaceBackend::Candle(analysis) => {
                // candle `analyze` returns every face largest-first with embeddings; take `[0]`.
                // (`largest_face` *errors* on no face — `analyze` returns an empty list instead, the
                // clean N/A funnel both other worker face paths rely on.)
                let faces = analysis
                    .analyze(image)
                    .map_err(|error| WorkerError::Engine(format!("face analyze: {error}")))?;
                Ok(faces.into_iter().next().map(|f| (f.det_score, f.embedding)))
            }
        }
    }
}

/// The generator-agnostic identity-likeness scorer (sc-4407). Construct it once per job from the loaded
/// face stack + the source face image — the source embedding is computed ONCE here and cached — then
/// call [`score`](Self::score) for each generated image. The source is never re-embedded.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct FaceLikenessScorer {
    backend: FaceBackend,
    /// The cached, L2-normalized source embedding. `None` ⇒ no detectable source face: every `score`
    /// call short-circuits to an N/A result (`NoSourceFace`) — the whole job is N/A, but never an error.
    source_normalized: Option<Vec<f32>>,
    /// Count of source embeddings actually computed (always ≤ 1). Lets a test assert the source is
    /// embedded exactly once across N `score` calls — the explicit caching acceptance criterion.
    source_embed_count: usize,
}

#[cfg(target_os = "macos")]
impl FaceLikenessScorer {
    /// Load the MLX SCRFD + ArcFace stack from `weights_dir` (the same converted bundle InstantID /
    /// kps / dataset-face provision) and embed the source face once. Runs `!Send` MLX work — call
    /// inside `spawn_blocking`.
    pub(crate) fn load_mlx(weights_dir: &std::path::Path, source: &Image) -> WorkerResult<Self> {
        use mlx_gen::weights::Weights;
        let scrfd = Weights::from_file(weights_dir.join(crate::image_jobs::INSTANTID_SCRFD_FILE))
            .map_err(|error| WorkerError::Engine(format!("SCRFD weights: {error}")))?;
        let arcface =
            Weights::from_file(weights_dir.join(crate::image_jobs::INSTANTID_ARCFACE_FILE))
                .map_err(|error| WorkerError::Engine(format!("ArcFace weights: {error}")))?;
        let analysis = mlx_gen_face::FaceAnalysis::load(&scrfd, &arcface)
            .map_err(|error| WorkerError::Engine(format!("face stack load: {error}")))?;
        Self::with_backend(FaceBackend::Mlx(analysis), source)
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl FaceLikenessScorer {
    /// Load the candle SCRFD + ArcFace stack from `weights_dir` and embed the source face once.
    pub(crate) fn load_candle(weights_dir: &std::path::Path, source: &Image) -> WorkerResult<Self> {
        let analysis = candle_gen_face::load(weights_dir)
            .map_err(|error| WorkerError::Engine(format!("face stack load: {error}")))?;
        Self::with_backend(FaceBackend::Candle(Box::new(analysis)), source)
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl FaceLikenessScorer {
    /// Build the scorer from an already-loaded backend, embedding + caching the source face exactly
    /// once. A source with no detectable face is NOT an error — it yields a scorer whose every `score`
    /// returns the `NoSourceFace` N/A (the whole job is N/A, but generation is never blocked).
    fn with_backend(backend: FaceBackend, source: &Image) -> WorkerResult<Self> {
        let (source_normalized, source_embed_count) = match backend.largest_face(source)? {
            Some((_det_score, raw)) => (l2_normalized(&raw), 1),
            None => (None, 1),
        };
        Ok(Self {
            backend,
            source_normalized,
            source_embed_count,
        })
    }

    /// Score one generated image against the CACHED source embedding. Re-embeds only the generated
    /// image — never the source. Returns an N/A [`LikenessResult`] (not an `Err`) for the no-face /
    /// low-confidence / no-source-face cases; `Err` only for a genuine backend failure (which the
    /// caller turns into a logged `null` — see [`score_or_null`]). Runs `!Send` work; call inside
    /// `spawn_blocking`.
    pub(crate) fn score(&self, generated: &Image) -> WorkerResult<LikenessResult> {
        let Some(source_normalized) = self.source_normalized.as_deref() else {
            return Ok(LikenessResult::na(NoScoreReason::NoSourceFace));
        };
        let largest = self.backend.largest_face(generated)?;
        let generated = largest
            .as_ref()
            .map(|(det_score, raw)| (*det_score, raw.as_slice()));
        Ok(score_against_source(source_normalized, generated))
    }

    /// Score one generated image, turning a backend error into a logged `null` result — the
    /// "scoring errors are non-fatal; never block a generation" acceptance criterion. Use this from a
    /// generation post-pass; use [`score`](Self::score) when the caller wants to handle the error.
    pub(crate) fn score_or_null(&self, generated: &Image) -> LikenessResult {
        match self.score(generated) {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(error = %error, "face-likeness scoring failed; recording null");
                LikenessResult::na(NoScoreReason::EmbeddingError)
            }
        }
    }

    /// Whether the source face embedded — `false` ⇒ the whole job is N/A (no source face).
    pub(crate) fn has_source_face(&self) -> bool {
        self.source_normalized.is_some()
    }

    /// How many times the source was embedded (always ≤ 1). For the caching acceptance test.
    #[cfg(test)]
    pub(crate) fn source_embed_count(&self) -> usize {
        self.source_embed_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `l2_normalized` casts back to f32, so round-tripped components carry only ~1e-7 precision —
    // an f32-appropriate tolerance, not f64 epsilon.
    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-5
    }

    // -- pure cosine / normalization -------------------------------------------------

    #[test]
    fn l2_normalize_unit_and_degenerate() {
        let n = l2_normalized(&[3.0, 4.0]).expect("non-zero");
        assert!(approx(f64::from(n[0]), 0.6) && approx(f64::from(n[1]), 0.8));
        assert!(l2_normalized(&[]).is_none(), "empty ⇒ None");
        assert!(l2_normalized(&[0.0, 0.0]).is_none(), "zero-norm ⇒ None");
    }

    #[test]
    fn cosine_is_dot_of_normalized() {
        // Identical direction ⇒ 1.0; orthogonal ⇒ 0.0; opposite ⇒ -1.0.
        let a = l2_normalized(&[1.0, 1.0]).unwrap();
        assert!(approx(cosine_normalized(&a, &a).unwrap(), 1.0));
        let x = l2_normalized(&[1.0, 0.0]).unwrap();
        let y = l2_normalized(&[0.0, 1.0]).unwrap();
        assert!(approx(cosine_normalized(&x, &y).unwrap(), 0.0));
        let neg = l2_normalized(&[-1.0, 0.0]).unwrap();
        assert!(approx(cosine_normalized(&x, &neg).unwrap(), -1.0));
    }

    #[test]
    fn cosine_length_mismatch_is_none() {
        assert!(cosine_normalized(&[1.0, 0.0], &[1.0, 0.0, 0.0]).is_none());
        assert!(cosine_normalized(&[], &[]).is_none());
    }

    // -- the N/A decision (the result-honesty acceptance) ---------------------------

    #[test]
    fn same_identity_scores_high() {
        // A source embedding and a near-identical generated embedding ⇒ a high cosine, detected.
        let source = l2_normalized(&[1.0, 0.2, 0.1, 0.05]).unwrap();
        let generated_raw = [0.98, 0.21, 0.11, 0.04];
        let result = score_against_source(&source, Some((0.95, &generated_raw)));
        assert!(result.detected);
        let score = result.score.expect("scored");
        assert!(score > 0.99, "same-identity cosine should be high: {score}");
        assert!(result.reason.is_none());
    }

    #[test]
    fn different_identity_scores_low() {
        // A nearly-orthogonal generated embedding ⇒ a low (near-zero) cosine — still detected (a face
        // WAS found), just not a match. This is a real low score, NOT the N/A branch.
        let source = l2_normalized(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        let generated_raw = [0.02, 1.0, 0.0, 0.0];
        let result = score_against_source(&source, Some((0.92, &generated_raw)));
        assert!(result.detected);
        let score = result.score.expect("scored");
        assert!(
            score < 0.1,
            "different-identity cosine should be low: {score}"
        );
    }

    #[test]
    fn no_generated_face_is_na_not_a_low_score() {
        // The honesty linchpin: a generation with no detectable face must be detected:false /
        // score:null with reason no_face — NEVER a misleading low number.
        let source = l2_normalized(&[1.0, 0.0]).unwrap();
        let result = score_against_source(&source, None);
        assert!(!result.detected);
        assert!(result.score.is_none());
        assert_eq!(result.reason, Some(NoScoreReason::NoFace));
    }

    #[test]
    fn low_confidence_detection_is_na() {
        // A face below MIN_DET_SCORE (extreme profile) ⇒ N/A low_confidence, not a noisy score.
        let source = l2_normalized(&[1.0, 0.0]).unwrap();
        let generated_raw = [1.0, 0.0];
        let result = score_against_source(&source, Some((MIN_DET_SCORE - 0.01, &generated_raw)));
        assert!(!result.detected);
        assert!(result.score.is_none());
        assert_eq!(result.reason, Some(NoScoreReason::LowConfidence));
    }

    #[test]
    fn at_threshold_is_scored() {
        // Exactly at the floor counts as a reliable detection (>= is the boundary).
        let source = l2_normalized(&[1.0, 0.0]).unwrap();
        let generated_raw = [1.0, 0.0];
        let result = score_against_source(&source, Some((MIN_DET_SCORE, &generated_raw)));
        assert!(result.detected);
        assert!(approx(result.score.unwrap(), 1.0));
    }

    #[test]
    fn unnormalizable_generated_embedding_is_embedding_error() {
        let source = l2_normalized(&[1.0, 0.0]).unwrap();
        let zero = [0.0f32, 0.0];
        let result = score_against_source(&source, Some((0.9, &zero)));
        assert_eq!(result.reason, Some(NoScoreReason::EmbeddingError));
        assert!(result.score.is_none());
    }

    // -- result serialization (the acceptance result shape) -------------------------

    #[test]
    fn scored_result_json_shape() {
        let result = LikenessResult::scored(0.873);
        let json = Value::Object(result.to_json(Some("asset_123")));
        assert_eq!(
            json,
            json!({
                "score": 0.873,
                "detected": true,
                "method": "arcface_antelopev2",
                "sourceRef": "asset_123",
            })
        );
    }

    #[test]
    fn na_result_json_shape_carries_reason_and_null_score() {
        let result = LikenessResult::na(NoScoreReason::NoFace);
        let json = Value::Object(result.to_json(Some("asset_123")));
        assert_eq!(
            json,
            json!({
                "score": Value::Null,
                "detected": false,
                "method": "arcface_antelopev2",
                "sourceRef": "asset_123",
                "reason": "no_face",
            })
        );
    }

    #[test]
    fn na_reason_strings_are_stable() {
        assert_eq!(NoScoreReason::NoFace.as_str(), "no_face");
        assert_eq!(NoScoreReason::LowConfidence.as_str(), "low_confidence");
        assert_eq!(NoScoreReason::NoSourceFace.as_str(), "no_source_face");
        assert_eq!(NoScoreReason::EmbeddingError.as_str(), "embedding_error");
    }

    /// Real-weight scorer integration (sc-4407): proves the worker binary links + loads the MLX
    /// `FaceAnalysis` and that the SHARED scorer (a) scores a same-identity pair high (≈ the recorded
    /// InstantID ArcFace baselines), (b) scores a different-identity pair low, (c) returns
    /// `detected: false` for a no-face image, and — the explicit caching AC — (d) embeds the SOURCE
    /// face exactly ONCE across N generated-image `score` calls. `#[ignore]` per convention (weights
    /// live outside CI). Run on a Mac with the bundle staged + Metal:
    ///   SCENEWORKS_INSTANTID_WEIGHTS=/path/to/instantid-mlx \
    ///   SCENEWORKS_TEST_FACE=/path/to/personA.png \
    ///   SCENEWORKS_TEST_FACE_B=/path/to/personB.png \
    ///     cargo test -p sceneworks-worker --lib -- --ignored face_likeness_real_weights --nocapture
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight: needs the SceneWorks/instantid-mlx SCRFD+ArcFace bundle + Metal + face photos"]
    fn face_likeness_real_weights_scores_and_caches_source_once() {
        use std::path::PathBuf;
        let home = std::env::var("HOME").expect("HOME");
        let bundle = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(&home)
                    .join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
            });

        let load = |path: &str| -> Image {
            let decoded = image::open(path)
                .unwrap_or_else(|e| panic!("face {path}: {e}"))
                .to_rgb8();
            Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            }
        };

        let face_a =
            std::env::var("SCENEWORKS_TEST_FACE").expect("set SCENEWORKS_TEST_FACE to person A");
        let source = load(&face_a);
        let scorer = FaceLikenessScorer::load_mlx(&bundle, &source).expect("scorer loads");
        assert!(scorer.has_source_face(), "source has a detectable face");
        assert_eq!(scorer.source_embed_count(), 1, "source embedded once");

        // (a) same identity (source vs itself) scores high — ≈ the recorded InstantID baselines.
        let same = scorer.score(&source).expect("score runs");
        assert!(same.detected, "same-identity detected");
        let same_score = same.score.expect("same-identity scored");
        assert!(
            same_score > 0.8,
            "same-identity cosine ≈ InstantID baseline (>0.8): {same_score}"
        );

        // (d) caching: after N score() calls the SOURCE is still embedded exactly once.
        for _ in 0..3 {
            let _ = scorer.score(&source).expect("score runs");
        }
        assert_eq!(
            scorer.source_embed_count(),
            1,
            "source embedded ONCE across N generated-image scores"
        );

        // (b) different identity scores low (if a second photo is supplied).
        if let Ok(face_b) = std::env::var("SCENEWORKS_TEST_FACE_B") {
            let other = load(&face_b);
            let diff = scorer.score(&other).expect("score runs");
            if diff.detected {
                let diff_score = diff.score.expect("different-identity scored");
                assert!(
                    diff_score < same_score,
                    "different identity {diff_score} below same identity {same_score}"
                );
            }
        }

        // (c) a no-face image (synthetic gradient) returns detected:false / score:null.
        let mut gradient = image::RgbImage::new(128, 128);
        for (x, y, px) in gradient.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 2) as u8, (y * 2) as u8, 96]);
        }
        let gradient = Image {
            width: gradient.width(),
            height: gradient.height(),
            pixels: gradient.into_raw(),
        };
        let none = scorer.score(&gradient).expect("score runs");
        assert!(!none.detected, "no-face ⇒ detected:false");
        assert!(none.score.is_none(), "no-face ⇒ score:null");
        assert_eq!(none.reason, Some(NoScoreReason::NoFace));
        println!("face-likeness ok: same={same_score:.4}");
    }
}
