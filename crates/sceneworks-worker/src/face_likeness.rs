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

// `Image` resolves to `mlx_gen::Image` on macOS and `gen_core::Image` under the candle backend —
// the same cfg-gated split the sibling job modules use (sensenova_jobs.rs / video_jobs.rs). The whole
// scorer (and its test stub) only exists under these two configs, so on the plain Linux parity build
// (no `backend-candle`) `Image` is intentionally absent and nothing here is compiled.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use gen_core::Image;
#[cfg(target_os = "macos")]
use mlx_gen::Image;

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

/// The `rawAdapterSettings` key the likeness block lands under in a generated asset's sidecar
/// (sc-4408). `build_image_sidecar_parts` passes `rawAdapterSettings` through verbatim into
/// `recipe.rawAdapterSettings`, so attaching the block here is the whole persistence pathway.
pub(crate) const FACE_LIKENESS_FACT_KEY: &str = "faceLikeness";

/// Build the `faceLikeness` fact block for a generated asset's `rawAdapterSettings` (sc-4408).
///
/// This is the carrier seam the scoring surfaces (sc-4409 Angles / sc-4410 Poses / sc-4411
/// With-Character) attach to each scored asset — NOT the pure scorer (that is sc-4407's
/// [`LikenessResult`]). The persisted shape matches the asset-sidecar contract:
/// `{ score, detected, method, sourceAssetId }` (+ a `reason` on the N/A branch). It maps the
/// scorer's `sourceRef` to the sidecar's `sourceAssetId` field name.
///
/// A scored result carries the cosine `score` + `detected: true`; an N/A result carries
/// `score: null`, `detected: false` and the `reason`, so a profile / no-face generation persists an
/// honest `detected: false` block rather than a misleading low number (the sc-4407 result-honesty
/// contract, carried intact into the sidecar).
pub(crate) fn face_likeness_fact(
    result: &LikenessResult,
    source_asset_id: Option<&str>,
) -> JsonObject {
    let mut object = JsonObject::new();
    object.insert(
        "score".to_owned(),
        match result.score {
            Some(score) => json!(score),
            None => Value::Null,
        },
    );
    object.insert("detected".to_owned(), Value::Bool(result.detected));
    object.insert("method".to_owned(), json!(LIKENESS_METHOD));
    object.insert(
        "sourceAssetId".to_owned(),
        match source_asset_id {
            Some(id) => json!(id),
            None => Value::Null,
        },
    );
    if let Some(reason) = result.reason {
        object.insert("reason".to_owned(), json!(reason.as_str()));
    }
    object
}

/// Attach a face-likeness result to a generated asset's `rawAdapterSettings` under
/// [`FACE_LIKENESS_FACT_KEY`] (sc-4408). The omit-when-absent seam: a `None` result (scoring was
/// skipped / not applicable / hard-failed before a [`LikenessResult`] was produced) leaves
/// `rawAdapterSettings` untouched, so the field is OMITTED entirely from the sidecar — no `null`
/// clutter, no crash. A `Some(result)` writes the [`face_likeness_fact`] block (including the
/// explicit `detected: false` N/A block). The scoring surfaces call this once per generated asset
/// just before [`write_image_asset`](crate::image_jobs::write_image_asset) builds its fact.
pub(crate) fn attach_face_likeness(
    raw_settings: &mut JsonObject,
    result: Option<&LikenessResult>,
    source_asset_id: Option<&str>,
) {
    if let Some(result) = result {
        raw_settings.insert(
            FACE_LIKENESS_FACT_KEY.to_owned(),
            Value::Object(face_likeness_fact(result, source_asset_id)),
        );
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
// The test-only zero-cost `Stub` variant is tiny next to the real `Mlx` backend; the size disparity
// is irrelevant for a per-job singleton (one scorer per generation), and it only exists under `test`.
#[cfg_attr(test, allow(clippy::large_enum_variant))]
enum FaceBackend {
    #[cfg(target_os = "macos")]
    Mlx(mlx_gen_face::FaceAnalysis),
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    Candle(Box<dyn gen_core::FaceEmbedder>),
    /// Weight-free deterministic backend for the wiring tests (sc-4409): keyed on the image's first
    /// pixel byte so a test can map a synthetic image to a face / no-face / low-confidence detection
    /// and count how often the backend is hit, WITHOUT staging the antelopev2 weights in CI. Never
    /// compiled outside `cargo test`. Gated on the face-backend configs too (the whole scorer is), so
    /// the plain Linux parity build — which has no `Image` type in scope — doesn't try to compile it.
    #[cfg(test)]
    Stub(StubFaceBackend),
}

/// The canned detection the [`FaceBackend::Stub`] returns for a given image, selected by the image's
/// first pixel byte (the test encodes intent in pixel 0). Mirrors the three real-backend outcomes.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
#[derive(Clone, Copy)]
enum StubFace {
    /// A reliable face at `det_score`, embedded as the canned vector (drives a real cosine).
    Face(f32),
    /// No detectable face ⇒ `Ok(None)` (the clean N/A funnel).
    NoFace,
}

/// Counts every `largest_face` call so a test can prove the SOURCE is embedded exactly once across N
/// generated-image scores (the explicit caching acceptance criterion), independent of weights.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
struct StubFaceBackend {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
impl StubFaceBackend {
    /// Map an image to its canned detection by pixel 0: `0` ⇒ no face; otherwise a face whose
    /// `det_score` scales with pixel 0 (`0.5 + px/512`, so distinct identities produce distinct
    /// embeddings). The score crosses [`MIN_DET_SCORE`] (0.65) at px ≈ 77, so the scorer treats a
    /// low pixel value (px `1..=76`, det_score `< 0.65`) as a LowConfidence N/A and a high one
    /// (px `>= 77`) as a reliable, scored detection — letting a test exercise all three outcomes.
    fn classify(image: &Image) -> StubFace {
        match image.pixels.first().copied().unwrap_or(0) {
            0 => StubFace::NoFace,
            other => StubFace::Face(0.5 + f32::from(other) / 512.0),
        }
    }

    /// A canned, identity-bearing embedding keyed off pixel 0 so same-pixel images cosine-match high
    /// and different-pixel images lower — enough to exercise the scoring + persistence wiring.
    fn embedding(image: &Image) -> Vec<f32> {
        let key = f32::from(image.pixels.first().copied().unwrap_or(0));
        vec![key, 1.0, 0.25, 0.1]
    }
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
            #[cfg(test)]
            FaceBackend::Stub(stub) => {
                stub.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(match StubFaceBackend::classify(image) {
                    StubFace::NoFace => None,
                    StubFace::Face(det) => Some((det, StubFaceBackend::embedding(image))),
                })
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

    /// Build a scorer over the weight-free [`FaceBackend::Stub`] from a synthetic `source` image
    /// (sc-4409 wiring tests). Returns the scorer plus the shared backend-call counter so a test can
    /// prove the source embedded exactly once and assert exactly how many generated-image scores hit
    /// the backend — the caching acceptance criterion, verifiable without antelopev2 weights in CI.
    #[cfg(test)]
    pub(crate) fn with_stub_source(
        source: &Image,
    ) -> (Self, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend = FaceBackend::Stub(StubFaceBackend {
            calls: calls.clone(),
        });
        let scorer = Self::with_backend(backend, source).expect("stub backend never errors");
        (scorer, calls)
    }

    /// Platform-neutral load: the MLX stack on macOS, the candle stack off-Mac. Lets the shared
    /// angle-set seam ([`build_face_likeness_scorer`]) construct a scorer without each call site
    /// cfg-branching on the backend. Embeds the source face once (the caching contract).
    fn load_for_weights_dir(weights_dir: &std::path::Path, source: &Image) -> WorkerResult<Self> {
        #[cfg(target_os = "macos")]
        {
            Self::load_mlx(weights_dir, source)
        }
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            Self::load_candle(weights_dir, source)
        }
    }
}

/// Build the per-job identity-likeness scorer for a character set (sc-4409 angles / sc-4410 poses),
/// generator-agnostically.
///
/// This is the SHARED seam every character-set producer calls — InstantID, FLUX.2 edit, Qwen-Edit, and
/// SenseNova-U1 angle sets (sc-4409), AND every pose-library route (sc-4410: InstantID pose set, the
/// FLUX.2 / Qwen edit pose tier, and the Z-Image / Qwen / FLUX.2-dev / FLUX.1-dev strict-control pose
/// lanes) all route their job to their own engine, but each scores its finished images through this one
/// path (the scorer is model-independent by design — sc-4407). The source identity face is embedded
/// exactly ONCE here and reused across every angle / pose.
///
/// **Non-fatal construction** (the explicit AC carried from the sc-4407 review): a missing/corrupt
/// weights bundle or a source image with no detectable face must NEVER abort the generation. Any
/// error is logged and becomes `None` — the set still renders, the scores are simply absent (the field
/// is omitted from each sidecar). A `None` here ⇒ pass `None` to [`score_generated_image`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn build_face_likeness_scorer(
    weights_dir: &std::path::Path,
    source: &Image,
) -> Option<FaceLikenessScorer> {
    match FaceLikenessScorer::load_for_weights_dir(weights_dir, source) {
        Ok(scorer) => Some(scorer),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "character-set face-likeness scorer construction failed; scores will be omitted \
                 (generation continues)"
            );
            None
        }
    }
}

/// Score one finished generated image (an angle or a pose) against the per-job cached source embedding
/// and build the persisted `faceLikeness` sidecar block (sc-4408/sc-4409/sc-4410) — the SHARED
/// per-image post-pass for every character-set producer. For pose-library routes that run a face-restore
/// pass (InstantID), the caller passes the FINAL post-restore image, so the score reflects what the user
/// sees (sc-4410). `None` scorer (no set, or a failed [`build_face_likeness_scorer`]) ⇒ `None` ⇒ the
/// field is omitted entirely. Per-image scoring is non-fatal ([`score_or_null`]); a full-body / turned /
/// profile pose with no reliable frontal face records an honest `detected: false` N/A block, never a
/// misleading low number.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn score_generated_image(
    scorer: Option<&FaceLikenessScorer>,
    image: &Image,
    source_asset_id: Option<&str>,
) -> Option<JsonObject> {
    scorer.map(|scorer| {
        let result = scorer.score_or_null(image);
        face_likeness_fact(&result, source_asset_id)
    })
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

    // -- sc-4408 carrier: faceLikeness → rawAdapterSettings persistence seam --------

    #[test]
    fn face_likeness_fact_scored_shape_uses_source_asset_id() {
        // The sidecar contract field is `sourceAssetId` (sc-4407's scorer emits `sourceRef`); the
        // carrier maps it. A scored result is the full `{score, detected, method, sourceAssetId}`.
        let fact = face_likeness_fact(&LikenessResult::scored(0.873), Some("asset_src1"));
        assert_eq!(
            Value::Object(fact),
            json!({
                "score": 0.873,
                "detected": true,
                "method": "arcface_antelopev2",
                "sourceAssetId": "asset_src1",
            })
        );
    }

    #[test]
    fn face_likeness_fact_na_block_carries_detected_false_and_reason() {
        // An N/A generation persists an explicit detected:false block (with reason), NOT a low number.
        let fact = face_likeness_fact(
            &LikenessResult::na(NoScoreReason::NoFace),
            Some("asset_src1"),
        );
        assert_eq!(
            Value::Object(fact),
            json!({
                "score": Value::Null,
                "detected": false,
                "method": "arcface_antelopev2",
                "sourceAssetId": "asset_src1",
                "reason": "no_face",
            })
        );
    }

    #[test]
    fn attach_face_likeness_writes_block_into_raw_settings() {
        // The end-to-end persistence pathway: attaching a scored result puts the `faceLikeness` block
        // into `rawAdapterSettings` (which `build_image_sidecar_parts` passes through verbatim) while
        // leaving any pre-existing settings intact.
        let mut raw = JsonObject::new();
        raw.insert("steps".to_owned(), json!(8));
        attach_face_likeness(
            &mut raw,
            Some(&LikenessResult::scored(0.91)),
            Some("asset_src1"),
        );
        assert_eq!(raw.get("steps"), Some(&json!(8)));
        assert_eq!(
            raw.get(FACE_LIKENESS_FACT_KEY),
            Some(&json!({
                "score": 0.91,
                "detected": true,
                "method": "arcface_antelopev2",
                "sourceAssetId": "asset_src1",
            }))
        );
    }

    #[test]
    fn attach_face_likeness_omits_field_when_absent() {
        // The omit-when-absent contract: a None result (scoring skipped/failed) leaves
        // rawAdapterSettings untouched — the field is omitted entirely, no null clutter, no crash.
        let mut raw = JsonObject::new();
        raw.insert("steps".to_owned(), json!(8));
        attach_face_likeness(&mut raw, None, Some("asset_src1"));
        assert!(
            !raw.contains_key(FACE_LIKENESS_FACT_KEY),
            "absent score ⇒ faceLikeness omitted"
        );
        assert_eq!(raw.get("steps"), Some(&json!(8)));
    }

    #[test]
    fn na_reason_strings_are_stable() {
        assert_eq!(NoScoreReason::NoFace.as_str(), "no_face");
        assert_eq!(NoScoreReason::LowConfidence.as_str(), "low_confidence");
        assert_eq!(NoScoreReason::NoSourceFace.as_str(), "no_source_face");
        assert_eq!(NoScoreReason::EmbeddingError.as_str(), "embedding_error");
    }

    // -- sc-4409: angle-set wiring over the weight-free stub backend -----------------
    // These exercise the live `FaceLikenessScorer` (caching + N/A + non-fatal construction +
    // per-image persistence) WITHOUT antelopev2 weights, so they run in CI on the face-backend
    // build (macOS, or candle off-Mac). The real-weight end-to-end remains the `#[ignore]` test
    // above. The angle-set producer (`generate_instantid_stream`) calls exactly this surface:
    // construct once from the source, then `score_or_null` + `face_likeness_fact` per finished view.

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    mod angle_set_wiring {
        use super::*;
        use std::sync::atomic::Ordering;

        /// A synthetic image whose first pixel byte selects the stub detection (see `StubFaceBackend`):
        /// `0` ⇒ no face; `>=77` ⇒ a reliable face; `1..=76` ⇒ a low-confidence face.
        fn image(pixel0: u8) -> Image {
            Image {
                width: 1,
                height: 1,
                pixels: vec![pixel0, 0, 0],
            }
        }

        #[test]
        fn source_embedded_once_across_n_angles_each_attaches_a_block() {
            // The explicit caching AC: one source embed at construction, reused across N angle scores
            // (the source is NEVER re-embedded), and every scored angle yields an attachable block.
            let source = image(200);
            let (scorer, calls) = FaceLikenessScorer::with_stub_source(&source);
            assert!(scorer.has_source_face(), "source has a detectable face");
            assert_eq!(scorer.source_embed_count(), 1, "source embedded once");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "construction hit the backend exactly once (the source embed)"
            );

            // Score an 11-view-style set; every reliable-face view attaches a faceLikeness block.
            let angles = [200u8, 180, 160, 140, 120, 100, 90, 200, 180, 160, 140];
            for (i, px) in angles.iter().enumerate() {
                let result = scorer.score_or_null(&image(*px));
                assert!(result.detected, "angle {i} (px {px}) detected");
                let mut raw = JsonObject::new();
                attach_face_likeness(&mut raw, Some(&result), Some("char_src"));
                let block = raw
                    .get(FACE_LIKENESS_FACT_KEY)
                    .and_then(Value::as_object)
                    .expect("faceLikeness block attached");
                assert_eq!(block.get("detected"), Some(&Value::Bool(true)));
                assert_eq!(block.get("sourceAssetId"), Some(&json!("char_src")));
                assert!(block.get("score").map(Value::is_number).unwrap_or(false));
            }

            // The source was embedded ONCE; the backend was hit only the source embed + one detect per
            // angle (N+1) — never a second source embed.
            assert_eq!(
                scorer.source_embed_count(),
                1,
                "still one source embed after N scores"
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1 + angles.len(),
                "backend hit = 1 source embed + 1 detect per angle (no source re-embed)"
            );
        }

        #[test]
        fn profile_no_face_angle_records_detected_false_not_a_low_number() {
            // A profile / up / down view with no detectable frontal face must persist an honest
            // detected:false N/A block (score null + reason), NOT a misleading low number.
            let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(200));
            let result = scorer.score_or_null(&image(0)); // pixel 0 == 0 ⇒ no face
            assert!(!result.detected, "no-face angle ⇒ detected:false");
            assert!(
                result.score.is_none(),
                "no-face angle ⇒ score:null (not a low number)"
            );
            assert_eq!(result.reason, Some(NoScoreReason::NoFace));

            let mut raw = JsonObject::new();
            attach_face_likeness(&mut raw, Some(&result), Some("char_src"));
            assert_eq!(
                raw.get(FACE_LIKENESS_FACT_KEY),
                Some(&json!({
                    "score": Value::Null,
                    "detected": false,
                    "method": LIKENESS_METHOD,
                    "sourceAssetId": "char_src",
                    "reason": "no_face",
                })),
                "the N/A block persists detected:false + reason, not a low score"
            );
        }

        #[test]
        fn no_source_face_makes_every_angle_na_but_never_fails_the_job() {
            // Construction-side non-fatal guard surrogate: a source with no detectable face yields a
            // scorer (NOT an error) whose every angle is the NoSourceFace N/A — the whole set is N/A,
            // generation is never blocked. (The producer additionally maps a hard construction error
            // to a `None` scorer ⇒ the field is omitted; see `omitted_when_scorer_absent`.)
            let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(0)); // no source face
            assert!(!scorer.has_source_face(), "no detectable source face");
            let result = scorer.score_or_null(&image(200)); // a real face in the generation
            assert!(!result.detected, "no source ⇒ every angle N/A");
            assert_eq!(result.reason, Some(NoScoreReason::NoSourceFace));
        }

        #[test]
        fn omitted_when_scorer_absent() {
            // The producer wraps scorer CONSTRUCTION so a hard failure → `None` scorer → no block is
            // built → the field is omitted entirely from the sidecar (no null clutter, no crash).
            // This is the `attach_face_likeness(None)` omit-when-absent path the angle producer uses
            // when `scorer` is `None`.
            let mut raw = JsonObject::new();
            raw.insert("steps".to_owned(), json!(8));
            attach_face_likeness(&mut raw, None, Some("char_src"));
            assert!(
                !raw.contains_key(FACE_LIKENESS_FACT_KEY),
                "absent scorer ⇒ faceLikeness omitted, generation unaffected"
            );
        }

        #[test]
        fn low_confidence_angle_records_low_confidence_na() {
            // A low-confidence detection (an extreme-but-detected angle below MIN_DET_SCORE) records an
            // honest low_confidence N/A, not a noisy number — exercises the stub's `1..=76` tier.
            let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(200));
            let result = scorer.score_or_null(&image(40)); // det_score 0.5+40/512 ≈ 0.578 < 0.65
            assert!(!result.detected, "low-confidence angle ⇒ detected:false");
            assert!(result.score.is_none());
            assert_eq!(result.reason, Some(NoScoreReason::LowConfidence));
        }

        // -- the shared seam every angle-set producer calls (InstantID / FLUX.2 / Qwen / SenseNova) --
        // `build_face_likeness_scorer` needs real weights, so it is covered by the `#[ignore]` real-weight
        // test; `score_generated_image` (the per-image half all four lanes call identically) is fully
        // covered here over the stub scorer.

        /// Build a stub scorer the way `score_generated_image` consumes it (`Option<&_>`).
        fn stub_scorer(
            source_px: u8,
        ) -> (
            FaceLikenessScorer,
            std::sync::Arc<std::sync::atomic::AtomicUsize>,
        ) {
            FaceLikenessScorer::with_stub_source(&image(source_px))
        }

        #[test]
        fn score_generated_image_attaches_block_and_embeds_source_once_across_a_set() {
            // The generator-agnostic per-image post-pass: each finished angle yields a faceLikeness
            // block (detected:true for reliable views), and across an N-angle set the SOURCE is embedded
            // exactly once — proving the FLUX.2 / Qwen / SenseNova lanes reuse one cached source embed
            // (they build the scorer once, then call this per image), not re-embed per angle.
            let (scorer, calls) = stub_scorer(200);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "construction embedded the source once"
            );

            let angles = [200u8, 180, 160, 140, 120, 100, 90, 200, 180, 160, 140];
            for px in angles {
                let block = score_generated_image(Some(&scorer), &image(px), Some("char_src"))
                    .expect("a scorer present ⇒ a block built");
                assert_eq!(block.get("detected"), Some(&Value::Bool(true)));
                assert_eq!(block.get("sourceAssetId"), Some(&json!("char_src")));
                assert_eq!(block.get("method"), Some(&json!(LIKENESS_METHOD)));
            }
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1 + angles.len(),
                "1 source embed + 1 detect per angle — the source is NEVER re-embedded across the set"
            );
        }

        #[test]
        fn score_generated_image_no_face_attaches_detected_false_block() {
            // A profile / up / down angle (no detectable frontal face) attaches an honest detected:false
            // N/A block through the shared seam — never a low number, never a skipped asset.
            let (scorer, _calls) = stub_scorer(200);
            let block = score_generated_image(Some(&scorer), &image(0), Some("char_src"))
                .expect("a scorer present ⇒ a block (the N/A block) built");
            assert_eq!(
                Value::Object(block),
                json!({
                    "score": Value::Null,
                    "detected": false,
                    "method": LIKENESS_METHOD,
                    "sourceAssetId": "char_src",
                    "reason": "no_face",
                })
            );
        }

        #[test]
        fn score_generated_image_none_scorer_omits_block() {
            // No angle set / non-fatal construction failure ⇒ `None` scorer ⇒ `None` ⇒ the consumer
            // omits the field entirely (the same contract the producer relies on for non-angle paths).
            assert!(
                score_generated_image(None, &image(200), Some("char_src")).is_none(),
                "None scorer ⇒ no block ⇒ field omitted"
            );
        }

        // -- sc-4410: the SAME shared seam carries pose-library scoring ------------------
        // Pose-library routes (InstantID pose set, the FLUX.2 / Qwen edit pose tier, and the strict-
        // control pose lanes) call `build_face_likeness_scorer` + `score_generated_image` exactly as the
        // angle lanes do — there is ONE shared seam, no pose/angle fork. These assert the pose-specific
        // contract over the weight-free stub: source embedded once across N poses, and a full-body /
        // turned pose with no reliable frontal face records an honest N/A (not a low number).

        #[test]
        fn source_embedded_once_across_n_poses_each_attaches_a_block() {
            // The caching AC under the pose-library shape: one source embed at construction, reused
            // across N pose scores (the strict-control + edit pose lanes build the scorer once, then call
            // `score_generated_image` per finished pose). Every reliable-face pose attaches a block.
            let (scorer, calls) = stub_scorer(200);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "source embedded once at construction"
            );

            // A pose-library set (front-facing poses → reliable frontal face).
            let poses = [200u8, 190, 180, 170, 160, 150];
            for px in poses {
                let block = score_generated_image(Some(&scorer), &image(px), Some("char_src"))
                    .expect("a scorer present ⇒ a block built");
                assert_eq!(block.get("detected"), Some(&Value::Bool(true)));
                assert_eq!(block.get("sourceAssetId"), Some(&json!("char_src")));
                assert!(block.get("score").map(Value::is_number).unwrap_or(false));
            }
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1 + poses.len(),
                "1 source embed + 1 detect per pose — the source is NEVER re-embedded across the set"
            );
            assert_eq!(
                scorer.source_embed_count(),
                1,
                "still one source embed after N poses"
            );
        }

        #[test]
        fn full_body_or_turned_pose_records_detected_false_not_a_low_number() {
            // A full-body / turned pose with no detectable frontal face (the AC's explicit `detected:
            // false` case) attaches an honest N/A block through the shared seam — never a low number,
            // never a skipped pose asset.
            let (scorer, _calls) = stub_scorer(200);
            let block = score_generated_image(Some(&scorer), &image(0), Some("char_src"))
                .expect("a scorer present ⇒ the N/A block built");
            assert_eq!(
                Value::Object(block),
                json!({
                    "score": Value::Null,
                    "detected": false,
                    "method": LIKENESS_METHOD,
                    "sourceAssetId": "char_src",
                    "reason": "no_face",
                }),
                "a full-body / turned pose persists detected:false + reason, not a low score"
            );
        }

        // -- sc-4411: the SAME shared seam carries Image Studio "With Character" (plain) scoring -------
        // The PLAIN With-Character generation (`character_image` + a `referenceAssetId`, no angle/pose)
        // routes to its identity generator (Z-Image identity init, Flux IP-adapter, InstantID identity,
        // PuLID, Kolors IP, SenseNova / FLUX.2 / Qwen plain edit) and scores each finished image through
        // `build_face_likeness_scorer` + `score_generated_image` — the IDENTICAL seam the angle/pose
        // lanes use, no fork. These assert the sc-4411 ACs over the weight-free stub (generator-
        // agnostic): a block is attached vs the reference, the source is embedded once across N images,
        // CHANGING the reference changes the scored source, and a non-frontal result is an honest N/A.

        #[test]
        fn with_character_attaches_block_and_embeds_source_once_across_n_images() {
            // A With-Character generation produces N regular images "with" the character; each attaches a
            // faceLikeness block vs the chosen reference, and across the N-image batch the SOURCE is
            // embedded exactly once (the per-job scorer is built once from the reference, then reused).
            let (scorer, calls) = stub_scorer(200);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "construction embedded the reference source once"
            );
            // Distinct per-image seeds ⇒ distinct generations, but all "with" the one character.
            let generated = [200u8, 150, 120, 90, 80];
            for px in generated {
                let block = score_generated_image(Some(&scorer), &image(px), Some("char_ref_1"))
                    .expect("a scorer present ⇒ a block built");
                assert_eq!(block.get("detected"), Some(&Value::Bool(true)));
                assert_eq!(
                    block.get("sourceAssetId"),
                    Some(&json!("char_ref_1")),
                    "the block attributes the score to the chosen reference asset"
                );
                assert!(block.get("score").map(Value::is_number).unwrap_or(false));
            }
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1 + generated.len(),
                "1 source embed + 1 detect per image — the source is NEVER re-embedded across the batch"
            );
            assert_eq!(scorer.source_embed_count(), 1, "still one source embed");
        }

        #[test]
        fn changing_the_reference_asset_changes_the_scored_source() {
            // The explicit AC: changing the reference asset changes the source the score is computed
            // against. Two scorers built from DIFFERENT reference faces (the per-job source is the
            // current `referenceAssetId`, never cached across jobs) score the SAME generated image to
            // different cosines — same-identity high, different-identity lower.
            let generated = image(200); // a generation matching reference A's identity (pixel 200)
            let (scorer_a, _) = FaceLikenessScorer::with_stub_source(&image(200)); // reference A
            let (scorer_b, _) = FaceLikenessScorer::with_stub_source(&image(80)); // reference B (changed)

            let a = scorer_a.score_or_null(&generated);
            let b = scorer_b.score_or_null(&generated);
            let (sa, sb) = (a.score.expect("A scored"), b.score.expect("B scored"));
            assert!(
                sa > sb,
                "same reference (A) cosine {sa} exceeds the changed reference (B) cosine {sb} — \
                 the source IS derived from the current reference asset"
            );

            // And the persisted block records WHICH reference each score was computed against.
            let block_a =
                score_generated_image(Some(&scorer_a), &generated, Some("ref_a")).expect("block A");
            let block_b =
                score_generated_image(Some(&scorer_b), &generated, Some("ref_b")).expect("block B");
            assert_eq!(block_a.get("sourceAssetId"), Some(&json!("ref_a")));
            assert_eq!(block_b.get("sourceAssetId"), Some(&json!("ref_b")));
        }

        #[test]
        fn with_character_non_frontal_result_is_na_per_metric_honesty() {
            // The AC's metric-honesty rule: a non-frontal / no-face With-Character result returns N/A
            // (detected:false, score:null + reason) through the shared seam — never a misleading low
            // number. (pixel 0 ⇒ the stub's no-face funnel.)
            let (scorer, _calls) = stub_scorer(200);
            let block = score_generated_image(Some(&scorer), &image(0), Some("char_ref_1"))
                .expect("a scorer present ⇒ the N/A block built");
            assert_eq!(
                Value::Object(block),
                json!({
                    "score": Value::Null,
                    "detected": false,
                    "method": LIKENESS_METHOD,
                    "sourceAssetId": "char_ref_1",
                    "reason": "no_face",
                }),
                "a non-frontal With-Character result is an honest N/A, not a low score"
            );
        }

        #[test]
        fn with_character_non_fatal_when_reference_has_no_face() {
            // Non-fatal construction surrogate (the sc-4407 contract carried to sc-4411): a reference
            // image with no detectable face yields a scorer (NOT an error) whose every generated image
            // is the NoSourceFace N/A — the With-Character generation still renders, scores just absent.
            let (scorer, _calls) = FaceLikenessScorer::with_stub_source(&image(0)); // no face in ref
            assert!(!scorer.has_source_face(), "no detectable reference face");
            let result = scorer.score_or_null(&image(200));
            assert!(!result.detected, "no reference face ⇒ every image N/A");
            assert_eq!(result.reason, Some(NoScoreReason::NoSourceFace));
            // And the producer's `None`-scorer path (a hard construction failure) omits the field.
            assert!(
                score_generated_image(None, &image(200), Some("char_ref_1")).is_none(),
                "a failed scorer construction ⇒ no block ⇒ field omitted (generation unaffected)"
            );
        }
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
