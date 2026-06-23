//! Tier-0 dataset quality evaluation (epic 6529 "Dataset Doctor", sc-6532).
//!
//! This is the **pure** half of Tier-0: the result types plus all decision logic. Given per-item
//! scalars (already extracted from pixels by the worker — see
//! `sceneworks_worker::dataset_quality`), the item dimensions (sc-6531), the content hash
//! (sc-6531), and the dataset context (kind, target bucket, preset minimum), it derives typed
//! per-item quality flags and dataset-level findings.
//!
//! No image decoding happens here, so this module stays codec-free and is fully unit-testable with
//! synthetic scalars — no fixture images required. The pixel extraction that *does* need a decode
//! lives in the worker, behind [`Tier0Scalars`].
//!
//! Catalog, thresholds, and the warn-not-block framing come from the spike,
//! `docs/sc-6530/dataset-doctor-metrics.md`.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::contracts::string_enum;

string_enum! {
    /// A Tier-0 quality check — the cheap, no-model checks from the spike's catalog.
    pub enum QualityCheck {
        Resolution => "resolution",
        CropLoss => "crop_loss",
        Blur => "blur",
        Exposure => "exposure",
        ExactDuplicate => "exact_duplicate",
        NearDuplicate => "near_duplicate",
        Count => "count",
        // Decode: the image could not be decoded, so no pixel scalars exist for it (rare — uploads
        // are normalized to png/jpeg/webp). Surfaced so an undecodable image can't quietly count as
        // "technically fine" in the readiness rollup.
        Decode => "decode",
        // Tier-1 (epic 6529 P2, sc-6536): embedding-based "training usefulness" findings, computed by
        // the dataset-analysis job from CLIP image embeddings. Advisory only (never Fatal) and NOT
        // counted toward the technical sub-score — they describe the dataset's usefulness, not each
        // image's technical quality.
        // NearDuplicateEmbedding: CLIP-cosine near-duplicate — catches a burst of near-identical
        // frames that the pHash `NearDuplicate` check misses (different crop/exposure, same content).
        NearDuplicateEmbedding => "near_duplicate_embedding",
        // LowDiversity: dataset-level — the set clusters too tightly in embedding space (same
        // pose/angle/lighting/background), so the LoRA won't generalize. Carries the "add some from
        // other angles" recommendation.
        LowDiversity => "low_diversity",
        // LowAesthetic: dataset-level, STYLE datasets ONLY (sc-6537) — the set scores low on the LAION
        // CLIP+MLP aesthetic predictor. Advisory, never gates: low-aesthetic candids are often the best
        // identity shots, so this is never raised on person/object datasets (documented bias).
        LowAesthetic => "low_aesthetic",
    }
}

string_enum! {
    /// Severity of a flag. Drives the thumbnail badge (no flag / `Info` → ✓, `Warn` → ⚠,
    /// `Fatal` → ✕) and the readiness gate. Bias to `Warn`; reserve `Fatal` for genuinely
    /// untrainable inputs (the spike's block-vs-warn policy). Declared worst-last so the derived
    /// `Ord` ranks `Info < Warn < Fatal`.
    pub enum Severity {
        Info => "info",
        Warn => "warn",
        Fatal => "fatal",
    }
}

string_enum! {
    /// What a LoRA is being taught. Changes what "good" means, so several thresholds vary by it
    /// (the spike's per-kind column). Maps from the training preset's `recommendedFor` tags.
    pub enum DatasetKind {
        Person => "person",
        Style => "style",
        Object => "object",
    }
}

/// Per-image scalars extracted from pixels by the worker. Blur + exposure are measured on the
/// center-crop→bucket-resize the trainer actually feeds (so they are comparable across source
/// resolutions and capture upscale-to-mush); the perceptual hash is taken on the full image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tier0Scalars {
    /// Variance of the Laplacian on the bucket-resized grayscale crop. Higher = sharper.
    pub blur_variance: f64,
    /// Fraction of pixels crushed to black (luma ≤ cutoff), in `[0, 1]`.
    pub shadow_clip: f64,
    /// Fraction of pixels blown to white (luma ≥ cutoff), in `[0, 1]`.
    pub highlight_clip: f64,
    /// Perceptual-hash bytes from the one pinned `HasherConfig` (fixed length). Hamming distance
    /// over these drives near-duplicate clustering.
    pub phash: Vec<u8>,
}

/// A single quality finding on an item (or, for the dataset-level `Count` check, the dataset).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityFlag {
    pub check: QualityCheck,
    pub severity: Severity,
    /// The measured value behind the flag — Laplacian variance, clip fraction, short-edge px,
    /// crop-loss fraction, item count, or Hamming distance. The "evidence" the readout shows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// The threshold `value` was judged against (for "x of y"-style copy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Peer item ids for relational checks (exact / near-duplicate clusters). Empty otherwise —
    /// lets the UI say *which* photos are duplicates and lets 6533 reconcile pHash with CLIP pairs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<String>,
    /// The user has dismissed this finding for this image (sc-6534 per-image override). An
    /// acknowledged flag is kept in the report so the UI can show it struck-through, but is
    /// **excluded from every rollup** (the item's badge severity, the technical sub-score, the
    /// severity counts, and the gate). Only non-`Fatal` findings can be acknowledged — a decode
    /// failure is genuinely untrainable, so this stays `false` for it (enforced in `evaluate_tier0`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub acknowledged: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// All Tier-0 flags raised for one item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemQualityFlags {
    pub item_id: String,
    pub flags: Vec<QualityFlag>,
}

/// Result of evaluating Tier-0 over a dataset: per-item flags plus dataset-level findings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tier0Evaluation {
    pub items: Vec<ItemQualityFlags>,
    pub dataset: Vec<QualityFlag>,
}

/// One item's inputs to Tier-0 evaluation: identity, dimensions + content hash (sc-6531), and the
/// pixel scalars (worker-extracted; `None` until they have been computed).
#[derive(Debug, Clone)]
pub struct ItemQualityInput {
    pub item_id: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub content_hash: Option<String>,
    pub scalars: Option<Tier0Scalars>,
    /// Checks the user has dismissed for this image (sc-6534). A flag whose check is listed here is
    /// marked acknowledged (and so dropped from every rollup) — but only when it is not `Fatal`, so
    /// a decode failure cannot be waved through. Resolved by the API from the item's persisted ack,
    /// already filtered to the current content hash.
    pub acknowledged: Vec<QualityCheck>,
}

/// Tier-0 thresholds. Defaults from the spike (`docs/sc-6530`), varying by kind where the kind
/// changes what "good" means. One config surface — calibrate here, not at call sites (sc-6530 §8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tier0Thresholds {
    /// Below `min_resolution_ratio × bucket_edge` on the short side ⇒ upscale-to-mush warning.
    pub min_resolution_ratio: f64,
    /// Center-crop dropping more than this fraction of the long side ⇒ crop-loss warning.
    pub crop_loss_fraction: f64,
    /// Absolute Laplacian-variance floor: below this is "soft" regardless of the dataset.
    pub blur_floor: f64,
    /// An image below `blur_relative_factor × dataset_median` is a soft outlier within a sharp set.
    pub blur_relative_factor: f64,
    /// More than this fraction of pixels clipped at black or white ⇒ exposure warning.
    pub exposure_clip_fraction: f64,
    /// pHash Hamming distance ≤ this ⇒ near-duplicate.
    pub near_dup_hamming: u32,
}

impl Tier0Thresholds {
    /// Spike defaults, tuned per kind. Style tolerates softer images (texture/bokeh); person and
    /// object are stricter on sharpness.
    pub fn for_kind(kind: &DatasetKind) -> Self {
        let blur_floor = match kind {
            DatasetKind::Style => 60.0,
            _ => 100.0,
        };
        Self {
            min_resolution_ratio: 0.75,
            crop_loss_fraction: 0.35,
            blur_floor,
            blur_relative_factor: 0.5,
            exposure_clip_fraction: 0.05,
            near_dup_hamming: 6,
        }
    }
}

/// Below this many usable images a set cannot train at all — the only count that blocks rather
/// than warns (sc-6530 §5).
pub const HARD_MIN_ITEMS: u32 = 4;

/// Evaluate Tier-0 over a dataset. Pure: no IO, no decode. `bucket_edge` is the trainer's target
/// square resolution (the size the blur/exposure scalars were measured at); `min_items` is the
/// chosen preset's recommended minimum.
///
/// Degenerate inputs are handled: an empty set yields only the `Count` finding; a single image has
/// no blur median and forms no duplicate cluster, so neither the relative-blur nor the duplicate
/// checks fire.
pub fn evaluate_tier0(
    items: &[ItemQualityInput],
    bucket_edge: u32,
    min_items: u32,
    thresholds: &Tier0Thresholds,
) -> Tier0Evaluation {
    let mut per_item: Vec<ItemQualityFlags> = items
        .iter()
        .map(|item| ItemQualityFlags {
            item_id: item.item_id.clone(),
            flags: Vec::new(),
        })
        .collect();

    let blur_median = median(
        items
            .iter()
            .filter_map(|item| item.scalars.as_ref().map(|s| s.blur_variance)),
    );

    // Per-image, dataset-independent checks.
    for (idx, item) in items.iter().enumerate() {
        let flags = &mut per_item[idx].flags;
        push_resolution_flags(flags, item, bucket_edge, thresholds);
        push_scalar_flags(flags, item, blur_median, thresholds);
    }

    // Relational checks: exact duplicates (content hash) then near duplicates (pHash).
    push_exact_duplicate_flags(items, &mut per_item);
    push_near_duplicate_flags(items, &mut per_item, thresholds.near_dup_hamming);

    // Per-image overrides (sc-6534): mark dismissed findings acknowledged once all flags exist. A
    // `Fatal` finding is never acknowledgeable — a decode failure is untrainable regardless of the
    // user's wishes — so the gate can't be waved past genuinely broken inputs. Relational checks are
    // marked per side: dismissing a duplicate on image A leaves B's flag standing (B is still a dup
    // of something), which is the intended asymmetry.
    for (idx, item) in items.iter().enumerate() {
        if item.acknowledged.is_empty() {
            continue;
        }
        for flag in &mut per_item[idx].flags {
            if flag.severity != Severity::Fatal && item.acknowledged.contains(&flag.check) {
                flag.acknowledged = true;
            }
        }
    }

    // Dataset-level: too few images for the preset.
    let mut dataset = Vec::new();
    let count = items.len() as u32;
    if count < min_items {
        let severity = if count < HARD_MIN_ITEMS {
            Severity::Fatal
        } else {
            Severity::Warn
        };
        dataset.push(QualityFlag {
            check: QualityCheck::Count,
            severity,
            value: Some(count as f64),
            threshold: Some(min_items as f64),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    Tier0Evaluation {
        items: per_item,
        dataset,
    }
}

fn push_resolution_flags(
    flags: &mut Vec<QualityFlag>,
    item: &ItemQualityInput,
    bucket_edge: u32,
    thresholds: &Tier0Thresholds,
) {
    let (Some(width), Some(height)) = (item.width, item.height) else {
        return;
    };
    let short = f64::from(width.min(height));
    let long = f64::from(width.max(height));
    let target = f64::from(bucket_edge);

    if target > 0.0 && short < target {
        // Below the bucket is a mild nudge; far below means it will be upscaled to mush.
        let severity = if short < thresholds.min_resolution_ratio * target {
            Severity::Warn
        } else {
            Severity::Info
        };
        flags.push(QualityFlag {
            check: QualityCheck::Resolution,
            severity,
            value: Some(short),
            threshold: Some(target),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    if long > 0.0 {
        let crop_loss = (long - short) / long;
        if crop_loss > thresholds.crop_loss_fraction {
            flags.push(QualityFlag {
                check: QualityCheck::CropLoss,
                severity: Severity::Warn,
                value: Some(crop_loss),
                threshold: Some(thresholds.crop_loss_fraction),
                peers: Vec::new(),
                acknowledged: false,
            });
        }
    }
}

fn push_scalar_flags(
    flags: &mut Vec<QualityFlag>,
    item: &ItemQualityInput,
    blur_median: Option<f64>,
    thresholds: &Tier0Thresholds,
) {
    let Some(scalars) = &item.scalars else {
        return;
    };

    // Soft if below the absolute floor OR a clear outlier below the dataset median (the spike's
    // "floor AND relative" rule — relative alone would pass a uniformly-soft set).
    let below_floor = scalars.blur_variance < thresholds.blur_floor;
    let below_relative = blur_median
        .is_some_and(|median| scalars.blur_variance < thresholds.blur_relative_factor * median);
    if below_floor || below_relative {
        flags.push(QualityFlag {
            check: QualityCheck::Blur,
            severity: Severity::Warn,
            value: Some(scalars.blur_variance),
            threshold: Some(thresholds.blur_floor),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    let clip = scalars.shadow_clip.max(scalars.highlight_clip);
    if clip > thresholds.exposure_clip_fraction {
        flags.push(QualityFlag {
            check: QualityCheck::Exposure,
            severity: Severity::Warn,
            value: Some(clip),
            threshold: Some(thresholds.exposure_clip_fraction),
            peers: Vec::new(),
            acknowledged: false,
        });
    }
}

fn push_exact_duplicate_flags(items: &[ItemQualityInput], per_item: &mut [ItemQualityFlags]) {
    let mut by_hash: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        if let Some(hash) = item.content_hash.as_deref() {
            by_hash.entry(hash).or_default().push(idx);
        }
    }
    for group in by_hash.values().filter(|group| group.len() > 1) {
        for &idx in group {
            let peers = group
                .iter()
                .filter(|&&other| other != idx)
                .map(|&other| items[other].item_id.clone())
                .collect();
            per_item[idx].flags.push(QualityFlag {
                check: QualityCheck::ExactDuplicate,
                severity: Severity::Warn,
                value: Some(0.0),
                threshold: None,
                peers,
                acknowledged: false,
            });
        }
    }
}

fn push_near_duplicate_flags(
    items: &[ItemQualityInput],
    per_item: &mut [ItemQualityFlags],
    near_dup_hamming: u32,
) {
    let hashes: Vec<Option<&[u8]>> = items
        .iter()
        .map(|item| item.scalars.as_ref().map(|s| s.phash.as_slice()))
        .collect();

    // Union-find over pHash pairs within the Hamming threshold.
    let mut uf = UnionFind::new(items.len());
    for (i, hi) in hashes.iter().copied().enumerate() {
        let Some(hi) = hi else { continue };
        for (j, hj) in hashes.iter().copied().enumerate().skip(i + 1) {
            let Some(hj) = hj else { continue };
            if hamming(hi, hj).is_some_and(|distance| distance <= near_dup_hamming) {
                uf.union(i, j);
            }
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, hash) in hashes.iter().enumerate() {
        if hash.is_some() {
            clusters.entry(uf.find(i)).or_default().push(i);
        }
    }

    for cluster in clusters.values().filter(|cluster| cluster.len() > 1) {
        for &idx in cluster {
            // Exclude byte-identical peers — those are reported as exact duplicates, not near
            // ones, so the same pair is never flagged twice (sc-6530 §2).
            let mut peers = Vec::new();
            let mut nearest = u32::MAX;
            for &other in cluster {
                if other == idx || same_content_hash(items, idx, other) {
                    continue;
                }
                if let (Some(a), Some(b)) = (hashes[idx], hashes[other]) {
                    if let Some(distance) = hamming(a, b) {
                        nearest = nearest.min(distance);
                    }
                }
                peers.push(items[other].item_id.clone());
            }
            if !peers.is_empty() {
                per_item[idx].flags.push(QualityFlag {
                    check: QualityCheck::NearDuplicate,
                    severity: Severity::Warn,
                    value: (nearest != u32::MAX).then_some(f64::from(nearest)),
                    threshold: Some(f64::from(near_dup_hamming)),
                    peers,
                    acknowledged: false,
                });
            }
        }
    }
}

fn same_content_hash(items: &[ItemQualityInput], a: usize, b: usize) -> bool {
    match (
        items[a].content_hash.as_deref(),
        items[b].content_hash.as_deref(),
    ) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Hamming distance between two equal-length byte hashes, or `None` if the lengths differ (which
/// only happens if two images were hashed with different configs — they never are in practice).
fn hamming(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum())
}

/// Median of a sample, or `None` when empty. Even-length samples average the two middle values.
fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

/// Minimal union-find for near-duplicate clustering.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]]; // path halving
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

// ---------------------------------------------------------------------------
// Readiness report (sc-6533) — the dataset-level rollup the training screens read.
// ---------------------------------------------------------------------------

/// A cached Tier-0 extraction stored on a dataset item. Keyed by **both** the content hash and the
/// bucket edge: blur + exposure are measured on the center-crop→bucket-resize, so the same image at
/// a different training resolution has different scalars. Reuse only when both still match (sc-6533).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedTier0Scalars {
    pub content_hash: String,
    pub bucket_edge: u32,
    pub scalars: Tier0Scalars,
}

impl CachedTier0Scalars {
    /// True when this cache entry still applies to an item: same image bytes (content hash) **and**
    /// same bucket edge (blur/exposure are measured at the bucket, so a resolution change
    /// invalidates them). The single source of truth for cache reuse — kept here, pure and tested,
    /// rather than in the API layer.
    pub fn valid_for(&self, content_hash: Option<&str>, bucket_edge: u32) -> bool {
        self.bucket_edge == bucket_edge && content_hash == Some(self.content_hash.as_str())
    }
}

/// A user's per-image quality override (sc-6534): the set of checks they have dismissed for one
/// image. Keyed by content hash — the same precedent as [`CachedTier0Scalars`] — so a dismissal
/// cannot silently apply after the image bytes change (you can't pre-acknowledge a photo you never
/// saw). Stored on the dataset item; resolved to effective checks by the API before evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityAck {
    pub content_hash: String,
    pub checks: Vec<QualityCheck>,
}

impl QualityAck {
    /// True when this ack still applies to an item: same image bytes (content hash). A mismatch
    /// (the image was replaced) silently voids the ack.
    pub fn valid_for(&self, content_hash: Option<&str>) -> bool {
        content_hash == Some(self.content_hash.as_str())
    }

    /// The dismissed checks that still apply, or empty when the ack is stale. The single source of
    /// truth for ack reuse — kept here, pure and tested, not in the API layer.
    pub fn effective_checks(&self, content_hash: Option<&str>) -> Vec<QualityCheck> {
        if self.valid_for(content_hash) {
            self.checks.clone()
        } else {
            Vec::new()
        }
    }
}

string_enum! {
    /// The discrete readiness gate (sc-6530 §4) — deliberately NOT a 0–100 score. Drives whether
    /// Train is enabled and what the readout says.
    pub enum ReadinessGate {
        // Ready: enough usable images, nothing worth surfacing — train freely.
        Ready => "ready",
        // NeedsAttention: trainable, but warnings are present (soft/dup/low-res). Train stays
        // enabled; the readout explains what would make it stronger. The default for most real sets.
        NeedsAttention => "needs_attention",
        // Blocked: genuinely untrainable (a fatal flag, e.g. too few images). Train disabled.
        Blocked => "blocked",
    }
}

/// Interpretable sub-scores (sc-6530 §4) — each a plain share the UI can name, never blended into
/// one number. `technical` comes from Tier-0; the rest are Tier-1 (`None` until the embedding job
/// lands, keeping the report forward-compatible).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadinessSubScores {
    /// Share of items with no technical-quality warning (resolution/crop/blur/exposure/dup/decode).
    pub technical: f64,
    /// Embedding spread in `[0, 1]` (sc-6536): `1 - mean pairwise cosine similarity`. Higher = more
    /// varied. `None` until the dataset-analysis job runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diversity: Option<f64>,
    /// Face-embedding consistency (sc-6529 face stack — a *different* encoder than CLIP). Reserved;
    /// CLIP must not fill this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<f64>,
    /// Caption↔image CLIP alignment (sc-6537). `None` until increment-2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alignment: Option<f64>,
    /// Aesthetic score (sc-6537, style datasets only — advisory). `None` until increment-2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aesthetic: Option<f64>,
}

/// Per-item readiness: the worst severity (for the thumbnail badge) and the flags behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemReadiness {
    pub item_id: String,
    /// Worst flag severity on the item, or `None` when clean (badge ✓).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<Severity>,
    pub flags: Vec<QualityFlag>,
}

/// Flag counts by severity across items + dataset — the readout's "2 blurry, 3 near-dup" line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeverityCounts {
    pub info: u32,
    pub warn: u32,
    pub fatal: u32,
}

/// One metric's spread across the dataset, for the Advanced surface's per-metric distribution
/// (sc-6534). The raw per-item values let the UI draw a histogram; `threshold` marks the line a
/// flag is judged against, and `higher_is_better` orients it (sharpness improves upward, clip
/// fractions downward). Carried in the report payload so the web needs no second fetch for scalars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricDistribution {
    pub values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    pub higher_is_better: bool,
}

/// Per-metric distributions over the items that have pixel scalars (sc-6534). `None` on the report
/// until any scalars exist (an unassessed or empty set).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadinessDistributions {
    pub blur_variance: MetricDistribution,
    pub shadow_clip: MetricDistribution,
    pub highlight_clip: MetricDistribution,
}

/// The complete readiness report the training screens render from one payload (sc-6533). Tier-1
/// fields stay `None`/empty until the embedding job attaches them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetReadinessReport {
    pub gate: ReadinessGate,
    pub sub_scores: ReadinessSubScores,
    pub counts: SeverityCounts,
    pub item_count: u32,
    pub items: Vec<ItemReadiness>,
    /// Dataset-level findings not tied to a single item (e.g. too few images).
    pub dataset_flags: Vec<QualityFlag>,
    /// Per-metric value spreads for the Advanced distribution view (sc-6534). Populated by the
    /// extraction layer (`sceneworks-image-quality`), which holds the scalars; the pure core rollup
    /// leaves it `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributions: Option<ReadinessDistributions>,
}

/// Resolved inputs for an evaluation, derived from the chosen training target/preset and the
/// dataset's character.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadinessContext {
    pub kind: DatasetKind,
    pub bucket_edge: u32,
    pub min_items: u32,
    pub thresholds: Tier0Thresholds,
}

/// Target resolution assumed when no training target is selected yet (a dataset can be assessed
/// before Teach locks a target).
pub const DEFAULT_BUCKET_EDGE: u32 = 1024;

/// Resolve the evaluation context from the chosen target/preset + the dataset's character. Pure and
/// unit-tested so this easy-to-get-wrong mapping doesn't live only in the (here-unbuildable) API
/// layer. `recommended_for` are the preset/target tags (`"character"`/`"style"`); `character_type`
/// is the dataset's character kind (e.g. `"person"`); `bucket_edge` is floored to a multiple of 32
/// to match the trainer.
pub fn readiness_context(
    target_resolution: Option<u32>,
    recommended_for: &[String],
    character_type: Option<&str>,
    configured_min_items: Option<u32>,
) -> ReadinessContext {
    let kind = resolve_kind(recommended_for, character_type);
    let bucket_edge = target_resolution
        .map(|res| (res / 32).max(1) * 32)
        .unwrap_or(DEFAULT_BUCKET_EDGE);
    let min_items = configured_min_items.unwrap_or_else(|| default_min_items(&kind));
    let thresholds = Tier0Thresholds::for_kind(&kind);
    ReadinessContext {
        kind,
        bucket_edge,
        min_items,
        thresholds,
    }
}

fn resolve_kind(recommended_for: &[String], character_type: Option<&str>) -> DatasetKind {
    let tagged = |tag: &str| {
        recommended_for
            .iter()
            .any(|value| value.eq_ignore_ascii_case(tag))
    };
    if character_type.is_some_and(|value| value.eq_ignore_ascii_case("person"))
        || tagged("character")
        || tagged("person")
    {
        DatasetKind::Person
    } else if tagged("style") {
        DatasetKind::Style
    } else {
        DatasetKind::Object
    }
}

/// Per-kind minimum item count (sc-6530 §3) when the preset doesn't pin one.
fn default_min_items(kind: &DatasetKind) -> u32 {
    match kind {
        DatasetKind::Person => 15,
        DatasetKind::Style => 20,
        DatasetKind::Object => 10,
        DatasetKind::Unknown(_) => 12,
    }
}

/// Roll a Tier-0 evaluation (plus optional Tier-1 embedding findings, sc-6536) up into the dataset
/// readiness report (sc-6533). Pure: no IO. The gate follows the spike's block-vs-warn policy —
/// `Blocked` on any fatal flag, else `NeedsAttention` on any warning, else `Ready`. The `technical`
/// share counts only **technical** checks (resolution/crop/blur/exposure/dup/decode) — embedding
/// "usefulness" findings describe the *set*, not an image's technical quality, so they raise the
/// badge and the gate but never drag `technical` down. Pass `None` for `tier1` when only Tier-0 ran.
pub fn build_readiness_report(
    evaluation: Tier0Evaluation,
    tier1: Option<&Tier1Evaluation>,
    aesthetic: Option<&AestheticEvaluation>,
) -> DatasetReadinessReport {
    let item_count = evaluation.items.len() as u32;
    let mut counts = SeverityCounts::default();
    let mut items = Vec::with_capacity(evaluation.items.len());
    let mut technically_clean = 0_u32;

    // Per-item Tier-1 flags keyed by item id, merged onto the authoritative Tier-0 item set.
    let tier1_by_item: HashMap<&str, &[QualityFlag]> = tier1
        .map(|t| {
            t.items
                .iter()
                .map(|entry| (entry.item_id.as_str(), entry.flags.as_slice()))
                .collect()
        })
        .unwrap_or_default();

    for entry in &evaluation.items {
        let mut flags = entry.flags.clone();
        if let Some(extra) = tier1_by_item.get(entry.item_id.as_str()) {
            flags.extend(extra.iter().cloned());
        }
        // Acknowledged findings (sc-6534) are dropped from every rollup — counts, the badge, the
        // technical share, and (via counts) the gate — but kept in `flags` for struck-through display.
        for flag in flags.iter().filter(|flag| !flag.acknowledged) {
            bump(&mut counts, &flag.severity);
        }
        // Technical share: a decode failure (Decode warn) correctly counts; an embedding near-dup
        // (sc-6536) does not.
        let has_technical_problem = flags
            .iter()
            .filter(|flag| !flag.acknowledged && is_technical_check(&flag.check))
            .any(|flag| matches!(flag.severity, Severity::Warn | Severity::Fatal));
        if !has_technical_problem {
            technically_clean += 1;
        }
        let severity = worst_severity(&flags);
        items.push(ItemReadiness {
            item_id: entry.item_id.clone(),
            severity,
            flags,
        });
    }

    let mut dataset_flags = evaluation.dataset;
    if let Some(t) = tier1 {
        dataset_flags.extend(t.dataset.iter().cloned());
    }
    if let Some(a) = aesthetic {
        dataset_flags.extend(a.dataset.iter().cloned());
    }
    for flag in dataset_flags.iter().filter(|flag| !flag.acknowledged) {
        bump(&mut counts, &flag.severity);
    }

    let technical = if item_count == 0 {
        1.0
    } else {
        f64::from(technically_clean) / f64::from(item_count)
    };
    let gate = if counts.fatal > 0 {
        ReadinessGate::Blocked
    } else if counts.warn > 0 {
        ReadinessGate::NeedsAttention
    } else {
        ReadinessGate::Ready
    };

    DatasetReadinessReport {
        gate,
        sub_scores: ReadinessSubScores {
            technical,
            diversity: tier1.map(|t| t.diversity),
            identity: None,
            alignment: None,
            aesthetic: aesthetic.map(|a| a.score),
        },
        counts,
        item_count,
        items,
        dataset_flags,
        distributions: None,
    }
}

/// Technical-quality checks (Tier-0) vs. the "training usefulness" checks (Tier-1, sc-6536). Only the
/// former feed the `technical` sub-score. Unknown/future checks are treated as non-technical.
fn is_technical_check(check: &QualityCheck) -> bool {
    matches!(
        check,
        QualityCheck::Resolution
            | QualityCheck::CropLoss
            | QualityCheck::Blur
            | QualityCheck::Exposure
            | QualityCheck::ExactDuplicate
            | QualityCheck::NearDuplicate
            | QualityCheck::Count
            | QualityCheck::Decode
    )
}

// ---------------------------------------------------------------------------
// Tier-1 embedding analysis (epic 6529 P2, sc-6536) — pure math over CLIP image embeddings. The
// embeddings are produced by the (Metal-only) dataset-analysis job; everything here operates on
// `Vec<f32>` and is fully unit-testable with synthetic vectors, exactly like `evaluate_tier0`.
// ---------------------------------------------------------------------------

/// One item's CLIP image embedding (raw — `evaluate_tier1` L2-normalizes internally).
#[derive(Debug, Clone, PartialEq)]
pub struct ItemEmbedding {
    pub item_id: String,
    pub embedding: Vec<f32>,
    /// Per-image overrides (sc-6534): the checks the user dismissed for this item. `evaluate_tier1`
    /// marks the matching non-`Fatal` per-item flag (`near_duplicate_embedding`) acknowledged so it
    /// drops from the rollups but stays for struck-through display — mirrors `ItemQualityInput`.
    pub acknowledged: Vec<QualityCheck>,
}

/// Persisted CLIP image embeddings for a dataset (sc-6535) — too large to inline in the manifest
/// (768×f32 per item), so they live in a content-hash-keyed sidecar file
/// (`dataset.sceneworks.embeddings.json`). Reused across edits: an item whose bytes are unchanged
/// keeps its embedding. The readiness path maps each item's `content_hash` → embedding → [`ItemEmbedding`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetEmbeddings {
    /// The embedding space (e.g. `"clip-vit-l14"`); guards against mixing encoders on reuse.
    pub space: String,
    /// `content_hash` → raw embedding (the encoder's un-normalized output).
    pub embeddings: std::collections::BTreeMap<String, Vec<f32>>,
}

/// Tier-1 thresholds (sc-6536). Calibrated in sc-6535 against real CLIP ViT-L/14 embeddings over the
/// Google DreamBooth benchmark (30 subject sets), two style sets (hokusai, monkey-island), and a real
/// 64-image person set: `near_dup_cosine 0.95` validated; the non-style diversity floor dropped
/// `0.18 → 0.12` (0.18 flagged 22/30 of the canonical subject benchmark, incl. a healthy real person
/// set at 0.188); and `diversity_min_items` gates the low-diversity *warning* to sets large enough for
/// it to mean redundancy rather than just smallness. The Style floor (0.10) is provisional — only two
/// style sets sampled, and they disagree (0.144 vs 0.284).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tier1Thresholds {
    /// Cosine similarity ≥ this ⇒ embedding near-duplicate.
    pub near_dup_cosine: f64,
    /// Diversity (`1 - mean pairwise cosine`) below this ⇒ a `LowDiversity` warning.
    pub diversity_floor: f64,
    /// Minimum comparable items before the low-diversity *warning* fires. Below this a dataset is too
    /// small for "diversity" to mean anything (a 5-image subject set is supposed to be tight) — near-
    /// duplicate detection carries the redundancy signal instead. The diversity *score* is still
    /// computed for the variety meter; only the warning is gated. (sc-6535 calibration.)
    pub diversity_min_items: usize,
}

impl Tier1Thresholds {
    pub fn for_kind(kind: &DatasetKind) -> Self {
        // Style sets are intentionally more uniform (one aesthetic), so tolerate lower diversity;
        // person/object want pose/angle variety. The 0.12 non-style floor is calibrated (sc-6535):
        // it clears healthy real person sets (~0.19) while still catching genuinely degenerate large
        // sets; the prior 0.18 flagged most of the canonical DreamBooth subject benchmark.
        let diversity_floor = match kind {
            DatasetKind::Style => 0.10,
            _ => 0.12,
        };
        Self {
            near_dup_cosine: 0.95,
            diversity_floor,
            diversity_min_items: 15,
        }
    }
}

/// Result of Tier-1 evaluation: per-item embedding flags + dataset-level findings + the diversity
/// score (`[0, 1]`) that fills `ReadinessSubScores.diversity`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tier1Evaluation {
    pub items: Vec<ItemQualityFlags>,
    pub dataset: Vec<QualityFlag>,
    pub diversity: f64,
}

/// Evaluate Tier-1 over a dataset's CLIP image embeddings (sc-6536). Pure: no IO, no model. Degenerate
/// inputs are handled — fewer than two comparable items form no clusters and have an undefined spread,
/// so `diversity` defaults to 1.0 (nothing to be redundant with) and no flags fire.
pub fn evaluate_tier1(items: &[ItemEmbedding], thresholds: &Tier1Thresholds) -> Tier1Evaluation {
    let mut per_item: Vec<ItemQualityFlags> = items
        .iter()
        .map(|item| ItemQualityFlags {
            item_id: item.item_id.clone(),
            flags: Vec::new(),
        })
        .collect();

    // L2-normalize once; a zero/degenerate vector → `None`, excluded from cosine.
    let normalized: Vec<Option<Vec<f32>>> = items
        .iter()
        .map(|item| l2_normalize(&item.embedding))
        .collect();

    // Pairwise cosine drives both the near-dup union-find and the diversity mean.
    let mut uf = UnionFind::new(items.len());
    let mut sim_sum = 0.0_f64;
    let mut sim_count = 0_u64;
    for (i, ni) in normalized.iter().enumerate() {
        let Some(a) = ni else { continue };
        for (j, nj) in normalized.iter().enumerate().skip(i + 1) {
            let Some(b) = nj else { continue };
            let cos = dot(a, b);
            sim_sum += cos;
            sim_count += 1;
            if cos >= thresholds.near_dup_cosine {
                uf.union(i, j);
            }
        }
    }

    // Near-dup clusters → per-item `NearDuplicateEmbedding` flags with peers + the cluster's max cosine.
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, hash) in normalized.iter().enumerate() {
        if hash.is_some() {
            clusters.entry(uf.find(i)).or_default().push(i);
        }
    }
    for cluster in clusters.values().filter(|cluster| cluster.len() > 1) {
        for &idx in cluster {
            let mut peers = Vec::new();
            let mut max_cos = 0.0_f64;
            for &other in cluster {
                if other == idx {
                    continue;
                }
                if let (Some(a), Some(b)) = (&normalized[idx], &normalized[other]) {
                    max_cos = max_cos.max(dot(a, b));
                }
                peers.push(items[other].item_id.clone());
            }
            if !peers.is_empty() {
                per_item[idx].flags.push(QualityFlag {
                    check: QualityCheck::NearDuplicateEmbedding,
                    severity: Severity::Warn,
                    value: Some(max_cos),
                    threshold: Some(thresholds.near_dup_cosine),
                    peers,
                    acknowledged: false,
                });
            }
        }
    }

    // Per-image overrides (sc-6534): a dismissed `near_duplicate_embedding` drops from the rollups
    // but stays for struck-through display — the same post-pass as `evaluate_tier0`. `low_diversity`
    // is dataset-level, so it is not per-item-acknowledgeable. (NearDuplicateEmbedding is `Warn`, so
    // the non-`Fatal` guard never blocks it; it is kept for parity with Tier-0.)
    for (idx, item) in items.iter().enumerate() {
        if item.acknowledged.is_empty() {
            continue;
        }
        for flag in &mut per_item[idx].flags {
            if flag.severity != Severity::Fatal && item.acknowledged.contains(&flag.check) {
                flag.acknowledged = true;
            }
        }
    }

    // Diversity = 1 − mean pairwise cosine (clamped). Undefined for <2 comparable items → 1.0.
    let diversity = if sim_count == 0 {
        1.0
    } else {
        (1.0 - sim_sum / sim_count as f64).clamp(0.0, 1.0)
    };
    // Low-diversity is a dataset-level *warning*, gated by size: below `diversity_min_items` a tight
    // set is expected (a small subject LoRA), so near-dup carries the signal and this stays quiet.
    let comparable = normalized.iter().filter(|entry| entry.is_some()).count();
    let mut dataset = Vec::new();
    if comparable >= thresholds.diversity_min_items && diversity < thresholds.diversity_floor {
        dataset.push(QualityFlag {
            check: QualityCheck::LowDiversity,
            severity: Severity::Warn,
            value: Some(diversity),
            threshold: Some(thresholds.diversity_floor),
            peers: Vec::new(),
            acknowledged: false,
        });
    }

    Tier1Evaluation {
        items: per_item,
        dataset,
        diversity,
    }
}

/// L2-normalize a vector for cosine, or `None` when it has no magnitude (a degenerate embedding).
fn l2_normalize(values: &[f32]) -> Option<Vec<f32>> {
    let norm = values
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return None;
    }
    Some(
        values
            .iter()
            .map(|x| (f64::from(*x) / norm) as f32)
            .collect(),
    )
}

/// Cosine similarity of two already-L2-normalized vectors (their dot product).
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// Worst severity among the item's *active* flags — acknowledged findings (sc-6534) and the
/// forward-compat `Unknown` variant are skipped, so an all-dismissed item badges clean and a future
/// unknown severity can never rank as "worst" (the derived `Ord` would otherwise sort it after
/// `Fatal`).
fn worst_severity(flags: &[QualityFlag]) -> Option<Severity> {
    flags
        .iter()
        .filter(|flag| !flag.acknowledged)
        .map(|flag| &flag.severity)
        .filter(|severity| !matches!(severity, Severity::Unknown(_)))
        .max()
        .cloned()
}

fn bump(counts: &mut SeverityCounts, severity: &Severity) {
    match severity {
        Severity::Info => counts.info += 1,
        Severity::Warn => counts.warn += 1,
        Severity::Fatal => counts.fatal += 1,
        Severity::Unknown(_) => {}
    }
}

// ---------------------------------------------------------------------------
// Aesthetic scoring (epic 6529 P2, sc-6537) — the LAION-Aesthetics V2 CLIP+MLP predictor, run
// host-side over the persisted CLIP image embeddings. STYLE datasets only, advisory (never gates):
// low-aesthetic candids are often the best identity shots, so it is never raised on person/object.
// Pure `Vec<f32>` math + a tiny dep-free safetensors reader for the bundled MLP head (the asset is
// `include_bytes!`d by the caller, `sceneworks-image-quality`).
// ---------------------------------------------------------------------------

/// One `Linear` layer of the aesthetic MLP head.
#[derive(Debug, Clone)]
struct AestheticLayer {
    /// `[out_dim, in_dim]` row-major — the PyTorch / safetensors `Linear.weight` layout.
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

/// The LAION-Aesthetics V2 MLP head (`layers.{0,2,4,6,7}`): a stack of `Linear` layers with **no
/// activations** (the `linearMSE` predictor is purely linear at inference), applied to the
/// L2-normalized CLIP ViT-L/14 image embedding to predict an aesthetic score (~`[1, 10]`).
#[derive(Debug, Clone)]
pub struct AestheticPredictor {
    layers: Vec<AestheticLayer>,
}

/// Failure loading an [`AestheticPredictor`] from safetensors bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AestheticLoadError(pub String);

impl std::fmt::Display for AestheticLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "aesthetic predictor load failed: {}", self.0)
    }
}
impl std::error::Error for AestheticLoadError {}

impl AestheticPredictor {
    /// Parse the MLP head from a safetensors blob holding `layers.{N}.weight` / `layers.{N}.bias`
    /// (`F32`). The format is an 8-byte little-endian header length, a JSON header, then the raw
    /// little-endian tensor bytes — so this needs only `serde_json`, no tensor backend.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, AestheticLoadError> {
        let err = |m: &str| AestheticLoadError(m.to_owned());
        let len_bytes: [u8; 8] = bytes
            .get(0..8)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| err("truncated header length"))?;
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| err("header length past end of file"))?;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])
            .map_err(|e| AestheticLoadError(format!("header json: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| err("header is not an object"))?;
        let data = &bytes[header_end..];

        let read = |name: &str| -> Result<(Vec<f32>, Vec<usize>), AestheticLoadError> {
            let t = obj
                .get(name)
                .ok_or_else(|| AestheticLoadError(format!("missing tensor {name}")))?;
            if t.get("dtype").and_then(serde_json::Value::as_str) != Some("F32") {
                return Err(AestheticLoadError(format!("{name}: dtype must be F32")));
            }
            let shape: Vec<usize> = t
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| AestheticLoadError(format!("{name}: no shape")))?
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect();
            let offs = t
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| AestheticLoadError(format!("{name}: no data_offsets")))?;
            let a = offs
                .first()
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let b = offs.get(1).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
            let raw = data
                .get(a..b)
                .filter(|r| r.len() % 4 == 0)
                .ok_or_else(|| AestheticLoadError(format!("{name}: bad data range")))?;
            let vals = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((vals, shape))
        };

        let mut indices: Vec<usize> = obj
            .keys()
            .filter_map(|k| k.strip_prefix("layers.")?.strip_suffix(".weight"))
            .filter_map(|n| n.parse::<usize>().ok())
            .collect();
        indices.sort_unstable();
        if indices.is_empty() {
            return Err(err("no `layers.N.weight` tensors in the safetensors"));
        }

        let mut layers = Vec::with_capacity(indices.len());
        for i in indices {
            let (weight, wshape) = read(&format!("layers.{i}.weight"))?;
            let (bias, _) = read(&format!("layers.{i}.bias"))?;
            let [out_dim, in_dim] = wshape[..] else {
                return Err(AestheticLoadError(format!("layers.{i}.weight is not 2-D")));
            };
            if weight.len() != out_dim * in_dim || bias.len() != out_dim {
                return Err(AestheticLoadError(format!(
                    "layers.{i}: weight/bias shape mismatch"
                )));
            }
            layers.push(AestheticLayer {
                weight,
                bias,
                in_dim,
                out_dim,
            });
        }
        if layers.windows(2).any(|w| w[0].out_dim != w[1].in_dim) {
            return Err(err("layer dimensions do not chain"));
        }
        Ok(Self { layers })
    }

    /// The embedding dimension this predictor expects (768 for ViT-L/14).
    pub fn input_dim(&self) -> usize {
        self.layers.first().map_or(0, |l| l.in_dim)
    }

    /// Predict the LAION aesthetic score (~`[1, 10]`) for one CLIP image embedding. The embedding is
    /// L2-normalized first (the predictor was trained on normalized CLIP features). Returns `None` for
    /// a degenerate (zero-magnitude) or wrong-dimension embedding.
    pub fn predict(&self, embedding: &[f32]) -> Option<f64> {
        let norm = embedding
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            return None;
        }
        let mut x: Vec<f64> = embedding.iter().map(|v| f64::from(*v) / norm).collect();
        for layer in &self.layers {
            if x.len() != layer.in_dim {
                return None;
            }
            let mut y = Vec::with_capacity(layer.out_dim);
            for o in 0..layer.out_dim {
                let row = &layer.weight[o * layer.in_dim..(o + 1) * layer.in_dim];
                let acc = row
                    .iter()
                    .zip(&x)
                    .map(|(w, xi)| f64::from(*w) * xi)
                    .sum::<f64>()
                    + f64::from(layer.bias[o]);
                y.push(acc);
            }
            x = y;
        }
        x.first().copied()
    }
}

/// Aesthetic-score thresholds (sc-6537). **Placeholder pending a style-dataset sweep** (§8) — the LAION
/// score is roughly `[1, 10]`; this floor is a guess, not tuned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AestheticThresholds {
    /// Mean aesthetic below this ⇒ a `LowAesthetic` advisory. Placeholder.
    pub floor: f64,
}

impl Default for AestheticThresholds {
    fn default() -> Self {
        Self { floor: 5.0 }
    }
}

/// Result of aesthetic scoring (sc-6537): the mean LAION score (fills `ReadinessSubScores.aesthetic`)
/// plus dataset-level advisory flags (`LowAesthetic`). Only ever produced for `DatasetKind::Style`.
#[derive(Debug, Clone, PartialEq)]
pub struct AestheticEvaluation {
    pub score: f64,
    pub dataset: Vec<QualityFlag>,
}

/// Score a **style** dataset's images with the LAION aesthetic predictor over their CLIP embeddings.
/// Returns `None` for non-style kinds (aesthetic never applies to person/object) or when nothing
/// scored. The `LowAesthetic` flag is `Info`, never `Warn`: aesthetic is explicitly non-blocking
/// (low-aesthetic candids are often the best shots) and the floor is an uncalibrated placeholder, so it
/// surfaces the sub-score + a heads-up without ever changing the readiness gate.
pub fn evaluate_aesthetic(
    items: &[ItemEmbedding],
    predictor: &AestheticPredictor,
    kind: &DatasetKind,
    thresholds: &AestheticThresholds,
) -> Option<AestheticEvaluation> {
    if *kind != DatasetKind::Style {
        return None;
    }
    let scores: Vec<f64> = items
        .iter()
        .filter_map(|item| predictor.predict(&item.embedding))
        .collect();
    if scores.is_empty() {
        return None;
    }
    let score = scores.iter().sum::<f64>() / scores.len() as f64;
    let mut dataset = Vec::new();
    if score < thresholds.floor {
        dataset.push(QualityFlag {
            check: QualityCheck::LowAesthetic,
            severity: Severity::Info,
            value: Some(score),
            threshold: Some(thresholds.floor),
            peers: Vec::new(),
            acknowledged: false,
        });
    }
    Some(AestheticEvaluation { score, dataset })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> Tier0Thresholds {
        Tier0Thresholds::for_kind(&DatasetKind::Person)
    }

    fn item(id: &str) -> ItemQualityInput {
        ItemQualityInput {
            item_id: id.to_owned(),
            width: Some(512),
            height: Some(512),
            content_hash: None,
            scalars: None,
            acknowledged: Vec::new(),
        }
    }

    fn scalars(blur: f64, shadow: f64, highlight: f64, phash: Vec<u8>) -> Tier0Scalars {
        Tier0Scalars {
            blur_variance: blur,
            shadow_clip: shadow,
            highlight_clip: highlight,
            phash,
        }
    }

    fn flags_for<'a>(eval: &'a Tier0Evaluation, id: &str) -> &'a [QualityFlag] {
        &eval
            .items
            .iter()
            .find(|entry| entry.item_id == id)
            .expect("item present")
            .flags
    }

    fn has(eval: &Tier0Evaluation, id: &str, check: QualityCheck) -> bool {
        flags_for(eval, id).iter().any(|f| f.check == check)
    }

    #[test]
    fn small_and_wide_images_flag_resolution_and_crop_loss() {
        let mut tiny = item("tiny");
        tiny.width = Some(128);
        tiny.height = Some(128);
        let mut wide = item("wide");
        wide.width = Some(1000);
        wide.height = Some(300);

        let eval = evaluate_tier0(&[tiny, wide], 512, 1, &thresholds());

        assert!(has(&eval, "tiny", QualityCheck::Resolution));
        assert!(has(&eval, "wide", QualityCheck::CropLoss));
    }

    #[test]
    fn square_bucket_sized_image_has_no_resolution_or_crop_flags() {
        let mut ok = item("ok");
        ok.width = Some(512);
        ok.height = Some(512);
        let eval = evaluate_tier0(std::slice::from_ref(&ok), 512, 1, &thresholds());
        assert!(!has(&eval, "ok", QualityCheck::Resolution));
        assert!(!has(&eval, "ok", QualityCheck::CropLoss));
    }

    #[test]
    fn blur_fires_on_absolute_floor() {
        let mut soft = item("soft");
        soft.scalars = Some(scalars(10.0, 0.0, 0.0, vec![0; 8]));
        let eval = evaluate_tier0(std::slice::from_ref(&soft), 512, 1, &thresholds());
        assert!(has(&eval, "soft", QualityCheck::Blur));
    }

    #[test]
    fn blur_fires_on_relative_outlier_even_above_floor() {
        // A custom low floor so only the relative-to-median rule can trip.
        let mut th = thresholds();
        th.blur_floor = 1.0;
        let sharp = |id: &str, v: f64| {
            let mut it = item(id);
            it.scalars = Some(scalars(v, 0.0, 0.0, vec![0; 8]));
            it
        };
        // Median ≈ 1000; the outlier at 100 is < 0.5 × median but well above the floor.
        let items = [
            sharp("a", 1000.0),
            sharp("b", 1100.0),
            sharp("c", 1050.0),
            sharp("outlier", 100.0),
        ];
        let eval = evaluate_tier0(&items, 512, 1, &th);
        assert!(has(&eval, "outlier", QualityCheck::Blur));
        assert!(!has(&eval, "a", QualityCheck::Blur));
    }

    #[test]
    fn uniformly_soft_set_still_flags_via_floor() {
        // Every image is soft, so the median is soft too — the absolute floor must still catch it.
        let soft = |id: &str| {
            let mut it = item(id);
            it.scalars = Some(scalars(20.0, 0.0, 0.0, vec![0; 8]));
            it
        };
        let eval = evaluate_tier0(&[soft("a"), soft("b"), soft("c")], 512, 1, &thresholds());
        assert!(has(&eval, "a", QualityCheck::Blur));
    }

    #[test]
    fn exposure_fires_on_clipping() {
        let mut dark = item("dark");
        dark.scalars = Some(scalars(5000.0, 0.4, 0.0, vec![1; 8]));
        let eval = evaluate_tier0(std::slice::from_ref(&dark), 512, 1, &thresholds());
        let flag = flags_for(&eval, "dark")
            .iter()
            .find(|f| f.check == QualityCheck::Exposure)
            .expect("exposure flag");
        assert_eq!(flag.value, Some(0.4));
    }

    #[test]
    fn exact_duplicates_reference_each_other() {
        let mut a = item("a");
        a.content_hash = Some("samehash".to_owned());
        let mut b = item("b");
        b.content_hash = Some("samehash".to_owned());
        let mut c = item("c");
        c.content_hash = Some("different".to_owned());

        let eval = evaluate_tier0(&[a, b, c], 512, 1, &thresholds());
        let a_dup = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::ExactDuplicate)
            .expect("exact-dup flag on a");
        assert_eq!(a_dup.peers, vec!["b".to_owned()]);
        assert!(!has(&eval, "c", QualityCheck::ExactDuplicate));
    }

    #[test]
    fn near_duplicates_cluster_by_hamming_but_not_exact_pairs() {
        // a/b differ by 1 bit (near); c is far; d is byte-identical to a (exact, not near).
        let near = |id: &str, hash: u8, content: &str| {
            let mut it = item(id);
            it.content_hash = Some(content.to_owned());
            it.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![hash, 0, 0, 0, 0, 0, 0, 0]));
            it
        };
        let items = [
            near("a", 0b0000_0000, "ha"),
            near("b", 0b0000_0001, "hb"),
            near("c", 0b1111_1111, "hc"),
            near("d", 0b0000_0000, "ha"), // same hash AND same content as a
        ];
        let eval = evaluate_tier0(&items, 512, 1, &thresholds());

        let a_near = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("near-dup flag on a");
        assert!(a_near.peers.contains(&"b".to_owned()));
        // d is an exact duplicate of a, so it must NOT also appear as a near-duplicate peer.
        assert!(!a_near.peers.contains(&"d".to_owned()));
        assert!(a_near.peers.iter().all(|p| p != "c"));
    }

    #[test]
    fn count_warns_then_blocks_below_hard_floor() {
        let warn = evaluate_tier0(
            &[item("a"), item("b"), item("c"), item("d"), item("e")],
            512,
            12,
            &thresholds(),
        );
        let warn_flag = warn
            .dataset
            .iter()
            .find(|f| f.check == QualityCheck::Count)
            .expect("count flag");
        assert_eq!(warn_flag.severity, Severity::Warn);

        let block = evaluate_tier0(&[item("a"), item("b")], 512, 12, &thresholds());
        let block_flag = block
            .dataset
            .iter()
            .find(|f| f.check == QualityCheck::Count)
            .expect("count flag");
        assert_eq!(block_flag.severity, Severity::Fatal);
    }

    #[test]
    fn empty_dataset_only_reports_count() {
        let eval = evaluate_tier0(&[], 512, 10, &thresholds());
        assert!(eval.items.is_empty());
        assert_eq!(eval.dataset.len(), 1);
        assert_eq!(eval.dataset[0].check, QualityCheck::Count);
    }

    #[test]
    fn flags_round_trip_as_camelcase_json() {
        let flag = QualityFlag {
            check: QualityCheck::NearDuplicate,
            severity: Severity::Warn,
            value: Some(2.0),
            threshold: Some(6.0),
            peers: vec!["other".to_owned()],
            acknowledged: false,
        };
        let json = serde_json::to_string(&flag).expect("serialize");
        // Field names are camelCase; the check enum value is the snake_case string the rest of
        // the contract crate uses for string enums.
        assert!(json.contains("\"check\":\"near_duplicate\""));
        assert!(json.contains("\"peers\":[\"other\"]"));
        let back: QualityFlag = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, flag);
    }

    // --- sc-6533 readiness report ---

    fn sharp(id: &str, phash: u8) -> ItemQualityInput {
        let mut it = item(id);
        it.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![phash; 8]));
        it
    }

    #[test]
    fn gate_is_ready_when_every_item_is_clean() {
        let eval = evaluate_tier0(&[sharp("a", 0xAA), sharp("b", 0x55)], 512, 1, &thresholds());
        let report = build_readiness_report(eval, None, None);
        assert_eq!(report.gate, ReadinessGate::Ready);
        assert!((report.sub_scores.technical - 1.0).abs() < 1e-9);
        assert!(report.sub_scores.diversity.is_none()); // Tier-1 not computed yet
        assert_eq!(report.counts, SeverityCounts::default());
        assert!(report.items.iter().all(|i| i.severity.is_none()));
    }

    #[test]
    fn gate_needs_attention_and_technical_share_drops_on_warning() {
        let mut soft = item("soft");
        soft.scalars = Some(scalars(10.0, 0.0, 0.0, vec![0x0F; 8]));
        let eval = evaluate_tier0(&[soft, sharp("ok", 0xF0)], 512, 1, &thresholds());
        let report = build_readiness_report(eval, None, None);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
        assert!(report.counts.warn >= 1);
        assert!((report.sub_scores.technical - 0.5).abs() < 1e-9);
        let soft_item = report.items.iter().find(|i| i.item_id == "soft").unwrap();
        assert_eq!(soft_item.severity, Some(Severity::Warn));
        let ok_item = report.items.iter().find(|i| i.item_id == "ok").unwrap();
        assert_eq!(ok_item.severity, None);
    }

    #[test]
    fn gate_blocks_when_too_few_images() {
        let eval = evaluate_tier0(&[item("a"), item("b")], 512, 12, &thresholds());
        let report = build_readiness_report(eval, None, None);
        assert_eq!(report.gate, ReadinessGate::Blocked);
        assert!(report.counts.fatal >= 1);
        assert!(report
            .dataset_flags
            .iter()
            .any(|f| f.check == QualityCheck::Count));
    }

    #[test]
    fn decode_failure_flag_keeps_item_out_of_the_technical_share() {
        // An undecodable item carries a Decode warning (injected upstream) and must not count as
        // "fine" even though it has no scalar-derived flags.
        let mut broken = item("broken");
        broken.scalars = None;
        let mut eval = evaluate_tier0(&[broken, sharp("ok", 0x3C)], 512, 1, &thresholds());
        eval.items
            .iter_mut()
            .find(|e| e.item_id == "broken")
            .unwrap()
            .flags
            .push(QualityFlag {
                check: QualityCheck::Decode,
                severity: Severity::Warn,
                value: None,
                threshold: None,
                peers: Vec::new(),
                acknowledged: false,
            });
        let report = build_readiness_report(eval, None, None);
        assert!((report.sub_scores.technical - 0.5).abs() < 1e-9);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
    }

    #[test]
    fn readiness_context_resolves_kind_bucket_and_min_items() {
        let person = readiness_context(Some(1024), &["character".to_owned()], Some("person"), None);
        assert_eq!(person.kind, DatasetKind::Person);
        assert_eq!(person.bucket_edge, 1024);
        assert_eq!(person.min_items, 15);

        // 1000 floors to a multiple of 32; style raises the default minimum.
        let style = readiness_context(Some(1000), &["style".to_owned()], None, None);
        assert_eq!(style.kind, DatasetKind::Style);
        assert_eq!(style.bucket_edge, 992);
        assert_eq!(style.min_items, 20);

        // No target → default bucket; explicit min wins over the per-kind default.
        let object = readiness_context(None, &[], None, Some(8));
        assert_eq!(object.kind, DatasetKind::Object);
        assert_eq!(object.bucket_edge, DEFAULT_BUCKET_EDGE);
        assert_eq!(object.min_items, 8);
    }

    #[test]
    fn report_round_trips_as_camelcase_json() {
        let eval = evaluate_tier0(&[sharp("a", 0x11)], 512, 1, &thresholds());
        let report = build_readiness_report(eval, None, None);
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(json.contains("\"subScores\""));
        assert!(json.contains("\"itemCount\""));
        let back: DatasetReadinessReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);
    }

    #[test]
    fn cached_scalars_valid_only_for_same_hash_and_bucket() {
        let cache = CachedTier0Scalars {
            content_hash: "abc".to_owned(),
            bucket_edge: 512,
            scalars: scalars(5000.0, 0.0, 0.0, vec![0; 8]),
        };
        assert!(cache.valid_for(Some("abc"), 512));
        assert!(
            !cache.valid_for(Some("abc"), 1024),
            "bucket change invalidates"
        );
        assert!(
            !cache.valid_for(Some("xyz"), 512),
            "content change invalidates"
        );
        assert!(!cache.valid_for(None, 512), "missing hash invalidates");
    }

    #[test]
    fn acknowledged_findings_drop_from_every_rollup() {
        // Two soft images; the user dismisses blur on `a` only. Far-apart pHashes keep them from
        // also registering as near-duplicates, so blur is the only finding in play.
        let soft = |id: &str, phash: u8, ack: &[QualityCheck]| {
            let mut it = item(id);
            it.scalars = Some(scalars(10.0, 0.0, 0.0, vec![phash; 8]));
            it.acknowledged = ack.to_vec();
            it
        };
        let eval = evaluate_tier0(
            &[soft("a", 0x00, &[QualityCheck::Blur]), soft("b", 0xFF, &[])],
            512,
            1,
            &thresholds(),
        );
        // The flag is kept on `a` for display, but marked acknowledged; `b`'s stands.
        let a_blur = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::Blur)
            .expect("blur flag on a");
        assert!(a_blur.acknowledged);
        let b_blur = flags_for(&eval, "b")
            .iter()
            .find(|f| f.check == QualityCheck::Blur)
            .expect("blur flag on b");
        assert!(!b_blur.acknowledged);

        let report = build_readiness_report(eval, None, None);
        assert_eq!(report.counts.warn, 1, "only b's warning counts");
        assert!(
            (report.sub_scores.technical - 0.5).abs() < 1e-9,
            "a counts as clean once dismissed"
        );
        let a = report.items.iter().find(|i| i.item_id == "a").expect("a");
        assert_eq!(a.severity, None, "a badges clean");
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
    }

    #[test]
    fn all_findings_acknowledged_reads_ready() {
        let mut soft = item("solo");
        soft.scalars = Some(scalars(10.0, 0.0, 0.0, vec![0; 8]));
        soft.acknowledged = vec![QualityCheck::Blur];
        let report = build_readiness_report(
            evaluate_tier0(std::slice::from_ref(&soft), 512, 1, &thresholds()),
            None,
            None,
        );
        assert_eq!(report.gate, ReadinessGate::Ready);
        assert_eq!(report.counts.warn, 0);
        assert!((report.sub_scores.technical - 1.0).abs() < 1e-9);
        // The dismissed flag is still in the payload so the UI can show it struck-through.
        assert!(report.items[0]
            .flags
            .iter()
            .any(|f| f.check == QualityCheck::Blur && f.acknowledged));
    }

    #[test]
    fn dismissing_a_duplicate_is_asymmetric() {
        // a and b are near-dupes (1 bit apart); the user dismisses only on a, so b still counts.
        let near = |id: &str, hash: u8, ack: &[QualityCheck]| {
            let mut it = item(id);
            it.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![hash, 0, 0, 0, 0, 0, 0, 0]));
            it.acknowledged = ack.to_vec();
            it
        };
        let eval = evaluate_tier0(
            &[
                near("a", 0b0000_0000, &[QualityCheck::NearDuplicate]),
                near("b", 0b0000_0001, &[]),
            ],
            512,
            2,
            &thresholds(),
        );
        let a = flags_for(&eval, "a")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("near-dup on a");
        assert!(a.acknowledged, "a's dup is dismissed");
        let b = flags_for(&eval, "b")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("near-dup on b");
        assert!(!b.acknowledged, "b is still a duplicate of something");
    }

    #[test]
    fn quality_ack_voids_when_content_hash_changes() {
        let ack = QualityAck {
            content_hash: "abc".to_owned(),
            checks: vec![QualityCheck::Blur],
        };
        assert!(ack.valid_for(Some("abc")));
        assert_eq!(ack.effective_checks(Some("abc")), vec![QualityCheck::Blur]);
        // Replaced image (new hash) or missing hash → the dismissal no longer applies.
        assert!(!ack.valid_for(Some("xyz")));
        assert!(ack.effective_checks(Some("xyz")).is_empty());
        assert!(ack.effective_checks(None).is_empty());
    }

    // ----- Tier-1 embedding analysis (sc-6536) -----

    fn emb(id: &str, vector: &[f32]) -> ItemEmbedding {
        ItemEmbedding {
            item_id: id.to_owned(),
            embedding: vector.to_vec(),
            acknowledged: Vec::new(),
        }
    }

    fn tier1_thresholds() -> Tier1Thresholds {
        Tier1Thresholds::for_kind(&DatasetKind::Person)
    }

    /// Build a minimal valid safetensors blob from `(name, shape, f32 values)` tuples.
    fn safetensors_bytes(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, vals) in tensors {
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [data.len(), data.len() + bytes.len()],
                }),
            );
            data.extend(bytes);
        }
        let hjson = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
        out.extend(hjson);
        out.extend(data);
        out
    }

    #[test]
    fn aesthetic_predictor_parses_and_forwards() {
        // layers.0 = identity (2->2); layers.2 = sum + 0.5 (2->1). Gaps in indices mirror the real
        // head (dropout layers carry no params); the parser orders by index and chains the dims.
        let st = safetensors_bytes(&[
            ("layers.0.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            ("layers.0.bias", vec![2], vec![0.0, 0.0]),
            ("layers.2.weight", vec![1, 2], vec![1.0, 1.0]),
            ("layers.2.bias", vec![1], vec![0.5]),
        ]);
        let predictor = AestheticPredictor::from_safetensors_bytes(&st).expect("parse");
        assert_eq!(predictor.input_dim(), 2);
        // [3,4] L2-normalizes to [0.6,0.8] -> identity -> 0.6 + 0.8 + 0.5 = 1.9.
        let score = predictor.predict(&[3.0, 4.0]).expect("score");
        assert!((score - 1.9).abs() < 1e-9, "got {score}");
        assert!(
            predictor.predict(&[0.0, 0.0]).is_none(),
            "degenerate -> None"
        );
        assert!(
            predictor.predict(&[1.0, 2.0, 3.0]).is_none(),
            "wrong dim -> None"
        );
    }

    #[test]
    fn aesthetic_is_style_only_and_advisory() {
        // Constant predictor: weight zeros, bias 3.0 -> always 3.0 regardless of input.
        let st = safetensors_bytes(&[
            ("layers.0.weight", vec![1, 2], vec![0.0, 0.0]),
            ("layers.0.bias", vec![1], vec![3.0]),
        ]);
        let predictor = AestheticPredictor::from_safetensors_bytes(&st).expect("parse");
        let items = vec![
            ItemEmbedding {
                item_id: "a".into(),
                embedding: vec![1.0, 0.0],
                acknowledged: vec![],
            },
            ItemEmbedding {
                item_id: "b".into(),
                embedding: vec![0.0, 1.0],
                acknowledged: vec![],
            },
        ];
        let thresholds = AestheticThresholds { floor: 5.0 };
        // Aesthetic never applies to person/object.
        assert!(
            evaluate_aesthetic(&items, &predictor, &DatasetKind::Person, &thresholds).is_none()
        );
        assert!(
            evaluate_aesthetic(&items, &predictor, &DatasetKind::Object, &thresholds).is_none()
        );
        // Style: mean 3.0 < floor 5.0 -> a LowAesthetic advisory (Info — never gates).
        let eval = evaluate_aesthetic(&items, &predictor, &DatasetKind::Style, &thresholds)
            .expect("style");
        assert!((eval.score - 3.0).abs() < 1e-9);
        assert!(eval
            .dataset
            .iter()
            .any(|f| f.check == QualityCheck::LowAesthetic && f.severity == Severity::Info));
        // Above the floor -> score still reported, no flag.
        let lenient = AestheticThresholds { floor: 2.0 };
        let eval =
            evaluate_aesthetic(&items, &predictor, &DatasetKind::Style, &lenient).expect("style");
        assert!(eval.dataset.is_empty());
    }

    #[test]
    fn tier1_clusters_near_identical_embeddings_and_scores_low_diversity() {
        // A large burst of near-identical vectors clusters and reads as undiverse. Above the size
        // gate (>= diversity_min_items) the dataset-level LowDiversity warning fires.
        let items: Vec<_> = (0..16)
            .map(|i| emb(&format!("v{i}"), &[1.0, i as f32 * 0.001, 0.0, 0.0]))
            .collect();
        let eval = evaluate_tier1(&items, &tier1_thresholds());

        assert!(
            eval.items.iter().all(|e| e
                .flags
                .iter()
                .any(|f| f.check == QualityCheck::NearDuplicateEmbedding)),
            "every near-identical item should be an embedding near-duplicate"
        );
        assert!(
            eval.diversity < 0.1,
            "near-identical set has near-zero diversity"
        );
        assert!(eval
            .dataset
            .iter()
            .any(|f| f.check == QualityCheck::LowDiversity));
    }

    #[test]
    fn tier1_small_set_does_not_warn_low_diversity() {
        // Below the size gate a tight set is expected (a small subject LoRA), so the dataset-level
        // LowDiversity warning is suppressed even though the score is low — near-dup still flags the
        // redundancy. (sc-6535 calibration.)
        let items = [
            emb("a", &[1.0, 0.0, 0.0, 0.0]),
            emb("b", &[0.999, 0.01, 0.0, 0.0]),
            emb("c", &[1.0, 0.0, 0.0, 0.0]),
        ];
        let eval = evaluate_tier1(&items, &tier1_thresholds());
        assert!(
            eval.diversity < 0.1,
            "score is still computed for the variety meter"
        );
        assert!(
            eval.items.iter().all(|e| e
                .flags
                .iter()
                .any(|f| f.check == QualityCheck::NearDuplicateEmbedding)),
            "near-dup still flags the redundancy"
        );
        assert!(
            !eval
                .dataset
                .iter()
                .any(|f| f.check == QualityCheck::LowDiversity),
            "small set: the low-diversity warning is gated off"
        );
    }

    #[test]
    fn tier1_orthogonal_embeddings_are_diverse_and_unclustered() {
        // Pairwise-orthogonal unit vectors: cosine 0 → no near-dup, maximal diversity.
        let items = [
            emb("a", &[1.0, 0.0, 0.0, 0.0]),
            emb("b", &[0.0, 1.0, 0.0, 0.0]),
            emb("c", &[0.0, 0.0, 1.0, 0.0]),
        ];
        let eval = evaluate_tier1(&items, &tier1_thresholds());
        assert!(eval.items.iter().all(|e| e.flags.is_empty()));
        assert!(eval.dataset.is_empty());
        assert!((eval.diversity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tier1_degenerate_inputs_are_safe() {
        // A single item (no pairs) and a zero vector (no magnitude) raise nothing.
        let single = evaluate_tier1(&[emb("solo", &[1.0, 2.0, 3.0])], &tier1_thresholds());
        assert!((single.diversity - 1.0).abs() < 1e-9);
        assert!(single.items[0].flags.is_empty());

        let zero = evaluate_tier1(
            &[emb("z1", &[0.0, 0.0, 0.0]), emb("z2", &[0.0, 0.0, 0.0])],
            &tier1_thresholds(),
        );
        assert!(zero.items.iter().all(|e| e.flags.is_empty()));
        assert!((zero.diversity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tier1_findings_merge_into_the_report_without_touching_technical() {
        // Two technically-perfect images that are embedding near-duplicates.
        let mut a = item("a");
        a.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![0x00; 8]));
        let mut b = item("b");
        b.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![0xFF; 8])); // distinct pHash → no Tier-0 dup
        let tier0 = evaluate_tier0(&[a, b], 512, 1, &thresholds());

        let tier1 = evaluate_tier1(
            &[emb("a", &[1.0, 0.0, 0.0]), emb("b", &[1.0, 0.0, 0.0])],
            &tier1_thresholds(),
        );
        let report = build_readiness_report(tier0, Some(&tier1), None);

        // Embedding dup raises the badge + gate, but the images are technically fine → technical 1.0.
        assert!((report.sub_scores.technical - 1.0).abs() < 1e-9);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
        assert!(report.sub_scores.diversity.expect("diversity set") < 0.1);
        let a = report.items.iter().find(|i| i.item_id == "a").expect("a");
        assert_eq!(a.severity, Some(Severity::Warn));
        assert!(a
            .flags
            .iter()
            .any(|f| f.check == QualityCheck::NearDuplicateEmbedding));
    }

    #[test]
    fn report_without_tier1_leaves_diversity_none() {
        let mut ok = item("ok");
        ok.scalars = Some(scalars(5000.0, 0.0, 0.0, vec![1; 8]));
        let report = build_readiness_report(
            evaluate_tier0(std::slice::from_ref(&ok), 512, 1, &thresholds()),
            None,
            None,
        );
        assert_eq!(report.sub_scores.diversity, None);
        assert_eq!(report.sub_scores.aesthetic, None);
    }

    #[test]
    fn tier1_acknowledged_near_duplicate_is_marked_and_kept_per_side() {
        // The user dismisses the embedding near-dup on "a"; "b" still flags (per-side dismissal,
        // like Tier-0). The acked flag is retained for struck-through display — and
        // build_readiness_report already excludes acknowledged flags from every rollup.
        let a = ItemEmbedding {
            item_id: "a".to_owned(),
            embedding: vec![1.0, 0.0, 0.0],
            acknowledged: vec![QualityCheck::NearDuplicateEmbedding],
        };
        let eval = evaluate_tier1(&[a, emb("b", &[1.0, 0.0, 0.0])], &tier1_thresholds());

        let a_flag = eval
            .items
            .iter()
            .find(|e| e.item_id == "a")
            .expect("a")
            .flags
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicateEmbedding)
            .expect("a keeps the flag for display");
        assert!(
            a_flag.acknowledged,
            "a's dismissed near-dup is acknowledged"
        );

        let b_flag = eval
            .items
            .iter()
            .find(|e| e.item_id == "b")
            .expect("b")
            .flags
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicateEmbedding)
            .expect("b flags");
        assert!(
            !b_flag.acknowledged,
            "b's near-dup stands (per-side dismissal)"
        );
    }
}
