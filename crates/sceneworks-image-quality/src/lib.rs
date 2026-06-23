//! Tier-0 pixel extraction (epic 6529 "Dataset Doctor", sc-6532).
//!
//! The decode-bearing half of Tier-0: turn an image into the per-image scalars that the *pure*
//! decision logic in [`sceneworks_core::dataset_quality`] consumes. It lives in its own crate —
//! not the decode-free core, not the GPU-laden worker — so it builds and tests anywhere on just
//! the pure-Rust `image` (png/jpeg/webp) + `imageproc` + `image_hasher` stack, with no Metal/CUDA.
//! Both the worker and the API's synchronous readiness path can depend on it.
//!
//! Blur + exposure are measured on the **center-crop→bucket-resize the trainer actually feeds**
//! (mirrors candle-gen's `train/dataset.rs` kernel): that makes the variance comparable across
//! source resolutions and folds in upscale-to-mush (a tiny image blown up to the bucket reads as
//! soft, which is the correct signal). The perceptual hash is taken on the full image — it
//! self-normalizes by downscaling internally. See `docs/sc-6530/dataset-doctor-metrics.md`.

use std::path::{Path, PathBuf};

use image::{imageops::FilterType, DynamicImage, GrayImage, Luma, RgbImage};
use image_hasher::{HashAlg, Hasher, HasherConfig};
use sceneworks_core::dataset_quality::{
    build_readiness_report, evaluate_tier0, AestheticEvaluation, AestheticPredictor,
    DatasetReadinessReport, ItemQualityInput, MetricDistribution, QualityCheck, QualityFlag,
    ReadinessDistributions, Severity, Tier0Scalars, Tier0Thresholds, Tier1Evaluation,
};

/// Luma at or below this counts as crushed-to-black (8-bit).
const SHADOW_CUTOFF: u8 = 4;
/// Luma at or above this counts as blown-to-white (8-bit).
const HIGHLIGHT_CUTOFF: u8 = 251;

/// The one pinned perceptual-hash configuration. A fixed algorithm + size means every stored hash
/// is the same byte length and Hamming distances are comparable across the whole dataset — the
/// invariant the near-duplicate clustering in `sceneworks_core::dataset_quality` relies on.
fn dataset_hasher() -> Hasher {
    HasherConfig::new()
        .hash_alg(HashAlg::Gradient) // dHash: cheap, robust to small edits, good for near-dup
        .hash_size(8, 8)
        .to_hasher()
}

/// The bundled LAION-Aesthetics V2 MLP head (sc-6537), parsed once. Extracted (head only) from
/// `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE` (Apache-2.0; see
/// `assets/README.md`) — it scores the L2-normalized CLIP ViT-L/14 image embedding produced by the
/// dataset-analysis job. Lives here (the GPU-free host-eval crate) so the readiness path can score
/// without a model server.
pub fn aesthetic_predictor() -> &'static AestheticPredictor {
    static PREDICTOR: std::sync::OnceLock<AestheticPredictor> = std::sync::OnceLock::new();
    PREDICTOR.get_or_init(|| {
        const BYTES: &[u8] =
            include_bytes!("../assets/aesthetic-v2-sac-logos-ava1-l14.safetensors");
        AestheticPredictor::from_safetensors_bytes(BYTES)
            .expect("the bundled aesthetic predictor asset parses")
    })
}

/// Decode the image at `path` and extract its Tier-0 scalars. `bucket_edge` is the trainer's
/// target square resolution (the size blur + exposure are measured at).
pub fn extract_tier0_scalars(path: &Path, bucket_edge: u32) -> image::ImageResult<Tier0Scalars> {
    Ok(scalars_from_image(&image::open(path)?, bucket_edge))
}

/// Extract Tier-0 scalars from an already-decoded image — the testable core, no IO.
pub fn scalars_from_image(image: &DynamicImage, bucket_edge: u32) -> Tier0Scalars {
    let phash = dataset_hasher().hash_image(image).as_bytes().to_vec();

    // Measure sharpness + exposure on exactly what the trainer feeds in.
    let trainer_gray = trainer_grayscale(image, bucket_edge.max(1));
    let (shadow_clip, highlight_clip) = exposure_clip(&trainer_gray);
    Tier0Scalars {
        blur_variance: laplacian_variance(&trainer_gray),
        shadow_clip,
        highlight_clip,
        phash,
    }
}

/// One item's inputs for [`compute_readiness`]. `cached_scalars` is the validated cache the caller
/// chose to reuse (via `CachedTier0Scalars::valid_for`); when it is `None`, the image at
/// `image_path` is decoded.
pub struct ReadinessItem {
    pub item_id: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub content_hash: Option<String>,
    pub image_path: Option<PathBuf>,
    pub cached_scalars: Option<Tier0Scalars>,
    /// Checks the user dismissed for this image (sc-6534), already resolved by the caller against the
    /// current content hash (`QualityAck::effective_checks`). Threaded into the evaluation so a
    /// dismissed finding drops out of the rollups.
    pub acknowledged: Vec<QualityCheck>,
}

/// Compute the dataset readiness report (sc-6533), decoding Tier-0 scalars for any item without a
/// reusable cached value. Returns the report plus the freshly-extracted `(item_id, scalars)` pairs
/// so the caller can persist them as the content-hash + bucket-keyed cache. An item whose image
/// fails to decode gets a `Decode` warning, so it can never pass as "technically fine".
pub fn compute_readiness(
    items: &[ReadinessItem],
    bucket_edge: u32,
    min_items: u32,
    thresholds: &Tier0Thresholds,
    tier1: Option<&Tier1Evaluation>,
    aesthetic: Option<&AestheticEvaluation>,
) -> (DatasetReadinessReport, Vec<(String, Tier0Scalars)>) {
    let mut inputs = Vec::with_capacity(items.len());
    let mut extracted = Vec::new();
    let mut decode_failed = Vec::new();

    for item in items {
        let scalars = if let Some(cached) = &item.cached_scalars {
            Some(cached.clone())
        } else if let Some(path) = &item.image_path {
            match extract_tier0_scalars(path, bucket_edge) {
                Ok(scalars) => {
                    extracted.push((item.item_id.clone(), scalars.clone()));
                    Some(scalars)
                }
                Err(_) => {
                    decode_failed.push(item.item_id.clone());
                    None
                }
            }
        } else {
            None
        };
        inputs.push(ItemQualityInput {
            item_id: item.item_id.clone(),
            width: item.width,
            height: item.height,
            content_hash: item.content_hash.clone(),
            scalars,
            acknowledged: item.acknowledged.clone(),
        });
    }

    let mut evaluation = evaluate_tier0(&inputs, bucket_edge, min_items, thresholds);
    for entry in &mut evaluation.items {
        if decode_failed.contains(&entry.item_id) {
            // Injected after the ack post-pass, so a decode failure is never acknowledgeable here —
            // and the API also strips `Decode` from stored acks. An undecodable image stays a real
            // warning that drags the technical share down (it can't pass as "technically fine").
            entry.flags.push(QualityFlag {
                check: QualityCheck::Decode,
                severity: Severity::Warn,
                value: None,
                threshold: None,
                peers: Vec::new(),
                acknowledged: false,
            });
        }
    }

    let mut report = build_readiness_report(evaluation, tier1, aesthetic);
    report.distributions = build_distributions(&inputs, thresholds);
    (report, extracted)
}

/// Per-metric distributions for the Advanced view (sc-6534), built from the scalars already gathered
/// for evaluation — so the report carries them and the web needs no second fetch. `None` when no
/// item has scalars (empty or wholly-undecodable set).
fn build_distributions(
    inputs: &[ItemQualityInput],
    thresholds: &Tier0Thresholds,
) -> Option<ReadinessDistributions> {
    let scalars: Vec<&Tier0Scalars> = inputs.iter().filter_map(|i| i.scalars.as_ref()).collect();
    if scalars.is_empty() {
        return None;
    }
    Some(ReadinessDistributions {
        // Sharpness: higher is better, judged against the absolute blur floor.
        blur_variance: MetricDistribution {
            values: scalars.iter().map(|s| s.blur_variance).collect(),
            threshold: Some(thresholds.blur_floor),
            higher_is_better: true,
        },
        // Clip fractions: lower is better, judged against the exposure-clip ceiling.
        shadow_clip: MetricDistribution {
            values: scalars.iter().map(|s| s.shadow_clip).collect(),
            threshold: Some(thresholds.exposure_clip_fraction),
            higher_is_better: false,
        },
        highlight_clip: MetricDistribution {
            values: scalars.iter().map(|s| s.highlight_clip).collect(),
            threshold: Some(thresholds.exposure_clip_fraction),
            higher_is_better: false,
        },
    })
}

/// Center-crop to a square and resize to `edge`×`edge` (Lanczos3), then grayscale — mirroring the
/// trainer's `load_image_tensor` kernel so the scalars describe what training sees.
fn trainer_grayscale(image: &DynamicImage, edge: u32) -> GrayImage {
    let rgb: RgbImage = image.to_rgb8();
    let side = rgb.width().min(rgb.height()).max(1);
    let x0 = (rgb.width() - side) / 2;
    let y0 = (rgb.height() - side) / 2;
    let cropped = image::imageops::crop_imm(&rgb, x0, y0, side, side).to_image();
    let resized = image::imageops::resize(&cropped, edge, edge, FilterType::Lanczos3);
    DynamicImage::ImageRgb8(resized).into_luma8()
}

/// Variance of the (4-connected) Laplacian response — the classic focus/blur measure. Higher means
/// more high-frequency detail (sharper); a flat or out-of-focus image responds near zero.
fn laplacian_variance(gray: &GrayImage) -> f64 {
    let kernel: [i32; 9] = [0, 1, 0, 1, -4, 1, 0, 1, 0];
    let response: image::ImageBuffer<Luma<i32>, Vec<i32>> =
        imageproc::filter::filter3x3(gray, &kernel);

    let count = f64::from(response.width()) * f64::from(response.height());
    if count == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for pixel in response.pixels() {
        let value = f64::from(pixel[0]);
        sum += value;
        sum_sq += value * value;
    }
    let mean = sum / count;
    (sum_sq / count - mean * mean).max(0.0)
}

/// Fraction of pixels crushed to black and blown to white (luma histogram tails).
fn exposure_clip(gray: &GrayImage) -> (f64, f64) {
    let total = u64::from(gray.width()) * u64::from(gray.height());
    if total == 0 {
        return (0.0, 0.0);
    }
    let mut shadow = 0_u64;
    let mut highlight = 0_u64;
    for pixel in gray.pixels() {
        if pixel[0] <= SHADOW_CUTOFF {
            shadow += 1;
        }
        if pixel[0] >= HIGHLIGHT_CUTOFF {
            highlight += 1;
        }
    }
    (
        shadow as f64 / total as f64,
        highlight as f64 / total as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)))
    }

    #[test]
    fn bundled_aesthetic_predictor_loads_and_scores() {
        // The bundled LAION head parses, expects a 768-d CLIP embedding, and scores one to a finite
        // value. (A real-image sanity band is validated end-to-end with the MLX CLIP embedder.)
        let predictor = aesthetic_predictor();
        assert_eq!(
            predictor.input_dim(),
            768,
            "ViT-L/14 image_embeds are 768-d"
        );
        let embedding = vec![0.1_f32; 768];
        let score = predictor.predict(&embedding).expect("finite score");
        assert!(score.is_finite());
        // Same predictor instance is returned (parsed once).
        assert!(std::ptr::eq(predictor, aesthetic_predictor()));
    }

    /// A 1px checkerboard — maximally high-frequency, so its Laplacian variance is large.
    fn checkerboard(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            }
        }))
    }

    #[test]
    fn sharp_image_has_higher_blur_variance_than_flat() {
        let flat = scalars_from_image(&solid(64, 64, [128, 128, 128]), 64);
        let sharp = scalars_from_image(&checkerboard(64, 64), 64);
        assert!(flat.blur_variance < 1.0, "flat field responds ~0");
        assert!(
            sharp.blur_variance > flat.blur_variance,
            "checkerboard ({}) should be far sharper than flat ({})",
            sharp.blur_variance,
            flat.blur_variance
        );
    }

    #[test]
    fn black_image_reads_as_shadow_clipped() {
        let scalars = scalars_from_image(&solid(32, 32, [0, 0, 0]), 32);
        assert!(scalars.shadow_clip > 0.99);
        assert!(scalars.highlight_clip < 0.01);
    }

    #[test]
    fn white_image_reads_as_highlight_clipped() {
        let scalars = scalars_from_image(&solid(32, 32, [255, 255, 255]), 32);
        assert!(scalars.highlight_clip > 0.99);
        assert!(scalars.shadow_clip < 0.01);
    }

    #[test]
    fn midtone_image_is_not_clipped() {
        let scalars = scalars_from_image(&solid(32, 32, [128, 128, 128]), 32);
        assert!(scalars.shadow_clip < 0.01);
        assert!(scalars.highlight_clip < 0.01);
    }

    #[test]
    fn identical_images_hash_identically() {
        let a = scalars_from_image(&checkerboard(48, 48), 64);
        let b = scalars_from_image(&checkerboard(48, 48), 64);
        assert!(!a.phash.is_empty());
        assert_eq!(a.phash, b.phash);
    }

    #[test]
    fn phash_length_is_stable_across_image_sizes() {
        let small = scalars_from_image(&solid(10, 10, [1, 2, 3]), 8);
        let large = scalars_from_image(&checkerboard(100, 40), 64);
        assert_eq!(small.phash.len(), large.phash.len());
    }

    #[test]
    fn extract_from_path_decodes_and_scores() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("checker.png");
        checkerboard(32, 32).save(&path).expect("write png");
        let scalars = extract_tier0_scalars(&path, 32).expect("decode + score");
        assert!(!scalars.phash.is_empty());
        assert!(scalars.blur_variance > 1.0);
    }

    /// A mid-contrast checkerboard: sharp edges (high Laplacian variance) but no exposure clipping
    /// — a clean "good" image, unlike the 0/255 checkerboard which is fully clipped.
    fn mid_checkerboard(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([64, 64, 64])
            } else {
                Rgb([192, 192, 192])
            }
        }))
    }

    /// Top 90% black, bottom 10% mid — non-uniform, so it exercises exposure clipping *through* the
    /// Lanczos resize (a solid would be resize-invariant).
    fn mostly_black(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |_, y| {
            if y < height * 9 / 10 {
                Rgb([0, 0, 0])
            } else {
                Rgb([128, 128, 128])
            }
        }))
    }

    /// A strictly-increasing left→right grey ramp (peaks ~200 so a small `offset` never clamps).
    /// Two ramps differing only by `offset` share a gradient/dHash → Hamming 0 → near-duplicates.
    fn gradient(width: u32, height: u32, offset: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, _| {
            let base = (x * 200 / width.max(1)) as u8;
            let v = base.saturating_add(offset);
            Rgb([v, v, v])
        }))
    }

    /// The acceptance check: run *real images* through extract → evaluate and assert the resulting
    /// `QualityFlag`s. This is the only test where the worker-side metric *scale* (real
    /// `blur_variance`) meets the core thresholds (`blur_floor`) — a gross units mismatch would
    /// flag everything or nothing while every isolated test still passed. (Tuning is sc-6530 §8;
    /// this is just sanity that the floor sits between real sharp and soft values.)
    #[test]
    fn extract_then_evaluate_flags_real_images() {
        use sceneworks_core::dataset_quality::{
            evaluate_tier0, DatasetKind, ItemQualityInput, QualityCheck, Tier0Thresholds,
        };

        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let to_item = |id: &str, image: &DynamicImage, content: &str| ItemQualityInput {
            item_id: id.to_owned(),
            width: Some(image.width()),
            height: Some(image.height()),
            content_hash: Some(content.to_owned()),
            scalars: Some(scalars_from_image(image, bucket)),
            acknowledged: Vec::new(),
        };

        let sharp = mid_checkerboard(64, 64);
        let flat = solid(64, 64, [128, 128, 128]);
        let dark = mostly_black(64, 64);
        let ramp_a = gradient(64, 64, 0);
        let ramp_b = gradient(64, 64, 12);

        let items = [
            to_item("sharp", &sharp, "h_sharp"),
            to_item("flat", &flat, "h_flat"),
            to_item("dark", &dark, "h_dark"),
            to_item("ramp_a", &ramp_a, "h_ramp_a"),
            to_item("ramp_b", &ramp_b, "h_ramp_b"),
        ];
        let eval = evaluate_tier0(&items, bucket, 1, &thresholds);

        let flags_of = |id: &str| {
            &eval
                .items
                .iter()
                .find(|entry| entry.item_id == id)
                .expect("item present")
                .flags
        };
        let has = |id: &str, check: QualityCheck| flags_of(id).iter().any(|f| f.check == check);

        // Metric scale: a real sharp image clears the blur floor; a flat field does not.
        assert!(
            !has("sharp", QualityCheck::Blur),
            "a sharp image's real variance must exceed blur_floor"
        );
        assert!(
            has("flat", QualityCheck::Blur),
            "a flat field must read as soft"
        );
        // Exposure clipping survives the Lanczos resize on a non-uniform image.
        assert!(has("dark", QualityCheck::Exposure));
        // Near-duplicate: two ramps differing only in brightness share a dHash.
        let near = flags_of("ramp_a")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("ramp_a is a near-duplicate of ramp_b");
        assert!(near.peers.contains(&"ramp_b".to_owned()));
    }

    #[test]
    fn compute_readiness_extracts_evaluates_and_reports() {
        use sceneworks_core::dataset_quality::{
            DatasetKind, QualityCheck, ReadinessGate, Tier0Thresholds,
        };

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);

        let sharp_path = dir.path().join("sharp.png");
        mid_checkerboard(64, 64)
            .save(&sharp_path)
            .expect("write sharp");
        let flat_path = dir.path().join("flat.png");
        solid(64, 64, [128, 128, 128])
            .save(&flat_path)
            .expect("write flat");

        let items = vec![
            ReadinessItem {
                item_id: "sharp".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_sharp".to_owned()),
                image_path: Some(sharp_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "flat".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_flat".to_owned()),
                image_path: Some(flat_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, extracted) = compute_readiness(&items, bucket, 1, &thresholds, None, None);

        // Both decoded fresh (returned for the caller to cache); flat reads soft → NeedsAttention.
        assert_eq!(extracted.len(), 2);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
        let flat = report
            .items
            .iter()
            .find(|i| i.item_id == "flat")
            .expect("flat");
        assert!(flat.flags.iter().any(|f| f.check == QualityCheck::Blur));
    }

    #[test]
    fn compute_readiness_reuses_cache_and_flags_decode_failure() {
        use sceneworks_core::dataset_quality::{DatasetKind, QualityCheck, Tier0Thresholds};

        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let cached = scalars_from_image(&mid_checkerboard(64, 64), bucket);

        let items = vec![
            ReadinessItem {
                item_id: "cached".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h".to_owned()),
                image_path: None, // no path → must use the cache
                cached_scalars: Some(cached),
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "broken".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_broken".to_owned()),
                image_path: Some(PathBuf::from("/no/such/file.png")),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, extracted) = compute_readiness(&items, bucket, 1, &thresholds, None, None);

        // The cached item is not re-extracted; the broken path yields nothing to cache.
        assert!(extracted.is_empty());
        let broken = report
            .items
            .iter()
            .find(|i| i.item_id == "broken")
            .expect("broken");
        assert!(broken.flags.iter().any(|f| f.check == QualityCheck::Decode));
    }

    #[test]
    fn acknowledging_a_finding_drops_it_from_the_rollup() {
        use sceneworks_core::dataset_quality::{
            DatasetKind, QualityCheck, ReadinessGate, Tier0Thresholds,
        };

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let flat_path = dir.path().join("flat.png");
        solid(64, 64, [128, 128, 128])
            .save(&flat_path)
            .expect("write flat");

        let items = vec![ReadinessItem {
            item_id: "flat".to_owned(),
            width: Some(64),
            height: Some(64),
            content_hash: Some("h_flat".to_owned()),
            image_path: Some(flat_path),
            cached_scalars: None,
            acknowledged: vec![QualityCheck::Blur],
        }];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None);
        // Blur is the only finding and the user dismissed it → Ready, badge clean…
        assert_eq!(report.gate, ReadinessGate::Ready);
        assert_eq!(report.items[0].severity, None);
        // …but the flag is still in the payload, marked acknowledged for the struck-through display.
        let blur = report.items[0]
            .flags
            .iter()
            .find(|f| f.check == QualityCheck::Blur)
            .expect("blur kept");
        assert!(blur.acknowledged);
    }

    #[test]
    fn decode_failure_cannot_be_acknowledged() {
        use sceneworks_core::dataset_quality::{
            DatasetKind, QualityCheck, ReadinessGate, Tier0Thresholds,
        };

        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let items = vec![ReadinessItem {
            item_id: "broken".to_owned(),
            width: Some(64),
            height: Some(64),
            content_hash: Some("h_broken".to_owned()),
            image_path: Some(PathBuf::from("/no/such/file.png")),
            cached_scalars: None,
            acknowledged: vec![QualityCheck::Decode], // even if asked, a decode failure stands
        }];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None);
        let decode = report.items[0]
            .flags
            .iter()
            .find(|f| f.check == QualityCheck::Decode)
            .expect("decode flag");
        assert!(!decode.acknowledged);
        assert_ne!(report.gate, ReadinessGate::Ready);
    }

    #[test]
    fn compute_readiness_emits_per_metric_distributions() {
        use sceneworks_core::dataset_quality::{DatasetKind, Tier0Thresholds};

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let a_path = dir.path().join("a.png");
        mid_checkerboard(64, 64).save(&a_path).expect("write a");
        let b_path = dir.path().join("b.png");
        solid(64, 64, [128, 128, 128])
            .save(&b_path)
            .expect("write b");

        let items = vec![
            ReadinessItem {
                item_id: "a".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("ha".to_owned()),
                image_path: Some(a_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "b".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("hb".to_owned()),
                image_path: Some(b_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None);
        let dist = report.distributions.expect("distributions present");
        // One value per decodable item, oriented + thresholded for the chart.
        assert_eq!(dist.blur_variance.values.len(), 2);
        assert!(dist.blur_variance.higher_is_better);
        assert_eq!(dist.blur_variance.threshold, Some(thresholds.blur_floor));
        assert!(!dist.shadow_clip.higher_is_better);
        assert_eq!(dist.shadow_clip.values.len(), 2);
    }
}
