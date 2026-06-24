# sc-6541 — Closed-loop: correlating dataset signals with trained-LoRA quality

**Research protocol for epic 6529.** sc-6541 is the empirical endgame: "usefulness" is
ultimately decided by the trained LoRA, so the only ground truth is to *train one and look*.
This document fixes the experiment **before** any GPU time is spent — it is the artifact that
makes the story a bounded chore (2 pts, time-boxed) rather than an open-ended experiment.

It feeds recommended threshold updates back into the design spike,
[`docs/sc-6530/dataset-doctor-metrics.md`](../sc-6530/dataset-doctor-metrics.md) (§3 defaults,
§8 calibration). Read that first — every threshold this study touches is defined there.

> **Honesty caveat, stated up front.** With one LoRA per dataset condition the sample size is
> N≈3 (pilot) to N≈5 (full matrix). That is far too small to *learn* thresholds by regression.
> This is a **pilot calibration**: it estimates the *direction and separability* of each signal
> against trained-LoRA quality, and recommends a threshold change **only** when the effect is
> large, monotone, and matches the metric's intended mechanism. The deliverable language is
> "supported adjustment" / "no evidence to change" — never "learned optimum." Anything stronger
> would manufacture false precision in exactly the place the epic warned against.

---

## 1. The experiment as a regression

We treat each dataset condition as one observation in a tiny designed experiment:

- **X — the predictor:** the Dataset Doctor *readiness signal vector* computed over the
  dataset, today, by the shipped pipeline (sub-scores + per-check raw metrics). This is the
  thing we ship and want to validate.
- **Y — the response:** the *output quality* of a LoRA trained on that dataset, measured on a
  fixed prompt grid against a held-out reference of the true subject. Multi-axis on purpose
  (§4) — a single identity number would mislead.

The study asks one question per signal: **does X move Y in the direction the spike assumes,
and far enough to separate a clean dataset from a deliberately-degraded one?** Signals that do
are validated; signals that don't are flagged for pruning (sc-6541's "prune checks that don't
predict outcomes").

---

## 2. Scope — pilot loop now, full matrix documented

Training is **native Rust/MLX** — `gen_core::load_trainer("z_image_lora",
&LoadSpec::Dir(base)).train(req, …)` (the `ZImageTurboTrainer` in `mlx-gen-z-image`), driven
from an `#[ignore]` worker test exactly like the eval harness; **no Python/torch path is
involved** (that is the legacy off-Mac fallback). Generation for the eval grid is likewise the
native Z-Image generator with the trained LoRA as an adapter. Production character presets are
**3000 steps** (`crates/sceneworks-core/src/training.rs`); on Apple-Silicon MLX that is
multi-hour per LoRA, so the study uses a **reduced step budget** (locked by the calibration run,
§2.1) and a **reduced pilot** matrix — the right cut for a 2-pt time box:

| Condition | Axis exercised | Pilot? | Rationale |
|---|---|---|---|
| **Clean** (Basim train pool) | baseline | ✅ | Must land "Ready" with high Y — the control. |
| **Blurry** (simulated degradation) | technical-quality (blur floor) | ✅ | Tests whether the blur signal predicts *soft outputs*. |
| **Low-diversity / near-dup-heavy** | training-usefulness (diversity, near-dup) | ✅ | Tests whether the diversity/near-dup signal predicts *narrow output variety*. Non-tautological with identity. Built from **near-dups (slight perturbation), not exact copies** — exact copies trip the SHA exact-dup path, not the near-dup/diversity signal this variant must exercise. |
| **Wrong-person** (identity contamination) | identity (`same_person_cosine`) | ⏸ follow-up | Re-tests the identity check almost tautologically; defer to the full matrix. |
| **Low-diversity (pose/angle collapse, distinct from near-dup)** | coverage | ⏸ follow-up | Second diversity flavor; only worth it once the instrument is proven. |

**Gate to the full matrix:** only commit training time to the two follow-up conditions
**after** the pilot proves Y is *separable* across clean/blurry/low-diversity. If the
measurement instrument can't tell three obviously-different datasets apart, more conditions add
cost without signal. This gate is itself a finding worth writing down.

The two follow-up conditions are documented as **planned follow-up, not gaps in acceptance** —
sc-6541's acceptance is "a written analysis + recommended threshold updates," which the pilot
satisfies.

### 2.1 Stage 0 — calibration LoRA + no-LoRA baseline, before the pilot matrix

The dominant risk is **underfit confounding**: a reduced-step LoRA can underfit so badly that no
variant learns the subject, collapsing all Y to the floor and yielding a null study. This is
settled *empirically, not by argument*. Before training the pilot variants:

1. **Train one clean calibration LoRA** — a ~20-image subset of the train pool at a candidate
   config. Eval it. **Gate (a):** it must produce a coherent person with identity fidelity
   *meaningfully above the no-LoRA baseline*. **Gate (b):** read the step budget that achieves
   this and **lock it for every variant**. If the clean LoRA can't separate from baseline, no
   degraded variant will — learned for one run's cost, not four.
2. **The no-LoRA baseline is a required control, not optional.** ArcFace was trained on real
   photographs; Z-Image-Turbo generations carry a domain gap, so even a perfect LoRA will not
   reach the **~0.73** photo-to-photo band measured in [results §0](./results.md). Generate the
   same prompt grid from the **base model with no adapter** and record its identity fidelity to
   the reference centroid. That floor is what every LoRA's Y must beat — without it an identity
   cosine of "0.5" is uninterpretable.

Subset size (~20) is a *starting* config, not a locked constant — the calibration run measures
whether it learns identity at the reduced budget; adjust and re-run if not. Whatever is chosen,
hold it fixed across all variants (§8).

---

## 3. X — the dataset signal vector

Computed by the shipped Dataset Doctor pipeline over each variant (per
[spike §4](../sc-6530/dataset-doctor-metrics.md)). Recorded per dataset:

**Sub-scores** (interpretable ratios, not a blended number):
- `technical` — share of items with no quality flag (resolution/crop/blur/exposure/dup)
- `diversity` — CLIP-embedding coverage spread
- `identity` — share consistent with the set's face centroid (person/character)
- `alignment` — caption↔image agreement (first-sentence CLIP score, sc-6537)

**Raw per-check distributions** (the numbers behind the flags — what calibration actually
tunes against): Laplacian-variance distribution + median; pHash Hamming histogram; CLIP
near-dup cosine clusters; CLIP diversity spread; per-image caption-alignment scores; ArcFace
cosine-to-centroid distribution; face-fraction distribution; resolution-vs-bucket.

**The discrete gate** (Ready / Needs-attention / Blocked) each variant resolves to.

These come straight from the existing analysis passes — the CLIP `dataset_analysis` job
(`crates/sceneworks-worker/src/dataset_analysis_jobs.rs`) and the `dataset_face_analysis` job
(`face_analysis_jobs.rs`) — plus the pure rollup in
`crates/sceneworks-core/src/dataset_quality.rs`. No new measurement code on the X side; the
`calibrate_thresholds` / `sweep_caption_alignment` `#[ignore]` harnesses already dump most of
these distributions and are the template.

---

## 4. Y — the trained-LoRA output quality vector

Measured on the fixed prompt grid (§7), aggregated over 8–16 generated images per LoRA so each
metric estimates a *distribution*, not a single point. **Multi-axis by design** — Codex's key
correction: identity alone would let a blurry-but-faithful LoRA look "good."

| Y metric | Method | What it catches | Reuses |
|---|---|---|---|
| **Face-detect rate** | SCRFD over each output; fraction with ≥1 face | LoRA that stopped producing a coherent person at all | `*-gen-face` detect |
| **Identity fidelity** | mean ArcFace cosine of each output's largest face to the **held-out reference centroid** | does it look like the actual subject | `*-gen-face` embed + the cosine math in `dataset_quality.rs` |
| **Output sharpness** | Laplacian variance of each output | blurry training → soft generations (the blur-axis confound) | worker blur path |
| **Prompt adherence** | CLIP image↔text cosine, output vs its prompt | low-diversity/near-dup → ignores the prompt, collapses to memorized frames | the CLIP ViT-L/14 tower (sc-6535) |
| **Output spread** | mean pairwise CLIP distance across the grid | mode collapse / no variety | CLIP embeddings |

**Explicitly out of scope:** aesthetic score. The spike scopes aesthetic to `style` datasets
only (`training.rs`, the style-kind gate); a person LoRA must not be graded on it.

**Reference:** the ArcFace target is the **held-out reference pool** (§5), never the training
images — otherwise identity fidelity rewards memorization, not generalization.

**Baseline floor:** every Y metric is read *relative to the no-LoRA baseline* (§2.1) — the same
grid generated from the base model with no adapter. The domain gap between ArcFace (real photos)
and Z-Image generations means raw cosines are uninterpretable in absolute terms; the question is
always "how far above the no-LoRA floor."

**Seam needed:** the identity scorer in `dataset_quality.rs` currently consumes *training-set*
face records. The harness feeds it **generated-image** embeddings instead. That is the one
genuinely new code path (§6); it reuses the embedding + cosine logic, not the dataset-scoped
entry point.

---

## 5. The Basim split — done once, before any variant exists

`~/Datasets/Basim` is 64 real photos of one person. **Reference leakage is the silent
killer**: if a reference image also trained the LoRA, identity fidelity measures memorization.

So, **once**, deterministically (sorted filename, fixed stride — no randomness, reproducible):

- **Training pool** — **51 images**. *Every* variant is derived from this pool only.
- **Held-out reference pool** — **13 images** (stratified across both capture ranges). Never
  enters any training set. The sole ArcFace target for identity fidelity.

This split is **frozen** in [`basim-split.md`](./basim-split.md) (the exact filename lists +
the rule), done before variant construction so the whole study is reproducible from the doc.
Validated in [results §0](./results.md): all 13 reference images detect a face (no fallback
needed) and cluster at cosine ≈ 0.73.

---

## 6. The eval harness — standalone, reproducible, JSON-emitting

A **standalone `#[ignore]` research harness** in `crates/sceneworks-worker` (the established
calibration shape — `CALIB_DIR`, `RUST_TEST_THREADS=1`, `--ignored --nocapture`), **not** the
job/API/web path. The product path adds queue/asset/UI noise that contributes nothing to
evaluation logic, and the "Test Character" UI (`apps/web/.../characterPanels.jsx`) is a manual
review surface with no scoring. Rationale recorded so a later reader doesn't "productize" it.

**Inputs:** a trained LoRA path, the held-out reference-pool path, the fixed prompt grid, the
fixed seed list, the frozen inference config.

**Per output image** it computes the five Y metrics (§4) and emits **JSON** (per-image rows +
per-LoRA aggregates), so the X→Y join and the write-up are pure post-processing, never a
re-run. Schema sketch:

```json
{
  "lora": "basim-clean",
  "config": { "steps": 600, "lr": 5e-5, "rank": 8, "weight": 1.0, "resolution": 768, "trigger": "..." },
  "reference_pool": ["IMG_0461.JPG", "..."],
  "outputs": [
    { "prompt_id": "p03", "seed": 42, "face_detected": true,
      "identity_cosine": 0.41, "sharpness": 182.4, "prompt_adherence": 0.27, "image": "..." }
  ],
  "aggregates": { "face_detect_rate": 1.0, "identity_cosine_mean": 0.39,
                  "identity_cosine_std": 0.06, "sharpness_mean": 175.2,
                  "prompt_adherence_mean": 0.26, "output_spread": 0.31 }
}
```

The same harness runs against every LoRA unchanged — only the LoRA path and label differ.

---

## 7. The fixed prompt grid

A small, fixed, kind-appropriate grid that probes the axes Y measures — held **identical**
across every LoRA (it is part of the controlled config, §8). Person grid, ~8 prompts spanning:
a plain identity portrait (identity floor), varied settings/lighting/wardrobe (prompt
adherence + spread), a couple of compositions the training set did *not* contain (diversity
generalization). Derived from the trigger word the way the trainer's own `generate_samples`
does (`samplePromptsFromTrigger`, `apps/web/.../trainingConfig.js`) so it matches how the
product would sample. Each prompt × a fixed seed list → the 8–16 outputs per LoRA.

The exact grid + seed list are frozen in an appendix of the results doc when the run executes.

---

## 8. Controlled constants — the only variable is the dataset

The experiment is only valid if *everything but the training data* is held fixed. Locked
across all conditions (Codex blind-spot #1 + #3):

- **Training:** identical step count, LR, rank/alpha, optimizer, resolution, trigger word,
  save cadence. **Reduced steps** (target ~400–800 vs the 3000-step preset) for tractability —
  a deliberate, documented methodology choice; underfit is acceptable *as long as it is equal
  across conditions* (an equal underfit cancels out of the X→Y comparison).
- **Dataset shape held constant across variants:** same item count, same captions, same trigger,
  same resolution. Degradation changes only *pixels/selection*, never the manifest shape — else
  X is confounded with count/caption effects.
- **Inference:** identical sampler, steps, guidance, LoRA weight, resolution, and the §7 prompt
  grid + seed list.
- **Face-centered crop preprocessing (fixed across all variants):** every training image is
  SCRFD-face-centered-cropped to a padded square *before* training (`face_center_crop_dir`).
  The trainer's own `center_crop_square` drops the face on tall full-body shots (verified — see
  [results §2](./results.md)), which would decouple a face degradation from what the LoRA learns.
  Pre-cropping keeps the face in-frame and dominant for **every** variant, so blur/low-diversity
  land on the face the LoRA actually trains on. (This is the same idea as Dataset Doctor's own
  `crop_loss`/`subject_prominence` checks, applied upstream.)

If a constant has to move (e.g. count, because near-dup construction removes images), it is
called out explicitly as a known confound in the write-up.

---

## 9. Statistical framing — what N≈3–5 can and cannot say

- **Unit of observation is the output image, not the LoRA.** 8–16 outputs/LoRA → each Y metric
  is a distribution (mean + std), so we can speak to *within-LoRA* spread and *between-LoRA*
  separation honestly even at N=3 LoRAs.
- **No regression-learned thresholds.** We report, per signal: the *direction* of the X→Y
  relationship, whether clean vs degraded Y distributions *separate* (non-overlapping at ~1 std),
  and whether that matches the metric's intended mechanism.
- **Threshold recommendations** are made only when all three hold, and are phrased as a bounded
  adjustment with the evidence, e.g. *"blur floor: outputs from the blurred set were
  X× softer with no identity loss — supported: keep the absolute floor, it predicts a real
  output effect"* or *"no evidence to change."*
- **Seed variance (optional, if compute remains):** re-train the clean + worst-degraded
  condition once more with a different seed to sanity-check that between-condition gaps exceed
  between-seed noise. Not run by default inside the time box.

---

## 10. Threshold map — falsifiable prediction per condition

Each condition makes a prediction; the study confirms or refutes it.

| Condition | Spike §3 threshold(s) in play | Prediction (the falsifiable claim) |
|---|---|---|
| Clean | — (control) | Gate = Ready; highest identity fidelity, sharpness, spread. |
| Blurry | Blur floor (abs + 0.5× median) | `technical` sub-score drops; **output sharpness drops**; identity may survive → validates blur as an *output-quality* predictor distinct from identity. |
| Low-diversity / near-dup | Near-dup pHash ≤6 / CLIP ≥0.95; `diversity` sub-score | `diversity` drops, near-dup flags fire; **output spread + prompt adherence drop**; identity may stay high → validates diversity as a *variety* predictor. |
| *(follow-up)* Wrong-person | `same_person_cosine` 0.40 | `identity` sub-score drops; identity fidelity drops — expected, near-tautological. |
| *(follow-up)* Subject-prominence | `min_face_fraction` 0.02 | face-fraction flags fire; face-detect rate / identity drop. |

Refutation is a *result*: a signal whose degradation does **not** move Y is a prune candidate.

---

## 11. Deliverables (maps to sc-6541 acceptance)

1. **This protocol** (`docs/sc-6541/closed-loop-protocol.md`) — the design, frozen before runs.
2. **The eval harness** — the standalone `#[ignore]` worker harness (§6), reusable beyond this
   study.
3. **The X and Y matrices** — JSON dumps from the analysis passes and the harness.
4. **The written analysis** (`docs/sc-6541/results.md`, authored when runs complete) — per-signal
   direction + separability + threshold support/no-support, the separability gate verdict, and
   explicit follow-up call-outs (wrong-person, second diversity flavor).
5. **Spike feedback** — pilot-supported edits applied to
   [`docs/sc-6530`](../sc-6530/dataset-doctor-metrics.md) §3/§8, in the "supported adjustment"
   register.

**Optional stretch (per the story):** per-image influence estimate (which training images moved
the result) — out of the time box unless the pilot finishes with compute to spare.

---

## 12. Execution checklist

- [x] Freeze this protocol.
- [x] Split Basim → train pool + held-out reference pool ([`basim-split.md`](./basim-split.md)).
- [x] Build + unit-test the standalone eval harness; **validate on real weights** ([results §0](./results.md)).
- [ ] Stage the Z-Image-Turbo base snapshot; build the native-Rust train + generate `#[ignore]` driver.
- [ ] **[Stage 0 — calibration]** Generate the **no-LoRA baseline** grid; train **one clean
      calibration LoRA** (~20-img subset); eval both. **Gate:** clean LoRA's identity fidelity
      separates above the baseline floor → lock the step budget. (Else adjust subset/steps, re-run.)
- [ ] Construct the 3 variant datasets from the train pool (clean / blurry / low-diversity near-dups).
- [ ] Run the readiness analysis passes over all 3 → **X matrix**.
- [ ] Train 3 LoRAs at the **locked** identical config; eval each via the harness → **Y matrix**.
- [ ] Write `results.md`; apply supported edits to spike §3/§8; mark follow-ups.
- [ ] If Y is separable and compute remains: run the two follow-up conditions.

---

## 13. Risks & how they are controlled

| Risk (Codex) | Control |
|---|---|
| **Train-step confounding** — underfit swamps dataset effects | Hold steps/LR/weight/inference identical; equal underfit cancels. **And** settle the budget empirically with the Stage-0 calibration run before committing the variants (§2.1). |
| **Domain-gap floor** — ArcFace cosines uninterpretable in absolute terms | The **no-LoRA baseline** (§2.1) is a required control; every Y is read relative to it. |
| **Reference leakage** — identity rewards memorization | Held-out reference pool split off *before* variants; never trained on. |
| **Variant construction drift** — X confounded with count/caption | Manifest shape (count/captions/trigger/resolution) fixed; only pixels/selection change; any forced change called out. |
| **Over-investment** — full 5-variant matrix before the instrument is proven | Pilot-first; full matrix gated on demonstrated Y separability. |
| **False precision** — "learned thresholds" at N≈5 | Pilot-calibration register; recommend only large/monotone/mechanism-matched effects. |
