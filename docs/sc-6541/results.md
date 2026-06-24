# sc-6541 — Closed-loop results

Companion to [`closed-loop-protocol.md`](./closed-loop-protocol.md). Results are appended as the
pipeline executes. The final analysis (per-signal correlation + threshold recommendations) lands
in §3 once the LoRAs are trained and evaluated.

---

## §0. Instrument validation (2026-06-24) — the eval harness works on real weights

Before spending any GPU/training time, the scoring harness
(`crates/sceneworks-worker/src/lora_eval_harness.rs`, test-only) was validated end-to-end on
real weights in **self-consistency mode** (score the held-out reference pool against its own
ArcFace centroid). This proves every Y axis is wired correctly before the study depends on it.

Run:
```sh
REF_DIR=~/Datasets/lora-eval/basim-reference \
RUST_TEST_THREADS=1 cargo test -p sceneworks-worker --lib eval_lora_outputs -- --ignored --nocapture
```
(weights: `openai/clip-vit-large-patch14` HF snapshot + `SceneWorks/instantid-mlx` SCRFD+ArcFace
bundle, auto-pulled to the app-managed cache; run on Apple-Silicon MLX.)

**Reference pool (13 images, the held-out split):**

| Aggregate | Value | Reading |
|---|---|---|
| `face_detect_rate` | **1.0** | Every reference image yields a face — incl. the lo-res PNGs, so the JPG-only fallback from `basim-split.md` is **not** needed. |
| `identity_cosine_mean` | **0.733** (std 0.058) | Same-person ArcFace cosines to the centroid sit in **0.66–0.82** — a tight cluster, far above the `same_person_cosine` 0.40 threshold. Face + centroid + cosine math confirmed. |
| `sharpness_mean` | 292 (std 313) | Wide by design — hi-res JPGs (~100–700) ≫ lo-res PNGs (one at 7.7). The `blur_variance` seam works. |
| `output_spread` | 0.179 | Mean pairwise CLIP distance across the pool — one person, varied shots. CLIP image tower works. |
| `prompt_adherence` | n/a | No prompts supplied in smoke mode (correct). |

**Calibration value already banked:** the same-person operating point for this subject is
**cosine ≈ 0.73 ± 0.06** (photo-to-photo). This is the yardstick for reading LoRA identity
fidelity: a faithful person LoRA's generations should land in a comparable band (likely a bit
lower, generated ≠ photograph); a wrong-person/degraded LoRA should fall well below it. It also
sanity-checks that `same_person_cosine` 0.40 is a *lenient* floor for this subject — consistent
with the spike's "bias to warn" stance, to be revisited per-subject in the full analysis.

**Pure-function coverage:** the scoring core (normalize/cosine/centroid/mean/std/spread/aggregate)
has synthetic-vector unit tests; `fmt --check` + `clippy -D warnings` clean on the worker.

---

## §0.5. Pipeline live + no-LoRA baseline floor (2026-06-24)

The full native-Rust loop runs end-to-end on this Mac (M5 Max, 64 GB): base model
`Tongyi-MAI/Z-Image-Turbo` staged (31 GB), generation via `mlx_gen_z_image::load(spec).generate`
(~23 s/image at 1024²), training via `gen_core::load_trainer("z_image_turbo", …).train`, scoring
via the eval harness. (Generation uses the direct provider `load`, not the `gen_core` registry —
the worker's test build links a stub `z_image_turbo` registration that makes the registry `load`
panic on a duplicate id; the trainer registry is unaffected.)

**No-LoRA baseline (8 prompts × seed 42, the §2.1 control floor):**

| Y aggregate | Value | Reading |
|---|---|---|
| `face_detect_rate` | 0.875 | 7/8 base generations yield a face. |
| **`identity_cosine_mean`** | **−0.058** (±0.024) | **The floor.** A generic base-model person has ≈0 cosine to the Basim subject — so "looks like the subject" must beat ~0, *not* the 0.73 photo-band (the ArcFace domain gap, exactly why this control is required). |
| `sharpness_mean` | 425 (±459) | — |
| `prompt_adherence_mean` | 0.244 | base-model prompt following. |
| `output_spread` | 0.382 | high — unconstrained base outputs are varied. |

Visual check: the base-model portrait is a coherent, photorealistic *generic* man (not the
subject) — confirming generation quality and that identity ≈ 0 is the right floor.

## §1. The variant study — X signals → Y outcomes (2026-06-24)

Three LoRAs at the **locked config** (clean / blurry / low-diversity), each on 21 face-centered
crops; evaluated on the 8-prompt grid at **2 seeds (42, 1234) = 16 paired outputs**. No-LoRA
baseline = the floor.

**X — dataset signals (training set, the shipped Dataset Doctor metrics):**

| variant | `blur_variance` min/med/mean | `clip_diversity` | near-dup pairs (CLIP ≥ 0.95) |
|---|---|---|---|
| **clean** | 8.8 / 203 / 347 | 0.172 | 2 |
| **blurry** | 2.0 / **2.7** / 2.9 | 0.174 | 2 |
| **low-div** | 204 / 562 / 528 | **0.097** | **45** |

Each degradation moves **only its own signal**: blur collapses `blur_variance` ~**75×**
(203 → 2.7) while diversity/near-dup are untouched; low-diversity drops diversity **44%**
(0.172 → 0.097) and explodes near-dup (2 → 45) while `blur_variance` stays high. Clean isolation —
the X side behaves exactly as the spike's check catalog intends.

**Y — LoRA output quality (8 prompts × 2 seeds = 16 outputs):**

| variant | identity (mean ± std) | prompt-adh | `output_spread`\* | `same_prompt_spread`† |
|---|---|---|---|---|
| baseline (floor) | −0.057 ± 0.038 | 0.244 | 0.364 | 0.090 |
| **clean** | **+0.284 ± 0.14** | 0.256 | 0.416 | **0.172** |
| **blurry** | **+0.178 ± 0.09** | 0.255 | 0.414 | 0.154 |
| **low-div** | **+0.259 ± 0.16** | 0.236 | 0.398 | **0.130** |

\* `output_spread` is across *different* prompts → dominated by prompt variance, not a clean
variety signal. † `same_prompt_spread` (within-prompt variety across the 2 seeds) **is** the
mode-collapse measure — low-div collapses it most (0.172 → 0.130). _(Don't compare these to the
baseline: the no-LoRA turbo model is seed-stable for a fixed prompt for unrelated reasons; the
apples-to-apples comparison is clean-vs-degraded LoRA.)_

**Paired clean-vs-degraded** (per image = prompt + seed, N = 16; +meanΔ = clean better; two-sided
sign-test p; identity n = 14 because ~2/16 generations had no detected face):

| | identity | prompt-adherence |
|---|---|---|
| clean vs **blurry** | meanΔ +0.106, **11/14, p = 0.057** | ≈0 (9/16, p = 0.80) |
| clean vs **low-div** | +0.024, 8/14, p = 0.79 (null) | +0.019, **12/16, p = 0.077** |

**Result (N = 16 firms the directions to *marginal* significance):** blur → identity/fidelity loss
(p ≈ 0.06); low-diversity → prompt-adherence loss (p ≈ 0.08) **and** within-prompt variety loss
(`same_prompt_spread` 0.172 → 0.130), with identity **unchanged** (p = 0.79). Two degradations, two
*different* output-harm axes. (Output *sharpness* stayed too content-dominated, std ~1200, to use.)

## §2. LoRA output quality (Y) — calibration gate (2026-06-24)

**Clean calibration LoRA** (21 imgs, rank 16, res 512, 400 steps, bf16, scale 0.8 at gen; ~7 min
train, ~1 s/step):

| | `identity_cosine_mean` | `face_detect` | `prompt_adherence` | `output_spread` |
|---|---|---|---|---|
| No-LoRA baseline (floor) | −0.058 (±0.024) | 0.875 | 0.244 | 0.382 |
| **Clean LoRA** | **+0.156 (±0.090)** | 0.875 | 0.253 | 0.453 |

**Gate: PASS (separation), but thin headroom.** Identity moved from ≈0 to +0.156 — cleanly above
the floor (even the weakest LoRA sample ≈0.07 > the best baseline sample ≈−0.03), so the
instrument detects a learned identity. But +0.21 of dynamic range is thin for resolving
*degraded*-variant drops against the ±0.09 per-sample spread.

**Two findings that bear on the study design:**
1. **The "Basim" set is 3D game-character renders, not real photos** (Assassin's Creed's Basim) —
   mostly full-body action poses. ArcFace still detects + clusters the rendered faces (ref
   self-consistency 0.73), so identity scoring is valid, but it's a stylized-render subject.
2. **Suspected face-crop loss:** the trainer center-crops each image to a square before the
   512 resize; for a *tall full-body* render the face sits at the top and may be cropped out,
   starving face-identity learning — a plausible cause of the weak +0.156 (alongside the
   render→generation domain gap). _This is itself a subject-prominence signal worth noting._

**Crop confound — confirmed and fixed (the load-bearing decision, settled with zero training).**
Of the 21 cal images, 4 are tall (h/w > 1.3); center-crop-square on `IMG_0401` (1132×2101) yields
**torso-and-legs, no face** — the trainer would learn zero facial identity from it, and any
blur/degradation on that (absent) face couldn't affect the LoRA. The ~square images (e.g.
`IMG_0408`) crop to a face-filling close-up, so the majority are fine. Per the advisor this is the
fork-decider: a face-crop fix is needed (not more steps), or the variant degradations decouple
from the outcome.

**Fix applied:** a new `face_center_crop_dir` step (SCRFD-detect the largest face → padded
square centered on it → resize) preprocesses the training set. On the 21 cal images: **0 no-face
fallbacks** — SCRFD finds a face in every *full* image (the square crop, not detection, was what
lost them), and `IMG_0401` now yields a proper head-and-shoulders face crop. The clean LoRA is
being **retrained on the face-cropped set** at the same config (rank 16 / 512 / 400 / bf16); the
3 variants will derive from these face-cropped images so faces are always in-frame and
degradations land on them.

> Subject note (user-confirmed): "Basim" is Assassin's-Creed game-character renders, not photos —
> kept (paired analysis + face-crops handle it); the render→generation domain gap attenuates
> absolute identity but not the paired degradation comparison. The naive `center_crop_square` is
> itself a finding: it's exactly what Dataset Doctor's `crop_loss` (smart-crop) +
> `subject_prominence` checks warn about — a face-aware crop / aspect-bucketing in the trainer is
> a worthwhile follow-up.

**Calibration gate — PASSED, config LOCKED.** Retraining on the face-cropped set lifted identity:

| LoRA | `identity_cosine_mean` | vs floor (−0.058) |
|---|---|---|
| full-frame crops | 0.156 (±0.090) | +0.21 |
| **face-centered crops** | **0.285 (±0.16)** | **+0.34** |

Face-cropping ~doubled identity and the portrait now reads as the bearded character, not a
generic man. +0.34 of range above the floor is comfortable headroom for *paired* degradation
detection. Per the time-box, not chasing higher absolute identity.

**Locked config (held fixed for all 3 variants):** SCRFD face-centered crops (pad 2.2 → 768²) →
rank 16, res 512, 400 steps, bf16, trigger `sks man`; generate at 1024² scale 0.8, seed 42 (the
variant runs add seeds for paired power).

## §3. Analysis & threshold recommendations

### What's solid vs what's directional

Two tiers, kept deliberately apart (the Y effects are underpowered — see "Honest scope"):

**Solid:**
- **X-side isolation is decisive.** Each degradation moved *only* its own signal (blur:
  `blur_variance` 203 → 2.7, diversity/near-dup untouched; low-div: diversity 0.172 → 0.097 +
  near-dup 2 → 45, blur untouched). The shipped checks cleanly separate known-bad from clean.
- **The center-crop confound — the strongest concrete finding.** The trainer's blind
  `center_crop_square` dropped faces on tall full-body inputs; face-centering *doubled* learned
  identity (0.156 → 0.285). A mechanism-level confirmation that `crop_loss` + `subject_prominence`
  flag real training harm.
- **The absolute blur floor is load-bearing (design fact, N-independent):** the blur set is
  *uniformly* soft, so relative-to-median passes it — only the absolute floor catches it.

**Marginal (N = 16, 2 seeds — borderline significant, p ≈ 0.06–0.08; not p < 0.05, don't say "proves"):**
- **Blur → identity/fidelity loss.** 0.284 → 0.178 (paired clean-wins **11/14, sign-test p = 0.057**;
  meanΔ +0.106). The 2nd seed moved this from p ≈ 0.23 (n = 7) to borderline. A technical-quality
  degradation hits the fidelity axis.
- **Low-diversity → variety + adherence loss, NOT identity.** Identity unchanged (8/14, p = 0.79 —
  correct, same subject). Prompt-adherence down (**12/16, p = 0.077**). And the proper variety
  metric now sees it: `same_prompt_spread` (within-prompt across seeds) collapses **0.172 → 0.130**,
  most of any LoRA — i.e. the low-diversity LoRA gives *more similar* outputs for the same prompt
  (mode collapse). Directional (8 prompt-pairs), but it's the effect `output_spread` structurally
  couldn't see.

So the two-axis framing is *supported at marginal significance* (blur leans on fidelity, low-div
leaves identity alone and hits variety/adherence) — borderline (p ≈ 0.06–0.08), not p < 0.05.

### Threshold recommendations (pilot — separation is on the *signal* side; the output-quality link is directional)

1. **Blur — keep the dual floor; the *absolute* floor is load-bearing (confirms spike §2).**
   The blur set is *uniformly* soft (every image ~2.7), so a relative-to-median rule passes it
   (nothing deviates from the median) — exactly the failure the spike's §2 warned about. Only the
   **absolute** floor catches a uniformly-soft set. This is a *design* fact (N-independent); the
   added pilot signal — that the blurred LoRA *also* read lower identity — is borderline (11/14,
   p ≈ 0.057 at N = 16), not yet p < 0.05. The clean set's softest image is 8.8 and median 203; the blur set
   sits at ~3 — an absolute floor in the **~10–50** band separates them *here*, but `blur_variance`
   is resolution/content/subject-dependent, so **calibrate the constant across multiple subjects
   before fixing it** (do not hard-code 10–50 from one render set).
2. **Diversity floor (non-style 0.12) — no evidence to change, now with a supporting output link.**
   Clean 0.172 passes, low-div 0.097 flags, 0.12 sits cleanly between. The 2-seed run *does* show
   an output consequence: the low-div LoRA's within-prompt variety (`same_prompt_spread`) collapsed
   0.172 → 0.130 and prompt-adherence dropped (12/16, p ≈ 0.08). Keep 0.12; the variety harm is
   now directionally demonstrated (was unmeasurable single-seed).
3. **Near-dup (CLIP ≥ 0.95) — clean signal-side separation.** Clean 2 pairs vs low-div 45 pairs at
   the existing threshold; ties to the same variety/adherence outcome as #2.
4. **Methodology / prune note:** *output* sharpness (`blur_variance` on generations) is too
   content-dominated (std ~1200) to use as a LoRA-quality axis — identity + spread + prompt
   adherence carry the signal. (This is an eval-harness note, not a Dataset Doctor threshold.)
5. **Subject-prominence / crop:** the trainer's `center_crop_square` dropped faces on tall
   full-body inputs and measurably halved learned identity (0.156 → 0.285 once face-cropped) — a
   live confirmation that `crop_loss` + `subject_prominence` flag real harm, and an argument for a
   face-aware crop / aspect-bucketing in the trainer itself.

### Honest scope

N = 16 paired observations (8 prompts × 2 seeds), **reduced-step** (400) LoRAs, **one** subject
that is a **game-character render** (the ArcFace domain gap attenuates absolute identity — every
LoRA reads 0.1–0.3, not the 0.73 photo-band). The X-signal separation is *decisive*; the Y
directions are *consistent and now borderline-significant* (blur→identity p ≈ 0.057, low-div→
adherence p ≈ 0.077) with the variety effect demonstrated via `same_prompt_spread`. **Still a
marginal-significance pilot, not regression-learned thresholds** (single subject; p just above
0.05; multiple metrics tested without correction). Remaining strengthening, in order: the two
deferred variants (wrong-person → identity tautology check;
pose/angle-collapse → a second diversity flavor), then a real-person photo subject to remove the
render domain gap.

### Fed back to the spike

[`docs/sc-6530/dataset-doctor-metrics.md`](../sc-6530/dataset-doctor-metrics.md) §3 + §8 updated
with these pilot-supported results (blur dual-floor validated; diversity 0.12 + near-dup 0.95
supported; §8 gains a worked closed-loop example).
