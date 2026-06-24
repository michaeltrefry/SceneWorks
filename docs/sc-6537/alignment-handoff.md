# Dataset Doctor — sc-6537 ② Caption↔Image Alignment: HANDOFF

**Audience:** a fresh agent (Codex) finishing epic 6529 (Dataset Doctor) cold, with no prior context.
**Status:** Everything except sc-6537 ② (alignment) is shipped. This doc is the spec + map to finish ②.
**Author note:** written at a session boundary (usage limit); treat the "verify, don't trust" section as
load-bearing — several non-obvious things bit the previous agent.

---

## 0. TL;DR

Dataset Doctor adds pre-training image QA to SceneWorks LoRA training (epic 6529, 4 phases). P1 (Tier-0
heuristics + UI) and P2-① (CLIP embedding pipeline: near-dup, diversity, **aesthetic**) are **done and in
review**. The **only** remaining work is **sc-6537 ②: caption↔image alignment** — a CLIP score between
each item's auto-generated caption and its image, to flag weak/mislabeled captions and wire a re-caption
action. It is **advisory** (like aesthetic), and it is the larger, **cross-repo** half (it needs a new
CLIP-L **text**-projection head that does not exist yet in mlx-gen/candle-gen).

When ② merges, epic 6529 is complete.

---

## 1. Repos, branches, PRs (the 3-PR stack + 2 backend PRs)

| Repo | Path | Branch | PR | Contains |
|---|---|---|---|---|
| SceneWorks | `~/Code/SceneWorks/SceneWorks` | `sc-6529-dataset-doctor-p1` | **#866** → `main` | P1 (Tier-0 + readiness API + UI) |
| SceneWorks | ″ | `sc-6535-dataset-doctor-p2` | **#867** → #866 | P2 doc, increment-1a math, the `dataset_analysis` job pipeline, **Tier-1 calibration** (commit `bc056b45`) |
| SceneWorks | ″ | `sc-6537-dataset-doctor-aesthetic` | **#870** → #867 | **sc-6537 ① aesthetic** (commit `842bb3fb`) — **the closest pattern to mirror for ②** |
| mlx-gen | `~/Code/SceneWorks/mlx-gen` | — | **#540 (MERGED)** | `gen-core` `ImageEmbedder` contract + `mlx-gen-clip` provider |
| candle-gen | `~/Code/SceneWorks/candle-gen` | `claude/candle-gen-clip` | **#135** → `main` | `candle-gen-clip` (the candle twin of mlx-gen-clip) |

**Merge order:** #866 → #867 → #870 (GitHub retargets each to its next base as the parent merges).
**② should be a NEW branch** `sc-6537-dataset-doctor-alignment` stacked on **#870** (so it sees the aesthetic
code it mirrors), OR off `main` once the stack lands — your call based on what's merged when you start.
**Shortcut:** epic 6529; story **sc-6537** is **In Progress** (do NOT mark Done until ② lands). PRs reference
story IDs. Create PRs with `gh pr create` directly (the GitHub MCP connector fails with "Resource not
accessible by integration").

---

## 2. How Dataset Doctor works — the seams ② plugs into

Read `docs/sc-6535/dataset-doctor-p2.md` end-to-end first (esp. **§1 scope split, §2 sub-score model, §3
the encoder + the verified text gap, §4 the contract, §8 calibration**). Then skim PR #870's diff — it is
the template.

**The embedding pipeline (already built for the image side):**
```
worker `dataset_analysis` job (MLX/candle CLIP) ──embeds images──▶ POST /analysis-embeddings
   ──▶ content-hash-keyed sidecar `dataset.sceneworks.embeddings.json`
   ──▶ rust-api readiness report reads the sidecar ──▶ HOST-SIDE pure-Rust eval
        (`evaluate_tier1` = near-dup+diversity; `evaluate_aesthetic` = LAION MLP)
   ──▶ `ReadinessSubScores` + `QualityFlag`s ──▶ web meters + copy
```

**Key design principle:** heavy model work (embedding) runs in the **worker** (GPU lane, downloads weights);
**derived scoring** runs **host-side** in `sceneworks-core` over the persisted vectors (pure `Vec<f32>`,
no GPU, fully unit-testable). ② follows the same split: embed captions in the worker, score caption↔image
cosine host-side.

**The contract layer** lives in the **mlx-gen** repo: package `sceneworks-gen-core`, lib `gen_core`
(`~/Code/SceneWorks/mlx-gen/gen-core/`). It is backend-neutral (host types only: `Vec<f32>`, `&[u8]`, no
tensors), builds on Linux, and uses **`inventory`** link-time registration. `mlx-gen` and `candle-gen`
providers depend only on it. The worker pins all mlx-gen crates (and candle-gen) at one rev.

---

## 3. The remaining work — ② alignment, in detail

### 3.1 What it is (Shortcut sc-6537 acceptance)
- **Caption↔image alignment**: CLIP cosine between each item's caption (auto-captioned with JoyCaption)
  and its image, in CLIP's **joint contrastive space**. Flags mislabeled/weak captions → re-caption action.
- **Advisory, never blocking.** Surface an `alignment` sub-score + a `CaptionAlignment` finding with a
  re-caption action. Per-kind floor (CLIPScore is unnormalized).

### 3.2 The verified gap (why this needs new model code) — `docs/sc-6535/dataset-doctor-p2.md` §3
- Image side: **done.** `mlx-gen-clip` / `candle-gen-clip` produce `visual_projection(CLS)` = the canonical
  768-d CLIP image embedding (`image_embeds`). That is one half of the joint space.
- Text side: **MISSING.** `mlx-gen-flux/src/text_encoder.rs::ClipTextEncoder` (line ~87–155) returns the
  **conditioning-pooled EOS hidden state** `[1,768]` for FLUX conditioning — **not** `text_projection(eot)`.
  SDXL's projected text encoder (TE2) is CLIP-**bigG** (1280-d), not matched ViT-L/14. So neither existing
  text path gives the **joint-space** text vector. You must add a CLIP-L `text_projection` head: take the
  EOS-pooled hidden state and apply the checkpoint's `text_projection.weight` (Linear 768→768, no bias) →
  the contrastive text embedding that is directly comparable (cosine) to `image_embeds`.

### 3.3 The plan (mirror aesthetic ① + the original 1b cross-repo pattern)

This is a contract + provider + worker + host + web change, like increment 1b. Sequence:

**A. gen-core contract (mlx-gen repo, `gen-core/src/`).** Add a **text** embedder contract mirroring
`image_embed.rs`. Two options — pick one and document it:
  - (preferred) a new `TextEmbedder` trait: `descriptor()` + `embed_text(&str) -> Result<Vec<f32>>` (raw,
    caller L2-normalizes), with `TextEmbedderDescriptor { id, family, backend, embedding_dim, space,
    mac_only }`. Register exactly like `ImageEmbedderRegistration` (`inventory::collect!` +
    `load_text_embedder(id, spec)`). Use the **same `space`** string as the image embedder
    (`"clip-vit-l14"`) so cosine is meaningful cross-modally.
  - Files to mirror: `gen-core/src/image_embed.rs`, `gen-core/src/registry.rs`, `gen-core/src/lib.rs`
    (re-exports). **No `Send + Sync`** on the trait (matches `ImageEmbedder`/`Captioner` — MLX `Array`'s
    raw handle isn't `Sync`; the worker runs it in one `spawn_blocking`).
  - This is a **gen-core change → new gen-core rev → cross-repo re-pin** (see §4 skew gate).

**B. mlx-gen provider (mlx-gen repo).** Either extend `mlx-gen-clip` or add `mlx-gen-clip-text`. Reuse
`mlx-gen-flux`'s `ClipTextEncoder` + the CLIP **tokenizer** (CLIP BPE, 77-token context, the EOS/argmax
pooling — note the line ~151 comment about a pad-vs-EOS tie bug) and add a local `text_projection` head
loaded from the snapshot's `text_projection.weight`. Mirror exactly how `mlx-gen-clip` added a local
`visual_projection` to `mlx-gen-sdxl`'s `ClipVisionEncoder`. `id` e.g. `"clip_vit_l14_text"`, 768-d,
`space "clip-vit-l14"`, backend `"mlx"`. Real-weights test (`#[ignore]`) against the cached
`openai/clip-vit-large-patch14` snapshot: a caption like "a photo of a dog" should have **higher** cosine
to a dog image than to an unrelated one (relative check; CLIPScore is unnormalized so don't assert an
absolute value).

**C. candle-gen provider (candle-gen repo).** The candle twin (mirror `candle-gen-clip`, PR #135). NOTE:
`candle-gen-sdxl/src/` has only `vision_encoder.rs` — the candle CLIP **text** tower lives elsewhere
(check `candle-gen-flux`, `candle-gen-sdxl` other modules, or `candle-transformers::models::clip`).
**Locate it before coding.** Register backend `"candle"`, same id/space. Then in SceneWorks re-pin
candle-gen + force-link under `cfg(backend-candle)`.

**D. SceneWorks worker.** Embed each item's caption text. Either extend the `dataset_analysis` job
(`crates/sceneworks-worker/src/dataset_analysis_jobs.rs`) to also embed captions, or add a sibling job.
Read the caption from the item's caption sidecar (see §3.4). Force-link the text provider
(`use mlx_gen_clip_text as _;` analog). Persist the text vectors (extend the embedding sidecar with a
text section, or a parallel sidecar) keyed by content hash (or item id).

**E. SceneWorks host eval (`crates/sceneworks-core/src/dataset_quality.rs`).** Add:
  - `CaptionAlignment` to the `QualityCheck` `string_enum!` (it is **NOT** there yet — only `LowAesthetic`
    was added; `near_duplicate_embedding`, `low_diversity`, `low_aesthetic` exist). Advisory, not in
    `is_technical_check`.
  - An `AlignmentThresholds { floor: f64 }` with `for_kind` (per-kind floor; **placeholder**, flag it).
  - `evaluate_alignment(items_with_image_and_text_embeds, thresholds, kind) -> AlignmentEvaluation` —
    per-item cosine(L2norm(image), L2norm(text)); items below the floor get a per-item `CaptionAlignment`
    `Info`/`Warn` flag carrying a re-caption hint; the mean fills `ReadinessSubScores.alignment` (already
    reserved as `Option<f64>`, currently always `None`). **Mirror `evaluate_aesthetic` exactly** (it's the
    freshest example, ~line 1264 of `dataset_quality.rs`).
  - Thread it into `build_readiness_report` (add an `alignment: Option<&AlignmentEvaluation>` param exactly
    like the `tier1`/`aesthetic` params; set `alignment: alignment.map(|a| a.score)`; fold its flags into
    `dataset_flags`/per-item). Update ALL callers (core tests, `sceneworks-image-quality::compute_readiness`,
    `apps/rust-api/src/training.rs::dataset_readiness_report`). The aesthetic PR #870 shows every call site.

**F. rust-api wiring (`apps/rust-api/src/training.rs`).** In `dataset_readiness_report`, build the
per-item (image_embed, text_embed) pairs from the persisted sidecars and call `evaluate_alignment`, pass
into `compute_readiness`. Mirror the aesthetic block (the `embeddings` hoist + `evaluate_aesthetic` call).

**G. web (`apps/web/src/`).** `datasetReadiness.js`: `CHECK_REASON`/`METRIC_LABEL`/`CHECK_PHRASE` +
`PHRASE_ORDER` for `caption_alignment`, an `alignmentPercent`/`alignmentScore` helper, summary copy, and a
**re-caption action** wired from a low-alignment flag (the story explicitly requires this — find how the
caption job is triggered from the UI and surface a "re-caption these" affordance). `DatasetDoctor.jsx` +
`styles.css`: an "Alignment" meter (mirror the "Aesthetic"/"Variety" meters). Add tests in
`datasetReadiness.test.js` (mirror the aesthetic tests) — keep `npm run lint` 0-error and the vitest suite
green.

### 3.4 Where captions live
Per-item captions are `.txt` sidecars next to the image (`training_store.rs`: `caption_path`,
`source_path.with_extension("txt")`, `CaptionInput`, `caption: Option<CaptionInput>` on the item;
`write_training_dataset_caption_sidecars` in `apps/rust-api/src/training.rs:284`). The JoyCaption job
(`crates/sceneworks-worker/src/caption_jobs.rs`) produces them. The worker reads the caption text to embed
it; the host reads the persisted text vectors. (`caption_jobs.rs` is also the **template** for any new
worker job — the `dataset_analysis` job was cloned from it.)

---

## 4. Cross-repo discipline (gotchas that bit the last agent)

- **The gen-core skew gate (CRITICAL).** The worker's `--features backend-candle` lane must resolve **one**
  `sceneworks-gen-core` rev. All mlx-gen crates are pinned at one rev (→ one gen-core); candle-gen pins its
  own gen-core; the worker pins both. Adding a contract to gen-core (step A) means a **new gen-core rev** →
  re-pin **mlx-gen** (`[workspace.dependencies]` one-liner), **candle-gen** (its single
  `sceneworks-gen-core` pin + the `-testkit` dev-dep in lockstep), and the **worker**
  (`crates/sceneworks-worker/Cargo.toml`, all ~26 mlx-gen crate revs + candle-gen). See
  `candle-gen/scripts/check-gen-core-skew.sh` and the long comment block atop `candle-gen/Cargo.toml`.
- **VERIFY PIN ANCESTRY WITH `git merge-base`, DO NOT ASSUME DIRECTION.** The last agent's brief assumed
  candle-gen lagged the worker; `git merge-base 43aa2bf 42a6415 == 42a6415` proved candle-gen *leads* the
  worker (its `43aa2bf` descends from the worker's `42a6415`). Bumping the "wrong" way would have *removed*
  an unrelated feature (sc-7491). Always check which rev is the descendant before re-pinning.
- **Provider force-linking.** A provider self-registers via `inventory::submit!` **only if linked**. A
  consumer that depends on it purely for the registration side-effect must `use <crate> as _;` or the
  linker drops the statics ("no embedder registered" at runtime). The worker has one such line per provider.
- **Native build (no Docker).** Metal Toolchain is installed (2026-06-23). Recipe: `. "$HOME/.cargo/env"`
  then `cargo build/clippy/test -p <crate>`. mlx-gen/worker/rust-api build natively on this Mac. See memory
  `[[sceneworks-no-local-rust-toolchain]]`.
- **Gates:** `cargo clippy --all-targets -- -D warnings` (warnings are errors) **and** `cargo fmt --all
  --check` for every Rust crate; `npm run lint` (0 errors) + `npm test` for web. `RUST_TEST_THREADS=1` is
  forced for MLX tests (shared Metal device isn't thread-safe). Rust toolchain pinned at 1.96.0.
- **Commits/PRs:** end commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context)
  <noreply@anthropic.com>` (adapt to your own attribution), end PR bodies with the tool's generated-with
  line, branch convention `sc-XXXX-...` or `claude/sc-XXXX-...`. Commit/push only when asked.
- **No-vendor convention:** SceneWorks downloads weights from HF (manifest + `hf_home` cache), never vendors
  them — the aesthetic MLP head was a *documented exception* (3.7 MB extracted subset, Apache-2.0, see
  `crates/sceneworks-image-quality/assets/README.md`). For ②'s text head, prefer the HF-download path
  (it's the convention) unless you hit the same "tiny extracted subset" situation.

---

## 5. Verification recipe + tooling

- **Test datasets** are on disk in `~/Datasets/`: `dreambooth/dataset/` (30 subject sets), `hokusai-style/`
  (46), `monkey-island/` (44), `Basim/` (64 person photos). Pulled with
  `scratchpad/hf_pull.py` (HF token in `~/.env` as `HUGGINGFACE_API_TOKEN`). These are gold for calibrating
  the alignment floor and for real-weights validation.
- **CLIP snapshot** cached at
  `~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/<sha>/` (has
  `model.safetensors` with `vision_model.*`, `text_model.*`, `visual_projection.weight`,
  `text_projection.weight`). The text head loads `text_model.*` + `text_projection.weight` from here.
- **Throwaway-validation pattern** (used heavily, keeps the tree clean): add an `#[ignore]`d test to the
  worker's `dataset_analysis_jobs.rs` `real_weights_tests` mod that loads real weights + embeds, run with
  `RUST_TEST_THREADS=1 cargo test -p sceneworks-worker <name> -- --ignored --nocapture`, eyeball, then
  `git checkout -- <file>` to drop it. The calibration tool `calibrate_thresholds` (CALIB_DIR-driven,
  committed in #867) is the model for a sweep; `scratchpad/extract_aesthetic.py` shows range-fetching an
  HF safetensors header + extracting tensors without downloading the whole file.
- **Expected test counts before ②** (so you know nothing regressed): core 279 lib, image-quality 14,
  rust-api 131, web 683; all clippy/fmt clean.

---

## 6. Open placeholders / decisions (be honest about these in the PR)

- **Alignment floor: `0.15`, provisional/photographic-only** (was a `0.10` placeholder). Production-faithful sweep:
  12 dreambooth photos captioned by the **real JoyCaption job** ("Descriptive/long"), CLIP-L cosines,
  matched (own caption) vs cross-subject mismatched. **Key finding — embed the caption's _first sentence_,
  not the whole paragraph.** Full 256-token captions put matched & mismatched at the *same* median (≈0.11,
  zero separation: AUC ≈ 0.5) because CLIP truncates at 77 tokens and the generic scene/lighting prose
  shared across single-subject photos washes the subject noun out. First-sentence embedding
  (`dataset_quality::caption_alignment_text`, applied in the worker before `embed_text`) restores a clean
  gap: matched min 0.1754 / median 0.2490 vs mismatched max 0.1621 / p95 0.1531. At floor `0.15`: 0/12
  correct captions flagged, ~93% of cross-subject mismatches caught, headroom below the matched min.
  Reproduce: `CALIB_DIR=~/Datasets/dreambooth/dataset … cargo test -p sceneworks-worker
  sweep_caption_alignment -- --ignored --nocapture` (prints `[FULL CAPTION]` vs `[FIRST SENTENCE]` tables).
  **Caveat (don't oversell in the PR):** the 4 subjects are maximally distinct, so this validates
  *gross wrong-subject* detection, not subtle mis-caption, and only on single-subject **photos**.
  CLIP scores style/illustration lower → a correct first sentence there could dip below `0.15` and
  false-flag; mark it provisional like the aesthetic/diversity floors, broader/style (or per-kind) sweep
  pending. Stays a dismissable advisory `Warn` (verified: gate is `Blocked` only on a *fatal* flag, so a
  low-alignment Warn yields at most `NeedsAttention` — never blocks training).
- **Aesthetic floor `5.0`** (PR #870) is a placeholder — real style sets scored hokusai 6.30 /
  monkey-island 5.47 means; needs a style-set sweep. (Not your task, but same calibration debt.)
- **Style diversity floor `0.10`** (in #867) is provisional — only 2 style sets sampled (0.144 vs 0.284).
- **Contract shape (step A):** new `TextEmbedder` trait vs extending `ImageEmbedder` — pick the cleaner;
  the previous agent leaned toward a parallel `TextEmbedder` for symmetry.

---

## 7. "Verify, don't trust" — lessons that cost the last agent time

1. **Check formulas against the production code, not memory/docs.** `evaluate_tier1` diversity was assumed
   to be `1 − mean cosine`; it had to be confirmed by reading the function (it was, but the near-dup-cluster
   interaction was the real question). Read `evaluate_aesthetic`/`evaluate_tier1` source before mirroring.
2. **Validate on the RIGHT data.** Aesthetic was first validated on a *person* set, but it only runs for
   *style* — the meaningful check was the style sets. For ②, validate alignment on captioned images where
   you know the caption is good vs deliberately wrong, not just "it returns a number".
3. **Advisory flags should be `Info`, not `Warn`.** A non-dismissable, dataset-level flag on an uncalibrated
   floor should not downgrade the readiness gate. `LowAesthetic` was changed `Warn`→`Info` for exactly this
   reason. `CaptionAlignment` is borderline (a wrong caption IS a real defect the user can fix) — but it's a
   per-item, *dismissable* flag, so `Warn` may be OK there; decide deliberately, don't inherit.
4. **Verify licenses at the source.** The aesthetic head was assumed MIT; it's actually **Apache-2.0**
   (upstream `christophschuhmann/improved-aesthetic-predictor`). Check the actual upstream repo's license
   before redistributing any weights.
5. **Use the advisor / a second-opinion pass before committing** a multi-crate feature — it caught #2, #3,
   the missing rust-api e2e coverage, and the PR-scoping/license issues on the aesthetic work.

---

## 8. Definition of done for ② (and the epic)

- New CLIP-L text-projection embedder in gen-core + mlx-gen + candle-gen (real-weights `#[ignore]` test
  shows correct caption↔image relative cosine).
- Worker embeds captions + persists them; rust-api computes `evaluate_alignment` for the report.
- `alignment` sub-score + `CaptionAlignment` finding + **re-caption action** in the UI, framed advisory.
- All gates green (clippy `-D warnings`, fmt, web lint 0-error, all test suites), with a few `#[ignore]`d
  real-weights tests for the model path.
- PR `sc-6537-dataset-doctor-alignment` stacked appropriately; Shortcut sc-6537 → In Review/Done; the epic
  (6529) is then complete.

Good luck. The aesthetic PR (#870) is your Rosetta Stone — almost every host-side step has a direct analog
there.
