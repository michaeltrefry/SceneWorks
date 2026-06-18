# SceneWorks Research Decision Log

> **Story:** [sc-1172 — Maintain SceneWorks research decision log](https://app.shortcut.com/trefry/story/1172)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-06-18
> **Status:** Living document — the capstone of epic 1093. Update when a v1 research decision changes.

## Purpose

A single record of the research outcomes that shape the SceneWorks v1 implementation path. Each
decision lists **decision · rationale · alternatives considered · follow-up risks · Rust backend
impact**, plus an explicit **provenance** tag so a reader knows how strongly each conclusion is held:

- ⚙️ **Empirical** — spiked in code on the dev machine (Apple M5 Max, 64 GB, macOS 26.5.1); numbers are measured.
- 🌐 **Web-verified** — confirmed against June-2026 primary sources (HF cards, LICENSE files, benchmarks); uncertain items flagged.
- 📄 **Code-grounded** — verified against the shipped Rust contracts / manifest / renderer.

This log is the synthesis; the per-decision detail lives in the sibling research docs produced by the
epic's other stories, linked below. It supersedes an earlier draft that stated these as settled
conclusions before the research tracks were run — that draft was withdrawn; this version is grounded.

## Summary

| # | Decision area | v1 choice | Provenance | Backend change |
|---|---|---|---|---|
| 1 | Image adapter | `z_image_turbo` (primary) | ⚙️🌐📄 | none (catalog add: klein-4B) |
| 2 | Video adapter | `ltx_2_3` (primary), Wan2.2 fallback | ⚙️🌐📄 | manifest `resources` (shipped) |
| 3 | Timeline library | own model + FFmpeg export | ⚙️📄 | none for v1 |
| 4 | Replacement pipeline | job/track-first, **Face Only** default | 📄 (not re-spiked) | none for Face Only |
| 5 | Apple runtime | native in-process **Rust + MLX** | ⚙️📄 | typed peak-memory (recommended) |
| 6 | Manifest schema | `schemaVersion: 1`, typed fields, capability enum | 📄 | reconcile capability skew |

---

## 1. Image adapter — `z_image_turbo` (primary)
*Detail: [`IMAGE_MODEL_FEASIBILITY_MATRIX.md`](IMAGE_MODEL_FEASIBILITY_MATRIX.md) (sc-1173)*

- **Decision.** `z_image_turbo` is the v1 default; `qwen_image` + `qwen_image_edit_2511` (+ Lightning)
  is the quality/text/edit tier; recommend **adding `flux2_klein_4b`** for a commercial-safe,
  24 GB-friendly, cross-platform option.
- **Rationale.** ⚙️ Ran natively on M5 Max (512²/8-step/Q8: 7.1 s wall, 28.6 GB peak unified). 🌐 Only
  candidate that fits 24 GB at full bf16 (~14–16 GB), Apache-2.0/ungated, 8-step distilled, strong MLX
  story.
- **Alternatives.** Qwen-Image/Edit-2511 (best benchmarks/text/edit, needs fp8 on 24 GB — kept as
  quality tier); FLUX.2-klein-4B (Apache, ungated — *recommend adding*); FLUX.2-dev (Mac-only, gated,
  non-commercial); HunyuanImage-3.0 (rejected: 80B MoE, ≥3×80 GB, region-restricted license);
  Wan2.2 single-frame (rejected as workhorse: community hack, slow, no still-edit).
- **Follow-up risks.** No dedicated Z-Image-Edit checkpoint shipped (`z_image_edit` is a stopgap
  img2img); Z-Image LoRA ecosystem is young; **catalog ships `flux2_klein_9b` (gated/NC) not the
  Apache klein-4B** — a real gap. Several 🌐 figures unverifiable (see sc-1173 caveats).
- **Rust backend impact.** 📄 Maps onto existing `ModelManifestEntry`; **no schema change**. ⚠️ Fix the
  `image_edit` vs `edit_image` capability-string skew (manifest value deserializes to
  `Unknown`). Omit gating fields for Apache models; set them for `flux2_dev`/`flux2_klein_9b`.

## 2. Video adapter — `ltx_2_3` (primary), Wan2.2 fallback
*Detail: [`VIDEO_MODEL_FEASIBILITY_MATRIX.md`](VIDEO_MODEL_FEASIBILITY_MATRIX.md) (sc-1174)*

- **Decision.** `ltx_2_3` primary; **Wan2.2 TI2V-5B** (Apache, fits 24 GB) as the safe fallback;
  Wan2.2 A14B as quality/multi-GPU tier.
- **Rationale.** ⚙️ Ran natively on M5 Max (q4 + Gemma-12B, 256²/9f/+audio: 15.6 s wall, 53.4 GB
  peak). 🌐 ~5.7× faster than Wan A14B on 24 GB (distilled), only option with native synced audio,
  richest conditioning, best Apple story.
- **Alternatives.** Wan2.2 TI2V-5B (Apache, low-VRAM — the safe fallback) and A14B (quality, needs
  fp8/offload on 24 GB, multi-GPU path). ComfyUI/hosted-API/network rendering are explicit non-goals.
- **Follow-up risks.** 🌐 **LTX-2.3 is NOT Apache** — custom Community License (free commercial only
  under $10M ARR + anti-compete); a tracked commercial obligation. ⚙️ Memory-heavy: 53.4 GB peak at a
  *minimal* clip → 64 GB Macs have little headroom; manifest `minMemoryGb: 31` undercounts ~1.7×.
  Story claims corrected: "LTX ≤15 s" is a **sweet-spot not a ceiling** (~20 s max); "Wan loops ~7 s"
  is **unverified/likely false** (native ~5 s).
- **Rust backend impact.** 📄 The `resources` multi-file manifest block + `VideoRequest` conditioning
  already cover the full LTX/Wan surface (shipped). ⚠️ Capability skew (`first_last_frame`/`extend_clip`
  not in enum; `video_extend` ≠ `extend_clip`). Recommend per-file `sha256`, a typed VRAM block, and
  surfacing the LTX audio track on the job result.

## 3. Timeline library — own model + FFmpeg export
*Detail: [`FFMPEG_TIMELINE_EXPORT_MVP.md`](FFMPEG_TIMELINE_EXPORT_MVP.md) (sc-1175)*

- **Decision.** SceneWorks-owned timeline data model + lightweight React editor; **FFmpeg is the
  authoritative export renderer**. No third-party editor SDK, no browser/canvas recording.
- **Rationale.** ⚙️ Every export primitive the renderer emits (still loop, trim+`setpts` speed,
  scale/pad aspect-fit, fade, `xfade` crossfade, concat, libx264 encode) was replicated and passes on
  ffmpeg 8.1.2. Keeps timeline JSON portable; avoids a young/license-sensitive editor SDK.
- **Alternatives.** React-Video-Editor-Timeline, Remotion, Twick — all rejected (dependency/license
  risk; export is backend FFmpeg regardless).
- **Follow-up risks.** Scope creep (multi-track compositing, audio-mix UI, persistent undo) — all
  explicitly deferred post-v1.
- **Rust backend impact.** 📄 Fully expressible on existing `Timeline*` contracts + the
  `timeline_export` job / `timeline-exporter` role. **No new versioned contract change for v1.**
  Confirmed: zero field-name divergence between `TimelineItem` and the renderer.

## 4. Replacement pipeline — job/track-first, Face Only default
*Detail: [`REPLACE_PERSON_RESEARCH.md`](REPLACE_PERSON_RESEARCH.md) · risks in [`V1_RISK_REGISTER.md`](V1_RISK_REGISTER.md) (sc-1177)*

- **Decision.** Job-based, track-first workflow; **Face Only is the default/safest real adapter path**;
  Full-Person modes are gated per-model; procedural preview wires the full flow without a model.
- **Rationale.** 📄 Face Only has the narrowest mask, smallest temporal burden, clearest failure mode;
  the job/track contract stays stable while model adapters evolve. (On Mac, replace_person is reported
  complete via native Wan-VACE + a SAM2 MLX segmenter.)
- **Alternatives.** Full-Person Keep/Replace Outfit — kept but gated until they clear a quality bar.
- **Follow-up risks.** Mask correction deferred (box tracks only); Full-Person quality is the biggest
  open quality question. **Provenance note:** this decision is 📄 from prior research and shipped
  code — it was **not independently re-spiked** in this pass (the empirical effort focused on
  image/video/runtime/export). Treat the Replace-Person quality bar as still-to-validate.
- **Rust backend impact.** 📄 `person_track`/`person_replace` jobs + `PersonTrack*` contracts +
  sidecar lineage already exist; Full-Person needs honest per-model capability flags.

## 5. Apple runtime — native in-process Rust + MLX
*Detail: [`APPLE_RUNTIME_FEASIBILITY.md`](APPLE_RUNTIME_FEASIBILITY.md) (sc-1176)*

- **Decision.** Apple is a **first-class v1 runtime** via the native, in-process Rust + MLX worker
  (`mlx-gen`) — no Python/venv/subprocess/Docker-GPU. CPU/Candle/CoreML are not the Mac path.
- **Rationale.** ⚙️ On M5 Max / macOS 26.5: the worker builds Apple MLX from source (2m22s, 0 warn),
  `nax_guard` passes (16-bit NAX kernels present **and numerically correct**), and **both flagships run
  in-process** (Z-Image 28.6 GB; LTX 53.4 GB). 📄 Docker on macOS can't pass through Metal, so native
  MLX is required.
- **Alternatives.** `ort`+CoreML (rejected: hangs on YOLO11), Candle/CPU on Mac (rejected; Candle is a
  future Win/Linux target), torch-on-Mac (rejected as destination; kept on Win/Linux for unported
  models).
- **Follow-up risks.** ⚙️ Memory-bound (LTX 53.4/64 GB at minimal clip); macOS 26.2 floor for correct
  16-bit kernels; torch-only holdouts (each with a porting epic); AuraSR dropped on Mac.
- **Rust backend impact.** 📄 CUDA-free seams already ship (engine-id table, `cfg(target_os)` gate,
  `mac_rust_supported` oracle, `ModelMacSupport` surface, warn-only `SCENEWORKS_MLX_REQUIRED`).
  **Recommended add:** a typed precision-aware peak-memory field (the measured LTX peak was ~1.7× the
  manifest estimate) so admission checks fit-by-precision instead of OOMing.

## 6. Manifest schema — `schemaVersion: 1`, typed fields, capability enum
*Detail: cross-cutting; grounded in `crates/sceneworks-core/src/contracts.rs` + `config/manifests/builtin.models.jsonc`*

- **Decision.** Integer `schemaVersion` (currently 1) on manifests + sidecars; split into
  `builtin.models`/`user.models`/`builtin.loras`/`user.loras`; typed `ModelManifestEntry` with a
  `#[serde(flatten)] extra` passthrough for additive keys; capability flags as a forward-compatible
  `ModelCapability` enum; gating via `gated`/`credentialHost`/`licenseUrl`.
- **Rationale.** 📄 Already shipped and validated against `packages/schemas/model-manifest.schema.json`.
  The `extra` flatten means additive keys (`recommended`, `autoDownload`, `resources`, new `ui.*`)
  need no Rust change; CI capability audits (sc-2018) enforce capability honesty.
- **Alternatives.** N/A — settled and shipped (the plan's "what should the schema be?" open question is
  stale).
- **Follow-up risks.** ⚠️ **Capability-string skew** — manifest entries use strings not in the enum
  (`edit_image`, `first_last_frame`, `extend_clip`), which deserialize to `Unknown`; works today only
  because nothing strictly matches the typed variants. Reconcile before adding capability-gated routing.
  Keep the `extra` passthrough disciplined (CI audits are the guard).
- **Rust backend impact.** 📄 The schema *is* the backend contract. Recommended additive changes (no
  breaking change): typed precision-aware VRAM/peak field; per-file `sha256`/size on downloads; enum
  reconciliation for capabilities.

---

## Provenance & coverage summary

| # | Decision | Empirically spiked this pass? | Notes |
|---|---|---|---|
| 1 Image | ✅ Z-Image-Turbo run | comparison set web-verified | klein-4B catalog gap |
| 2 Video | ✅ LTX-2.3 run | Wan2.2 web-verified | LTX license ≠ Apache; memory-heavy |
| 3 Timeline | ✅ all ffmpeg chains | — | no contract change for v1 |
| 4 Replace Person | ❌ not re-spiked | 📄 prior research/code only | quality bar still to validate |
| 5 Apple runtime | ✅ build + nax_guard + both flagships | — | strongest empirical story |
| 6 Manifest | ✅ verified against shipped schema | — | fix capability skew |

**Open recommendations carried forward (see [`V1_RISK_REGISTER.md`](V1_RISK_REGISTER.md)):** typed
precision-aware peak-memory field; reconcile capability-flag enum skew; add Apache `flux2_klein_4b`;
per-file download hashes; surface LTX audio track; validate Replace-Person Full-Person quality bar.

## Sources
Per-decision docs (this folder): `IMAGE_MODEL_FEASIBILITY_MATRIX.md`, `VIDEO_MODEL_FEASIBILITY_MATRIX.md`,
`FFMPEG_TIMELINE_EXPORT_MVP.md`, `APPLE_RUNTIME_FEASIBILITY.md`, `V1_RISK_REGISTER.md`, plus prior
`IMAGE_MODEL_RESEARCH.md`, `VIDEO_MODEL_RESEARCH.md`, `EPIC_NATIVE_LTX23_VIDEO_ADAPTER.md`,
`TIMELINE_LIBRARY_RESEARCH.md`, `REPLACE_PERSON_RESEARCH.md`, `MULTI_MODEL_REFERENCE_CONDITIONING.md`.
Engineering records: `docs/rust-mlx-build.md`, `docs/mac-rust-gaps.md`, `docs/sc-3633-mlx-port.md`.
Shipped contracts/schema: `crates/sceneworks-core/src/contracts.rs`, `crates/sceneworks-core/src/jobs_store.rs`,
`config/manifests/builtin.models.jsonc`, `packages/schemas/model-manifest.schema.json`. Empirical figures
measured on Apple M5 Max / macOS 26.5.1.
