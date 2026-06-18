# Image-Model Feasibility Matrix (sc-1173)

> **Story:** [sc-1173 — Validate image model feasibility matrix](https://app.shortcut.com/trefry/story/1173)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-06-18
> **Status:** Validated — chosen model spiked empirically; comparison set web-verified (June 2026).

**Provenance:** ⚙️ = empirically run on this machine (Apple M5 Max) · 🌐 = web-verified June 2026 (HF cards / repo LICENSEs / benchmarks; uncertain items flagged) · 📄 = SceneWorks code/manifest.

## Recommendation (v1 ordering)

Target hardware: **Windows/NVIDIA 24 GB first, Apple/MLX second.**

1. **`z_image_turbo` — primary T2I workhorse.** Only candidate that fits 24 GB at full bf16
   (~14–16 GB), 8-step distilled, **Apache-2.0 / ungated**, strong MLX story. Already the manifest's
   `recommended: true` default. ⚙️ Ran natively here (see below). Lowest-risk default.
2. **`qwen_image` (T2I) + `qwen_image_edit_2511` (edit) + `…_lightning` (speed) — quality/text/edit
   tier.** Best open benchmarks, best text rendering, best editing/character-consistency and
   ControlNet ecosystem, Apache-2.0/ungated. Cost: needs fp8/GGUF on 24 GB. Already shipped.
3. **`flux2_klein_4b` — commercial-safe photoreal + cross-platform. ⚠️ NOT in the manifest today —
   recommend adding.** Apache-2.0, **ungated**, ~8–13 GB VRAM, 4-step, native multi-reference edit,
   excellent Apple support. The catalog currently ships only `flux2_klein_9b` (gated, **FLUX
   Non-Commercial**, Mac-only) — klein-4B fills the commercial + Windows/24 GB gap.
4. **`flux2_dev` — Mac-only "max quality" tier (keep as-is).** SOTA but **gated + FLUX
   Non-Commercial + huge** (32B + 24B encoder). Never the commercial default.
5. **`wan_2_2` single-frame — future "cinematic" style mode, not the workhorse.** No official image
   mode (community frames=1 hack); slow, no still-edit/control story. Correctly typed `video`.

## Reconciliation: brief vs. shipped reality 🌐📄

The story's candidate names don't all match what shipped — surfaced so the decision log is accurate:

| Brief named | Reality (June 2026) |
|---|---|
| FLUX.2-klein-4B | Real & **commercial-safe**: Apache-2.0, ungated, 7.75 GB DiT. But a **klein-9B** also exists (FLUX Non-Commercial, gated) — and the manifest ships **9B**, not 4B (`flux2_klein_9b`, `builtin.models.jsonc:1004`). Gap. |
| Qwen-Image-Edit-2509 | Two generations old. Current is **2511** (#1 open edit, LMArena), already shipped (`qwen_image_edit_2511`, `:231`). |
| HunyuanImage-3.0 | **80B MoE, needs ≥3×80 GB GPUs**; restrictive Tencent license (100M-MAU cap, **excludes EU/UK/KR**). Correctly **absent** from the manifest. |
| Wan2.2 single-frame | No official T2I mode — community hack on a video model; registered `type: "video"` (`:2394`). |
| LightX2V Qwen-Image-Lightning | A ~0.85–1.7 GB step-distill **LoRA** (4/8-step), Apache-2.0; shipped fused (`qwen_image_edit_2511_lightning`, `:304`). |

## Empirical result — Z-Image-Turbo ⚙️

`zimage_real_weights_generates_one_image` on Apple M5 Max (512², 8-step, Q8, native MLX, in-process):

| Metric | Value |
|---|---|
| Gen time | 6.16 s (in-process) / 7.12 s wall |
| **Peak unified memory** | **28.6 GB** |
| Host RSS | 3.34 GB |

> Note on the two memory framings: the **28.6 GB** above is the **Apple unified peak** (Q8 weights +
> Qwen3-4B text encoder + VAE + activations, all resident). The matrix's "~14–16 GB" 24 GB-fit figure
> is the **CUDA bf16 transformer-only** VRAM. Both are correct at different scopes — on a 24 GB NVIDIA
> card the DiT fits at bf16; the Apple measurement is the whole-pipeline unified footprint.

## Comparison matrix 🌐

| Model | Params/Arch | Download (bf16) | License / Gated | Commercial | Steps | Fits 24 GB? | Edit | ControlNet | LoRA | Apple/MLX |
|---|---|---|---|---|---|---|---|---|---|---|
| **Z-Image-Turbo** | 6B S3-DiT | ~33 GB | **Apache-2.0 / no** | **Yes** | 8 (distil) | **Yes (~14–16 GB bf16)** | stopgap img2img (no edit ckpt) | community | young | **Strong** |
| **Qwen-Image** | 20B MMDiT | ~41 GB DiT | **Apache-2.0 / no** | **Yes** | 50 (8 w/Lightning) | fp8/GGUF | use edit family | **Yes (Union)** | **mature** | yes (heavy) |
| **Qwen-Image-Edit-2511** | 20B MMDiT edit | ~41 GB DiT | **Apache-2.0 / no** | **Yes** | 40 (8 w/Lightning) | fp8 (~86%) | **best open edit** | **native** | yes | partial |
| **Qwen-Image-Lightning** | LoRA / Qwen-Image | ~0.85–1.7 GB | **Apache-2.0 / no** | **Yes** | **4 / 8** | yes | (edit variants) | (via base) | **is the LoRA** | base on MLX |
| **HunyuanImage-3.0** | **80B MoE** | ~169 GB | Tencent Community | **No\*** (EU/UK/KR excl.) | 50 (8 distil) | **No (≥3×80 GB)** | Instruct branch | none found | not yet | **none** |
| **FLUX.2-dev** | 32B + Mistral-24B enc | ~64 GB DiT | **FLUX NC / GATED** | **No** | 28–50 | no (fp8/offload) | **native unified** | none found | emerging | impractical (32B) |
| **FLUX.2-klein-4B** ⚠️not shipped | 4B flow DiT | 7.75 GB DiT | **Apache-2.0 / no** | **Yes** | **4 (distil)** | **Yes (~8–13 GB)** | **native** | none found | **Strong** |
| **FLUX.2-klein-9B** (shipped) | 9B flow DiT | ~49–52 GB | **FLUX NC / GATED** | **No** | 4 (distil) | Mac MLX only | native | none found | MLX-only |
| **Wan2.2 TI2V-5B (1-frame)** | dense 5B video | ~10 GB | **Apache-2.0 / no** | **Yes** | 27–40 | **Yes (even 8 GB)** | enhance img2img | video-only | 2nd-class | experimental |

\* Hunyuan "commercial yes" is conditional (MAU cap + region exclusions).

## Quality / speed notes 🌐
- **Z-Image-Turbo:** LMArena ~#23 (excellent for 6B/8-step); 4090 speed reports conflict (2.3 s vs
  8–13 s) — benchmark on target HW. No dedicated edit checkpoint shipped; `z_image_edit` is a stopgap
  img2img on turbo weights (manifest documents this). No IP-Adapter → manifest correctly withholds
  `character_image` (sc-2005).
- **Qwen-Image / Edit-2511:** best benchmarks of the open field (GenEval ~0.91, DPG ~88.3), best
  bilingual text rendering, best character-consistency + ControlNet-Union; fp8 ~71 s/image on a 4090.
- **FLUX.2-klein-4B:** BFL claims it matches Qwen-Image / beats Z-Image — but **4B-specific
  GenEval/ELO are not independently published** (treat as unverified); sub-0.5 s on NVIDIA at 4-step.

## Rust backend target 📄

Every studied model maps onto the existing `ModelManifestEntry` (`contracts.rs:1344-1373`) — **no
schema change required**; these are population/wiring implications.

- **Manifest fields per entry:** `id, name, family, type, adapter, capabilities, downloads
  (provider/repo/files/platforms/estimatedSizeBytes), paths, defaults, limits, loraCompatibility, ui`.
  Gating is three optional fields (`gated`, `credentialHost`, `licenseUrl`) — **omit all three for
  Apache/ungated** models (Z-Image, all Qwen, Lightning, klein-4B, Wan); **set them for gated NC**
  models (`flux2_dev`, `flux2_klein_9b`).
- **Capability flags** (`ModelCapability`, `contracts.rs:542-552`): `text_to_image`, `image_edit`,
  `character_image`, `style_variations`. Z-Image → `["text_to_image","style_variations"]` (no
  `character_image`); Qwen-Edit / FLUX.2 edit-capable → add edit + `character_image`.
- **⚠️ Capability-string skew to fix:** the enum value is `image_edit` but manifest entries write
  `edit_image` in `capabilities`; since `ModelCapability` is a forward-compatible string enum,
  `"edit_image"` deserializes to `Unknown("edit_image")`, **not** `ImageEdit`. This works today only
  because nothing strictly matches the typed `ImageEdit` variant — confirm routing doesn't depend on
  it before adding new edit-capable entries.
- **Job-payload shape:** `JobType::ImageGenerate` / `ImageEdit` with an open `payload` JSON object
  (steps, guidanceScale, trueCfgScale, variationStrength, `referenceAssetId`); per-model knobs ride
  in `defaults`/`limits`/`ui` — no contract change to add a model.
- **Download / scheduling:** refresh `estimatedSizeBytes` from the live HF tree; **add a typed
  precision-aware VRAM/peak field** (the manifest's single `mlx.minMemoryGb` is coarse — see sc-1176
  where measured LTX peak was ~1.7× the estimate) so admission can check fit-by-precision instead of
  OOMing at runtime. Model-download jobs don't consume GPU slots.
- **Asset outputs:** `AssetFile` is already image-capable (`path, mime_type, width, height`) — no new
  fields needed.
- **Catalog action:** add `flux2_klein_4b` (Apache, ungated, Windows/CUDA adapter + MLX) to give the
  catalog a commercial-safe, 24 GB-friendly, cross-platform FLUX option it currently lacks.

## Caveats / could-not-verify 🌐 (carried forward, not gaps in this story)
Z-Image text encoder named "Qwen3-4B" is community-corroborated not card-verbatim; Z-Image 4090 speed
conflicts (2.3 s vs 8–13 s); klein-4B standalone GenEval/ELO unpublished; HunyuanImage HF gating
inferred (CLI worked) not page-confirmed; HunyuanImage per-image speed on 24 GB unverified; FLUX.2 fp8
disk sizes vary by source; ControlNet/IP-Adapter for FLUX.2 not found (unverified-absent); Wan2.2
single-frame numbers from one tester. Full source URLs and the complete uncertainty list are retained
in the research notes for this story.

## Sources
Empirical: `zimage_real_weights_generates_one_image` on M5 Max. Web (June 2026): HF model cards +
LICENSE files for Tongyi-MAI/Z-Image-Turbo, Qwen/Qwen-Image[-Edit-2511], lightx2v/Qwen-Image-Lightning,
tencent/HunyuanImage-3.0, black-forest-labs/FLUX.2-dev & FLUX.2-klein-4B, Wan-AI/Wan2.2-TI2V-5B, plus
LMArena/GenEval/DPG benchmarks and community benchmark reports. Code: `crates/sceneworks-core/src/contracts.rs`,
`config/manifests/builtin.models.jsonc`. Prior: `documents/IMAGE_MODEL_RESEARCH.md`.
