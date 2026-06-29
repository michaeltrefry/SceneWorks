//! Backend-neutral engine dispatch table + registry-derived capability advertisement
//! (sc-3723, epic 3720 Phase 0).
//!
//! [`MODEL_TABLE`] is the SceneWorks-id → mlx-gen-registry-id map plus the per-variant
//! defaults the worker needs that are NOT on the engine descriptor (HF repo, step/guidance
//! defaults, the asset `adapter` label). It is **all-targets** (no `#[cfg(target_os = …)]`):
//! the table is pure data, and keeping it neutral lets the registry-derived advertisement run
//! off-macOS (where the provider crates aren't linked, so the registry is empty and the derived
//! capability set is correctly empty — "absence, not runtime failure").
//!
//! The two descriptor-duplicating flags that used to live on each row
//! (`supports_guidance` / `supports_negative_prompt`) are gone: they are now read from the
//! linked gen_core descriptor through [`ResolvedModel`], so a row can never drift from the
//! engine's own advertised surface. A future candle backend lights up with **zero** worker
//! changes — it registers its descriptors into the same `inventory` registry and
//! [`registry_capabilities`] picks them up.

/// One engine-backed image family: how a SceneWorks model id maps onto the linked
/// mlx-gen registry, and the per-variant defaults (all chosen for parity with the
/// Python `MODEL_TARGETS` + the per-family MLX adapter). Adding a family = one row
/// here + its provider crate dep + a `use mlx_gen_<x> as _;` in `image_jobs.rs`.
pub(crate) struct ModelRow {
    /// SceneWorks model id (the job payload `model`).
    pub sceneworks_id: &'static str,
    /// registry id passed to `gen_core::load`.
    pub engine_id: &'static str,
    /// Default HuggingFace repo when the manifest entry omits `repo`.
    pub default_repo: &'static str,
    /// Default denoise steps (Python `MODEL_TARGETS[...]["steps"]`).
    pub default_steps: u32,
    /// Default guidance when supported and the request omits it.
    pub default_guidance: f32,
    /// The `adapter` id recorded on generated assets (the Python MLX adapter id).
    pub adapter_label: &'static str,
}

pub(crate) const MODEL_TABLE: &[ModelRow] = &[
    ModelRow {
        sceneworks_id: "z_image_turbo",
        engine_id: "z_image_turbo",
        default_repo: "Tongyi-MAI/Z-Image-Turbo",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_z_image",
    },
    // Base (non-distilled) Z-Image (epic 8236, sc-8320). The undistilled foundation model from the
    // `Tongyi-MAI/Z-Image` diffusers snapshot — the same `ZImageTransformer` as Turbo, but the
    // `z_image` engine descriptor uses a shift=6.0 schedule, ~50 default steps, and REAL CFG
    // (`supports_guidance` + negative prompt; the card recommends guidance 3.0–5.0, default 4.0) vs
    // Turbo's 4-step guidance-distilled CFG-free path. Ships its own fast `tokenizer/tokenizer.json`,
    // so it needs NO derived-tokenizer overlay. Routes to the base t2i path (not Turbo); strict-pose /
    // canny / depth control routes to the `z_image_control` engine variant (sc-8251).
    ModelRow {
        sceneworks_id: "z_image",
        engine_id: "z_image",
        default_repo: "Tongyi-MAI/Z-Image",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_z_image",
    },
    // Ideogram 4 (epic 4725) — native MLX, gated. Structured JSON-caption text-to-image; the
    // turnkey ships packed q4/ (default) + q8/ subdirs (resolve_ideogram_model_dir picks one).
    // V4_QUALITY_48 preset default (48 steps); asymmetric-CFG guidance 7.0.
    ModelRow {
        sceneworks_id: "ideogram_4",
        engine_id: "ideogram_4",
        default_repo: "SceneWorks/ideogram-4-mlx",
        default_steps: 48,
        default_guidance: 7.0,
        adapter_label: "mlx_ideogram",
    },
    // Ideogram 4 Turbo (mlx-gen #488) — the CFG-free, single-DiT few-step variant: the same
    // turnkey base (q4/q8 subdirs) plus the bundled ostris TurboTime LoRA the engine installs at
    // load. 8 steps; guidance is INERT (the `ideogram_4_turbo` descriptor advertises
    // supports_guidance=false, so `resolve_guidance` returns None and never forwards a value).
    ModelRow {
        sceneworks_id: "ideogram_4_turbo",
        engine_id: "ideogram_4_turbo",
        default_repo: "SceneWorks/ideogram-4-mlx",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_ideogram",
    },
    // Z-Image-Edit (epic 3529) — img2img/edit. No dedicated Edit checkpoint exists yet, so
    // (like the Python `MODEL_TARGETS` row) it runs the **Turbo weights** through the engine's
    // img2img path (`Conditioning::Reference` — VAE-encode the source + denoise from
    // `init_time_step(steps, strength)`), so it shares the `z_image_turbo` engine model. The
    // `z_image_turbo` `edit_image` mode resolves to the same img2img call (`resolve_zimage_edit_init`).
    ModelRow {
        sceneworks_id: "z_image_edit",
        engine_id: "z_image_turbo",
        default_repo: "Tongyi-MAI/Z-Image-Turbo",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_z_image",
    },
    ModelRow {
        sceneworks_id: "flux_schnell",
        engine_id: "flux1_schnell",
        default_repo: "black-forest-labs/FLUX.1-schnell",
        default_steps: 4,
        default_guidance: 0.0,
        adapter_label: "mlx_flux",
    },
    ModelRow {
        sceneworks_id: "flux_dev",
        engine_id: "flux1_dev",
        default_repo: "black-forest-labs/FLUX.1-dev",
        default_steps: 28,
        default_guidance: 3.5,
        adapter_label: "mlx_flux",
    },
    ModelRow {
        // Non-distilled true-CFG base: 20 steps + guidance 4.0 + negative prompt
        // (Python MODEL_TARGETS / MlxQwenAdapter). mlx-gen's own default is 4 steps,
        // so steps are passed explicitly. Edit moves to MLX (sc-3397, the `qwen_image_edit`
        // engine model below); base-Qwen strict-pose ControlNet routes to the
        // `qwen_image_control` engine variant when `advanced.poses` is present
        // (epic 3401 / sc-3575).
        sceneworks_id: "qwen_image",
        engine_id: "qwen_image",
        default_repo: "Qwen/Qwen-Image-2512",
        default_steps: 20,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    // Qwen-Image-Edit (sc-3397) — the three base edit ids all resolve to the engine's
    // single `qwen_image_edit` model (Reference/MultiReference, true CFG, LoRA/LoKr, Q4/Q8);
    // `qwen_image_edit`/`_2509` alias to the 2511 weights (Python MODEL_TARGETS, sc-2160).
    // 40 steps (engine's own default is 4 — passed explicitly, like the txt2img row). The
    // edit path resolves guidance from `trueCfgScale` (4.0), NOT `guidanceScale`; see
    // `resolve_qwen_edit_guidance`. The `_2511_lightning` distill (4-step, CFG-off) shares
    // these weights but adds the `lightning` sampler + the lightx2v distill LoRA — see the
    // row below and [`qwen_edit_lightning`] (sc-3398).
    ModelRow {
        sceneworks_id: "qwen_image_edit",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    ModelRow {
        sceneworks_id: "qwen_image_edit_2509",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    ModelRow {
        sceneworks_id: "qwen_image_edit_2511",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    // Lightning 4-step distill (sc-3398): same `qwen_image_edit` engine model + base
    // Qwen-Image-Edit-2511 weights as the rows above, but the generate path passes the
    // `lightning` sampler (static-shift schedule + CFG-off single forward) and stacks the
    // lightx2v distill LoRA ahead of any user LoRAs (see [`qwen_edit_lightning`] +
    // [`generate_qwen_edit_stream`]). Python parity (MODEL_TARGETS): 4 steps, guidance 1.0,
    // CFG off — so no negative prompt. The distill LoRA is a CFG-distilled adapter, so the
    // engine runs a single forward/step regardless of `default_guidance`.
    ModelRow {
        sceneworks_id: "qwen_image_edit_2511_lightning",
        engine_id: "qwen_image_edit",
        default_repo: "Qwen/Qwen-Image-Edit-2511",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_qwen",
    },
    // FLUX.2-klein (sc-3025) — MLX-only family (no torch fallback). All three SceneWorks
    // variants share the engine's single txt2img model `flux2_klein_9b` (edit + KV-cache
    // are the separate `*_edit`/`*_kv_edit` engine models, story sc-3029); the variants
    // differ only in their weights. Distilled klein runs guidance 1.0 (CFG-free) with no
    // negative prompt; the engine accepts guidance but rejects a negative prompt.
    ModelRow {
        sceneworks_id: "flux2_klein_9b",
        engine_id: "flux2_klein_9b",
        default_repo: "black-forest-labs/FLUX.2-klein-9B",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_flux2",
    },
    ModelRow {
        // Separately-distilled checkpoint, same architecture — its snapshot carries the
        // full diffusers tree, so txt2img loads through the base `flux2_klein_9b` loader.
        sceneworks_id: "flux2_klein_9b_kv",
        engine_id: "flux2_klein_9b",
        default_repo: "black-forest-labs/FLUX.2-klein-9b-kv",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_flux2",
    },
    ModelRow {
        // wikeeyang community fine-tune (sc-2220/2235): UNDISTILLED, so 24 steps. Its raw
        // repo is single-file (GGUF/safetensors) with no diffusers tree, so it loads from a
        // locally-assembled converted dir via the `modelPath` seam (manifest `modelPath`),
        // NOT the source repo below. The convert step is now native Rust/MLX
        // (mlx_gen_flux2::convert_and_assemble, sc-3136; run by the model_convert job).
        sceneworks_id: "flux2_klein_9b_true_v2",
        engine_id: "flux2_klein_9b",
        default_repo: "wikeeyang/Flux2-Klein-9B-True-V2",
        default_steps: 24,
        default_guidance: 1.0,
        adapter_label: "mlx_flux2",
    },
    // FLUX.2-dev (epic 5914) — the guidance-distilled 32B flagship. A SEPARATE engine
    // model `flux2_dev` (Mistral3 TE + 48/48/15360 DiT), NOT a klein weight variant, so it
    // maps to its own engine id. Embedded distilled guidance (FLUX.1-dev pattern, NOT
    // true-CFG): the descriptor advertises `supports_guidance` but not negative prompt, so
    // the engine takes the guidance scalar (default 4.0) over ~28 steps. On MLX it loads
    // from a pre-quantized Q4 dir assembled by the install-time `flux2_dev_quant` convert
    // job (sc-5917 / sc-5921). On the **candle** off-Mac lane (sc-7458) there is no packed
    // convert: it loads the dense `black-forest-labs/FLUX.2-dev` diffusers snapshot and
    // Q4-quantizes it at load — the 32B dense never lands on the GPU (CPU-staged in system
    // RAM, then `quantize_onto` the GPU; candle-gen-flux2 sc-7457). `resolve_quant` reads
    // the manifest `mlx.quantize: 4` so the candle descriptor's Q4 support drives it.
    ModelRow {
        sceneworks_id: "flux2_dev",
        engine_id: "flux2_dev",
        default_repo: "black-forest-labs/FLUX.2-dev",
        default_steps: 28,
        default_guidance: 4.0,
        adapter_label: "mlx_flux2",
    },
    // SDXL (sc-3026) — U-Net, real CFG (negative prompt + guidance 7.0), 30 steps.
    // `sdxl` and the `realvisxl` finetune share the engine's single `sdxl` model
    // (identical arch), differing only in weights. Replaces the in-process
    // _vendor/mlx_sd path. The engine supports Q4/Q8 (the Python vendored path had
    // none); Q8 is the default here (engine-validated; saves ~half the U-Net memory).
    ModelRow {
        sceneworks_id: "sdxl",
        engine_id: "sdxl",
        default_repo: "stabilityai/stable-diffusion-xl-base-1.0",
        default_steps: 30,
        default_guidance: 7.0,
        adapter_label: "mlx_sdxl",
    },
    ModelRow {
        sceneworks_id: "realvisxl",
        engine_id: "sdxl",
        default_repo: "SG161222/RealVisXL_V5.0",
        default_steps: 30,
        default_guidance: 7.0,
        adapter_label: "mlx_sdxl",
    },
    // RealVisXL Lightning (sc-6075) — standalone few-step *distilled* sibling of RealVisXL_V5.0.
    // Same SDXL arch, so it shares the `sdxl` engine via a weights swap; differs only in the
    // distilled checkpoint + the few-step recipe: ~5 steps at guidance 1.0 (CFG off). The
    // distillation is baked into the checkpoint (no acceleration LoRA), and the worker pins the
    // engine's `lightning` Euler-trailing sampler for this id (see `generate_stream`). txt2img only
    // (the accel sampler is engine-incompatible with reference/img2img conditioning).
    ModelRow {
        sceneworks_id: "realvisxl_lightning",
        engine_id: "sdxl",
        default_repo: "SG161222/RealVisXL_V5.0_Lightning",
        default_steps: 5,
        default_guidance: 1.0,
        adapter_label: "mlx_sdxl",
    },
    // Kolors (epic 3090, sc-3875) — Kwai-Kolors SDXL-architecture U-Net + ChatGLM3-6B text
    // encoder + SDXL VAE, EulerDiscrete sampler. Real CFG (negative prompt + guidance 5.0).
    // Python `MODEL_TARGETS` / `KolorsDiffusersAdapter` parity: 25 steps, guidance 5.0. The engine
    // `kolors` model (sc-3874) supports the full surface — img2img / ControlNet-pose /
    // IP-Adapter-Plus / Q8/Q4 / LoRA/LoKr — but this base row drives plain T2I (+ quant + LoRA)
    // through `generate_stream`; the advanced conditioning modes are gated to torch by
    // `kolors_mlx_eligible` until their dedicated streams land (subsequent epic-3090 slices).
    ModelRow {
        sceneworks_id: "kolors",
        engine_id: "kolors",
        default_repo: "Kwai-Kolors/Kolors-diffusers",
        default_steps: 25,
        default_guidance: 5.0,
        adapter_label: "mlx_kolors",
    },
    // Chroma (epic 3531, sc-3843) — FLUX.1-schnell-derived DiT, T5-only conditioning. The engine
    // is a TRUE-CFG family: its descriptor advertises `supports_guidance=false` +
    // `supports_negative_prompt=true`, so the CFG scale is forwarded as `true_cfg` (NOT the
    // distilled `guidance` scalar, which the engine rejects) — see [`uses_true_cfg`] /
    // [`resolve_true_cfg`]. HD/Base are full true-CFG (the manifest pre-fills 40 steps + guidance
    // 3.0; the engine's own defaults are 28 steps + 4.0 — the request carries the manifest values).
    // Each SceneWorks id maps 1:1 to the engine registry id of the same name.
    ModelRow {
        sceneworks_id: "chroma1_hd",
        engine_id: "chroma1_hd",
        default_repo: "lodestones/Chroma1-HD",
        default_steps: 40,
        default_guidance: 3.0,
        adapter_label: "mlx_chroma",
    },
    ModelRow {
        sceneworks_id: "chroma1_base",
        engine_id: "chroma1_base",
        default_repo: "lodestones/Chroma1-Base",
        default_steps: 40,
        default_guidance: 3.0,
        adapter_label: "mlx_chroma",
    },
    // Flash is the few-step distilled checkpoint: ~12 Heun steps, CFG baked toward 1.0 (single forward —
    // the negative prompt is effectively inert at true_cfg≈1). It shares the true-CFG descriptor,
    // so `true_cfg` still carries the scale (default 1.0).
    ModelRow {
        sceneworks_id: "chroma1_flash",
        engine_id: "chroma1_flash",
        default_repo: "lodestones/Chroma1-Flash",
        default_steps: 12,
        default_guidance: 1.0,
        adapter_label: "mlx_chroma",
    },
    // SenseNova-U1 (epic 3180, sc-3900) — NEO-Unify: a dense dual-path Qwen3-MoT AR LLM + a
    // flow-matching image generator (no separate VAE / text encoder). Unlike every other family
    // here it uses BOTH CFG knobs: the descriptor's `supports_guidance=true` carries the text CFG
    // via `guidance` (defaults 4.0 base / 1.0 fast), and `supports_true_cfg` carries the it2i
    // image-guidance via `true_cfg` (edit ≈ 1.0 / character ≈ 1.5) — so it is NOT a
    // [`uses_true_cfg`] family (which is for engines that read the *single* CFG knob from
    // `true_cfg`). The descriptor advertises no negative prompt. Plain T2I rides
    // [`generate_stream`]; edit (`Reference`) + Character Studio (`MultiReference`) divert to
    // [`generate_sensenova_edit_stream`] where the dual CFG + reference conditioning are built.
    // `_fast` is the same base weights with the 8-step distill LoRA merged internally at load
    // (`load_fast`); the worker only selects the engine id, the engine resolves + merges the
    // curated distill LoRA itself (no user LoRA slot — `supports_lora=false`). Both ids map 1:1 to
    // the engine registry id of the same name.
    ModelRow {
        sceneworks_id: "sensenova_u1_8b",
        engine_id: "sensenova_u1_8b",
        default_repo: "sensenova/SenseNova-U1-8B-MoT",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_sensenova",
    },
    ModelRow {
        sceneworks_id: "sensenova_u1_8b_fast",
        engine_id: "sensenova_u1_8b_fast",
        default_repo: "sensenova/SenseNova-U1-8B-MoT",
        default_steps: 8,
        default_guidance: 1.0,
        adapter_label: "mlx_sensenova",
    },
    // Microsoft Lens / Lens-Turbo (epic 3164 engine / sc-5105 cutover) — gpt-oss-20b MoE text
    // encoder + 48-layer dual-stream MMDiT + the Flux.2 VAE. Pure **T2I** (the descriptor advertises
    // no conditioning — no img2img / ControlNet / IP), so both ids ride the base [`generate_stream`]
    // path with quant (Q8 default) + LoRA/LoKr. Standard guidance family: `supports_guidance=true` +
    // `supports_negative_prompt=true` (NOT [`uses_true_cfg`]), so the CFG scale flows through
    // `guidance` and the negative prompt is forwarded. `mac_only` — there is no torch fallback on the
    // macOS path (the Python `/opt/lens-venv` sidecar is retired on Mac; Win/Linux/Docker keep it).
    // The two SceneWorks ids map 1:1 to the engine registry ids of the same name and differ only in
    // their step/guidance defaults: base `lens` is 20-step / CFG 5.0, distilled `lens_turbo` is
    // 4-step / guidance 1.0 (≈ no CFG) — Python `MODEL_TARGETS` parity. Each variant resolves its own
    // `microsoft/Lens` / `microsoft/Lens-Turbo` HF snapshot dir.
    ModelRow {
        sceneworks_id: "lens",
        engine_id: "lens",
        default_repo: "microsoft/Lens",
        default_steps: 20,
        default_guidance: 5.0,
        adapter_label: "mlx_lens",
    },
    ModelRow {
        sceneworks_id: "lens_turbo",
        engine_id: "lens_turbo",
        default_repo: "microsoft/Lens-Turbo",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_lens",
    },
    // Bernini still-image companion (epic 4699 / sc-5424) — the image-typed catalog id maps to the
    // SAME engine registry id (`bernini`) the video `bernini` id uses (`Modality::Both`), mirroring
    // the `z_image_edit → z_image_turbo` two-id/one-engine row above. The dedicated
    // `generate_bernini_image_stream` path (image_jobs/bernini.rs) builds the engine request itself
    // — forcing `frames:1` + `video_mode:"t2i"|"i2i"` so the engine returns a single still — so it
    // does NOT ride the generic `generate_stream`; this row supplies the `mlx_model` join the worker
    // uses for `adapter_id` / `mlx_weights_gap` / the descriptor-capability lookup. Engine defaults:
    // 40 steps, guidance (omega_txt) 4.0 (mlx-gen-bernini `FullDefaults`). No LoRA (descriptor
    // `supports_lora: false`). `default_repo` is the turnkey snapshot, but the dedicated path
    // resolves the dir via `resolve_bernini_model_dir` (env / app-managed / download), not this repo.
    ModelRow {
        sceneworks_id: "bernini_image",
        engine_id: "bernini",
        default_repo: "SceneWorks/bernini-mlx",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_bernini",
    },
    // Boogu-Image-0.1 (epic 6387) — native MLX, ungated (Apache-2.0). ~10.3B Lumina-Image-2.0 /
    // OmniGen2-lineage flow-matching DiT + Qwen3-VL-8B condition encoder + FLUX.1 VAE. Three variants,
    // one engine crate (`mlx-gen-boogu`); each id maps 1:1 to its gen_core descriptor id. The turnkey
    // `SceneWorks/boogu-image-mlx` ships pre-packed Q8 `base/ turbo/ edit/` subfolders (default) +
    // `*-bf16/`; `resolve_boogu_model_dir` (image_jobs/base.rs) points the engine at the variant
    // subfolder. The packed weights auto-detect their quant on load, so the worker's Q8 quant spec is
    // a no-op there. Base = true-CFG T2I (50 steps / guidance 4.0).
    ModelRow {
        sceneworks_id: "boogu_image",
        engine_id: "boogu_image",
        default_repo: "SceneWorks/boogu-image-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_boogu",
    },
    // Boogu Turbo — the DMD few-step, CFG-free distilled variant (`turbo/` checkpoint). 4 steps;
    // guidance is INERT (the `boogu_image_turbo` descriptor advertises supports_guidance=false, so
    // `resolve_guidance` returns None and never forwards a value).
    ModelRow {
        sceneworks_id: "boogu_image_turbo",
        engine_id: "boogu_image_turbo",
        default_repo: "SceneWorks/boogu-image-mlx",
        default_steps: 4,
        default_guidance: 0.0,
        adapter_label: "mlx_boogu",
    },
    // Boogu Edit — instruction image-edit (`edit/` checkpoint). The source image is read by the
    // Qwen3-VL vision tower + VAE-encoded into the DiT's spatial reference latent; the prompt is the
    // edit instruction. true-CFG (50 steps / guidance 4.0). The worker's `resolve_boogu_edit` builds
    // the source `Conditioning::Reference` (no mask path).
    ModelRow {
        sceneworks_id: "boogu_image_edit",
        engine_id: "boogu_image_edit",
        default_repo: "SceneWorks/boogu-image-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_boogu",
    },
    // Krea 2 Turbo (epic 7565) — native MLX, CFG-free rectified-flow few-step T2I (Qwen3-VL-4B TE +
    // 28-block single-stream DiT + Qwen-Image VAE). 8 steps; guidance is INERT (the `krea_2_turbo`
    // descriptor advertises supports_guidance=false, so `resolve_guidance` returns None and never
    // forwards a value). Loads the packed Q8 (default) / Q4 turnkey subdir (`krea_model_subdir`).
    ModelRow {
        sceneworks_id: "krea_2_turbo",
        engine_id: "krea_2_turbo",
        default_repo: "SceneWorks/krea-2-turbo-mlx",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_krea",
    },
    // Stable Diffusion 3.5 Large (epic 7841 / sc-7871) — native MLX, gated. 8B MMDiT + triple text
    // encoder (CLIP-L + CLIP-G + T5-XXL) + 16-ch VAE. True-CFG flagship: 28 steps / guidance 3.5 +
    // negative prompt (the `sd3_5_large` descriptor advertises supports_guidance + supports_negative
    // + supports_true_cfg). Installs a packed Q8 dir (`sd3_5_large_quant` converter, model_jobs.rs).
    ModelRow {
        sceneworks_id: "sd3_5_large",
        engine_id: "sd3_5_large",
        default_repo: "stabilityai/stable-diffusion-3.5-large",
        default_steps: 28,
        default_guidance: 3.5,
        adapter_label: "mlx_sd3",
    },
    // SD3.5 Large Turbo (epic 7841 / sc-7871) — the ADD-distilled few-step, CFG-free sibling: same 8B
    // MMDiT + triple TE + 16-ch VAE backbone + snapshot layout, distilled checkpoint. 4 steps; guidance
    // is INERT (the `sd3_5_large_turbo` descriptor advertises supports_guidance=false, so
    // `resolve_guidance` returns None and never forwards a value — the pipeline's `denoise_cfg` skips the
    // uncond forward at guidance 1.0). No negative prompt. The z_image_turbo / boogu / krea turbo pattern.
    ModelRow {
        sceneworks_id: "sd3_5_large_turbo",
        engine_id: "sd3_5_large_turbo",
        default_repo: "stabilityai/stable-diffusion-3.5-large-turbo",
        default_steps: 4,
        default_guidance: 0.0,
        adapter_label: "mlx_sd3",
    },
    // SD3.5 Medium (epic 7841 / sc-7869 M3, wired in sc-7871) — the MMDiT-X variant: 2.5B, 24 joint
    // blocks (first 13 dual-attention), hidden 1536, `pos_embed_max_size` 384. True-CFG like Large but a
    // distinct (smaller) transformer + its own recipe — 40 steps / guidance 5.0 (Stability's card notes
    // Medium is more guidance-sensitive than Large). The `sd3_5_medium` descriptor advertises
    // supports_guidance + supports_negative + supports_true_cfg. Installs a packed dir via the
    // `sd3_5_medium_quant` converter (model_jobs.rs). The generator self-registers via the shared
    // force-link (`use mlx_gen_sd3 as _;` in image_jobs.rs) now that M3 is on mlx-gen main (rev 1a784cd).
    ModelRow {
        sceneworks_id: "sd3_5_medium",
        engine_id: "sd3_5_medium",
        default_repo: "stabilityai/stable-diffusion-3.5-medium",
        default_steps: 40,
        default_guidance: 5.0,
        adapter_label: "mlx_sd3",
    },
    // SANA 1600M 1024px (epic 8485 / sc-8489) — native MLX, NVIDIA non-commercial. NVIDIA's efficient
    // Linear-DiT (ReLU linear-attn + Mix-FFN + NoPE) 1.6B trunk + a gemma-2-2b-it CHI caption encoder +
    // the 32× DC-AE (f32) decoder. True-CFG text-to-image: 20 steps / guidance 4.5 + negative prompt
    // (the `sana_1600m` descriptor advertises supports_guidance + supports_negative + supports_true_cfg).
    // Loads the un-gated `SceneWorks/Sana_1600M_1024px_mlx` MLX snapshot (transformer/ vae/ text_encoder/,
    // the latter bundling the SceneWorks/gemma-2-2b-it TE so the load path resolves one snapshot dir —
    // SanaTextEncoder::from_snapshot reads `<dir>/text_encoder/gemma-2-2b-it.safetensors` + tokenizer.json).
    // Generator self-registers via the shared force-link (`use mlx_gen_sana as _;` in image_jobs.rs);
    // reaches the generic MODEL_TABLE / `generate_stream` path. No runtime quant (the 2-bit quant is NOT
    // ported); ships dense bf16. 32× DC-AE divisor → width/height must be multiples of 32.
    ModelRow {
        sceneworks_id: "sana_1600m",
        engine_id: "sana_1600m",
        default_repo: "SceneWorks/Sana_1600M_1024px_mlx",
        default_steps: 20,
        default_guidance: 4.5,
        adapter_label: "mlx_sana",
    },
];

/// The mlx-gen registry ids of the video generators this worker serves (the engine ids
/// `wan_engine_id` / `ltx_engine_id` / `svd_engine_id` map TO). All-targets so the
/// registry-derived advertisement ([`registry_capabilities`]) can classify a `Video`
/// descriptor without the macOS-only video dispatch in scope.
pub(crate) const VIDEO_ENGINE_IDS: &[&str] = &[
    "wan2_2_ti2v_5b",
    "wan2_2_t2v_14b",
    "wan2_2_i2v_14b",
    "ltx_2_3",
    // The candle LTX provider registers a distinct engine id (`ltx_2_3_distilled`, not the MLX
    // `ltx_2_3`); listed here so the registry-derived `video_generate` advertisement
    // ([`registry_capabilities`]) picks up the candle LTX descriptor too (sc-5097).
    "ltx_2_3_distilled",
    "svd_xt",
];

/// The trainer registry ids this worker serves (the ids `engine_trainer_id` maps TO). Used by
/// [`registry_capabilities`] as the "is this a trainer the worker actually serves" filter: the
/// training capabilities light up when an enabled backend has a registered trainer whose id is one
/// of these. Trainer descriptors DO carry `backend` (sc-4906), so the derivation gates per-backend
/// — a candle trainer lights training up only under `backend_candle_enabled`, an mlx one only under
/// `backend_mlx_enabled` (see the gate in `registry_capabilities`). `lens` is the mlx (sc-5148) +
/// candle (sc-7817) Lens trainer. The mlx backend registers all of these; the candle backend only
/// the subset {`sdxl`, `z_image_turbo`, `lens`, `wan2_2_t2v_14b`} (the Wan 5B / I2V A14B + Kolors /
/// LTX have no candle trainer — `jobs_store::training_job_is_candle_eligible` keeps them off candle).
pub(crate) const TRAINER_IDS: &[&str] = &[
    "z_image_turbo",
    "sdxl",
    "kolors",
    "lens",
    // SD3.5 LoRA-training bases (epic 7841 T3 sc-7884): the engine registers the LoRA/LoKr trainer
    // under the same id as the inference generator of the training base — Large (sc-7883) and the
    // MMDiT-X Medium (sc-7885). mlx-only (no candle SD3 trainer; epic 7982).
    "sd3_5_large",
    "sd3_5_medium",
    "ltx_2_3",
    "wan2_2_ti2v_5b",
    "wan2_2_t2v_14b",
    "wan2_2_i2v_14b",
];

/// A [`ModelRow`] paired with the linked gen_core descriptor for its engine id — the merged
/// view the image path reads. The row supplies the worker-side defaults; the descriptor
/// supplies the capability surface (`supports_guidance` / `supports_negative_prompt` /
/// `backend`) so a row can never drift from the engine's own advertisement (sc-3723).
///
/// Compiled on the macOS MLX path AND the Windows candle lane (sc-5096): the join is purely
/// backend-neutral (`MODEL_TABLE` row + whichever provider crate registered the engine id), so the
/// candle `generate_candle_stream` reuses it exactly like the MLX `generate_stream` — `cfg(target_os)`
/// only decides which provider crate registered the descriptor, not how it is resolved.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct ResolvedModel {
    pub row: &'static ModelRow,
    pub descriptor: gen_core::ModelDescriptor,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ResolvedModel {
    pub fn engine_id(&self) -> &'static str {
        self.row.engine_id
    }
    pub fn default_repo(&self) -> &'static str {
        self.row.default_repo
    }
    pub fn default_steps(&self) -> u32 {
        self.row.default_steps
    }
    pub fn default_guidance(&self) -> f32 {
        self.row.default_guidance
    }
    // The MLX adapter label (`mlx_<family>`). The candle lane reports `candle_<family>` via the free
    // `image_jobs::candle_adapter_label` instead, so this accessor is MLX-path-only — silence the
    // dead-code lint on the candle-only build where the macOS dispatch is cfg'd out.
    #[cfg_attr(
        all(not(target_os = "macos"), feature = "backend-candle"),
        allow(dead_code)
    )]
    pub fn adapter_label(&self) -> &'static str {
        self.row.adapter_label
    }
    /// Whether the engine accepts a guidance scale (descriptor-derived; distilled variants
    /// — z-image-turbo, flux schnell — are `false`).
    pub fn supports_guidance(&self) -> bool {
        self.descriptor.capabilities.supports_guidance
    }
    /// Whether the engine accepts a negative prompt / true CFG (descriptor-derived).
    pub fn supports_negative_prompt(&self) -> bool {
        self.descriptor.capabilities.supports_negative_prompt
    }
    /// Whether the engine advertises any on-the-fly Q4/Q8 quantization (descriptor-derived). The
    /// candle SDXL / sc-5096 families advertise none (dense only); Lens advertises Q4/Q8 (sc-5126).
    /// Used on BOTH lanes: the candle lane has always gated quant on this; the MLX lane gates on it
    /// too as of sc-8489 so SANA (the lone generic-MLX family with `supported_quants: &[]`, whose
    /// `load` rejects any quant) loads dense, while every pre-existing family (all Q4/Q8) is
    /// unaffected.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    pub fn supports_quant(&self) -> bool {
        !self.descriptor.capabilities.supported_quants.is_empty()
    }
    /// Whether the engine accepts LoRA/LoKr adapters (descriptor-derived). Lens is the first candle
    /// family to advertise either (sc-5126); the others advertise neither. Candle-lane-only for the
    /// same reason as [`Self::supports_quant`].
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    pub fn supports_adapters(&self) -> bool {
        self.descriptor.capabilities.supports_lora || self.descriptor.capabilities.supports_lokr
    }
    /// The tensor backend that registered this engine (`"mlx"` | `"candle"`).
    pub fn backend(&self) -> &'static str {
        self.descriptor.backend
    }
}

/// The engine-backed family for a SceneWorks model id, if any — the row joined with its
/// linked gen_core descriptor. `None` when the id is not in [`MODEL_TABLE`] or no provider
/// crate registered its engine id (keeps the existing fail-loud-when-not-MLX behavior).
///
/// Backend-neutral despite the `mlx_` name (sc-5096): on the Windows candle lane the registry holds
/// the candle descriptors, so this resolves the candle engine for `request.model` the same way it
/// resolves the MLX engine on macOS — the candle `generate_candle_stream` calls it directly.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn mlx_model(sceneworks_id: &str) -> Option<ResolvedModel> {
    let row = MODEL_TABLE
        .iter()
        .find(|r| r.sceneworks_id == sceneworks_id)?;
    let descriptor = gen_core::registry::generators()
        .map(|reg| (reg.descriptor)())
        .find(|d| d.id == row.engine_id)?;
    Some(ResolvedModel { row, descriptor })
}

/// The registry-DERIVED subset of the MLX worker's capabilities (sc-3723): exactly the
/// capabilities backed by a linked generator/trainer/captioner descriptor whose backend is
/// enabled in `settings`. Off-macOS the provider crates aren't linked, so the registry is
/// empty and this returns an empty vec — the capability is *absent*, not a runtime failure.
/// A future candle backend lights up here with zero worker changes: it registers descriptors
/// with `backend = "candle"` and (with `backend_candle_enabled`) they are picked up.
///
/// The carve-outs the worker advertises that are NOT expressible as a single registered
/// generator descriptor (ImageEdit/ImageDetail/Vqa/Interleave, the advanced video modes,
/// pose/kps/upscale/person detect+track) stay hardcoded at the [`crate::gpu::mlx_gpu`] call
/// site; this function returns only the descriptor-derived core.
pub(crate) fn registry_capabilities(
    settings: &crate::Settings,
) -> Vec<sceneworks_core::contracts::WorkerCapability> {
    use sceneworks_core::contracts::WorkerCapability as Cap;

    let mut backends: Vec<&'static str> = Vec::new();
    if settings.backend_mlx_enabled {
        backends.push("mlx");
    }
    if settings.backend_candle_enabled {
        backends.push("candle");
    }

    let mut caps: Vec<Cap> = Vec::new();
    let push = |c: Cap, caps: &mut Vec<Cap>| {
        if !caps.contains(&c) {
            caps.push(c);
        }
    };

    for reg in gen_core::registry::generators() {
        let d = (reg.descriptor)();
        if !backends.contains(&d.backend) {
            continue;
        }
        let in_image = MODEL_TABLE.iter().any(|r| r.engine_id == d.id);
        let in_video = VIDEO_ENGINE_IDS.contains(&d.id);
        match d.modality {
            gen_core::Modality::Image if in_image => push(Cap::ImageGenerate, &mut caps),
            gen_core::Modality::Video if in_video => push(Cap::VideoGenerate, &mut caps),
            gen_core::Modality::Both => {
                if in_image {
                    push(Cap::ImageGenerate, &mut caps);
                }
                if in_video {
                    push(Cap::VideoGenerate, &mut caps);
                }
            }
            _ => {}
        }
    }

    // Trainers/captioners now carry `backend` (sc-4906), so gate them per-backend exactly like the
    // generators above — a candle-only trainer no longer lights up under `backend_mlx_enabled`
    // alone, and vice versa. `lora_train` (dry-run plan validation) and `lora_train_execute` (real
    // run) are both served in-process by the same trainer registry, so they light up together.
    if gen_core::registry::trainers().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && TRAINER_IDS.contains(&d.id)
    }) {
        push(Cap::LoraTrain, &mut caps);
        push(Cap::LoraTrainExecute, &mut caps);
    }
    // The JoyCaption captioner registers under the HF repo id (mlx-gen `JOY_CAPTION_MODEL_ID`),
    // not a short name.
    if gen_core::registry::captioners().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && d.id == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    }) {
        push(Cap::TrainingCaption, &mut caps);
    }
    // Dataset Doctor CLIP embedders (sc-6535/sc-6537): mlx-gen-clip registers paired image/text
    // embedders under `clip_vit_l14` + `clip_vit_l14_text`. Advertise `dataset_analysis` only when
    // both are registered on an enabled backend, so the worker cannot claim a caption-alignment job
    // with only half the CLIP pair linked.
    let has_clip_image = gen_core::registry::image_embedders().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && d.id == "clip_vit_l14"
    });
    let has_clip_text = gen_core::registry::text_embedders().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && d.id == "clip_vit_l14_text"
    });
    if has_clip_image && has_clip_text {
        push(Cap::DatasetAnalysis, &mut caps);
    }
    // Prompt-refinement (epic 7153). Both native lanes now run through the unified LLM engine — a
    // generic `core_llm::TextLlm` registered in core-llm's registry (NOT gen_core's), resolved
    // model-first: mlx-llm's `mlx-llama` on macOS (sc-7158), candle-llm's `candle-llama` on the
    // Windows/CUDA candle build (sc-7404). Light up `prompt_refine` when an enabled backend has a
    // core-llm text (non-vision) provider linked — the vision providers (mlx-joycaption / candle-llava)
    // set `supports_vision` and are excluded. The Python torch `PromptRefiner` stays the fallback on
    // platforms with neither.
    //
    // sc-8105: the `image_caption` task (reference image → Ideogram JSON caption) rides on this SAME
    // `PromptRefine` capability + job — it is a payload `task` discriminator, not a separate cap, so it
    // needs no capability-gate change. The gate below correctly keys on the WEIGHTLESS non-vision
    // descriptor because `image_caption` is served by the SAME text+Json provider as plain refinement:
    // `mlx-llama` statically advertises `supports_vision: false` + `[Constraint::Json]` and `can_load`s a
    // Qwen-VL (`qwen3_5`) snapshot, flipping `supports_vision` on only at LOAD time (mlx-llm
    // provider.rs:267). Its loaded `vision` tower then reads the `Content::Image` at GENERATE time.
    // Resolution itself must NOT demand vision (core-llm `select`/`meets` filters on the STATIC
    // descriptor, which has no vision+Json provider for a Qwen-VL snapshot); the worker resolves the
    // image_caption job on the JSON constraint alone (see `prompt_refine_jobs.rs`). Do NOT broaden this
    // gate to admit vision-only providers (e.g. `mlx-joycaption`) — they carry no constraints and would
    // over-advertise `prompt_refine` on a vision-only worker without serving the Json caption path.
    let native_prompt_refine = gen_core::core_llm::textllms().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend.as_str()) && !d.capabilities.supports_vision
    });
    if native_prompt_refine {
        push(Cap::PromptRefine, &mut caps);
    }
    caps
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::contracts::WorkerCapability as Cap;

    // A test Settings with the two backend toggles set; everything else is from_env defaults.
    // (Tests set no backend env vars, so from_env() yields mlx=on / candle=off by default; the
    // helper overrides both explicitly so each case is self-contained regardless of env.)
    fn settings_with_backends(mlx: bool, candle: bool) -> crate::Settings {
        let mut s = crate::Settings::from_env();
        s.backend_mlx_enabled = mlx;
        s.backend_candle_enabled = candle;
        s
    }

    // ── epic 7114 P5 / sc-7126 (+ sc-7432 bespoke coverage): manifest ⊆ engine drift guard ─────────
    // The builtin manifest's advertised sampler/scheduler menu for a model MUST be a subset of what
    // the linked engine actually honors on the ACTIVE backend, or the worker N3-falls the name back to
    // the default (sc-7127) — i.e. the UI offers a knob the engine silently ignores. This test parses
    // the embedded manifest and, for every image model (via `mlx_model` / MODEL_TABLE), every video
    // model (via `video_descriptor`, sc-7296), AND every bespoke out-of-MODEL_TABLE image model
    // (InstantID / PuLID via `bespoke_advertised`, sc-7432) with a source on the active backend, asserts
    // the per-backend-effective menu (base `limits` overridden by `<backend>.limits`) is honored. It
    // checks `mlx` on macOS (where the MLX provider crates are linked) and `candle` on the
    // `backend-candle` build — whichever registry is active — so each backend's truthfulness is enforced
    // on its own lane. `"default"` is the engine-default sentinel, always allowed.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn parse_builtin_models() -> serde_json::Value {
        let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1;
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(text);
        serde_json::from_str(&stripped).expect("parse builtin.models.jsonc")
    }

    // The effective `limits[key]` list for `backend`: the per-backend `<backend>.limits[key]` override
    // if present, else the base `limits[key]`. `None` => the model advertises no list for that axis.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn effective_list(model: &serde_json::Value, backend: &str, key: &str) -> Option<Vec<String>> {
        let pick = |scope: &serde_json::Value| {
            scope
                .get("limits")
                .and_then(|l| l.get(key))
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_owned))
                        .collect::<Vec<_>>()
                })
        };
        model.get(backend).and_then(pick).or_else(|| pick(model))
    }

    /// The engine registry id(s) a video manifest model resolves to, across backends — the union of
    /// the mlx maps (`wan_engine_id` / `ltx_engine_id` / `svd_engine_id` + the native VACE-Fun
    /// dispatch) and the candle map (`candle_video_engine_id`) in `video_jobs`. LTX is backend-split
    /// (`ltx_2_3` on mlx, `ltx_2_3_distilled` on candle) and `ltx_2_3_eros` shares the base engine id
    /// per backend; the resolver lists both and picks whichever the active registry actually holds.
    /// `wan_2_2_vace_fun_14b` is mlx-only (candle has no VACE engine), so it resolves to `None` on the
    /// candle lane and is skipped there.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn video_engine_ids(sceneworks_id: &str) -> &'static [&'static str] {
        match sceneworks_id {
            "wan_2_2" => &["wan2_2_ti2v_5b"],
            "wan_2_2_t2v_14b" => &["wan2_2_t2v_14b"],
            "wan_2_2_i2v_14b" => &["wan2_2_i2v_14b"],
            "wan_2_2_vace_fun_14b" => &["wan2_2_vace_fun_14b"],
            "svd" => &["svd_xt"],
            "ltx_2_3" | "ltx_2_3_eros" => &["ltx_2_3", "ltx_2_3_distilled"],
            _ => &[],
        }
    }

    /// The linked gen-core descriptor for a video manifest model on the ACTIVE backend, or `None`
    /// when no provider crate registered its engine id here. Mirrors [`mlx_model`]'s registry join
    /// for the video ids that live outside [`MODEL_TABLE`] (the image path).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn video_descriptor(sceneworks_id: &str) -> Option<gen_core::ModelDescriptor> {
        let ids = video_engine_ids(sceneworks_id);
        if ids.is_empty() {
            return None;
        }
        gen_core::registry::generators()
            .map(|reg| (reg.descriptor)())
            .find(|d| ids.contains(&d.id))
    }

    /// The `(samplers, schedulers)` menu the engine actually honors for a **bespoke** image model that
    /// lives OUTSIDE [`MODEL_TABLE`] (no `mlx_model` row, no video engine id) — sc-7432. These build
    /// CUSTOM request structs the worker N3-normalizes against [`crate::image_jobs::curated_image_menu`]
    /// (`instantid.rs` / `kolors_*` / `pulid*`), so the guard checks the manifest against the SAME source
    /// of truth and the two never disagree:
    ///   • `instantid_realvisxl` is a bespoke provider (`InstantId::load`) with NO `ModelDescriptor`;
    ///     both engines (mlx #538 / candle #130) honor the curated solver vocab via `Solver::from_name` /
    ///     the additive `denoise_curated` path, so the honored menu IS the curated vocab.
    ///   • `pulid_flux_dev` is the inventory-registered `pulid_flux` Generator on mlx (a real descriptor
    ///     advertising curated + flow_match/linear); on the candle lane it is the bespoke `PulidFlux`
    ///     provider (no descriptor), so fall back to the curated vocab + FLUX's native flow names. Either
    ///     way a superset of the manifest's `default`+curated menu.
    /// Kolors-conditioned is NOT here: `kolors` IS a `MODEL_TABLE` row, so the loop already resolves it
    /// via `mlx_model` and the existing descriptor check covers its (shared) menu.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn bespoke_advertised(sceneworks_id: &str) -> Option<(Vec<String>, Vec<String>)> {
        let curated = || {
            let (samplers, schedulers) = crate::image_jobs::curated_image_menu();
            (
                samplers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                schedulers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            )
        };
        match sceneworks_id {
            "instantid_realvisxl" => Some(curated()),
            "pulid_flux_dev" => gen_core::registry::generators()
                .map(|reg| (reg.descriptor)())
                .find(|d| d.id == "pulid_flux")
                .map(|d| {
                    (
                        d.capabilities
                            .samplers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                        d.capabilities
                            .schedulers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    )
                })
                .or_else(|| {
                    let (mut samplers, mut schedulers) = curated();
                    samplers.push("flow_match".to_string());
                    schedulers.push("linear".to_string());
                    Some((samplers, schedulers))
                }),
            _ => None,
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn manifest_menu_is_subset_of_descriptor() {
        // The active backend's registry: MLX on macOS, candle on the `backend-candle` build.
        let backend = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        };
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        let mut violations: Vec<String> = Vec::new();
        for model in models {
            let Some(id) = model["id"].as_str() else {
                continue;
            };
            // The advertised sampler/scheduler menu the engine honors, from whichever source applies:
            // image models via MODEL_TABLE (`mlx_model`); video models via their engine-id map
            // (`video_descriptor`); the bespoke out-of-MODEL_TABLE image models (InstantID / PuLID,
            // sc-7432) via `bespoke_advertised`. A model with no source on the active backend is skipped
            // (e.g. the mlx-only `wan_2_2_vace_fun_14b` on the candle lane).
            let Some((adv_samplers, adv_schedulers, adv_guidance)) = mlx_model(id)
                .map(|resolved| resolved.descriptor)
                .or_else(|| video_descriptor(id))
                .map(|descriptor| {
                    (
                        descriptor
                            .capabilities
                            .samplers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                        descriptor
                            .capabilities
                            .schedulers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                        // sc-7447: the guidance axis (epic 7434) — the manifest's per-backend
                        // `limits.guidanceMethods` MUST be a subset of what the engine descriptor
                        // honors, exactly like samplers/schedulers.
                        descriptor
                            .capabilities
                            .supported_guidance_methods
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    )
                })
                // Bespoke out-of-MODEL_TABLE models advertise no descriptor guidance vocab; they
                // also advertise no `limits.guidanceMethods` in the manifest, so the guidance axis is
                // simply absent for them (empty advertised set + `effective_list` => None).
                .or_else(|| bespoke_advertised(id).map(|(s, sc)| (s, sc, Vec::new())))
            else {
                continue;
            };
            for (axis, advertised) in [
                ("samplers", &adv_samplers),
                ("schedulers", &adv_schedulers),
                ("guidanceMethods", &adv_guidance),
            ] {
                if let Some(list) = effective_list(model, backend, axis) {
                    for name in list {
                        if name == "default" {
                            continue;
                        }
                        if !advertised.iter().any(|advertised| advertised == &name) {
                            violations.push(format!(
                                "{id}: {backend} {axis} {name:?} not honored by the engine (advertised: {advertised:?})"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            violations.is_empty(),
            "manifest advertises {} sampler/scheduler/guidance name(s) the {backend} engine does not honor:\n  {}",
            violations.len(),
            violations.join("\n  ")
        );
    }

    // sc-7447: the subset guard above only EXERCISES the guidance axis if a model actually advertises a
    // `limits.guidanceMethods` list — otherwise the new axis is vacuously green (the same trap sc-7432
    // closed for the bespoke sampler menus). Pin the CFG++ surface down on the MLX lane: `sdxl` +
    // `realvisxl` (epic 7434 / sc-8256) must advertise `cfg_pp`, and the linked engine descriptor must
    // honor it. Candle has no cfg_pp dispatch yet (sc-8257), so the base `limits` carries no guidance
    // vocab and this is a macOS-only assertion — the candle lane keeps the standard CFG-only surface.
    #[cfg(target_os = "macos")]
    #[test]
    fn sdxl_family_advertises_cfgpp_on_mlx() {
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        for id in ["sdxl", "realvisxl"] {
            let model = models
                .iter()
                .find(|m| m["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("{id}: manifest entry must exist"));
            let methods = effective_list(model, "mlx", "guidanceMethods")
                .unwrap_or_else(|| panic!("{id}: must advertise mlx limits.guidanceMethods"));
            assert!(
                methods.iter().any(|m| m == "cfg_pp"),
                "{id}: mlx must advertise cfg_pp (advertised: {methods:?})"
            );
            let descriptor = mlx_model(id)
                .map(|r| r.descriptor)
                .unwrap_or_else(|| panic!("{id}: must resolve an mlx descriptor"));
            assert!(
                descriptor
                    .capabilities
                    .supported_guidance_methods
                    .contains(&"cfg_pp"),
                "{id}: engine descriptor must honor cfg_pp (advertised: {:?})",
                descriptor.capabilities.supported_guidance_methods
            );
        }
    }

    // sc-7432: the subset guard above only EXERCISES the bespoke out-of-MODEL_TABLE models if
    // `bespoke_advertised` resolves a menu for them — a `None` would silently skip them and leave the
    // guard vacuously green. Pin that down: both bespoke ids must resolve a non-empty menu that includes
    // the solvers their manifest entries advertise (euler/heun), on whichever backend is active.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn bespoke_models_resolve_a_curated_menu() {
        for id in ["instantid_realvisxl", "pulid_flux_dev"] {
            let (samplers, schedulers) = bespoke_advertised(id)
                .unwrap_or_else(|| panic!("{id}: bespoke_advertised must resolve an engine menu"));
            assert!(!samplers.is_empty(), "{id}: sampler menu must be non-empty");
            assert!(
                !schedulers.is_empty(),
                "{id}: scheduler menu must be non-empty"
            );
            // The manifest advertises euler + heun for both; the engine must honor them.
            for solver in ["euler", "heun"] {
                assert!(
                    samplers.iter().any(|advertised| advertised == solver),
                    "{id}: engine must honor {solver:?} (advertised: {samplers:?})"
                );
            }
        }
    }

    // An MLX-backed stub generator registered into the same `inventory` registry the real
    // provider crates use, with an id that IS in MODEL_TABLE (`z_image_turbo`). On Linux/Windows
    // no real provider crate is linked, so this is the only generator the derivation sees — which
    // is exactly the point: it proves the derivation works off-macOS purely from the registry.
    fn stub_mlx_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "z_image_turbo",
            family: "test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }
    fn stub_mlx_load(_spec: &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    inventory::submit! {
        gen_core::registry::ModelRegistration { descriptor: stub_mlx_descriptor, load: stub_mlx_load }
    }

    // A candle-backed stub whose id is also in MODEL_TABLE (`sdxl`): proves a Windows/candle
    // backend lights up `image_generate` with zero worker code changes once its backend is
    // enabled — purely by registering a descriptor with `backend = "candle"`.
    fn stub_candle_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "sdxl",
            family: "test",
            backend: "candle",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }
    fn stub_candle_load(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    inventory::submit! {
        gen_core::registry::ModelRegistration { descriptor: stub_candle_descriptor, load: stub_candle_load }
    }

    // An MLX-backed stub whose id is NOT in MODEL_TABLE / VIDEO_ENGINE_IDS: proves an unknown
    // engine id contributes no capability (absence, not a runtime failure).
    fn stub_unknown_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "not_a_sceneworks_engine",
            family: "test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }
    fn stub_unknown_load(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    inventory::submit! {
        gen_core::registry::ModelRegistration { descriptor: stub_unknown_descriptor, load: stub_unknown_load }
    }

    // A candle-backed core-llm `TextLlm` stub (backend "candle", non-vision): proves the prompt-refine
    // derivation lights up `prompt_refine` purely from a registered `core_llm::TextLlm` descriptor on an
    // enabled backend (sc-7404), so the default (Linux) CI lane exercises it without linking a real
    // provider crate. The real lanes register mlx-llama / candle-llama into this SAME core-llm registry.
    fn stub_textllm_descriptor() -> gen_core::core_llm::TextLlmDescriptor {
        gen_core::core_llm::TextLlmDescriptor {
            id: "prompt_refine".to_string(),
            family: "llama".to_string(),
            backend: "candle".to_string(),
            capabilities: gen_core::core_llm::TextLlmCapabilities::default(),
        }
    }
    fn stub_textllm_load(
        _spec: &gen_core::core_llm::LoadSpec,
    ) -> gen_core::core_llm::Result<Box<dyn gen_core::core_llm::TextLlm>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    fn stub_textllm_can_load(_spec: &gen_core::core_llm::LoadSpec) -> bool {
        false
    }
    inventory::submit! {
        gen_core::core_llm::TextLlmRegistration {
            descriptor: stub_textllm_descriptor,
            load: stub_textllm_load,
            can_load: stub_textllm_can_load,
            // No per-snapshot vision probe (sc-8077 / mlx-llm 7041411): this candle text-only stub
            // never serves vision, so it stays as the static `supports_vision=false` descriptor says.
            weightless_vision: None,
        }
    }

    // A candle-backed stub `Trainer` (backend "candle") registered under an id that IS in TRAINER_IDS
    // (`sdxl`): proves a Windows/candle backend lights up `lora_train` + `lora_train_execute` from a
    // registered `backend = "candle"` trainer descriptor alone (sc-7817), so the default (Linux) CI
    // lane exercises the per-backend training gate without linking a real provider crate. The real
    // lanes register the candle-gen-{sdxl,z-image,lens,wan} trainers into this SAME registry.
    //
    // Compiled out of the `backend-candle` build: there the REAL `candle-gen-sdxl` trainer registers
    // `sdxl` already (so the capability test still lights up off it), and a stub `sdxl` would be a
    // DUPLICATE — `load_trainer("sdxl")` is first-wins, so in --release (where the candle GPU smokes
    // run, debug_assert off) the smoke could resolve THIS `unimplemented!()` stub instead of the real
    // trainer. On macOS the real trainers are `backend = "mlx"`, so the candle stub is still needed
    // there to exercise the candle branch, and no macOS smoke loads `sdxl`, so no collision.
    #[cfg(any(target_os = "macos", not(feature = "backend-candle")))]
    fn stub_candle_trainer_descriptor() -> gen_core::TrainerDescriptor {
        gen_core::TrainerDescriptor {
            id: "sdxl",
            family: "test",
            backend: "candle",
            modality: gen_core::Modality::Image,
            supports_lora: true,
            supports_lokr: true,
        }
    }
    #[cfg(any(target_os = "macos", not(feature = "backend-candle")))]
    fn stub_candle_trainer_load(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Trainer>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    #[cfg(any(target_os = "macos", not(feature = "backend-candle")))]
    inventory::submit! {
        gen_core::registry::TrainerRegistration { descriptor: stub_candle_trainer_descriptor, load: stub_candle_trainer_load }
    }

    #[test]
    fn mlx_enabled_advertises_image_generate_from_registry() {
        let caps = registry_capabilities(&settings_with_backends(true, false));
        assert!(
            caps.contains(&Cap::ImageGenerate),
            "MLX stub generator (z_image_turbo) should derive image_generate"
        );
    }

    #[test]
    fn mlx_disabled_drops_mlx_derived_image_generate() {
        // With both backends off, the mlx + candle stubs are filtered out → no image_generate.
        let caps = registry_capabilities(&settings_with_backends(false, false));
        assert!(
            !caps.contains(&Cap::ImageGenerate),
            "no enabled backend ⇒ no derived image_generate"
        );
    }

    // sc-8320: the base (non-distilled) `z_image` t2i row resolves to its OWN engine id (not Turbo's),
    // points at the `Tongyi-MAI/Z-Image` base snapshot, and carries the undistilled defaults (real CFG
    // guidance 4.0, ~50 steps) — proving it is selectable and routes to the base path distinct from
    // `z_image_turbo` (which stays 8-step, CFG-free at `Tongyi-MAI/Z-Image-Turbo`).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn z_image_base_row_is_distinct_from_turbo() {
        let base = MODEL_TABLE
            .iter()
            .find(|row| row.sceneworks_id == "z_image")
            .expect("z_image base MODEL_TABLE row");
        assert_eq!(base.engine_id, "z_image");
        assert_eq!(base.default_repo, "Tongyi-MAI/Z-Image");
        assert_eq!(base.default_steps, 50);
        assert!((base.default_guidance - 4.0).abs() < f32::EPSILON);
        assert_eq!(base.adapter_label, "mlx_z_image");

        let turbo = MODEL_TABLE
            .iter()
            .find(|row| row.sceneworks_id == "z_image_turbo")
            .expect("z_image_turbo MODEL_TABLE row");
        // The base must NOT collapse onto the Turbo engine id / repo / step+CFG defaults.
        assert_ne!(base.engine_id, turbo.engine_id);
        assert_ne!(base.default_repo, turbo.default_repo);
        assert_ne!(base.default_steps, turbo.default_steps);
        assert_ne!(base.default_guidance, turbo.default_guidance);
    }

    #[test]
    fn candle_backend_lights_up_with_zero_worker_changes() {
        // candle enabled (mlx off) ⇒ the candle `sdxl` stub alone derives image_generate.
        let on = registry_capabilities(&settings_with_backends(false, true));
        assert!(
            on.contains(&Cap::ImageGenerate),
            "an enabled candle backend should derive image_generate from its descriptor alone"
        );
        // candle disabled ⇒ that descriptor contributes nothing.
        let off = registry_capabilities(&settings_with_backends(false, false));
        assert!(!off.contains(&Cap::ImageGenerate));
    }

    #[test]
    fn candle_textllm_lights_up_prompt_refine() {
        // candle enabled ⇒ the candle core-llm `TextLlm` stub (non-vision) derives the PromptRefine cap.
        let on = registry_capabilities(&settings_with_backends(false, true));
        assert!(
            on.contains(&Cap::PromptRefine),
            "an enabled candle backend should derive prompt_refine from its core-llm TextLlm descriptor"
        );
        // both off ⇒ nothing (neither the candle stub nor — on macOS — the real mlx twin is enabled).
        let off = registry_capabilities(&settings_with_backends(false, false));
        assert!(!off.contains(&Cap::PromptRefine));
    }

    #[test]
    fn candle_backend_lights_up_lora_train_execute() {
        // candle enabled ⇒ the candle `sdxl` trainer stub (backend "candle", id in TRAINER_IDS)
        // derives BOTH the dry-run `lora_train` and the real-run `lora_train_execute` caps — the
        // off-Mac training cutover (sc-7817). They light up together (same in-process trainer registry).
        let on = registry_capabilities(&settings_with_backends(false, true));
        assert!(
            on.contains(&Cap::LoraTrain),
            "an enabled candle backend with a registered trainer should derive lora_train"
        );
        assert!(
            on.contains(&Cap::LoraTrainExecute),
            "an enabled candle backend with a registered trainer should derive lora_train_execute"
        );
        // both off ⇒ no training caps from the candle trainer (and on macOS the real mlx trainers are
        // filtered out too, since neither backend is enabled).
        let off = registry_capabilities(&settings_with_backends(false, false));
        assert!(!off.contains(&Cap::LoraTrain));
        assert!(!off.contains(&Cap::LoraTrainExecute));
    }

    // Off-macOS the ONLY trainer linked is the candle stub above (backend "candle"); no real mlx
    // trainer crate is linked. So an mlx-only worker must NOT advertise training off a candle trainer
    // — proving the per-backend gate (sc-4906) actually keys on `descriptor.backend`. On macOS the
    // real mlx trainers ARE linked, so an mlx-only worker legitimately advertises training and this
    // isolation no longer holds (hence the cfg gate).
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn mlx_only_does_not_advertise_training_from_a_candle_trainer() {
        let mlx_only = registry_capabilities(&settings_with_backends(true, false));
        assert!(
            !mlx_only.contains(&Cap::LoraTrainExecute),
            "off-macOS the only trainer is candle-backed, so an mlx-only worker must not advertise \
             lora_train_execute"
        );
    }

    // Off-macOS no real mlx prompt_refine provider is linked (only the candle stub above), so an
    // mlx-only worker must not advertise prompt_refine.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn mlx_only_does_not_advertise_prompt_refine_without_the_mlx_twin() {
        let mlx_only = registry_capabilities(&settings_with_backends(true, false));
        assert!(
            !mlx_only.contains(&Cap::PromptRefine),
            "off-macOS there is no mlx prompt_refine provider, so an mlx-only worker must not \
             advertise it"
        );
    }

    // On macOS the native mlx prompt_refine provider (sc-5552, force-linked in prompt_refine_jobs.rs)
    // is in the registry, so an mlx-only worker DOES advertise prompt_refine — the MLX twin of the
    // candle path (sc-5525).
    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_only_advertises_prompt_refine_via_the_mlx_twin() {
        let mlx_only = registry_capabilities(&settings_with_backends(true, false));
        assert!(
            mlx_only.contains(&Cap::PromptRefine),
            "the sc-5552 mlx prompt_refine twin should light up prompt_refine on an mlx-only worker"
        );
    }

    // Off-macOS the registry holds ONLY these three stubs (no real provider crate is linked), so
    // we can prove the unknown-id stub contributes nothing: with mlx enabled the unknown Image
    // stub (id not in MODEL_TABLE) must not derive image_generate by itself, and since no
    // video stub exists, video_generate is absent entirely — absence, not a runtime failure. On
    // macOS the real Wan/LTX/SVD engines are linked, so video_generate legitimately exists and
    // this isolation no longer holds.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unknown_engine_id_contributes_no_capability() {
        let caps = registry_capabilities(&settings_with_backends(true, false));
        // image_generate is present here (the in-table z_image_turbo mlx stub), but the unknown
        // id adds nothing — and it never introduces video_generate (no video stub registered).
        assert!(
            !caps.contains(&Cap::VideoGenerate),
            "an unknown mlx engine id must not derive video_generate"
        );
    }
}
