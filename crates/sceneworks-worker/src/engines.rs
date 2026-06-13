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
        default_repo: "Qwen/Qwen-Image",
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
    // Flash is the few-step distilled checkpoint: ~8 steps, CFG baked toward 1.0 (single forward —
    // the negative prompt is effectively inert at true_cfg≈1). It shares the true-CFG descriptor,
    // so `true_cfg` still carries the scale (default 1.0).
    ModelRow {
        sceneworks_id: "chroma1_flash",
        engine_id: "chroma1_flash",
        default_repo: "lodestones/Chroma1-Flash",
        default_steps: 8,
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

/// The mlx-gen trainer registry ids this worker can train (the ids `engine_trainer_id`
/// maps TO). Trainer descriptors carry no `backend` field at this gen_core rev (a Phase-0
/// limitation), so the derivation gates the training capabilities on "any backend enabled"
/// + a registered trainer whose id is one of these, rather than on a per-trainer backend.
pub(crate) const TRAINER_IDS: &[&str] = &[
    "z_image_turbo",
    "sdxl",
    "kolors",
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
    all(target_os = "windows", feature = "backend-candle")
))]
pub(crate) struct ResolvedModel {
    pub row: &'static ModelRow,
    pub descriptor: gen_core::ModelDescriptor,
}

#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
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
        all(target_os = "windows", feature = "backend-candle"),
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
    all(target_os = "windows", feature = "backend-candle")
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
