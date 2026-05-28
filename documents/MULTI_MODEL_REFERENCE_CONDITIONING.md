# Multi-Model Reference Conditioning Contract

> **Status**: Shipped. Epic [2003](https://app.shortcut.com/trefry/epic/2003) (sc-2004 through sc-2018). Enforced by CI (sc-2018 audits).
>
> **What this is**: The model-agnostic contract that lets Character Studio render a recognizable subject from a single reference image across diverse backbone families — IP-Adapter (Kolors, SDXL, RealVisXL, FLUX), face-specialized identity (InstantID), and edit-as-reference (Qwen-Image-Edit / 2509). Distilled from the shipped code, not designed up-front.
>
> **Audience**: Engineers wiring a new reference-capable image backbone into SceneWorks.

---

## 1. The outcome

Character Studio's "With character" mode lets the user pick an approved reference image of a character and have any compatible backbone generate identity-preserving variations from it. From the user's perspective the experience is uniform — pick a character, pick a model, tune a slider or two, generate — but four distinct mechanism classes operate underneath:

| Mechanism | Backbones | Conditioning lever |
|---|---|---|
| **IP-Adapter** (image embedding into cross-attention) | Kolors, SDXL, RealVisXL, FLUX | `ipAdapterScale` (FLUX adds `trueCfgScale`) |
| **Face-specialized identity** (ArcFace embedding + IdentityNet ControlNet) | InstantID (RealVisXL) | `ipAdapterScale` + `controlnetConditioningScale` + view angle + pose library |
| **Edit-as-reference** (instruction-edit pipeline, reference fed as `image=`) | Qwen-Image-Edit / 2509 | `trueCfgScale` only |
| **(Declined)** Interleave / unified multimodal | SenseNova-U1 (sc-2015 NO-GO) | n/a — model card disclaims the use case |

The contract has four layers: worker engine, manifest declarations, web payload schema, and Rust passthrough. Each layer keeps backbones interchangeable without copy-pasting per-model branches.

---

## 2. Worker engine layer — `apps/worker/scene_worker/image_adapters.py`

### 2.1 Per-model registry (`MODEL_TARGETS`)

Each reference-capable model declares its engine via one of two optional blocks:

```python
"sdxl": {
    "label": "Stable Diffusion XL",
    "family": "sdxl",
    "adapter": "sdxl_diffusers",
    # IP-Adapter family: per-pipeline `pipe.load_ip_adapter(...)` + per-request scale.
    "ipAdapter": {
        "repo": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "encoderSubfolder": "models/image_encoder",
        # FLUX-specific extras (FluxIPAdapterMixin API differs from SDXL/Kolors):
        # "imageEncoderRepo": "openai/clip-vit-large-patch14",
        # "imageEncoderSubfolder": "",
    },
    ...
},

"instantid_realvisxl": {
    "label": "InstantID (RealVisXL)",
    "family": "sdxl",
    "adapter": "instantid_sdxl",
    # Face-identity family: ArcFace embedding from insightface + IdentityNet
    # ControlNet driving 5-point landmarks. Strictly reference-driven (no T2I).
    "instantId": {
        "repo": "InstantX/InstantID",
        "controlnetSubfolder": "ControlNetModel",
        "ipAdapter": "ip-adapter.bin",
    },
    ...
},

"qwen_image_edit_2509": {
    "label": "Qwen Image Edit (2509)",
    "family": "qwen-image",
    "adapter": "qwen_image",
    # Edit-as-reference family: no per-model block. Dispatch is mode-based; the
    # adapter's pipeline class is chosen by model id (_EDIT_PIPELINE_BY_MODEL).
    ...
},
```

### 2.2 Adapter class shape

Every reference-capable adapter exposes the same public surface so the runtime dispatcher (`create_image_adapter`) stays mechanism-agnostic:

```python
class XxxDiffusersAdapter:
    id = "xxx_diffusers"

    @staticmethod
    def _use_ip_adapter(request: ImageRequest) -> bool:
        # IP-Adapter / face-ID family gate.
        return request.mode != "edit_image" and bool(request.reference_asset_id)

    # ...or for edit-style backbones (Qwen):
    @staticmethod
    def _use_reference(request: ImageRequest) -> bool:
        return request.mode == "character_image" and bool(request.reference_asset_id)

    def _load_pipeline(self, ...):
        # Cache key includes the reference state — a plain-T2I pipe and an
        # IP-Adapter-loaded pipe are NOT interchangeable.
        use_ip_adapter = self._use_ip_adapter(request)
        text_state_matches = use_img2img or self._text_ip_adapter == use_ip_adapter
        if cached_pipe is not None and self._loaded_repo == repo and text_state_matches:
            return cached_pipe
        ...
        if use_ip_adapter:
            # Encoder load + load_ip_adapter + set_ip_adapter_scale.
            # API surface differs by family:
            #   - SDXL/Kolors: pass image_encoder=... to from_pretrained, then
            #     pipe.load_ip_adapter(repo, subfolder=..., weight_name=...,
            #     image_encoder_folder=None).
            #   - FLUX (FluxIPAdapterMixin): pipe.load_ip_adapter(repo,
            #     weight_name=..., image_encoder_pretrained_model_name_or_path=...,
            #     image_encoder_subfolder=..., image_encoder_dtype=...).
            ...
        self._text_ip_adapter = use_ip_adapter  # cache-key persistence

    def _run_pipeline(self, ..., project_path, ...):
        ...
        if self._use_ip_adapter(request):
            kwargs["ip_adapter_image"] = load_reference_image(project_path, request.reference_asset_id)
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
            # FLUX adds:
            # kwargs["negative_prompt"] = request.negative_prompt
            # kwargs["true_cfg_scale"] = self._true_cfg_scale(request)

    def _ip_adapter_scale(self, request: ImageRequest) -> float:
        try:
            scale = float(request.advanced.get("ipAdapterScale", DEFAULT))
        except (TypeError, ValueError):
            return DEFAULT
        return max(0.0, min(1.0, scale))
```

For Qwen the same skeleton applies, but `_run_pipeline` passes the reference via `image=` (the edit-pipeline kwarg) rather than `ip_adapter_image=`, and the variation knob is `true_cfg_scale` exclusively (the IP-Adapter scale slider would be a no-op).

### 2.3 Shared helpers

- **`load_reference_image(project_path, reference_asset_id) -> PIL.Image.Image`** — resolves an asset id to a `PIL.Image` via `find_asset_media_path`. Used by every reference branch. Native resolution, no resize (encoder handles preprocessing).
- **`filter_call_kwargs(pipe, kwargs)`** — introspects `pipe.__call__`'s named parameters and drops kwargs the pipeline can't accept. Critical for the reference branch because it lets us pass `ip_adapter_image=` / `true_cfg_scale=` unconditionally; pipelines without those params silently skip them. The flip side: **named params in the pipeline signature are required to test reference wiring** — `FakePipe(**kwargs)` strips everything because var-keyword params don't appear in `inspect.signature`. See the FLUX / Qwen test pattern in `tests/test_worker_runtime.py` (named-param `FakePipe` retains the kwargs).

---

## 3. Manifest layer — `config/manifests/builtin.models.jsonc`

The manifest carries the user-facing surface: capability flags, declarative UX hints, prompt guides, license / gating posture.

### 3.1 Capability

Every reference-capable model declares `character_image` in **both** `capabilities` and `ui.recommendedFor`. The picker filters models against `capabilities` when the user enters "With character" mode; `recommendedFor` drives the suggestion ranking.

```jsonc
"capabilities": ["text_to_image", "edit_image", "character_image", "style_variations"],
...
"ui": {
  "recommendedFor": ["text_to_image", "edit_image", "character_image", "style_variations"],
  ...
}
```

The audit in `tests/test_worker_runtime.py::test_character_image_capability_implies_engine_or_tuning_declaration` enforces that any model claiming this capability has the engine wiring or tuning declaration to back it up (sc-2018).

### 3.2 Declarative tuning hints (`ui.*`)

The picker dynamically renders sliders / controls based on per-model declarations. New backbones almost never require picker code changes — they declare their tuning shape:

| Key | Type | Adds to picker | Worker reads |
|---|---|---|---|
| `ui.referenceStrengthDefault` | `float` | Default for the IP-Adapter "Reference strength" slider | `advanced.ipAdapterScale` |
| `ui.identityStructure` | `{ label, default, min, max, step }` | Second slider for IdentityNet structure lock | `advanced.controlnetConditioningScale` |
| `ui.variationStrength` | `{ label, default, min, max, step }` | Variation slider (FLUX, Qwen) | `advanced.trueCfgScale` |
| `ui.hideReferenceStrength` | `boolean` | **Hides** the IP-Adapter Reference-strength slider (Qwen — model doesn't use it) | (drops `ipAdapterScale` from payload) |
| `ui.viewAngles` | `[{ id, label }]` | InstantID head-angle dropdown | `advanced.viewAngle` |
| `ui.poseLibrary` | `boolean` | Toggles InstantID OpenPose library panel | `advanced.poses`, `advanced.faceRestore`, `advanced.bodyPoseSet` |

The audit `test_hide_reference_strength_models_declare_a_variation_knob` (sc-2018) enforces that `hideReferenceStrength: true` is accompanied by `variationStrength` — otherwise the picker would surface no identity-tuning control.

### 3.3 License + gating

The manifest's `gated` / `credentialHost` / `licenseUrl` fields drive the desktop's gated-banner UX. For reference-capable models the license posture matters because the IP-Adapter weights typically inherit the base model's license:

- **Apache-2.0 (commercial-OK, ungated)**: Kolors IP-Adapter-Plus, RealVisXL openrail++, Qwen-Image-Edit / 2509.
- **CreativeML OpenRAIL++-M (commercial-OK, ungated)**: SDXL base + h94 IP-Adapter weights.
- **FLUX.1 [dev] NC + gated**: flux_dev base AND the XLabs IP-Adapter weights (which were trained on flux_dev). Treat the same — gated, NC label.
- **Non-commercial-only weight licenses are disqualifying for built-ins** (project rule, [SceneWorks Is Open Source](sceneworks_is_open_source.md)), except where SceneWorks already exposes the base model under that posture (flux_dev).

---

## 4. Web payload schema — `apps/web/src/screens/ImageStudio.jsx`

The Image Studio picker assembles `advanced.*` keys based on the per-model `ui.*` declarations. The contract is purely declarative: no per-model branching in JSX.

Submit payload (only the reference-relevant keys, gated on `mode === "character_image"` + a reference attached):

| `advanced.*` key | Sent when | Origin |
|---|---|---|
| `ipAdapterScale: float` | NOT `ui.hideReferenceStrength` | Reference-strength slider state |
| `controlnetConditioningScale: float` | `ui.identityStructure` declared | Identity-structure slider state |
| `trueCfgScale: float` | `ui.variationStrength` declared | Variation slider state |
| `viewAngle: string` | `ui.viewAngles` + a selection + no pose library override | View-angle dropdown |
| `poses: string[]`, `faceRestore: bool` | `ui.poseLibrary` + selections | Pose library state |
| `angleSet: bool` | Character Studio "Angle set" panel only (sc-2050) | Character Studio panel |

State is persisted per-workspace via `useStudioSettingsWriter` ([useStudioSettings.js](../apps/web/src/hooks/useStudioSettings.js)) so reference tuning survives across sessions.

---

## 5. Rust layer — `apps/rust-api/src/` + `crates/sceneworks-core/src/`

The Rust API is the queue and manifest server; it doesn't run inference. Two surfaces matter for the contract:

- **`lora_family.rs::model_capabilities_for_type_and_family(type, family)`** — fallback capability list when a custom model omits `capabilities`. Updated by sc-2005 to drop the dishonest `character_image` claim from z-image's default. The audit `test_models_with_engine_block_advertise_character_image` (sc-2018) guards against the reverse drift.
- **`ui.*` freeform passthrough** — Rust deserializes `ui` as `serde_json::Value` and ships it to the web client untouched. Adding a new `ui.someNewSlider` field requires no Rust changes; the picker reads it directly from the model entry.

The `referenceAssetId` field on the job DTO is the only non-freeform reference-conditioning hook: it's surfaced as a typed field on `CreateImageJobRequest` so the picker can pass it through (with permission validation) without packing it into `advanced`.

---

## 6. CI enforcement (sc-2018)

Three audits ride in `tests/test_worker_runtime.py`:

1. **`test_character_image_capability_implies_engine_or_tuning_declaration`** — for every builtin model declaring `character_image`, assert the worker has an engine block (`ipAdapter` / `instantId` in `MODEL_TARGETS`) or the manifest declares `ui.variationStrength`. Catches the "advertised but unwired" bug shape from sc-2005.
2. **`test_models_with_engine_block_advertise_character_image`** — for every `MODEL_TARGETS` entry with an engine block, assert the builtin manifest advertises `character_image`. Catches the "wired but hidden" bug shape.
3. **`test_hide_reference_strength_models_declare_a_variation_knob`** — `hideReferenceStrength: true` must be accompanied by `variationStrength`.

Plus `scripts/check-scaffold.mjs::assertCharacterImageTuningSurface()` runs audit #3 at build time so it fails before pytest does.

---

## 7. How to add a new reference backbone

Concrete recipe based on the sc-2007 → sc-2011 → sc-2014 cadence. Step-by-step for an IP-Adapter-family backbone; analogous for face-identity or edit-as-reference. Plan on 6–8 files touched, ~250–400 lines of new code, no Rust changes unless the backbone introduces a new typed payload field.

1. **Spike** (desktop-first per sc-2010 / sc-2013 pattern): pick the upstream adapter weights, confirm license posture, identify the diffusers API shape (`FluxIPAdapterMixin` vs `IPAdapterMixin` vs custom pipeline), estimate Mac MPS memory peak. Capture findings in a Shortcut story comment.
2. **Worker engine** ([apps/worker/scene_worker/image_adapters.py](../apps/worker/scene_worker/image_adapters.py)): add the `MODEL_TARGETS` entry with an `ipAdapter` block (or `instantId` for face-ID); add `_use_ip_adapter` / `_use_reference` static gate, `_text_ip_adapter` cache flag, `_ip_adapter_scale` + (for guidance-distilled backbones) `_true_cfg_scale` helpers; modify `_load_pipeline` for encoder load + `pipe.load_ip_adapter`; modify `_run_pipeline` to pass `ip_adapter_image=` (or `image=` for edit-style). Mirror the SDXL adapter as the template — it's the cleanest reference.
3. **Manifest** ([config/manifests/builtin.models.jsonc](../config/manifests/builtin.models.jsonc)): add the model entry with `character_image` in `capabilities` + `ui.recommendedFor`; add per-model `ui.*` tuning hints (`referenceStrengthDefault`, `variationStrength`, etc.) as the mechanism requires; declare license / gating posture.
4. **Web constants** ([apps/web/src/constants.js](../apps/web/src/constants.js)): mirror the manifest entry (fallback for desktop-disconnected mode).
5. **Prompt guide** (`apps/web/public/prompt-guides/<id>.md`): scene-flexible prompt shape, comparison vs other backbones, tunable defaults.
6. **Tests** ([tests/test_worker_runtime.py](../tests/test_worker_runtime.py)): mirror the SDXL or FLUX test set — `_model_target_defaults`, `_use_ip_adapter` gate, `_ip_adapter_scale` default + clamp, `_run_pipeline` torch-free verification with a **named-param** `FakePipe` (var-keyword params fail `filter_call_kwargs`).
7. **Run the sc-2018 audits** — they catch drift between (1)–(4). The scaffold check covers the picker-symmetry half at build time.
8. **PR** with the standard test plan checklist: CI passes, Mac MPS E2E owed (the engine code is correct without a real hardware run, but identity quality + memory budget can only be confirmed there).

If the backbone introduces a new payload field (e.g., a third tuning slider beyond `variationStrength`), the contract extension is:

- Pick the `advanced.*` key name.
- Add a declarative `ui.*` hint that mirrors `ui.variationStrength` or `ui.identityStructure`.
- Render the slider conditionally in `ImageStudio.jsx` (same shape as the existing two).
- Update the audit `test_hide_reference_strength_models_declare_a_variation_knob` if the new knob is the model's only tuning control.

Rust changes are only needed if the new payload field needs typed validation (e.g., a structured object rather than a scalar). The freeform `advanced` and `ui` passthrough handles the common case.

---

## Open work (epic 2003)

- **sc-2012 PuLID-FLUX** — face-identity engine for FLUX. Spike-prep posted; awaiting a hardware spike per [sc-2012 comment](https://app.shortcut.com/trefry/story/2012). Will follow the InstantID (sc-2009) pattern with a custom vendored pipeline.
- **SenseNova-U1** (sc-2015/2016) — declined for this epic per the [sc-2015 NO-GO](https://app.shortcut.com/trefry/story/2015) (model card disclaims subject-consistency use cases). Revisit if upstream releases a subject-consistency LoRA or graduates the interleave path from Beta.

## References

- [Epic 2003 — SceneWorks: Multi-Model Character Studio Reference & Identity Conditioning](https://app.shortcut.com/trefry/epic/2003)
- Engine templates: [SdxlDiffusersAdapter](../apps/worker/scene_worker/image_adapters.py), [InstantIDAdapter](../apps/worker/scene_worker/instantid_adapter.py), [FluxDiffusersAdapter](../apps/worker/scene_worker/image_adapters.py), [QwenImageAdapter](../apps/worker/scene_worker/image_adapters.py)
- Picker: [ImageStudio.jsx](../apps/web/src/screens/ImageStudio.jsx)
- Audits: [tests/test_worker_runtime.py](../tests/test_worker_runtime.py) (search for "Multi-model Character Studio reference matrix")
