# Vendored third-party packages

These are not pip-installable (no `pyproject.toml`/`setup.py`), so they are
vendored here and placed on `sys.path` by the adapter that needs them. This
directory is bundled into both the Docker worker image (via `PYTHONPATH`) and
the desktop Python sidecar (via `stage-python.mjs`, which copies `scene_worker`
wholesale).

> **Upstream commit pins** for every package here live in
> [`VENDOR_PINS.md`](./VENDOR_PINS.md) — the machine-checkable SHA ledger with
> blob-hash verification steps. Keep the two files in sync when re-vendoring.

## lens

Minimal inference package for Microsoft's **Lens / Lens-Turbo** text-to-image
model. Importing `lens` registers the custom `LensPipeline`,
`LensTransformer2DModel`, and `LensGptOssEncoder` classes into the `diffusers`
and `transformers` namespaces that the model's `model_index.json` references —
there is no published pip package and no `trust_remote_code` path.

- Source: https://github.com/microsoft/Lens
- Commit: `5bf0f0cea2f4bc32ebb2b7ed2ef96d5e88b701e0` (2026-05-22)
- License: MIT (see `lens/LICENSE`)
- Consumed by: `scene_worker/image_adapters.py::LensTurboAdapter`
- Requires (pinned in `requirements.txt`): `diffusers==0.38.0`,
  `transformers>=5.8,<6`, `torch>=2.11`, `einops`, and `kernels` (for the
  mxfp4 GPT-OSS text-encoder path; without Triton the encoder dequantizes to
  bf16).

To update: re-copy `lens/` from the upstream repo at the desired commit and
update the commit hash above.

## sensenova_u1

Inference package for **SenseNova-U1** (OpenSenseNova), a unified multimodal
NEO-unify model (Qwen3-based Mixture-of-Transformers; no separate VAE/encoder).
Importing `sensenova_u1` runs `register()`, which registers the custom
`neo_chat` `model_type` → `NEOChatModel`/`NEOChatConfig` into the `transformers`
Auto* registry, so the checkpoint loads via plain `AutoModel.from_pretrained`
(no `trust_remote_code`, no published pip package). Attention falls back to
torch SDPA when `flash_attn` is absent, so it runs on CUDA and MPS unchanged.

- Source: https://github.com/OpenSenseNova/SenseNova-U1
- Commit: `238d6cf3421d12989ec4a240b173d60c924a760b` (2026-05-23)
- License: Apache-2.0 (see `sensenova_u1/LICENSE`)
- Consumed by: `scene_worker/image_adapters.py::SenseNovaU1Adapter`
- Runs in the MAIN worker venv (no sidecar): its deps (torch 2.8,
  `transformers>=4.57,<4.58`, accelerate, sentencepiece, safetensors) match the
  worker stack. Pip-install is not viable — upstream `pyproject` requires
  Python <3.12 (the worker is 3.12) and pins a cu128 torch.
- Wired paths: `t2i_generate` (text-to-image), `it2i_generate` (instruction
  editing), `chat` (VQA), and `interleave_gen` (interleaved text-image
  documents). The `gguf` / `--vram_mode` offload paths are unused (the offload
  path is the only place upstream hard-excludes MPS).

### Local patches (re-apply on re-vendoring)

SceneWorks-local edits to `sensenova_u1/models/neo_unify/modeling_neo_chat.py`:

1. **`chat()` think flag (sc-1575):** added a `think` parameter; when `False` it
   primes `<think>\n\n</think>` so VQA returns the answer instead of the model's
   chain-of-thought. `SenseNovaU1Adapter.answer_question` passes `think=False`.
2. **`interleave_gen()` max_images fix (sc-1606):** removed a hardcoded
   `max_images = 10` that shadowed the caller's value (which had already sized
   `image_size_list`), leaving the per-image guard at 10 regardless — so any
   caller passing `max_images < 10` hit an `IndexError` once the model emitted
   more images than its cap. Control-flow bug, not numerics; affects all
   platforms.

To update: re-copy `src/sensenova_u1/` from the upstream repo at the desired
commit, update the commit hash above, and re-apply the local patches.

## instantid

InstantID SDXL face-identity pipeline + the `ip_adapter` support module it
imports. InstantID preserves a person's identity from one reference image via an
insightface ArcFace embedding + a 5-point-landmark ControlNet ("IdentityNet"),
while the prompt drives scene/pose. Selected over IP-Adapter / FaceID in the
sc-2009 A/B (the only method that held identity AND followed the prompt).

- `pipeline_stable_diffusion_xl_instantid.py`
  - Source: https://github.com/instantX-research/InstantID (`main`, 2026-05-27)
    — commit `2145b67f9607da6234702063692330185f374486` (see `VENDOR_PINS.md`)
  - License: Apache-2.0 (see `LICENSE`)
  - Imports clean against `diffusers==0.38.0`, with one local patch (re-apply on
    re-vendoring): a `MultiControlNetModel` import shim that imports from
    `diffusers.models.controlnets.multicontrolnet` and falls back to the legacy
    `diffusers.pipelines.controlnet.multicontrolnet` path (the old re-export
    errors on instantiation in diffusers ≥0.34).
- `ip_adapter/` (`resampler.py`, `utils.py`, `attention_processor.py`; empty
  `__init__.py` to avoid pulling upstream's diffusers-heavy `ip_adapter.py`)
  - Source: https://github.com/tencent-ailab/IP-Adapter (`main`, 2026-05-27)
    — commit `62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2` (verbatim; see `VENDOR_PINS.md`)
  - License: Apache-2.0
- Consumed by: `scene_worker/instantid_adapter.py::InstantIDAdapter`
- Runs in the MAIN worker venv. Extra deps in `requirements-instantid.txt`
  (insightface, onnxruntime, onnx, peft, einops). The adapter inserts this dir
  on `sys.path` (so the pipeline's top-level `from ip_adapter...` resolves) only
  when an InstantID job runs — all heavy imports are lazy, so registering the
  adapter is cheap on workers without the extras installed.
- Models (downloaded on demand): `InstantX/InstantID` (ControlNet + ip-adapter.bin)
  and the antelopev2 face pack (mirror `DIAMONIK7777/antelopev2`).

To update: re-copy both directories from their upstream repos at the desired
commit and update the dates above.

## kolors

Kolors strict-pose **ControlNet** pipeline + model for Character Studio
(sc-2264). Provides identity-stable pose conditioning on the Kolors SDXL base.

- `models/controlnet.py`, `pipelines/pipeline_controlnet_xl_kolors_img2img.py`
  - Source: https://github.com/Kwai-Kolors/Kolors (`master`, files under
    `controlnet/`, 2026-05-30) — commit
    `038818d244ed103056abd10f429729a26af4d239` (verbatim; see `VENDOR_PINS.md`)
  - License: Apache-2.0 (in-file headers only; no `LICENSE` file was vendored —
    add `kolors/LICENSE` on next re-vendor for redistribution compliance)
- Consumed by: the Kolors strict-pose ControlNet tier in Character Studio (sc-2264)
- Models (downloaded on demand): `Kwai-Kolors/Kolors-ControlNet-Pose` and the
  IP-Adapter-Plus weights.

To update: re-copy `kolors/` from the upstream repo at the desired commit and
update the commit hash above and in `VENDOR_PINS.md`.
