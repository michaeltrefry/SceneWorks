# Vendored upstream pins

Each vendored package under `apps/worker/scene_worker/_vendor/` is pinned to a
specific upstream commit. Prose, license details, and the re-vendor procedure
live in [`README.md`](./README.md); **this file is the machine-checkable SHA
ledger** — the authoritative record of *which* upstream commit each copy came
from and how to verify it.

Every pin below is **CONFIRMED**: the SHA was verified to exist upstream and the
vendored bytes either match the upstream tree blob exactly (`git hash-object`
equals the upstream blob SHA) or differ only by the local patches called out
explicitly. None of the four image/text packages tag releases, so a commit SHA
is the only stable handle.

## Quarterly review checklist

1. Confirm each upstream repo is still live and note its latest commit.
2. If re-vendoring, re-copy the directory at the new commit, bump the SHA here
   and in `README.md`, and **re-apply the local patches** listed below.
3. Re-run the verification (`git hash-object <file>` vs the recorded blob SHA)
   so the ledger stays trustworthy.

> Note: InstantID, IP-Adapter, and Kolors-controlnet are effectively frozen
> upstream (no commits for ~19–23 months as of vendoring). The "upstream bug
> fixes won't reach us" risk is largely theoretical for these three — there are
> no upstream fixes arriving, and the Torch-2.1 `attn.scale` TODO comments in
> `attention_processor.py` are upstream-original and will not be resolved
> upstream. Any change there will be a local patch, not a re-vendor.

---

## instantid

Two upstream sources are combined under `_vendor/instantid/`.

### `pipeline_stable_diffusion_xl_instantid.py`, `LICENSE`

- Upstream: <https://github.com/instantX-research/InstantID> (branch: `main`)
- Commit: `2145b67f9607da6234702063692330185f374486` **[CONFIRMED]**
  (upstream tip; repo frozen since 2024-07-18, no release tags)
- Vendored: 2026-05-27 (SceneWorks commit `166475a`, sc-2009)
- Local patches (re-apply on re-vendor):
  - `pipeline_stable_diffusion_xl_instantid.py` — `MultiControlNetModel` import
    shim: import from `diffusers.models.controlnets.multicontrolnet` with a
    fallback to the legacy `diffusers.pipelines.controlnet.multicontrolnet`
    path. The old re-export errors on instantiation in diffusers ≥0.34.
    **[CONFIRMED — 6-line diff; everything else identical to upstream tip]**
  - `LICENSE` — trailing newline added at EOF (cosmetic). **[CONFIRMED]**

### `ip_adapter/{attention_processor,resampler,utils}.py` (+ empty `__init__.py`)

- Upstream: <https://github.com/tencent-ailab/IP-Adapter> (branch: `main`)
- Commit: `62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2` **[CONFIRMED]**
  (upstream tip; repo frozen since 2024-06-28, no release tags)
- Vendored: 2026-05-27 (sc-2009)
- Local patches: none — 3/3 files are byte-identical to upstream.
- Verify (`git hash-object <file>` must equal):
  - `attention_processor.py` → `93592bb1a7e7b3329fc8a400c51920dad519a42f`
  - `resampler.py` → `24266671d02092438ae6576336a59659fef9c054`
  - `utils.py` → `6a273358585962fdf383d0bb7a0e1c654b4999b8`
- Note: the four `# TODO: add support for attn.scale when we move to Torch 2.1`
  comments (`attention_processor.py:258,370,386,547`) are upstream-original.

---

## kolors

- Files: `models/controlnet.py`,
  `pipelines/pipeline_controlnet_xl_kolors_img2img.py`
- Upstream: <https://github.com/Kwai-Kolors/Kolors> (branch: `master`, files
  under `controlnet/`)
- Commit: `038818d244ed103056abd10f429729a26af4d239` **[CONFIRMED]**
  (upstream tip; repo frozen since 2024-11-13, `controlnet/` subtree frozen
  since 2024-08-06 @ `140c22d`; no release tags)
- Vendored: 2026-05-30 (SceneWorks commit `1dd803c`, sc-2264)
- License: Apache-2.0 (in-file headers). **GAP:** no `LICENSE` file was
  vendored — add `kolors/LICENSE` on next re-vendor for redistribution compliance.
- Local patches: none — both files are byte-identical to upstream.
- Verify (`git hash-object <file>` must equal):
  - `models/controlnet.py` → `43b09968601697db1ffd2349f37f28a42cf5ccc2`
  - `pipelines/pipeline_controlnet_xl_kolors_img2img.py` → `04a02f08c12252a5e976af6e6d80caf361093a99`

---

## lens

Already pinned in [`README.md`](./README.md); restated here for completeness.

- Upstream: <https://github.com/microsoft/Lens>
- Commit: `5bf0f0cea2f4bc32ebb2b7ed2ef96d5e88b701e0` (2026-05-22) **[CONFIRMED]**
- License: MIT (see `lens/LICENSE`)

## sensenova_u1

Already pinned in [`README.md`](./README.md); restated here for completeness.

- Upstream: <https://github.com/OpenSenseNova/SenseNova-U1>
- Commit: `238d6cf3421d12989ec4a240b173d60c924a760b` (2026-05-18) **[CONFIRMED]**
- License: Apache-2.0 (see `sensenova_u1/LICENSE`)
- Local patches: sc-1575 (chat `think` flag), sc-1606 (`interleave_gen`
  `max_images`) — see `README.md` and `modeling_neo_chat.py`.
