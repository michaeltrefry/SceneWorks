# Bundled model asset — LAION-Aesthetics V2 MLP head (sc-6537)

`aesthetic-v2-sac-logos-ava1-l14.safetensors` is the **prediction head only** — the `layers.*` MLP
(768 → 1024 → 128 → 64 → 16 → 1, no activations) extracted from
[`shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE`](https://huggingface.co/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE),
a repackaging of the LAION **improved-aesthetic-predictor**.

- **Upstream:** [`christophschuhmann/improved-aesthetic-predictor`](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- **License:** **Apache-2.0** (the upstream's license; preserved here per its terms).
- **What it is:** an MLP that scores an **L2-normalized CLIP ViT-L/14 image embedding** (`image_embeds`)
  on the LAION aesthetic scale (~`[1, 10]`).
- **Why vendored:** the CLIP backbone is **not** included — SceneWorks already produces the embedding
  in the dataset-analysis job, so only this tiny head (3.7 MB) is needed. It is vendored (rather than
  downloaded like the GB-scale models) because it is a small *extracted* subset, not a standalone HF
  file, and the GPU-free host readiness path (`sceneworks_image_quality::aesthetic_predictor`) has no
  model-download path.
- **Provenance:** head-only extraction of the source `model.safetensors` `layers.*` tensors; see the
  `__metadata__` block inside the file.

Consumed by Dataset Doctor (epic 6529, sc-6537) for the **STYLE-only, advisory** aesthetic sub-score.
