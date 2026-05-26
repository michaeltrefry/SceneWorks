# SceneWorks Video Model Research

Use `ltx_2_3` as the first SceneWorks video target, with Wan2.2 present as the next fallback family once the adapter boundary can host multiple runtimes.

## Recommendation

- First adapter: LTX-2.3.
- Runtime path: start with the official Lightricks PyTorch/ComfyUI-compatible path and keep the worker contract isolated behind a SceneWorks adapter.
- First shipped implementation in this repo: `procedural_video`, which produces deterministic preview clips while exercising the real Video Studio, job, recipe, lineage, and Library asset contracts.
- Fallback family: Wan2.2, exposed in the manifest so UI and payload settings can already express Wan-aware limits.

## Why LTX-2.3 First

- Lightricks documents LTX-2.3 as an open-weights DiT audio-video model with multimodal inputs including text, image, video, audio, depth, and LoRA-based customization.
- The Hugging Face model card provides the practical local entry point, model IDs, checkpoint variants, and PyTorch repository requirements.
- The official usage guides support both image-to-video and text-to-video, with resolution and frame count guidance that maps well to SceneWorks controls.
- It is a better first SceneWorks target than Wan2.2 for the current product slice because it supports a single family for I2V, T2V, first/last-frame style conditioning, and future audio-aware workflows.

## Encoded Product Limits

- Keep SceneWorks oriented around short shots assembled later in the editor.
- Simple UI should recommend 4-8 seconds for fast iteration and keep 10 seconds as the normal LTX-2.3 ceiling.
- The broader product assumption from the plan says LTX2.3 is best at 15 seconds or less. Current official guides list 257 frames, roughly 10 seconds at 25fps, for the common I2V/T2V workflows, so the UI should favor 10 seconds for now and reserve longer durations for future adapter-specific support.
- Resolution dimensions must be divisible by 32 for LTX-2.3. Favor 768x512, 640x640, 1280x720, and 720x1280 presets.
- FPS controls should default to 25fps for LTX-2.3, with 24fps and 30fps available in advanced mode.
- Quality should map to raw adapter settings:
  - Fast: fewer frames/steps for iteration.
  - Balanced: default distilled settings.
  - Best: higher step budget and future multiscale/upscale path.

## Wan2.2 Notes

- Wan2.2 has official T2V, I2V, TI2V, and S2V model entries with 480P/720P support.
- Wan2.2 is valuable as a fallback and later adapter because it has broad video modes and Diffusers/ComfyUI ecosystem support.
- Keep Wan-aware UI guidance conservative: shorter clips around 5-7 seconds are recommended until local looping behavior is validated against the exact runtime.
- **Quantized A14B inference (sc-1982).** A14B is GPU-heavy at bf16 (~56GB of transformers), so the manifest declares quantization variants and Video Studio exposes a **Quantization** selector (Advanced panel). Two paths:
  - **GGUF (torch adapter, cross-platform):** the two experts load via `WanTransformer3DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(...))` (high-noise → `transformer`, low-noise → `transformer_2`) from `QuantStack/Wan2.2-{T2V,I2V}-A14B-GGUF`. The 5B (TI2V) has a single-transformer GGUF too. Defaults are per-platform: **Q8_0 on MPS** (trivial dequant, ~3× slower vs ~13× for k-quants — and the GGUF path runs fp32 on MPS because Wan's Conv3d has no bf16 Metal kernel), **Q4_K_M on CUDA** (smallest, fused kernel). `auto` follows the default; `none` forces the unquantized base.
  - **MLX-Q4 (preferred on Mac):** the `model_convert` job accepts `quantizeBits`/`quantizeGroupSize` (and a `--quantize-only` pass for turnkey bf16 MLX repos), and the MLX adapter prefers a locally-converted/quantized dir over the turnkey download. ~3.7× faster + ~2.6× less memory than GGUF-on-MPS in the sc-1950 spike (84s / 41GB peak), fitting a 64GB Mac.
  - Quantized experts still accept trained per-expert LoRAs (validated for GGUF in sc-1950; MLX via `loras_high`/`loras_low`). Weights are Apache-2.0.

## Sources

- LTX open source overview: https://docs.ltx.video/open-source-model/getting-started/overview
- LTX-2.3 Hugging Face model card: https://huggingface.co/Lightricks/LTX-2.3
- LTX image-to-video guide: https://docs.ltx.video/open-source-model/usage-guides/image-to-video
- LTX text-to-video guide: https://docs.ltx.video/open-source-model/usage-guides/text-to-video
- Wan2.2 Hugging Face model card: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
