# sc-5525 — worker cutover: `prompt_refine` → candle `TextLlm` provider

Production counterpart of **sc-5500** (the gen-core `TextLlm` contract + the candle
`candle-gen-prompt-refine` provider). Routes the worker's `prompt_refine` job to the native candle
`TextLlm` provider (Llama-3.2-3B-Instruct, `backend="candle"`) through the backend-neutral
`gen_core::load_textllm` seam — the Windows/CUDA worker now refines prompts with **zero torch**.

There is **no mlx twin** (greenfield — no native prompt-refine on the macOS path), so this is
candle-only off-Mac. The Python torch `PromptRefiner`
(`apps/worker/scene_worker/prompt_refine.py`) **stays as the fallback** for the Mac path and the
default, candle-less Desktop installer. See "Deletion is deferred" below.

## What changed

- **gen-core / candle-gen pin bump** (lockstep, skew gate): worker `sceneworks-gen-core` + `mlx-gen`
  (+ all providers) `41c95a1` → `af190c7` (mlx-gen #448, the `TextLlm` contract); candle-gen
  `f366f40` → `aada3fe` (whose gen-core pin is `af190c7`, == the mlx-gen pin) and adds the
  `candle-gen-prompt-refine` dep. The candle lane resolves **exactly one** gen-core rev
  (`scripts/check-gen-core-skew.sh` + `cargo tree --features backend-candle --target all`). mlx-rs is
  unchanged at `44929a0`, so MLX core rebuilds nothing.
- **Capability advertisement** (`engines::registry_capabilities`): a `prompt_refine` `TextLlm`
  registered with an enabled backend now derives `WorkerCapability::PromptRefine`. The candle worker
  advertises it when `backend_candle_enabled`; the mlx worker never does (no mlx twin).
- **Handler** (`crates/sceneworks-worker/src/prompt_refine_jobs.rs`): `run_prompt_refine_job` resolves
  `gen_core::load_textllm("prompt_refine", …)` and streams progress / honors cancellation like the
  candle caption path. The generic `TextLlm` contract means the prompt-refinement **product logic**
  moves here caller-side — `build_refine_system_prompt` (rewrite rules + image/video medium switch +
  guide → the request `system`) and `clean_refine_output` (reasoning-block / code-fence / quote
  cleanup over the reply), ports of the Python `build_system_prompt` / `clean_output`. Sampling
  (temp 0.7 / top_p 0.9 / max_new_tokens 512), the empty-output → error behavior, and the
  `{originalPrompt, refinedPrompt}` result shape match the Python path.
- **Routing**: unchanged in `jobs_store` — `prompt_refine` is routed purely by capability match
  (`required_capability` → `"prompt_refine"`); the candle confinement gate only filters
  image/video/caption shapes, so it is inert for `prompt_refine`.

## Two-level gating (as elsewhere)

1. **Build feature** `backend-candle` — pulls the optional `candle-gen-prompt-refine` crate (CUDA).
2. **Runtime flag** `SCENEWORKS_BACKEND_CANDLE_ENABLED=1` — until set, the provider is linked but the
   capability is not advertised, so nothing routes to it (production routing unchanged).

Build with **VS2022 BuildTools MSVC 14.44** (CUDA 12.9 rejects VS18/14.51), `CUDA_COMPUTE_CAP=120`.

## Live deployed-worker smoke (run on the Windows/CUDA box)

### Prerequisites
- Worker built `--features backend-candle` (MSVC 14.44 / CUDA 12.9) and deployed.
- `SCENEWORKS_BACKEND_CANDLE_ENABLED=1` in the worker environment.
- The refine model present in the HF cache. It is now a catalog artifact —
  `prompt_refine_llama_3_2_3b` in `config/manifests/builtin.models.jsonc` (sc-5605) —
  so download it from the **Models** screen (or it is offered inline when "Refine my
  prompt" runs before the model is provisioned). The native path only *resolves* an
  already-cached snapshot (`huggingface_snapshot_dir`); it does not auto-download like
  the retired Python `PromptRefiner`. Repo: `huihui-ai/Llama-3.2-3B-Instruct-abliterated`.
- No co-resident Python worker also advertising `prompt_refine` (else routing may pick either —
  preferring the candle worker is a future routing refinement; see below).

### Steps
1. Submit a `prompt_refine` job (e.g. via Prompt → "Refine"): `payload.prompt` set, optional
   `guide` + `workflow` (`image` | `video`). Expect: `loading_model` then `running` progress with
   backend **`candle`**, terminal `completed` with `result.refinedPrompt` an on-guide rewrite and
   `result.originalPrompt` echoed. No torch in the worker process.
2. Re-run and cancel mid-generation → job ends `Canceled` (the provider honors `CancelFlag`
   pre-inference and mid-decode, sc-5500).

### Pass criteria
- The rewrite is produced by the candle Llama-3.2-3B provider (backend `candle`, zero torch), parity
  with the torch `PromptRefiner` for the same rules/guide/medium, and cancellation is honored.

> The candle provider itself is already real-weights GPU-validated end-to-end in sc-5500
> (`textllm_conformance` + the `refine` example). This story adds the worker orchestration + routing
> + the ported product logic; the live job-routing smoke is executed on the deployed box.

## Deletion is deferred (intentional)

sc-5525's "Done when" lists deleting `prompt_refine.py`. The candle backend is **not** in the default
Desktop installer (it is a dedicated CUDA server/CI lane), and there is no mlx twin — so deleting the
Python refiner now would remove prompt refinement from **every default install** (Mac + default
Windows + Linux), a shipping-product regression. The story's own scope ("keep the torch path as the
fallback **until the candle provider is the default everywhere**") sequences the deletion after candle
is the default off-Mac. The physical deletion is therefore tracked as a follow-up gated on that
condition; this story delivers the native candle cutover + keeps the torch path as the documented
fallback.

A future routing refinement (preferring an idle candle worker over a co-resident Python worker for
`prompt_refine`, mirroring the MLX soft-deferral) would make the candle path the effective default on
a CUDA box without removing the fallback.
