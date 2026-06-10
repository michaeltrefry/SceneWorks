"""Prompt refinement (sc-2041).

Rewrites a user's prompt to follow the selected model's prompt guide, using a
small instruction LLM loaded **in-process** via transformers — the same shape as
JoyCaption captioning (`caption_adapters.py`). The model downloads on demand via
the HF cache on first use and runs on the worker's device (MPS on Apple Silicon),
so there is no external API endpoint and no env setup for users.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .image_adapters import (
    release_inference_memory,
    require_inference_backend_for_gpu_worker,
    select_torch_device,
    select_torch_dtype,
)

# Default refinement model: a small, text-only, uncensored (abliterated)
# instruction LLM. Loads via AutoModelForCausalLM on any recent transformers
# (model_type "llama"), unlike multimodal models that need a processor + an
# image-text-to-text class. Overridable via PROMPT_REFINE_MODEL; downloaded on
# demand (HF cache), like JoyCaption. NOTE: point this only at text causal LMs.
DEFAULT_REFINE_MODEL = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"


class PromptRefineError(RuntimeError):
    """Base class for prompt-refinement failures."""


class PromptRefineUnavailable(PromptRefineError):
    """The refinement model/backend could not be loaded."""


class PromptRefineMalformed(PromptRefineError):
    """The model returned an empty or unusable rewrite."""


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_BASE_RULES = """\
You are a prompt rewriter for a generative {medium} model.
Rewrite the user's input into a single, precise {medium} prompt that follows the model's prompt guide below.

Rules:
- Output exactly one rewritten prompt and nothing else — no explanations, reasoning, commentary, options, or labels.
- Preserve the user's intent: do not change the subjects, attributes, actions, relationships, or core setting they described. You may add concrete details only when they make the {medium} more coherent and stay consistent with the user's meaning.
- If the user's prompt is already detailed and on-guide, make only minimal edits for fluency.
- Follow the guide's recommended structure, phrasing, and what-to-avoid guidance.
- Match the user's language: if their prompt is not in English, respond in the same language.
- Do not wrap the output in quotes, markdown, JSON, or code fences unless those are part of the described scene."""


def build_system_prompt(guide: Optional[str], workflow: Optional[str]) -> str:
    medium = "video" if (workflow or "").strip().lower() == "video" else "image"
    rules = _BASE_RULES.format(medium=medium)
    guide = (guide or "").strip()
    if guide:
        return f"{rules}\n\n# Model prompt guide\n\n{guide}"
    return rules


def clean_output(text: str) -> str:
    """Strip reasoning blocks and surrounding decoration from a model reply."""
    text = (text or "").strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


class PromptRefiner:
    """Loads a small instruction LLM in-process and rewrites prompts. Mirrors
    `JoyCaptioner`: lazy `load()`, shared MPS-aware device/dtype helpers, and a
    `model.generate` call.

    Held as a long-lived resident adapter in the worker loop's adapter dict
    (sc-4191): ``load()`` is idempotent so repeat ``prompt_refine`` jobs reuse the
    already-resident 3B LLM instead of paying a fresh multi-GB
    ``from_pretrained`` per job, and ``unload()`` lets ``evict_other_image_adapters``
    free it before another family loads."""

    #: Stable adapter id used as the resident-dict key and evict keep-id.
    id = "prompt_refiner"

    def __init__(self, *, model_name_or_path: str, gpu_id: str, max_new_tokens: int = 512) -> None:
        self.model_name_or_path = model_name_or_path or DEFAULT_REFINE_MODEL
        self.gpu_id = gpu_id
        self.max_new_tokens = int(max_new_tokens)
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self.torch_dtype = None
        # Tracks which checkpoint is currently resident so load() can skip a
        # redundant reload and detect a within-job model switch.
        self._loaded_model_name: Optional[str] = None

    def loaded_models(self) -> list[str]:
        return [self.model_name_or_path] if self.model is not None else []

    def unload(self) -> bool:
        """Drop the resident model/tokenizer and return cached accelerator blocks
        to the OS. Returns True if anything was freed (so ``evict_other_image_adapters``
        can report it). gc.collect() must precede empty_cache() or the dropped
        ``nn.Module`` reference cycles keep the multi-GB weights resident."""
        if self.model is None and self.tokenizer is None:
            return False
        torch = self.torch
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self.torch_dtype = None
        self._loaded_model_name = None
        if torch is not None:
            release_inference_memory(torch)
        return True

    def load(self) -> None:
        # Resident reuse: the requested checkpoint is already loaded → no-op.
        if self.model is not None and self._loaded_model_name == self.model_name_or_path:
            return
        # A different refine model was requested: free the resident one first so
        # we never hold two multi-GB LLMs at once (mirrors a within-family switch).
        if self.model is not None:
            self.unload()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - import guard
            raise PromptRefineUnavailable(
                "transformers/torch are not available in this worker environment."
            ) from exc

        try:
            require_inference_backend_for_gpu_worker(torch, self.gpu_id)
            self.torch = torch
            self.device = select_torch_device(torch, self.gpu_id)
            self.torch_dtype = select_torch_dtype(torch, self.device, None)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
            )
            self.model.to(self.device)
            self.model.eval()
            self._loaded_model_name = self.model_name_or_path
        except PromptRefineError:
            raise
        except Exception as exc:
            raise PromptRefineUnavailable(
                f"Could not load the prompt-refinement model '{self.model_name_or_path}': {exc}"
            ) from exc

    def refine(self, prompt: str, *, guide: Optional[str], workflow: Optional[str]) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise PromptRefineMalformed("Prompt is empty.")
        if self.model is None or self.tokenizer is None:
            self.load()

        # Fold the instructions + guide into a single user turn. Some chat
        # templates (e.g. Gemma) reject a system role, so this stays portable.
        system_prompt = build_system_prompt(guide, workflow)
        content = f"{system_prompt}\n\n# Prompt to rewrite\n\n{prompt}"
        messages = [{"role": "user", "content": content}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        with self.torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=pad_token_id,
            )[0]
        generated = generated[input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        refined = clean_output(text)
        if not refined:
            raise PromptRefineMalformed("The refinement model returned an empty prompt.")
        return refined


def refine_prompt(
    prompt: str,
    *,
    guide: Optional[str],
    workflow: Optional[str],
    gpu_id: str,
    model: Optional[str] = None,
    max_new_tokens: int = 512,
) -> str:
    """Convenience: load the model and refine a single prompt in one call."""
    refiner = PromptRefiner(
        model_name_or_path=model or DEFAULT_REFINE_MODEL,
        gpu_id=gpu_id,
        max_new_tokens=max_new_tokens,
    )
    return refiner.refine(prompt, guide=guide, workflow=workflow)
