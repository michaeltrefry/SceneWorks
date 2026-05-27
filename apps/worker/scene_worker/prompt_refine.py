"""Prompt refinement (sc-2041).

Rewrites a user's prompt to follow the selected model's prompt guide, by calling
an OpenAI-compatible ``/chat/completions`` endpoint (e.g. a local gemma GGUF
served by Ollama / llama.cpp / LM Studio). This mirrors the calling approach of
the vendored Lens ``PromptReasoner`` (``_vendor/lens/reasoner.py``), but injects
the model's guide and the image/video workflow into the system prompt — which the
reasoner's fixed, image-only system prompt cannot do — and keeps the refine path
free of the reasoner's torch/diffusers import side effects.
"""

from __future__ import annotations

import re
from typing import Optional


class PromptRefineError(RuntimeError):
    """Base class for prompt-refinement failures."""


class PromptRefineUnavailable(PromptRefineError):
    """The refinement runtime is not configured or unreachable."""


class PromptRefineTimeout(PromptRefineError):
    """The refinement runtime did not respond in time."""


class PromptRefineMalformed(PromptRefineError):
    """The runtime returned an empty or unusable response."""


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


def refine_prompt(
    prompt: str,
    *,
    guide: Optional[str],
    workflow: Optional[str],
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
    max_tokens: int = 1024,
) -> str:
    """Refine ``prompt`` via an OpenAI-compatible endpoint and return the rewrite.

    Raises ``PromptRefineUnavailable`` when the runtime is unconfigured or cannot
    be reached, ``PromptRefineTimeout`` on timeout, and ``PromptRefineMalformed``
    when the response is empty.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise PromptRefineMalformed("Prompt is empty.")
    if not base_url or not model:
        raise PromptRefineUnavailable(
            "Prompt refinement runtime is not configured. Set PROMPT_REFINE_BASE_URL "
            "and PROMPT_REFINE_MODEL (and optionally PROMPT_REFINE_API_KEY) to an "
            "OpenAI-compatible endpoint."
        )

    try:
        from openai import OpenAI
        from openai import APIConnectionError, APITimeoutError
    except ImportError as exc:  # pragma: no cover - import guard
        raise PromptRefineUnavailable(
            "The 'openai' package is not installed in the worker environment."
        ) from exc

    # Many local servers (Ollama, llama.cpp) ignore the key but the client
    # requires a non-empty value, so fall back to a placeholder.
    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout)
    system_prompt = build_system_prompt(guide, workflow)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
    except APITimeoutError as exc:
        raise PromptRefineTimeout("The refinement runtime timed out.") from exc
    except APIConnectionError as exc:
        raise PromptRefineUnavailable(
            f"Could not reach the refinement runtime at {base_url}."
        ) from exc

    choices = getattr(response, "choices", None) or []
    raw = (choices[0].message.content if choices else "") or ""
    refined = clean_output(raw)
    if not refined:
        raise PromptRefineMalformed("The refinement runtime returned an empty prompt.")
    return refined
