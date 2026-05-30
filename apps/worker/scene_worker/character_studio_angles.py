"""Shared canonical-angle definitions for multi-backbone Character Studio (sc-2003).

InstantID owns the strict tier via landmark packs (`VIEW_ANGLE_KPS` in
``instantid_adapter``). Prompt-driven backbones (Qwen-Image-Edit, FLUX.2-klein,
SenseNova-U1) consume the same 11-angle list but augment the user's prompt
with the per-angle description below — the worker spike (sc-2003 hardware run)
verified that this is the only viable cross-backbone interface today, because
each backbone uses a different identity-conditioning mechanism (cross-attention
adapter / dual-control / it2i / KV-cached flow) and they don't share landmark
or ControlNet inputs.

Augments are phrased as continuation clauses (no leading punctuation) so they
append cleanly to whatever prompt the user typed. Per-backbone prompt shape
differences (Qwen edit instructions vs FLUX descriptive vs SenseNova edit)
are handled at the call site — each adapter chooses to prepend "Re-render the
same woman as a …" or to append " — …" around this base augment.
"""

from __future__ import annotations

# Same order as instantid_adapter.ANGLE_SET_ORDER (kept in sync so the angle
# count and per-index angle match across backbones for users comparing outputs).
# Importable as a sibling so callers can pick this up without pulling the
# InstantID-only landmark pack along with it.
ANGLE_SET_ORDER: tuple[str, ...] = (
    "front",
    "three_quarter_left",
    "three_quarter_right",
    "left_profile",
    "right_profile",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)


# Continuation clauses appended to the user's prompt for each canonical angle.
# Wording mirrors what the sc-2003 hardware spike used to validate angle
# compliance across Qwen-Lightning + FLUX.2-klein + SenseNova-U1. The strongest
# discriminator across the spike runs was including BOTH the descriptive name
# ("three-quarter left profile") AND the directional instruction ("head turned
# slightly to the left") — backbones followed the directional cue when present
# and ignored the name otherwise.
ANGLE_PROMPT_AUGMENTS: dict[str, str] = {
    "front": "frontal portrait, looking directly at the camera, head and shoulders, neutral expression",
    "three_quarter_left": "three-quarter left profile, head turned slightly to the left, three-quarter view",
    "three_quarter_right": "three-quarter right profile, head turned slightly to the right, three-quarter view",
    "left_profile": "full left profile, head turned 90 degrees to the left, side view of the head",
    "right_profile": "full right profile, head turned 90 degrees to the right, side view of the head",
    "up": "looking up, head tilted slightly upward toward the sky",
    "down": "looking down, head tilted slightly downward toward the floor",
    "up_left": "looking up and to the left, head tilted slightly upward and turned slightly to the left",
    "up_right": "looking up and to the right, head tilted slightly upward and turned slightly to the right",
    "down_left": "looking down and to the left, head tilted slightly downward and turned slightly to the left",
    "down_right": "looking down and to the right, head tilted slightly downward and turned slightly to the right",
}


def augment_prompt_for_angle(base_prompt: str, angle: str) -> str:
    """Append the per-angle continuation clause to the user's base prompt.

    Empty base + unknown angle → empty string (caller usually validates the
    angle is in ``ANGLE_SET_ORDER`` first, but be lenient here so a forward-
    compatible manifest with a new angle id doesn't crash the worker; the user
    still gets their base prompt).
    """
    augment = ANGLE_PROMPT_AUGMENTS.get(angle, "")
    base = (base_prompt or "").strip().rstrip(",.;")
    if base and augment:
        return f"{base}, {augment}"
    return augment or base


# Best-effort pose tier (sc-2250 spike / sc-2256): backbones without a pose
# ControlNet approximate a library pose by passing the rendered OpenPose skeleton
# as an extra reference image plus this prompt cue telling the model to adopt that
# pose. Qwen-Image-Edit drives it through its multi-image edit input (image=[ref,
# skeleton]). The strict tier (InstantID landmark+OpenPose ControlNet, Z-Image
# Fun-ControlNet) drives the pose structurally and does NOT use this cue.
POSE_SKELETON_PROMPT = "matching the exact body pose shown in the OpenPose skeleton reference image"


def augment_prompt_for_pose(base_prompt: str) -> str:
    """Append the pose-skeleton instruction to the user's base prompt for the
    best-effort multi-image pose tier."""
    base = (base_prompt or "").strip().rstrip(",.;")
    if base:
        return f"{base}, {POSE_SKELETON_PROMPT}"
    return POSE_SKELETON_PROMPT


__all__ = [
    "ANGLE_SET_ORDER",
    "ANGLE_PROMPT_AUGMENTS",
    "augment_prompt_for_angle",
    "POSE_SKELETON_PROMPT",
    "augment_prompt_for_pose",
]
