# Replace Person V1 Research

Source: Shortcut epic 1090, `SCENEWORKS_PLAN.md`, and `IMPLEMENTATION_PLAN.md`.

## Recommendation

Ship V1 as a job-based, track-first workflow:

1. Extract a representative frame from the source clip.
2. Detect selectable people on that frame.
3. Store the user's selected person as a reusable person track.
4. Run replacement against the source clip, person track, Character, mode, quality preset, model, seed, and settings.
5. Save the output as a normal video asset with sidecar lineage.
6. Compare original and replacement with side-by-side and A/B review.

This keeps the product contract stable while real model adapters evolve behind the worker boundary.

## Pipeline Assessment

### Face Only

Face-only replacement is the safest first real adapter path. It has the narrowest mask area, the smallest temporal consistency burden, and the clearest failure mode when the selected person turns away or is occluded.

V1 mode status: supported in product and sidecar metadata. The current local adapter produces a procedural replacement preview so the full job, asset, and comparison flow can be exercised without a model install.

### Full Person, Keep Outfit

Full-person keep-outfit replacement needs stronger person segmentation and identity conditioning. It is plausible as a second adapter path when masks are stable, because the clothing region can stay close to source video.

V1 mode status: supported in product and sidecar metadata; real adapter support should stay disabled or clearly explained per model.

### Full Person, Replace Outfit

Full-person replace-outfit has the highest identity, pose, clothing, and temporal consistency risk. It should remain a visible V1 option only for models that explicitly support it.

V1 mode status: supported in product and sidecar metadata; procedural adapter marks the selected replacement mode for review.

## Tracking Stack

The app should treat tracking as reusable project data rather than temporary generation input. A person track stores:

- Source clip asset ID.
- Representative frame asset ID.
- Selected detection box and confidence.
- Sampled frame boxes with confidence.
- Deferred mask slots.
- Correction slots for future manual edits.

The current V1 implementation stores box tracks and leaves mask correction as a deferred adapter concern. This is enough to preserve lineage and unblock replacement workflow UX.

## Fallback Plan

If full-person replacement quality is below the bar, SceneWorks should still ship Face Only as the default supported path. Full Person modes should remain selectable only when the chosen model declares support, and their sidecars should preserve enough settings for adapter upgrades later.

## Practical Limits

- Keep clips short for V1: 4-8 seconds recommended, with model-specific hard limits.
- Store masks sparsely or by reference when real segmentation lands.
- Prefer frame sampling plus correction metadata over writing dense per-frame mask files by default.
- Keep the original source clip available as an asset for comparison and timeline version history.
