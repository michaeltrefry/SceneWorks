# Timeline Library Research

Source: Shortcut story sc-1136 and Phase 10 of `documents/IMPLEMENTATION_PLAN.md`.

## Recommendation

Use a SceneWorks-owned timeline data model and lightweight in-app React editor for the first Phase 10 slice, with FFmpeg as the authoritative export renderer.

This keeps timeline JSON portable, avoids adopting a young or license-sensitive editor SDK before the product model settles, and gives SceneWorks a direct mapping from saved timeline items to MP4 output.

## Candidates Reviewed

### React Video Editor Timeline

Docs: https://docs.reactvideoeditor.com/core/components/timeline

Fit:
- Strongest conceptual match for CapCut-like interactions.
- Supports tracks, item movement/resizing, scrubbing, zooming, keyboard shortcuts, and undo/redo.
- Its documented example is a copyable component approach, which is useful later if SceneWorks wants a richer timeline surface without surrendering its data model.

Gaps:
- Active development and performance caveats are called out in the docs.
- Would need adaptation from its `start`/`end` item shape into SceneWorks `sourceIn`/`sourceOut` plus `timelineStart`/`timelineEnd`.
- Phase 10 can satisfy core acceptance criteria with less dependency risk.

### Remotion

Repo: https://github.com/remotion-dev/remotion

Fit:
- Mature React video rendering ecosystem.
- Useful for programmatic video composition and preview/render pipelines.

Gaps:
- It is primarily a renderer/framework, not a drop-in interactive editor timeline.
- License has commercial considerations.
- SceneWorks final export is already planned as backend FFmpeg, so Remotion is not needed for this slice.

### Twick

Repo: https://github.com/ncounterspecialist/twick

Fit:
- Full React editor SDK with timeline editing, canvas tools, and MP4 export.
- Interesting reference for a future richer editing surface.

Gaps:
- Larger product-shaped SDK than SceneWorks needs right now.
- Browser export support is browser-limited, while SceneWorks wants backend rendering.
- Its license and cloud-oriented architecture need a dedicated review before adoption.

## SceneWorks V1 Mapping

Timeline item fields:
- `assetId` points to a project asset sidecar rather than a media path.
- `sourceIn` and `sourceOut` describe the trim range.
- `timelineStart` and `timelineEnd` describe placement.
- `speed`, `transitionIn`, and `transitionOut` live on the item.

Export mapping:
- Main-track items are normalized into FFmpeg segments.
- Still images use `-loop 1` and a target duration.
- Video clips use `-ss`, `-t`, `setpts`, scale, pad, fps, and yuv420p normalization.
- `fade_from_black` and `fade_to_black` map to FFmpeg `fade`.
- `crossfade` maps to FFmpeg `xfade` when adjacent segments request it.
- Render outputs become `render` assets under `assets/renders` with lineage to the timeline and source assets.

Known deferred work:
- Rich multi-track compositing.
- Audio mixing UI.
- Persistent undo stack.
- Generation-aware timeline hooks such as bridge clips, frame extraction, extension, and nondestructive replacement.
