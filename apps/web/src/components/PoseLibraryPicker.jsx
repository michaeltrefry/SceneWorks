import React from "react";
import { usePoseLibrary } from "../poseLibrary.js";

// Multi-select gallery of OpenPose poses, grouped by category. Controlled: the parent
// owns `selectedIds` (array) and gets toggles via `onToggle(id)` / `onClear()`. Shared
// by the Character Studio pose panel and the Image Studio pose section.
export function PoseLibraryPicker({ selectedIds = [], onToggle, onClear, loadUserPoses }) {
  const { poses, categories, loading, error } = usePoseLibrary({ loadUserPoses });
  const selected = new Set(selectedIds);

  if (loading) {
    return <p className="muted">Loading pose library…</p>;
  }
  if (error) {
    return <p className="inline-warning">Pose library unavailable: {error}</p>;
  }
  if (!poses.length) {
    return <p className="inline-warning">No poses found in the library.</p>;
  }

  return (
    <div className="pose-library">
      <div className="pose-library-toolbar">
        <span className="muted">
          {selected.size ? `${selected.size} pose${selected.size === 1 ? "" : "s"} selected` : "Select one or more poses"}
        </span>
        {selected.size ? (
          <button className="link-button" onClick={onClear} type="button">
            Clear
          </button>
        ) : null}
      </div>
      {categories.map((category) => {
        const inCategory = poses.filter((pose) => pose.category === category);
        if (!inCategory.length) {
          return null;
        }
        return (
          <div className="pose-category" key={category}>
            <p className="eyebrow">{category}</p>
            <div className="pose-grid">
              {inCategory.map((pose) => {
                const isSelected = selected.has(pose.id);
                return (
                  <button
                    aria-label={`${isSelected ? "Deselect" : "Select"} pose ${pose.label}`}
                    aria-pressed={isSelected}
                    className={isSelected ? "pose-thumb selected" : "pose-thumb"}
                    key={pose.id}
                    onClick={() => onToggle?.(pose.id)}
                    title={pose.label}
                    type="button"
                  >
                    <img alt={pose.label} loading="lazy" src={pose.previewUrl ?? `/${pose.preview}`} />
                    <span className="pose-thumb-label">{pose.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
