import React, { useRef, useState } from "react";

// The layers panel (sc-6118, Workstream E of epic 6087): the user-facing control
// surface for the sc-6117 raster layer stack. Presentational — it reflects the
// `layers` / `activeLayerId` it is given and calls the handler props, which the
// editor wires to the layer-stack ops (each checkpointed for undo, sc-6106).
//
// Rows render TOP layer first (the reverse of the bottom→top stack array) — the
// standard layers-panel convention where the top of the list is the top of the
// stack. Thumbnails reuse each layer's live object URL, which the editor already
// lifecycle-manages (revoked when the layer is evicted), so the panel creates and
// revokes nothing itself.
export function LayersPanel({
  layers,
  activeLayerId,
  busy = false,
  onSelect,
  onToggleVisible,
  onSetOpacity,
  onRename,
  onReorder,
  onAdd,
  onDelete,
  onDuplicate,
}) {
  const [renamingId, setRenamingId] = useState(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [dragId, setDragId] = useState(null);
  // One undo checkpoint per opacity DRAG, not per tick: the first change of a
  // gesture commits (checkpoints), the rest just preview. Reset when the gesture
  // ends (pointer up / blur / key up), covering both mouse and keyboard input.
  const opacityGestureRef = useRef(false);

  const count = layers.length;
  const rows = layers.map((layer, index) => ({ layer, index })).reverse();

  const startRename = (layer) => {
    setRenamingId(layer.id);
    setRenameDraft(layer.name);
  };
  const commitRename = () => {
    if (renamingId) {
      const name = renameDraft.trim();
      if (name) onRename(renamingId, name);
    }
    setRenamingId(null);
  };

  const handleOpacityChange = (id, value) => {
    const start = !opacityGestureRef.current;
    opacityGestureRef.current = true;
    onSetOpacity(id, Number(value) / 100, start);
  };
  const endOpacityGesture = () => {
    opacityGestureRef.current = false;
  };

  // Drop row `dragId` onto `targetId` → move it to that target's stack index.
  const handleDrop = (targetId) => {
    if (dragId && dragId !== targetId) {
      const toIndex = layers.findIndex((layer) => layer.id === targetId);
      if (toIndex >= 0) onReorder(dragId, toIndex);
    }
    setDragId(null);
  };

  return (
    <aside className="image-editor-layers" aria-label="Layers">
      <div className="image-editor-layers-head">
        <span className="image-editor-layers-title">Layers</span>
        <button
          className="image-editor-layers-add"
          disabled={busy}
          onClick={onAdd}
          title="Add a blank layer"
          aria-label="Add layer"
          type="button"
        >
          + Layer
        </button>
      </div>
      <div className="image-editor-layers-list" role="list">
        {rows.map(({ layer, index }) => {
          const isActive = layer.id === activeLayerId;
          const pct = Math.round(layer.opacity * 100);
          return (
            <div
              key={layer.id}
              role="listitem"
              className={isActive ? "image-editor-layer active" : "image-editor-layer"}
              aria-current={isActive ? "true" : undefined}
              draggable={!busy && renamingId !== layer.id}
              onDragStart={() => setDragId(layer.id)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => handleDrop(layer.id)}
              onDragEnd={() => setDragId(null)}
              onClick={() => onSelect(layer.id)}
            >
              <button
                className="image-editor-layer-vis"
                aria-label={layer.visible ? "Hide layer" : "Show layer"}
                aria-pressed={layer.visible}
                onClick={(event) => {
                  event.stopPropagation();
                  onToggleVisible(layer.id);
                }}
                title={layer.visible ? "Hide layer" : "Show layer"}
                type="button"
              >
                {layer.visible ? "●" : "○"}
              </button>
              {layer.objectUrl ? (
                <img
                  className="image-editor-layer-thumb"
                  src={layer.objectUrl}
                  alt=""
                  draggable={false}
                />
              ) : (
                <span className="image-editor-layer-thumb image-editor-layer-thumb-empty" />
              )}
              <span className="image-editor-layer-body">
                {renamingId === layer.id ? (
                  <input
                    className="image-editor-layer-rename"
                    value={renameDraft}
                    autoFocus
                    onChange={(event) => setRenameDraft(event.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") commitRename();
                      else if (event.key === "Escape") setRenamingId(null);
                    }}
                    onClick={(event) => event.stopPropagation()}
                    aria-label="Layer name"
                  />
                ) : (
                  <span
                    className="image-editor-layer-name"
                    title="Double-click to rename"
                    onDoubleClick={(event) => {
                      event.stopPropagation();
                      startRename(layer);
                    }}
                  >
                    {layer.name}
                  </span>
                )}
                <label className="image-editor-layer-opacity" onClick={(event) => event.stopPropagation()}>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={pct}
                    aria-label={`${layer.name} opacity`}
                    onChange={(event) => handleOpacityChange(layer.id, event.target.value)}
                    onPointerUp={endOpacityGesture}
                    onKeyUp={endOpacityGesture}
                    onBlur={endOpacityGesture}
                  />
                  <span className="image-editor-layer-opacity-val">{pct}%</span>
                </label>
              </span>
              <span className="image-editor-layer-ops">
                <button
                  className="image-editor-layer-op"
                  disabled={busy || index === count - 1}
                  onClick={(event) => {
                    event.stopPropagation();
                    onReorder(layer.id, index + 1);
                  }}
                  title="Move up"
                  aria-label="Move layer up"
                  type="button"
                >
                  ▲
                </button>
                <button
                  className="image-editor-layer-op"
                  disabled={busy || index === 0}
                  onClick={(event) => {
                    event.stopPropagation();
                    onReorder(layer.id, index - 1);
                  }}
                  title="Move down"
                  aria-label="Move layer down"
                  type="button"
                >
                  ▼
                </button>
                <button
                  className="image-editor-layer-op"
                  disabled={busy}
                  onClick={(event) => {
                    event.stopPropagation();
                    onDuplicate(layer.id);
                  }}
                  title="Duplicate layer"
                  aria-label="Duplicate layer"
                  type="button"
                >
                  ⧉
                </button>
                <button
                  className="image-editor-layer-op"
                  disabled={busy || count <= 1}
                  onClick={(event) => {
                    event.stopPropagation();
                    onDelete(layer.id);
                  }}
                  title="Delete layer"
                  aria-label="Delete layer"
                  type="button"
                >
                  ✕
                </button>
              </span>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
