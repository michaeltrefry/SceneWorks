import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "../api.js";
import { useAppContext } from "../context/AppContext.js";
import { terminalStatuses } from "../jobTypes.js";
import { KpsOverlay } from "../components/KpsOverlay.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
import { assetUrl } from "../components/assetMedia.jsx";
import {
  BUILTIN_DEFAULT_COLLECTION_ID,
  GLOBAL_KEYPOINTS_PROJECT_ID,
  deleteKeypointCollection,
  deleteKeypointPreset,
  keypointSourceImageUrl,
  saveKeypointPreset,
  setDefaultKeypointCollection,
  stageKeypointSource,
  upsertKeypointCollection,
  useKeypointCollections,
  useKeypointPresets,
} from "../keypointLibrary.js";

// The Key Point Library screen (epic 4422, sc-4435) — the face-angle sibling of the Pose
// Library. Three tabs:
//  - "Library": browse every angle preset (built-in 11 + user captures) as its source image
//    with the 5 kps overlaid; delete user presets (built-ins are protected).
//  - "Capture": upload a photo → SCRFD extraction (kps_extract job, sc-4433) → preview the
//    landmarks → name + save as a reusable preset. Extraction failures are explained.
//  - "Collections": compose ordered angle-set collections from any presets, mark one the
//    default (what "generate angles" runs, sc-4450), and override per generation in the studio.
const TABS = [
  ["library", "Library"],
  ["capture", "Capture"],
  ["collections", "Collections"],
];

function presetSourceUrl(preset) {
  return preset?.builtin ? "" : keypointSourceImageUrl(preset?.sourceImageRef);
}

// --- Library tab ------------------------------------------------------------

function PresetCard({ preset, onDelete }) {
  return (
    <div className="keypoint-card">
      <div className="keypoint-card-figure">
        <KpsOverlay
          kps={preset.kps}
          imageUrl={presetSourceUrl(preset)}
          label={`${preset.name} face landmarks`}
        />
      </div>
      <div className="keypoint-card-meta">
        <span className="keypoint-card-name" title={preset.name}>
          {preset.name}
        </span>
        <span className={preset.builtin ? "keypoint-badge builtin" : "keypoint-badge custom"}>
          {preset.builtin ? "Built-in" : "Custom"}
        </span>
      </div>
      {!preset.builtin && onDelete ? (
        <button className="link-button danger" onClick={() => onDelete(preset)} type="button">
          Delete
        </button>
      ) : null}
    </div>
  );
}

function KeypointLibraryPanel({ hidden, presets, loading, error, onDelete }) {
  const builtins = presets.filter((preset) => preset.builtin);
  const custom = presets.filter((preset) => !preset.builtin);
  return (
    <div aria-labelledby="keypoint-tab-library" hidden={hidden} id="keypoint-panel-library" role="tabpanel">
      {error ? <p className="inline-warning">Key Point Library unavailable: {error}</p> : null}
      {loading && !presets.length ? (
        <div className="empty-panel">Loading presets…</div>
      ) : (
        <>
          <div className="keypoint-section">
            <p className="eyebrow">
              Built-in angles <span className="muted">({builtins.length})</span>
            </p>
            <div className="keypoint-grid">
              {builtins.map((preset) => (
                <PresetCard key={preset.id} preset={preset} />
              ))}
            </div>
          </div>
          <div className="keypoint-section">
            <p className="eyebrow">
              Your captures <span className="muted">({custom.length})</span>
            </p>
            {custom.length ? (
              <div className="keypoint-grid">
                {custom.map((preset) => (
                  <PresetCard key={preset.id} preset={preset} onDelete={onDelete} />
                ))}
              </div>
            ) : (
              <div className="empty-panel">
                No captured presets yet — add one from a photo in the Capture tab.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// --- Capture tab ------------------------------------------------------------

// Pick a source — a workspace asset, a character image, or a file upload (the shared
// single-select DatasetAddDialog, same as the Image Editor) → stage it → fire a kps_extract
// job → watch the live jobs list → preview the detected landmarks → name + save. Every source
// is staged to the transient keypoint-uploads area so the extraction + save path is uniform
// (asset picks are re-fetched and staged the same as an upload), and the saved preset retains
// its own copy of the source image.
function KeypointCapturePanel({ hidden, onSaved }) {
  const { token, requestedGpu, jobs = [], assets = [], characters = [] } = useAppContext();
  const [source, setSource] = useState(null); // { path, displayName, previewUrl, sourceAssetId? }
  const [phase, setPhase] = useState("idle"); // idle | extracting | review
  const [jobId, setJobId] = useState(null);
  const [extraction, setExtraction] = useState(null); // { kps, lowConfidence, sourceWidth, sourceHeight }
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);

  const objectUrlRef = useRef(null);
  const clearSource = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);
  useEffect(() => () => clearSource(), [clearSource]);

  const reset = useCallback(() => {
    clearSource();
    setSource(null);
    setExtraction(null);
    setName("");
    setPhase("idle");
    setJobId(null);
  }, [clearSource]);

  // Stage a source, seed the name, and fire the extraction. `previewUrl` backs the overlay
  // preview; `sourceAssetId` records provenance when the source was an existing asset.
  const beginExtraction = useCallback(
    async ({ file, displayName, previewUrl, sourceAssetId }) => {
      setError("");
      setExtraction(null);
      try {
        const staged = await stageKeypointSource(token, file);
        if (!staged?.path) {
          throw new Error("Upload did not return a staged path.");
        }
        const label = displayName ?? staged.displayName ?? file.name ?? "image";
        setSource({ path: staged.path, displayName: label, previewUrl, sourceAssetId });
        setName(stripExtension(label));
        const job = await postExtractJob(token, requestedGpu, staged.path);
        setJobId(job.id);
        setPhase("extracting");
      } catch (err) {
        clearSource();
        setSource(null);
        setError(String(err?.message ?? err));
      }
    },
    [token, requestedGpu, clearSource],
  );

  // File upload (DatasetAddDialog onImport): stage the File directly; preview from an objectUrl.
  const onImport = useCallback(
    async (files) => {
      const file = files?.[0];
      setPickerOpen(false);
      if (!file || !file.type?.startsWith("image/")) {
        return;
      }
      clearSource();
      const objectUrl = URL.createObjectURL(file);
      objectUrlRef.current = objectUrl;
      await beginExtraction({ file, displayName: file.name, previewUrl: objectUrl });
    },
    [beginExtraction, clearSource],
  );

  // Asset / character pick (DatasetAddDialog onAdd): re-fetch the asset's bytes and stage them
  // like an upload, so extraction + save are identical to the file path. Preview from its URL.
  const onAddAsset = useCallback(
    async (ids) => {
      const asset = assets.find((item) => item.id === ids?.[0]);
      setPickerOpen(false);
      if (!asset) {
        return;
      }
      clearSource();
      setError("");
      const url = assetUrl(asset);
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Could not load the selected image (${response.status}).`);
        }
        const blob = await response.blob();
        const file = new File([blob], asset.displayName ?? "image", { type: blob.type || "image/png" });
        await beginExtraction({
          file,
          displayName: asset.displayName ?? asset.id,
          previewUrl: url,
          sourceAssetId: asset.id,
        });
      } catch (err) {
        setError(String(err?.message ?? err));
      }
    },
    [assets, beginExtraction, clearSource],
  );

  // Watch the live (SSE-fed) jobs list for the extraction to finish.
  useEffect(() => {
    if (!jobId) return;
    const job = jobs.find((item) => item.id === jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    setJobId(null);
    if (job.status !== "completed") {
      setError(job.error ?? job.message ?? "Face detection failed.");
      setPhase("idle");
      return;
    }
    const result = job.result ?? {};
    if (!result.detected || !Array.isArray(result.kps)) {
      setError(
        "No usable face was found in that image. Try a clearer, front-facing photo where the face is large in frame.",
      );
      setPhase("idle");
      return;
    }
    setExtraction({
      kps: result.kps,
      lowConfidence: Boolean(result.lowConfidence),
      sourceWidth: result.sourceWidth ?? null,
      sourceHeight: result.sourceHeight ?? null,
    });
    setPhase("review");
  }, [jobId, jobs]);

  const save = useCallback(async () => {
    if (!source || !extraction || !name.trim()) return;
    setSaving(true);
    setError("");
    try {
      await saveKeypointPreset(token, {
        name: name.trim(),
        kps: extraction.kps,
        sourceUploadPath: source.path,
        sourceWidth: extraction.sourceWidth,
        sourceHeight: extraction.sourceHeight,
        ...(source.sourceAssetId ? { sourceAssetId: source.sourceAssetId } : {}),
      });
      reset();
      onSaved?.();
    } catch (err) {
      setError(String(err?.message ?? err));
    } finally {
      setSaving(false);
    }
  }, [source, extraction, name, token, reset, onSaved]);

  return (
    <div aria-labelledby="keypoint-tab-capture" hidden={hidden} id="keypoint-panel-capture" role="tabpanel">
      <div className="keypoint-capture">
        {error ? <p className="inline-warning">{error}</p> : null}
        <div className="toolbar">
          <button onClick={() => setPickerOpen(true)} type="button">
            {source ? "Choose another image" : "Choose image"}
          </button>
          {phase === "extracting" ? <span className="muted">Detecting face landmarks…</span> : null}
        </div>

        {phase === "review" && extraction ? (
          <div className="keypoint-capture-review">
            <div className="keypoint-card-figure large">
              <KpsOverlay kps={extraction.kps} imageUrl={source?.previewUrl} label="captured face landmarks" />
            </div>
            <div className="keypoint-capture-form">
              {extraction.lowConfidence ? (
                <p className="inline-warning">
                  Low detection confidence — the angle may be extreme or the face small. Check the overlay before
                  saving.
                </p>
              ) : null}
              <label>
                Preset name
                <input onChange={(event) => setName(event.target.value)} value={name} placeholder="e.g. My front" />
              </label>
              <div className="toolbar">
                <button className="primary-action" disabled={!name.trim() || saving} onClick={save} type="button">
                  {saving ? "Saving…" : "Save preset"}
                </button>
                <button onClick={reset} type="button">
                  Discard
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-panel">
            Choose a clear, front-facing image — from your assets, a character, or a file upload. We detect the face
            and capture its 5-point framing as a reusable angle preset.
          </div>
        )}
      </div>

      {pickerOpen ? (
        <DatasetAddDialog
          assets={assets}
          characters={characters}
          confirmLabel="Use image"
          eyebrow="Capture"
          fileAccept="image/*"
          fileHint="Drag an image here, or"
          multiple={false}
          onAdd={onAddAsset}
          onClose={() => setPickerOpen(false)}
          onImport={onImport}
          title="Choose a face image"
        />
      ) : null}
    </div>
  );
}

// --- Collections tab --------------------------------------------------------

function KeypointCollectionsPanel({ hidden, presets, collections, collectionsLoading, onChanged }) {
  const { token } = useAppContext();
  const [name, setName] = useState("");
  const [orderedIds, setOrderedIds] = useState([]);
  const [editingId, setEditingId] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const presetById = useMemo(() => Object.fromEntries(presets.map((preset) => [preset.id, preset])), [presets]);

  const resetBuilder = useCallback(() => {
    setName("");
    setOrderedIds([]);
    setEditingId(null);
    setError("");
  }, []);

  const startEdit = useCallback((collection) => {
    setEditingId(collection.id);
    setName(collection.name ?? "");
    setOrderedIds(Array.isArray(collection.orderedPresetIds) ? [...collection.orderedPresetIds] : []);
    setError("");
  }, []);

  const togglePreset = useCallback((id) => {
    setOrderedIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
  }, []);

  const move = useCallback((index, delta) => {
    setOrderedIds((prev) => {
      const next = [...prev];
      const target = index + delta;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }, []);

  const wrap = useCallback(
    async (action) => {
      setBusy(true);
      setError("");
      try {
        await action();
        onChanged?.();
      } catch (err) {
        setError(String(err?.message ?? err));
      } finally {
        setBusy(false);
      }
    },
    [onChanged],
  );

  const save = useCallback(async () => {
    if (!name.trim() || !orderedIds.length) return;
    await wrap(async () => {
      await upsertKeypointCollection(token, {
        ...(editingId ? { id: editingId } : {}),
        name: name.trim(),
        orderedPresetIds: orderedIds,
      });
      resetBuilder();
    });
  }, [name, orderedIds, editingId, token, wrap, resetBuilder]);

  return (
    <div aria-labelledby="keypoint-tab-collections" hidden={hidden} id="keypoint-panel-collections" role="tabpanel">
      <div className="keypoint-collections">
        {error ? <p className="inline-warning">{error}</p> : null}

        <div className="keypoint-section">
          <p className="eyebrow">Collections</p>
          <p className="muted">
            The default collection is what “Generate angle set” runs. Built-in defaults work with zero setup.
          </p>
          {collectionsLoading && !collections.length ? (
            <div className="empty-panel">Loading collections…</div>
          ) : (
            <ul className="keypoint-collection-list">
              {collections.map((collection) => (
                <li className="keypoint-collection-row" key={collection.id}>
                  <div>
                    <span className="keypoint-card-name">{collection.name}</span>
                    <span className="muted"> · {collection.orderedPresetIds?.length ?? 0} angles</span>
                    {collection.isDefault ? <span className="keypoint-badge builtin">Default</span> : null}
                  </div>
                  <div className="toolbar">
                    {!collection.isDefault ? (
                      <button
                        className="link-button"
                        disabled={busy}
                        onClick={() => wrap(() => setDefaultKeypointCollection(token, collection.id))}
                        type="button"
                      >
                        Set default
                      </button>
                    ) : null}
                    {collection.id !== BUILTIN_DEFAULT_COLLECTION_ID ? (
                      <>
                        <button className="link-button" onClick={() => startEdit(collection)} type="button">
                          Edit
                        </button>
                        <button
                          className="link-button danger"
                          disabled={busy}
                          onClick={() => wrap(() => deleteKeypointCollection(token, collection.id))}
                          type="button"
                        >
                          Delete
                        </button>
                      </>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="keypoint-section">
          <p className="eyebrow">{editingId ? "Edit collection" : "New collection"}</p>
          <label>
            Name
            <input onChange={(event) => setName(event.target.value)} value={name} placeholder="e.g. LoRA coverage" />
          </label>

          {orderedIds.length ? (
            <ol className="keypoint-selected-list">
              {orderedIds.map((id, index) => {
                const preset = presetById[id];
                return (
                  <li className="keypoint-selected-row" key={id}>
                    <div className="keypoint-card-figure tiny">
                      <KpsOverlay
                        kps={preset?.kps ?? []}
                        imageUrl={presetSourceUrl(preset)}
                        label={preset?.name ?? id}
                      />
                    </div>
                    <span className="keypoint-card-name">{preset?.name ?? id}</span>
                    <div className="toolbar">
                      <button disabled={index === 0} onClick={() => move(index, -1)} type="button" aria-label="Move up">
                        ↑
                      </button>
                      <button
                        disabled={index === orderedIds.length - 1}
                        onClick={() => move(index, 1)}
                        type="button"
                        aria-label="Move down"
                      >
                        ↓
                      </button>
                      <button className="link-button danger" onClick={() => togglePreset(id)} type="button">
                        Remove
                      </button>
                    </div>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="muted">Pick presets below to build an ordered set.</p>
          )}

          <p className="eyebrow">Add presets</p>
          <div className="keypoint-grid">
            {presets.map((preset) => {
              const selected = orderedIds.includes(preset.id);
              return (
                <button
                  aria-pressed={selected}
                  className={selected ? "keypoint-pick selected" : "keypoint-pick"}
                  key={preset.id}
                  onClick={() => togglePreset(preset.id)}
                  type="button"
                >
                  <div className="keypoint-card-figure">
                    <KpsOverlay kps={preset.kps} imageUrl={presetSourceUrl(preset)} label={preset.name} />
                  </div>
                  <span className="keypoint-card-name">{preset.name}</span>
                </button>
              );
            })}
          </div>

          <div className="toolbar">
            <button
              className="primary-action"
              disabled={!name.trim() || !orderedIds.length || busy}
              onClick={save}
              type="button"
            >
              {editingId ? "Save changes" : "Create collection"}
            </button>
            {editingId || name || orderedIds.length ? (
              <button onClick={resetBuilder} type="button">
                Cancel
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Job helpers ------------------------------------------------------------

function stripExtension(filename) {
  return String(filename || "").replace(/\.[^.]+$/, "");
}

async function postExtractJob(token, requestedGpu, sourcePath) {
  return apiFetch("/api/v1/jobs", token, {
    method: "POST",
    body: JSON.stringify({
      type: "kps_extract",
      // The reserved keypoints project owns the staged upload; the worker reads sourcePath.
      projectId: GLOBAL_KEYPOINTS_PROJECT_ID,
      projectName: "Key Point Library",
      requestedGpu,
      payload: { projectId: GLOBAL_KEYPOINTS_PROJECT_ID, sourcePath },
    }),
  });
}

// --- Screen -----------------------------------------------------------------

export function KeyPointLibraryScreen() {
  const [activeTab, setActiveTab] = useState("library");
  const { token } = useAppContext();
  const { presets, loading: presetsLoading, error: presetsError, reload: reloadPresets } = useKeypointPresets();
  const {
    collections,
    loading: collectionsLoading,
    reload: reloadCollections,
  } = useKeypointCollections();

  const deletePreset = useCallback(
    async (preset) => {
      await deleteKeypointPreset(token, preset.id);
      await reloadPresets();
      await reloadCollections();
    },
    [token, reloadPresets, reloadCollections],
  );

  return (
    <section className="main-surface library-surface keypoint-library-surface">
      <div className="surface-header hero">
        <div className="section-heading">
          <p className="eyebrow">Key Point Library</p>
          <h2>Face angles</h2>
          <p className="hero-blurb">
            Capture face-angle framing presets from photos and compose them into angle-set collections. The default
            collection drives Character Studio’s “Generate angle set”.
          </p>
        </div>
        <div className="segmented-control" role="tablist" aria-label="Key Point Library sections">
          {TABS.map(([id, label]) => (
            <button
              aria-controls={`keypoint-panel-${id}`}
              aria-selected={activeTab === id}
              className={activeTab === id ? "active" : ""}
              id={`keypoint-tab-${id}`}
              key={id}
              onClick={() => setActiveTab(id)}
              role="tab"
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <KeypointLibraryPanel
        hidden={activeTab !== "library"}
        presets={presets}
        loading={presetsLoading}
        error={presetsError}
        onDelete={deletePreset}
      />
      <KeypointCapturePanel
        hidden={activeTab !== "capture"}
        onSaved={() => {
          setActiveTab("library");
          reloadPresets();
        }}
      />
      <KeypointCollectionsPanel
        hidden={activeTab !== "collections"}
        presets={presets}
        collections={collections}
        collectionsLoading={collectionsLoading}
        onChanged={() => {
          reloadCollections();
          reloadPresets();
        }}
      />
    </section>
  );
}
