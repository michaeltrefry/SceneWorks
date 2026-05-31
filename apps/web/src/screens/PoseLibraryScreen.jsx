import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL, apiFetch, isAbortError } from "../api.js";
import { AssetDetail, AssetGrid, FullscreenPreview, emptyTrash } from "../components/assetPanels.jsx";
import { AssetThumbnail } from "../components/assetMedia.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
import { useAppContext } from "../context/AppContext.js";
import { terminalStatuses } from "../jobTypes.js";
import { GLOBAL_POSES_PROJECT_ID } from "../poseLibrary.js";

// The Pose Library screen (epic 2282). Two tabs:
//  - "Poses": manage the global pose store — user-created type:"pose" assets in the
//    reserved project, as an image grid + viewer + Trashcan (reusing the shared asset
//    panels). Built-in poses stay bundled (read-only) and surface in the generation
//    pose pickers, not here.
//  - "Create": photo -> DWPose -> categorize -> save (sc-2287).
// The reserved project is hidden from the project switcher, so these assets never
// appear in the Assets/Character views; we address it directly here.
const TABS = [
  ["poses", "Poses"],
  ["create", "Create"],
];

const UNCATEGORIZED = "uncategorized";

// --- Create tab helpers -----------------------------------------------------

function basename(path) {
  return String(path || "")
    .split(/[\\/]/)
    .pop();
}

// Flatten a completed pose_detect job result into per-person candidate cards.
// Each candidate carries everything the save endpoint needs: the cached skeleton
// (jobId + filename), the source dimensions, and the keypoint metadata.
function buildCandidates(job) {
  const out = [];
  const sources = job?.result?.sources ?? [];
  sources.forEach((source, sourceIndex) => {
    (source.poses ?? []).forEach((pose) => {
      out.push({
        key: `${sourceIndex}:${pose.personIndex ?? out.length}`,
        jobId: job.id,
        skeletonFile: basename(pose.skeletonPreview),
        sourceDisplayName: source.displayName ?? basename(source.sourcePath) ?? "image",
        width: source.sourceWidth ?? null,
        height: source.sourceHeight ?? null,
        facing: pose.facing ?? "front",
        pose: {
          personIndex: pose.personIndex ?? 0,
          bbox: pose.bbox ?? null,
          facing: pose.facing ?? "front",
          meanConf: pose.meanConf ?? null,
          keypoints: pose.keypoints ?? [],
          hands: pose.hands ?? [[], []],
          face: pose.face ?? [],
          sourceAspect: source.sourceAspect ?? null,
          sourceAssetId: source.sourceAssetId ?? null,
        },
        keep: true,
        category: "",
        tagsText: "",
      });
    });
  });
  return out;
}

// The "Create" tab body: pick photos (DatasetAddDialog), run DWPose, then review,
// categorize, and save one whole-body pose per detected person to the store.
function PoseCreatePanel({ hidden, categories, onSaved }) {
  const { token, activeProject, assets = [], characters = [], requestedGpu, jobs = [] } = useAppContext();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [sources, setSources] = useState([]); // selected source asset records
  const [phase, setPhase] = useState("idle"); // idle | detecting | review
  const [jobId, setJobId] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [importing, setImporting] = useState(false);

  // Sources are normalized entries (deduped by `key`): either an asset-backed pick
  // ({kind:"asset", asset, assetId}) or a transient File-Upload ({kind:"upload",
  // path, objectUrl}).
  const mergeSources = useCallback((entries) => {
    setSources((prev) => {
      const seen = new Set(prev.map((source) => source.key));
      return [...prev, ...entries.filter((source) => source && !seen.has(source.key))];
    });
  }, []);

  // File tab: stage each image to the TRANSIENT pose-source area (NOT a workspace
  // asset) — the worker reads it by path and deletes it after detection (epic 2282).
  const handleImport = useCallback(
    async (files) => {
      const images = Array.from(files).filter((file) => file.type?.startsWith("image/"));
      if (!images.length) return;
      setImporting(true);
      try {
        const body = new FormData();
        for (const file of images) body.append("file", file);
        const result = await apiFetch("/api/v1/poses/sources", token, { method: "POST", body });
        const staged = Array.isArray(result?.sources) ? result.sources : [];
        mergeSources(
          staged.map((src, index) => ({
            key: src.path,
            kind: "upload",
            path: src.path,
            displayName: src.displayName ?? images[index]?.name ?? "image",
            objectUrl: images[index] ? URL.createObjectURL(images[index]) : undefined,
          })),
        );
        setError("");
      } catch (err) {
        setError(String(err?.message ?? err));
      } finally {
        setImporting(false);
      }
    },
    [token, mergeSources],
  );

  // Library / Character tabs: resolve the picked ids to asset records.
  const handleAdd = useCallback(
    (selectedIds) => {
      mergeSources(
        selectedIds
          .map((id) => assets.find((asset) => asset.id === id))
          .filter(Boolean)
          .map((asset) => ({
            key: asset.id,
            kind: "asset",
            asset,
            assetId: asset.id,
            displayName: asset.displayName ?? asset.id,
          })),
      );
    },
    [assets, mergeSources],
  );

  const removeSource = (key) =>
    setSources((prev) => {
      const target = prev.find((source) => source.key === key);
      if (target?.objectUrl) URL.revokeObjectURL(target.objectUrl);
      return prev.filter((source) => source.key !== key);
    });

  // Revoke any object URLs on unmount (latest set via ref).
  const sourcesRef = useRef(sources);
  sourcesRef.current = sources;
  useEffect(
    () => () => {
      for (const source of sourcesRef.current) if (source.objectUrl) URL.revokeObjectURL(source.objectUrl);
    },
    [],
  );

  const generate = useCallback(async () => {
    if (!activeProject || !sources.length) return;
    setError("");
    setCandidates([]);
    try {
      const job = await apiFetch("/api/v1/jobs", token, {
        method: "POST",
        body: JSON.stringify({
          type: "pose_detect",
          projectId: activeProject.id,
          projectName: activeProject.name ?? null,
          requestedGpu,
          payload: {
            projectId: activeProject.id,
            sources: sources.map((source) =>
              source.kind === "upload"
                ? { path: source.path, displayName: source.displayName, temp: true }
                : { assetId: source.assetId, displayName: source.displayName },
            ),
          },
        }),
      });
      setJobId(job.id);
      setPhase("detecting");
    } catch (err) {
      setError(String(err?.message ?? err));
    }
  }, [activeProject, sources, token, requestedGpu]);

  // Watch the live (SSE-fed) jobs list for the fired detector job to finish.
  useEffect(() => {
    if (!jobId) return;
    const job = jobs.find((item) => item.id === jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    if (job.status === "completed") {
      const built = buildCandidates(job);
      setCandidates(built);
      setPhase("review");
      if (!built.length) setError("No people were detected in the selected images.");
    } else {
      setError(job.error ?? job.message ?? "Pose detection failed.");
      setPhase("idle");
    }
    setJobId(null);
  }, [jobId, jobs]);

  const updateCandidate = (key, changes) =>
    setCandidates((prev) => prev.map((candidate) => (candidate.key === key ? { ...candidate, ...changes } : candidate)));

  const keepCount = candidates.filter((candidate) => candidate.keep).length;

  const save = useCallback(async () => {
    const keep = candidates.filter((candidate) => candidate.keep);
    if (!keep.length) return;
    setSaving(true);
    setError("");
    try {
      const poses = keep.map((candidate) => ({
        jobId: candidate.jobId,
        skeletonFile: candidate.skeletonFile,
        category: candidate.category.trim() || null,
        tags: candidate.tagsText
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean),
        width: candidate.width,
        height: candidate.height,
        pose: candidate.pose,
      }));
      await apiFetch("/api/v1/poses", token, { method: "POST", body: JSON.stringify({ poses }) });
      setSources((prev) => {
        for (const source of prev) if (source.objectUrl) URL.revokeObjectURL(source.objectUrl);
        return [];
      });
      setCandidates([]);
      setPhase("idle");
      onSaved?.();
    } catch (err) {
      setError(String(err?.message ?? err));
    } finally {
      setSaving(false);
    }
  }, [candidates, token, onSaved]);

  return (
    <div aria-labelledby="pose-library-tab-create" hidden={hidden} id="pose-library-panel-create" role="tabpanel">
      {!activeProject ? (
        <div className="empty-panel">Open a workspace to create poses from photos.</div>
      ) : (
        <div className="pose-create">
          {error ? <p className="inline-warning">{error}</p> : null}

          <div className="toolbar">
            <button onClick={() => setDialogOpen(true)} type="button">
              Add images
            </button>
            <button
              className="primary-action"
              disabled={!sources.length || phase === "detecting"}
              onClick={generate}
              type="button"
            >
              {phase === "detecting"
                ? "Detecting…"
                : `Generate poses${sources.length ? ` (${sources.length})` : ""}`}
            </button>
          </div>

          {sources.length ? (
            <div className="dataset-add-grid">
              {sources.map((source) => (
                <div className="dataset-add-card" key={source.key}>
                  {source.kind === "upload" ? (
                    <img alt="" src={source.objectUrl} />
                  ) : (
                    <AssetThumbnail asset={source.asset} />
                  )}
                  <span>{source.displayName}</span>
                  <button onClick={() => removeSource(source.key)} type="button">
                    Remove
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-panel">Add one or more photos, then generate whole-body pose skeletons.</div>
          )}

          {phase === "review" && candidates.length ? (
            <>
              <div className="toolbar">
                <p className="eyebrow">
                  Review {candidates.length} candidate{candidates.length === 1 ? "" : "s"} — {keepCount} kept
                </p>
                <button className="primary-action" disabled={!keepCount || saving} onClick={save} type="button">
                  {saving ? "Saving…" : `Save ${keepCount} pose${keepCount === 1 ? "" : "s"}`}
                </button>
              </div>
              <datalist id="pose-category-suggestions">
                {categories.map((category) => (
                  <option key={category} value={category} />
                ))}
              </datalist>
              {/* Reuse the shared selectable-image grid (DatasetAddDialog convention):
                  the skeleton <img> is bounded by `.dataset-add-card img`, and the
                  `selected` ring marks candidates kept for saving. */}
              <div className="dataset-add-grid">
                {candidates.map((candidate) => (
                  <div
                    className={candidate.keep ? "dataset-add-card selected" : "dataset-add-card"}
                    key={candidate.key}
                  >
                    <img
                      alt={`Pose skeleton from ${candidate.sourceDisplayName}`}
                      src={`${API_BASE_URL}/api/v1/poses/preview/${encodeURIComponent(
                        candidate.jobId,
                      )}/${encodeURIComponent(candidate.skeletonFile)}`}
                    />
                    <span>
                      {candidate.sourceDisplayName} · {candidate.facing}
                    </span>
                    <label>
                      Category
                      <input
                        list="pose-category-suggestions"
                        onChange={(event) => updateCandidate(candidate.key, { category: event.target.value })}
                        placeholder="e.g. standing"
                        value={candidate.category}
                      />
                    </label>
                    <label>
                      Tags
                      <input
                        onChange={(event) => updateCandidate(candidate.key, { tagsText: event.target.value })}
                        placeholder="comma, separated"
                        value={candidate.tagsText}
                      />
                    </label>
                    <button onClick={() => updateCandidate(candidate.key, { keep: !candidate.keep })} type="button">
                      {candidate.keep ? "Discard" : "Keep"}
                    </button>
                  </div>
                ))}
              </div>
            </>
          ) : null}

          {dialogOpen ? (
            <DatasetAddDialog
              assets={assets}
              characters={characters}
              importing={importing}
              memberIds={sources.filter((source) => source.kind === "asset").map((source) => source.assetId)}
              onAdd={handleAdd}
              onClose={() => setDialogOpen(false)}
              onImport={handleImport}
            />
          ) : null}
        </div>
      )}
    </div>
  );
}

export function PoseLibraryScreen() {
  const { token } = useAppContext();
  const [activeTab, setActiveTab] = useState("poses");
  const [poses, setPoses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [previewId, setPreviewId] = useState(null);
  const [assetMode, setAssetMode] = useState("assets");
  const [categoryFilter, setCategoryFilter] = useState("all");

  const refresh = useCallback(
    async (signal) => {
      try {
        setLoading(true);
        const items = await apiFetch(
          `/api/v1/projects/${GLOBAL_POSES_PROJECT_ID}/assets?includeRejected=true&includeTrashed=true`,
          token,
          signal ? { signal } : {},
        );
        setPoses((Array.isArray(items) ? items : []).filter((asset) => asset.type === "pose"));
        setError("");
      } catch (err) {
        if (isAbortError(err)) return;
        setError(String(err?.message ?? err));
      } finally {
        setLoading(false);
      }
    },
    [token],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  // Mutations target the reserved project (asset.projectId) and refetch locally — the
  // app-level asset mutators refresh the *active* project, not this one.
  const updateAssetStatus = useCallback(
    async (asset, changes) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/status`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      await refresh();
    },
    [token, refresh],
  );
  const updateAssetTags = useCallback(
    async (asset, tags) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/tags`, token, {
        method: "PATCH",
        body: JSON.stringify({ tags }),
      });
      await refresh();
    },
    [token, refresh],
  );
  const deleteAsset = useCallback(
    async (asset) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}`, token, { method: "DELETE" });
      await refresh();
    },
    [token, refresh],
  );
  const purgeAsset = useCallback(
    async (asset) => {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/purge`, token, { method: "DELETE" });
      await refresh();
    },
    [token, refresh],
  );

  const categoryOf = (asset) => asset.pose?.category || UNCATEGORIZED;
  const categories = useMemo(() => [...new Set(poses.map(categoryOf))].sort(), [poses]);
  const availableTags = useMemo(
    () => [...new Set(poses.flatMap((asset) => (Array.isArray(asset.tags) ? asset.tags : [])))].sort(),
    [poses],
  );

  const inFilter = (asset) => categoryFilter === "all" || categoryOf(asset) === categoryFilter;
  const inMode = (asset) => (assetMode === "trashcan" ? Boolean(asset.status?.trashed) : !asset.status?.trashed);
  const visible = poses.filter((asset) => inFilter(asset) && inMode(asset));
  const trashedInView = poses.filter((asset) => inFilter(asset) && asset.status?.trashed);
  const selected = poses.find((asset) => asset.id === selectedId) ?? null;
  // Single click selects (side detail); preview opens the shared fullscreen modal —
  // same larger view as the rest of the asset library. Arrows step the visible set.
  const onPreview = (asset) => setPreviewId(asset.id);
  const previewAsset = poses.find((asset) => asset.id === previewId) ?? null;
  const previewIndex = visible.findIndex((asset) => asset.id === previewId);
  const previousAsset = previewIndex > 0 ? visible[previewIndex - 1] : null;
  const nextAsset = previewIndex >= 0 && previewIndex < visible.length - 1 ? visible[previewIndex + 1] : null;

  // Group the visible poses by category for the grid (category + tags shown per tile).
  const groups = useMemo(() => {
    const byCategory = new Map();
    for (const asset of visible) {
      const key = categoryOf(asset);
      if (!byCategory.has(key)) byCategory.set(key, []);
      byCategory.get(key).push(asset);
    }
    return [...byCategory.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [visible]);

  return (
    <section className="main-surface library-surface pose-library-surface">
      <div className="surface-header hero">
        <div className="section-heading">
          <p className="eyebrow">Pose Library</p>
          <h2>Poses</h2>
          <p className="hero-blurb">
            Manage your whole-body pose skeletons — discard, restore, tag, and categorize. Create new poses from photos
            in the Create tab.
          </p>
        </div>
        <div className="segmented-control" role="tablist" aria-label="Pose Library sections">
          {TABS.map(([id, label]) => (
            <button
              aria-controls={`pose-library-panel-${id}`}
              aria-selected={activeTab === id}
              className={activeTab === id ? "active" : ""}
              id={`pose-library-tab-${id}`}
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

      <div
        aria-labelledby="pose-library-tab-poses"
        hidden={activeTab !== "poses"}
        id="pose-library-panel-poses"
        role="tabpanel"
      >
        <div className="toolbar">
          <select
            aria-label="Pose category"
            onChange={(event) => setCategoryFilter(event.target.value)}
            value={categoryFilter}
          >
            <option value="all">All categories</option>
            {categories.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
          <div className="segmented-control" role="group" aria-label="Pose collection">
            <button className={assetMode === "assets" ? "active" : ""} onClick={() => setAssetMode("assets")} type="button">
              Poses
            </button>
            <button
              className={assetMode === "trashcan" ? "active" : ""}
              onClick={() => setAssetMode("trashcan")}
              type="button"
            >
              Trashcan
            </button>
          </div>
          {assetMode === "trashcan" ? (
            <button
              className="danger-action empty-trash-button"
              disabled={!trashedInView.length}
              onClick={() => emptyTrash(trashedInView, purgeAsset)}
              type="button"
            >
              Empty Trash ({trashedInView.length})
            </button>
          ) : null}
        </div>

        {error ? <p className="inline-warning">Pose library unavailable: {error}</p> : null}

        <div className="library-layout">
          <div className="pose-library-grids">
            {loading && !poses.length ? (
              <div className="empty-panel">Loading poses…</div>
            ) : !visible.length ? (
              <div className="empty-panel">
                {assetMode === "trashcan"
                  ? "Trashcan is empty."
                  : "No saved poses yet — create some from photos in the Create tab."}
              </div>
            ) : (
              groups.map(([category, items]) => (
                <div className="pose-category" key={category}>
                  <p className="eyebrow">
                    {category} <span className="muted">({items.length})</span>
                  </p>
                  <AssetGrid
                    assets={items}
                    onPreview={onPreview}
                    selectedAsset={selected}
                    setSelectedAssetId={setSelectedId}
                  />
                </div>
              ))
            )}
          </div>
          <AssetDetail
            asset={selected}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
            onPreview={onPreview}
            updateAssetStatus={updateAssetStatus}
            updateAssetTags={updateAssetTags}
            availableTags={availableTags}
          />
        </div>
      </div>

      <PoseCreatePanel
        categories={categories.filter((category) => category !== UNCATEGORIZED)}
        hidden={activeTab !== "create"}
        onSaved={() => {
          setActiveTab("poses");
          refresh();
        }}
      />

      {previewAsset ? (
        <FullscreenPreview
          asset={previewAsset}
          deleteAsset={deleteAsset}
          purgeAsset={purgeAsset}
          updateAssetStatus={updateAssetStatus}
          onClose={() => setPreviewId(null)}
          onPreviewAsset={(asset) => setPreviewId(asset.id)}
          previousAsset={previousAsset}
          nextAsset={nextAsset}
        />
      ) : null}
    </section>
  );
}
