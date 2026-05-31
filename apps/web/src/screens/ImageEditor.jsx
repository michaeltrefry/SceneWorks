import React, { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Transformer } from "react-konva";
import { apiFetch } from "../api.js";
import { terminalStatuses } from "../jobTypes.js";
import { useAppContext } from "../context/AppContext.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";

const MIN_SCALE = 0.05;
const MAX_SCALE = 16;
const ZOOM_STEP = 1.2;
const MIN_CROP_PX = 8;

// Tools still to come in epic 2427 — rendered as an inert scaffold so the frame
// (and the next slices' insertion points) are in place. Move + Crop + Upscale are live.
const UPCOMING_TOOLS = [
  { id: "edit", label: "AI Edit", story: "sc-2435" },
  { id: "detail", label: "Detail", story: "sc-2438" },
  { id: "color", label: "Color", story: "sc-2439" },
];

// Upscale engines + their valid factors (sc-2433). Mirrors the engines the worker
// supports (image_adapters.create_image_upscaler): Real-ESRGAN 2x/4x, AuraSR 4x.
const UPSCALE_ENGINES = [
  { key: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
  { key: "aura-sr", label: "AuraSR", factors: [4] },
];

export function upscaleFactorsForEngine(engineKey) {
  const found = UPSCALE_ENGINES.find((entry) => entry.key === engineKey);
  return found ? found.factors : [2, 4];
}

// The `POST /api/v1/jobs` body for a standalone image_upscale job (sc-2431). The
// worker reads sourceAssetId/factor/engine from the payload; displayName names the
// result. Pure for unit testing.
export function buildUpscaleJobBody({ project, requestedGpu, sourceAssetId, factor, engine, displayName }) {
  return {
    type: "image_upscale",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload: { projectId: project.id, sourceAssetId, factor, engine, displayName },
  };
}

// Filename for a Save / Download export (sc-2434): the source name with an
// "-edited" suffix before the extension, always .png — the working image is
// rasterized to PNG, so the original extension would be misleading. Pure.
export function editedFilename(source) {
  const base = (source?.name || "image").replace(/\.[^./\\]+$/, "").trim() || "image";
  return `${base}-edited.png`;
}

// Provenance for a saved edit, stored under the new asset's top-level `extra`
// (sc-2434): which source it was derived from + the ordered edit chain
// (crop/upscale/…) applied this session. Pure for unit testing.
export function buildSaveProvenance({ source, edits, width, height }) {
  return {
    editor: "image_editor",
    source: source?.assetId
      ? { kind: "asset", assetId: source.assetId, name: source.name ?? null }
      : { kind: "upload", name: source?.name ?? null },
    edits: edits ?? [],
    width: width ?? null,
    height: height ?? null,
  };
}

// Predefined crop ratios (width / height). Rotate swaps to the transpose; 1:1 and
// Freeform are unaffected.
const CROP_RATIOS = [
  { key: "free", label: "Freeform", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "5:7", label: "5:7", ratio: 5 / 7 },
  { key: "8:10", label: "8:10", ratio: 8 / 10 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// Resolve a ratio key (+ rotate) to a concrete width/height ratio, or null for
// freeform. Rotating transposes non-square ratios (3:4 → 4:3); 1:1 is a no-op.
export function cropRatioForKey(key, rotated) {
  const found = CROP_RATIOS.find((entry) => entry.key === key);
  const base = found ? found.ratio : null;
  if (base == null || base === 1) return base;
  return rotated ? 1 / base : base;
}

// Largest rect of the given ratio that fits in the image, centered. Freeform
// (null ratio) defaults to a centered 80% box. Returns image-pixel coords.
export function centeredCropRect(imgW, imgH, ratio) {
  if (ratio == null) {
    const w = imgW * 0.8;
    const h = imgH * 0.8;
    return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
  }
  let w = imgW;
  let h = w / ratio;
  if (h > imgH) {
    h = imgH;
    w = h * ratio;
  }
  return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
}

// The four dim rectangles that mask everything outside the crop rect (image coords).
function cropOverlayRects(imgW, imgH, rect) {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  return [
    { x: 0, y: 0, width: imgW, height: rect.y },
    { x: 0, y: bottom, width: imgW, height: imgH - bottom },
    { x: 0, y: rect.y, width: rect.x, height: rect.height },
    { x: right, y: rect.y, width: imgW - right, height: rect.height },
  ];
}

// Decode a blob into an HTMLImageElement via a same-origin object: URL. Asset
// files are served cross-origin from the API in local dev, so loading the bytes
// this way (rather than an <img crossOrigin> against the file URL) guarantees the
// Konva canvas is never tainted — later crop/export (sc-2430/sc-2434) need to read
// pixels back. Resolves { image, objectUrl }; caller owns revoking objectUrl.
function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => resolve({ image, objectUrl });
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not decode image"));
    };
    image.src = objectUrl;
  });
}

export function ImageEditor() {
  const {
    activeProject,
    assets,
    characters,
    setPreviewAsset,
    token,
    requestedGpu,
    jobs,
    importAsset,
    purgeAsset,
    registerLeaveGuard,
  } = useAppContext();

  // The working-image session: the single bitmap every tool operates on, plus its
  // provenance. This state is the contract consumed by crop/upscale/save and the
  // later AI tools (epic 2427). `objectUrl` is tracked so we can revoke it.
  const [working, setWorking] = useState(null);
  const [status, setStatus] = useState({ loading: false, error: "" });
  const [pickerOpen, setPickerOpen] = useState(false);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });

  // Crop tool (sc-2430): client-side, rasterized into a new working image on Apply.
  const [tool, setTool] = useState("move");
  const [ratioKey, setRatioKey] = useState("free");
  const [rotated, setRotated] = useState(false);
  const [cropRect, setCropRect] = useState(null); // image-pixel coords, or null

  // Upscale tool (sc-2433): engine + factor for the in-flight request.
  const [upscaleEngine, setUpscaleEngine] = useState("real-esrgan");
  const [upscaleFactor, setUpscaleFactor] = useState(2);

  // Save / export (sc-2434). `dirty` tracks edits not yet persisted to the Library;
  // `edits` is the ordered provenance chain; `savedAssetId` flags a completed Save
  // for the bar's "Saved" hint. A fresh open clears all three.
  const [dirty, setDirty] = useState(false);
  const [edits, setEdits] = useState([]);
  const [saving, setSaving] = useState(false);
  const [savedAssetId, setSavedAssetId] = useState(null);
  // An in-flight AI op (upscale now; AI-edit / detail later) on the working image.
  // The seam (sc-2432): stage the working bitmap as a scratch asset, run a worker
  // job against it, load the result back, then purge the scratch + result so the
  // session only persists on Save. { jobId, scratch (asset), source, label } | null.
  const [aiOp, setAiOp] = useState(null);

  const containerRef = useRef(null);
  const objectUrlRef = useRef(null);
  const needsFitRef = useRef(false);
  const cropRectRef = useRef(null);
  const transformerRef = useRef(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  const imageAssets = (assets ?? []).filter(assetCanRenderAsImage);

  // Track the container size so the Konva stage fills the available canvas area.
  // Measure once up front (a ResizeObserver alone can miss the first layout) and
  // then observe for later window / layout changes.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const measure = () => setStageSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Revoke the live object URL when the editor unmounts.
  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  const fitToView = useCallback(() => {
    if (!working || !stageSize.width || !stageSize.height) return;
    const scale = clamp(
      Math.min(stageSize.width / working.width, stageSize.height / working.height) * 0.92,
      MIN_SCALE,
      MAX_SCALE,
    );
    setView({
      scale,
      x: (stageSize.width - working.width * scale) / 2,
      y: (stageSize.height - working.height * scale) / 2,
    });
  }, [working, stageSize.width, stageSize.height]);

  // Fit a freshly loaded image once the stage has been measured (the stage may be
  // 0×0 on the first render before the ResizeObserver fires).
  useEffect(() => {
    if (needsFitRef.current && working && stageSize.width && stageSize.height) {
      needsFitRef.current = false;
      fitToView();
    }
  }, [working, stageSize.width, stageSize.height, fitToView]);

  const installWorkingImage = useCallback((image, objectUrl, source) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = objectUrl;
    needsFitRef.current = true;
    setTool("move");
    setCropRect(null);
    setWorking({
      image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      source,
    });
  }, []);

  const openFromBlob = useCallback(
    async (blob, source) => {
      setStatus({ loading: true, error: "" });
      try {
        const { image, objectUrl } = await blobToImage(blob);
        installWorkingImage(image, objectUrl, source);
        // A freshly opened image is a clean session — clear edit/provenance state.
        setEdits([]);
        setDirty(false);
        setSavedAssetId(null);
        setStatus({ loading: false, error: "" });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not open image" });
      }
    },
    [installWorkingImage],
  );

  const openAsset = useCallback(
    async (assetId) => {
      const asset = imageAssets.find((item) => item.id === assetId);
      if (!asset) return;
      const url = assetUrl(asset);
      if (!url) {
        setStatus({ loading: false, error: "Asset has no media file" });
        return;
      }
      setStatus({ loading: true, error: "" });
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to load asset (${res.status})`);
        const blob = await res.blob();
        await openFromBlob(blob, {
          kind: "asset",
          assetId: asset.id,
          name: asset.displayName ?? asset.id,
        });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not load asset" });
      }
    },
    [imageAssets, openFromBlob],
  );

  const openFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) {
        setStatus({ loading: false, error: "Please choose an image file" });
        return;
      }
      openFromBlob(file, { kind: "upload", name: file.name });
    },
    [openFromBlob],
  );

  function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file && confirmDiscardEdits()) openFile(file);
  }

  function handleWheel(event) {
    event.evt.preventDefault();
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer) return;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * (event.evt.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP), MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (pointer.x - view.x) / oldScale, y: (pointer.y - view.y) / oldScale };
    setView({ scale: newScale, x: pointer.x - mouseTo.x * newScale, y: pointer.y - mouseTo.y * newScale });
  }

  function zoomAtCenter(factor) {
    const cx = stageSize.width / 2;
    const cy = stageSize.height / 2;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * factor, MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (cx - view.x) / oldScale, y: (cy - view.y) / oldScale };
    setView({ scale: newScale, x: cx - mouseTo.x * newScale, y: cy - mouseTo.y * newScale });
  }

  function actualSize() {
    if (!working) return;
    setView({
      scale: 1,
      x: (stageSize.width - working.width) / 2,
      y: (stageSize.height - working.height) / 2,
    });
  }

  // ── Crop ────────────────────────────────────────────────────────────────
  function startCrop() {
    if (!working) return;
    setTool("crop");
    setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, rotated)));
  }

  function cancelCrop() {
    setTool("move");
    setCropRect(null);
  }

  function chooseRatio(key) {
    setRatioKey(key);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(key, rotated)));
  }

  function toggleRotate() {
    const next = !rotated;
    setRotated(next);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, next)));
  }

  function clampCropToImage(rect) {
    const width = clamp(rect.width, MIN_CROP_PX, working.width);
    const height = clamp(rect.height, MIN_CROP_PX, working.height);
    return {
      width,
      height,
      x: clamp(rect.x, 0, working.width - width),
      y: clamp(rect.y, 0, working.height - height),
    };
  }

  function handleCropDragEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({ ...cropRect, x: node.x(), y: node.y() });
    node.position({ x: next.x, y: next.y });
    setCropRect(next);
  }

  function handleCropTransformEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({
      x: node.x(),
      y: node.y(),
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY(),
    });
    node.scaleX(1);
    node.scaleY(1);
    node.setAttrs(next);
    setCropRect(next);
  }

  // Apply: rasterize the selected region into a fresh working image. The source
  // bitmap is blob-backed (never tainted), so reading pixels back is safe. The
  // result keeps the same source provenance so lineage survives to Save (sc-2434).
  const applyCrop = useCallback(async () => {
    if (!working || !cropRect) return;
    const sx = clamp(Math.round(cropRect.x), 0, working.width - 1);
    const sy = clamp(Math.round(cropRect.y), 0, working.height - 1);
    const sw = clamp(Math.round(cropRect.width), 1, working.width - sx);
    const sh = clamp(Math.round(cropRect.height), 1, working.height - sy);
    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    canvas.getContext("2d").drawImage(working.image, sx, sy, sw, sh, 0, 0, sw, sh);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    const { image, objectUrl } = await blobToImage(blob);
    installWorkingImage(image, objectUrl, working.source);
    setEdits((prev) => [...prev, { op: "crop", width: sw, height: sh }]);
    setDirty(true);
  }, [working, cropRect, installWorkingImage]);

  // Bind the transformer to the crop rect whenever crop mode is active.
  useEffect(() => {
    const transformer = transformerRef.current;
    const node = cropRectRef.current;
    if (tool === "crop" && transformer && node) {
      transformer.nodes([node]);
      transformer.getLayer()?.batchDraw();
    }
  }, [tool, cropRect]);

  // ── AI ops on the working image (sc-2432 seam) ────────────────────────────
  // Rasterize the current working image to a PNG File. `filename` overrides the
  // name (Save/Download use the "-edited" name; the AI-op scratch upload doesn't care).
  const workingImageToFile = useCallback(
    (filename) => {
      return new Promise((resolve, reject) => {
        if (!working) {
          reject(new Error("No working image."));
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = working.width;
        canvas.height = working.height;
        canvas.getContext("2d").drawImage(working.image, 0, 0);
        const base = (working.source.name || "image").replace(/\.[^./\\]+$/, "");
        const name = filename || `${base}.png`;
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error("Could not encode the working image."));
            return;
          }
          resolve(new File([blob], name, { type: "image/png" }));
        }, "image/png");
      });
    },
    [working],
  );

  // Stage the working image as a scratch asset, start a worker job against it, and
  // track it. The watcher below loads the result back and purges scratch + result —
  // intermediates never persist; only Save (sc-2434) lands a Library asset.
  const runAiOp = useCallback(
    async ({ buildBody, label, edit }) => {
      if (!working || aiOp || !activeProject) return;
      setStatus({ loading: false, error: "" });
      let scratch;
      try {
        const file = await workingImageToFile();
        scratch = await importAsset(file, { throwOnError: true });
      } catch (err) {
        setStatus({ loading: false, error: `Could not stage image: ${err.message || err}` });
        return;
      }
      try {
        const job = await apiFetch("/api/v1/jobs", token, {
          method: "POST",
          body: JSON.stringify(buildBody(scratch)),
        });
        setAiOp({ jobId: job.id, scratch, source: working.source, label, edit });
        setTool("move");
      } catch (err) {
        purgeAsset(scratch).catch(() => {});
        setStatus({ loading: false, error: `Could not start ${label}: ${err.message || err}` });
      }
    },
    [working, aiOp, activeProject, workingImageToFile, importAsset, token, purgeAsset],
  );

  function runUpscale() {
    const valid = upscaleFactorsForEngine(upscaleEngine);
    const factor = valid.includes(upscaleFactor) ? upscaleFactor : valid[0];
    runAiOp({
      label: "upscale",
      edit: { op: "upscale", engine: upscaleEngine, factor },
      buildBody: (scratch) =>
        buildUpscaleJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          factor,
          engine: upscaleEngine,
          displayName: working?.source?.name,
        }),
    });
  }

  // When the in-flight op's job terminates, load the result back into the working
  // image (on success) and purge the ephemeral scratch + result assets.
  useEffect(() => {
    if (!aiOp?.jobId) return;
    const job = jobs?.find((item) => item.id === aiOp.jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    const { scratch, source, edit } = aiOp;
    setAiOp(null); // stop tracking immediately so this can't re-enter on the next jobs tick
    const resultAsset = job.status === "completed" ? job.result?.assets?.[0] ?? null : null;
    (async () => {
      try {
        if (resultAsset) {
          const res = await fetch(assetUrl(resultAsset));
          if (!res.ok) throw new Error(`Failed to load result (${res.status})`);
          const { image, objectUrl } = await blobToImage(await res.blob());
          installWorkingImage(image, objectUrl, source);
          if (edit) setEdits((prev) => [...prev, edit]);
          setDirty(true);
        } else {
          setStatus({ loading: false, error: job.error ?? job.message ?? "The operation failed." });
        }
      } catch (err) {
        setStatus({ loading: false, error: err.message || "The operation failed." });
      } finally {
        if (scratch) purgeAsset(scratch).catch(() => {});
        if (resultAsset) purgeAsset(resultAsset).catch(() => {});
      }
    })();
  }, [aiOp, jobs, installWorkingImage, purgeAsset]);

  // ── Save / export (sc-2434) ───────────────────────────────────────────────
  // Persist the working image as a NEW Library asset, never overwriting the
  // source. Lineage links it back to the asset it was opened from (uploads have
  // no source to link); the edit chain rides along as provenance.
  const runSave = useCallback(async () => {
    if (!working || saving) return;
    setSaving(true);
    setStatus({ loading: false, error: "" });
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const saved = await importAsset(file, {
        throwOnError: true,
        sourceAssetId: working.source.assetId,
        provenance: buildSaveProvenance({
          source: working.source,
          edits,
          width: working.width,
          height: working.height,
        }),
      });
      setSavedAssetId(saved?.id ?? null);
      setDirty(false);
    } catch (err) {
      setStatus({ loading: false, error: `Could not save: ${err.message || err}` });
    } finally {
      setSaving(false);
    }
  }, [working, saving, workingImageToFile, importAsset, edits]);

  // Export the working image straight to disk as a PNG (no project involvement).
  const runDownload = useCallback(async () => {
    if (!working) return;
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const url = URL.createObjectURL(file);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = file.name;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setStatus({ loading: false, error: `Could not export: ${err.message || err}` });
    }
  }, [working, workingImageToFile]);

  // Confirm before an action that would discard unsaved edits (Open / drag-drop a
  // new image while dirty). Returns true when it's safe to proceed.
  function confirmDiscardEdits() {
    if (!dirty) return true;
    return (
      typeof window.confirm !== "function" ||
      window.confirm("You have unsaved edits. Open a new image and discard them?")
    );
  }

  // Warn before leaving with unsaved edits: a browser unload (close/refresh) and an
  // in-app navigation away (the App nav consults this guard, sc-2434).
  useEffect(() => {
    if (!dirty) return undefined;
    const onBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    const unregister = registerLeaveGuard?.(
      () =>
        typeof window.confirm !== "function" ||
        window.confirm("You have unsaved edits in the Image Editor. Leave and discard them?"),
    );
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      if (typeof unregister === "function") unregister();
    };
  }, [dirty, registerLeaveGuard]);

  const activeAiJob = aiOp ? jobs?.find((item) => item.id === aiOp.jobId) : null;

  return (
    <section className="main-surface image-editor-surface">
      <div className="image-editor-bar">
        <span className="image-editor-title" title={working ? working.source.name : undefined}>
          {working ? working.source.name : "No image open"}
        </span>
        <div className="image-editor-bar-actions">
          <button className={working ? "" : "primary"} onClick={() => setPickerOpen(true)} type="button">
            Open
          </button>
          {working && working.source.assetId ? (
            <button
              onClick={() => setPreviewAsset?.(imageAssets.find((item) => item.id === working.source.assetId))}
              title="Preview the source asset"
              type="button"
            >
              Source
            </button>
          ) : null}
          {working ? (
            <>
              <button onClick={runDownload} title="Download a PNG to your computer" type="button">
                Download
              </button>
              {savedAssetId && !dirty ? <span className="image-editor-saved">Saved ✓</span> : null}
              <button
                className="primary"
                disabled={!dirty || saving}
                onClick={runSave}
                title="Save a new image to the project Library"
                type="button"
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {status.error ? <div className="notice notice-error image-editor-notice">{status.error}</div> : null}

      <div
        className="image-editor-canvas-wrap"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        ref={containerRef}
      >
        {working && stageSize.width > 0 && stageSize.height > 0 ? (
          <Stage
            draggable={tool !== "crop"}
            height={stageSize.height}
            onDragEnd={(event) => {
              if (event.target !== event.target.getStage()) return;
              const stage = event.target.getStage();
              setView((prev) => ({ ...prev, x: stage.x(), y: stage.y() }));
            }}
            onWheel={handleWheel}
            scaleX={view.scale}
            scaleY={view.scale}
            width={stageSize.width}
            x={view.x}
            y={view.y}
          >
            <Layer>
              <Rect
                fill="#ffffff"
                height={working.height}
                shadowBlur={12}
                shadowColor="rgba(0,0,0,0.35)"
                width={working.width}
                x={0}
                y={0}
              />
              <KonvaImage height={working.height} image={working.image} width={working.width} x={0} y={0} />
              {tool === "crop" && cropRect ? (
                <>
                  {cropOverlayRects(working.width, working.height, cropRect).map((rect, index) => (
                    <Rect
                      key={index}
                      fill="rgba(0,0,0,0.55)"
                      height={rect.height}
                      listening={false}
                      width={rect.width}
                      x={rect.x}
                      y={rect.y}
                    />
                  ))}
                  <Rect
                    draggable
                    fill="rgba(255,255,255,0.01)"
                    height={cropRect.height}
                    onDragEnd={handleCropDragEnd}
                    onTransformEnd={handleCropTransformEnd}
                    ref={cropRectRef}
                    stroke="#ffffff"
                    strokeScaleEnabled={false}
                    strokeWidth={2}
                    width={cropRect.width}
                    x={cropRect.x}
                    y={cropRect.y}
                  />
                  <Transformer
                    anchorSize={8}
                    borderStroke="#ffffff"
                    boundBoxFunc={(oldBox, newBox) =>
                      newBox.width < MIN_CROP_PX || newBox.height < MIN_CROP_PX ? oldBox : newBox
                    }
                    enabledAnchors={
                      ratioKey === "free"
                        ? ["top-left", "top-center", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-center", "bottom-right"]
                        : ["top-left", "top-right", "bottom-left", "bottom-right"]
                    }
                    keepRatio={ratioKey !== "free"}
                    ref={transformerRef}
                    rotateEnabled={false}
                  />
                </>
              ) : null}
            </Layer>
          </Stage>
        ) : (
          <div className="image-editor-empty">
            {status.loading ? (
              <p>Loading image…</p>
            ) : (
              <>
                <p className="image-editor-empty-title">Open an image to start editing</p>
                <p className="image-editor-empty-hint">Drag &amp; drop an image here, or click Open.</p>
              </>
            )}
          </div>
        )}

        {working ? (
          <aside className="image-editor-toolbar" aria-label="Editor tools">
            <button
              className={tool === "move" ? "image-editor-tool active" : "image-editor-tool"}
              onClick={cancelCrop}
              title="Move / pan"
              type="button"
            >
              Move
            </button>
            <button
              className={tool === "crop" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={startCrop}
              title="Crop"
              type="button"
            >
              Crop
            </button>
            <button
              className={tool === "upscale" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={() => setTool("upscale")}
              title="Upscale"
              type="button"
            >
              Upscale
            </button>
            {UPCOMING_TOOLS.map((upcoming) => (
              <button
                className="image-editor-tool"
                disabled
                key={upcoming.id}
                title={`${upcoming.label} — coming soon (${upcoming.story})`}
                type="button"
              >
                {upcoming.label}
              </button>
            ))}
          </aside>
        ) : null}

        {tool === "crop" && cropRect ? (
          <div className="image-editor-cropbar">
            <div className="image-editor-ratios" role="group" aria-label="Crop ratio">
              {CROP_RATIOS.map((entry) => (
                <button
                  className={ratioKey === entry.key ? "active" : ""}
                  key={entry.key}
                  onClick={() => chooseRatio(entry.key)}
                  type="button"
                >
                  {entry.label}
                </button>
              ))}
            </div>
            <button
              className={rotated ? "active" : ""}
              disabled={ratioKey === "free" || ratioKey === "1:1"}
              onClick={toggleRotate}
              title="Rotate ratio (swap orientation)"
              type="button"
            >
              ⟲ Rotate
            </button>
            <span className="image-editor-cropdims">
              {Math.round(cropRect.width)} × {Math.round(cropRect.height)}
            </span>
            <button className="primary" onClick={applyCrop} type="button">
              Apply
            </button>
            <button onClick={cancelCrop} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "upscale" && working ? (
          <div className="image-editor-cropbar">
            <div className="image-editor-ratios" role="group" aria-label="Upscale engine">
              {UPSCALE_ENGINES.map((entry) => (
                <button
                  className={upscaleEngine === entry.key ? "active" : ""}
                  key={entry.key}
                  onClick={() => {
                    setUpscaleEngine(entry.key);
                    if (!entry.factors.includes(upscaleFactor)) setUpscaleFactor(entry.factors[0]);
                  }}
                  type="button"
                >
                  {entry.label}
                </button>
              ))}
            </div>
            <div className="image-editor-ratios" role="group" aria-label="Upscale factor">
              {upscaleFactorsForEngine(upscaleEngine).map((value) => (
                <button
                  className={upscaleFactor === value ? "active" : ""}
                  key={value}
                  onClick={() => setUpscaleFactor(value)}
                  type="button"
                >
                  {value}×
                </button>
              ))}
            </div>
            <span className="image-editor-cropdims">
              {working.width * upscaleFactor} × {working.height * upscaleFactor}
            </span>
            <button className="primary" disabled={!!aiOp} onClick={runUpscale} type="button">
              Upscale
            </button>
            <button onClick={() => setTool("move")} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {aiOp ? (
          <div className="image-editor-busy">
            <div className="image-editor-busy-card">
              <p className="image-editor-busy-title">
                {aiOp.label === "upscale" ? "Upscaling…" : "Working…"}
              </p>
              <p className="image-editor-busy-msg">
                {activeAiJob?.message ||
                  (activeAiJob?.status === "queued" ? "Queued — waiting for a worker." : "Processing…")}
              </p>
              {typeof activeAiJob?.progress === "number" ? (
                <div className="image-editor-busy-bar">
                  <span style={{ width: `${Math.round(activeAiJob.progress * 100)}%` }} />
                </div>
              ) : null}
            </div>
          </div>
        ) : null}

        {working ? (
          <div className="image-editor-viewbar">
            <button onClick={() => zoomAtCenter(1 / ZOOM_STEP)} title="Zoom out" type="button">
              −
            </button>
            <span className="image-editor-zoom">{Math.round(view.scale * 100)}%</span>
            <button onClick={() => zoomAtCenter(ZOOM_STEP)} title="Zoom in" type="button">
              +
            </button>
            <button onClick={fitToView} type="button">
              Fit
            </button>
            <button onClick={actualSize} type="button">
              100%
            </button>
            <span className="image-editor-dims">
              {working.width} × {working.height}
            </span>
          </div>
        ) : null}
      </div>

      {pickerOpen ? (
        <DatasetAddDialog
          assets={assets ?? []}
          characters={characters ?? []}
          confirmLabel="Open"
          eyebrow="Open"
          fileAccept="image/*"
          fileHint="Drag an image here, or"
          multiple={false}
          onAdd={(ids) => {
            setPickerOpen(false);
            if (ids[0] && confirmDiscardEdits()) openAsset(ids[0]);
          }}
          onClose={() => setPickerOpen(false)}
          onImport={(files) => {
            const file = files?.[0];
            setPickerOpen(false);
            if (file && confirmDiscardEdits()) openFile(file);
          }}
          title="Open image"
        />
      ) : null}
    </section>
  );
}
