import React, { useEffect, useState } from "react";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";
import { hasPresentCredential, loadCredentials, serverToken } from "../credentials.js";
import { extractFamilies, modelLoraFamilies, presetLoraId, presetLoras } from "../presetUtils.js";
import { useAppContext } from "../context/AppContext.js";
import { DEFAULT_MAC_CAPABILITIES, macModelBlock } from "../macGating.js";
import { apiFetch } from "../api.js";
import { isDesktop, tauriInvoke } from "../runtime.js";

// Wan A14B is a two-expert mixture; its LoRAs come as a high/low-noise pair. These
// base models accept the optional low-noise expert upload (sc-1991). The 5B model
// (wan_2_2) is dense and takes a single-file LoRA.
const WAN_MOE_BASE_MODELS = new Set(["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]);

function matchesFamily(item, familyFilter) {
  if (familyFilter === "all") {
    return true;
  }
  // Accept either a LoRA catalog entry or a lora_import job snapshot (whose
  // family metadata lives under payload.manifestEntry).
  const families = extractFamilies(item, { includeManifest: true });
  // Import jobs can briefly lack family metadata; completed catalog entries should not.
  return item.type === "lora_import" && families.length === 0 ? true : families.includes(familyFilter);
}

function loraImportKey(job) {
  return job.payload?.loraId ?? job.payload?.sourceUrl ?? job.payload?.sourcePath ?? job.payload?.name ?? null;
}

function completedLoraImportTimes(jobs) {
  const completed = new Map();
  jobs
    .filter((job) => job.type === "lora_import" && job.status === "completed")
    .forEach((job) => {
      const key = loraImportKey(job);
      if (!key || !job.createdAt) {
        return;
      }
      const previous = completed.get(key);
      if (!previous || job.createdAt.localeCompare(previous) > 0) {
        completed.set(key, job.createdAt);
      }
    });
  return completed;
}

function isSupersededLoraImport(job, completedTimes) {
  const key = loraImportKey(job);
  const completedAt = key ? completedTimes.get(key) : null;
  return Boolean(completedAt) && terminalStatuses.has(job.status) && job.status !== "completed" && completedAt.localeCompare(job.createdAt ?? "") > 0;
}

function downloadSizeText(model) {
  if (!model.downloadSizeLabel) {
    return "Unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

// MLX status text, keyed off the macOS catalog's mlxConversionState. Turnkey
// ("ready") models fetch their MLX weights automatically on first generation;
// convert-required models need the native checkpoint downloaded, then converted.
function mlxStatusText(model) {
  switch (model.mlxConversionState) {
    case "ready":
      return model.mlxInstallState === "installed"
        ? "MLX weights installed."
        : "MLX weights download automatically on first generation.";
    case "needs_source":
      return "Download the model first, then convert it to MLX.";
    case "needs_conversion":
      return "Native checkpoint downloaded — ready to convert to MLX.";
    case "converted":
      return "Converted to MLX and ready.";
    default:
      return "";
  }
}

const MODEL_TYPE_OPTIONS = [
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
  { value: "utility", label: "Utility" },
];

// sc-7081 (epic 7080): model upload/import is hidden + disabled on every platform until a
// real compatibility + conversion pipeline exists behind it. Today an imported checkpoint
// has no runnable engine (macOS is MLX-only with a compile-time engine table; off-Mac only
// loads full diffusers repos), so the form is kept in source but not rendered. The API
// refuses the request too. Flip to true once the pipeline gates imports on a compatibility
// verdict.
const MODEL_IMPORT_ENABLED = false;

// Models render in type-grouped sections. Order is fixed; any model whose `type`
// isn't listed here falls into a trailing "Other" group so nothing is hidden.
const MODEL_TYPE_GROUPS = [
  { type: "image", label: "Image Models" },
  { type: "video", label: "Video Models" },
  { type: "utility", label: "Utility Models" },
];

// Capability descriptors shown as chips on each model card. With models now grouped
// by `type`, the chips are what tell the user what a card actually does (plain
// text-to-image vs editing vs character reference, etc.). Unknown keys fall back to
// a humanized form so a new capability still reads sensibly without a code change.
const CAPABILITY_LABELS = {
  text_to_image: "Text to Image",
  image_to_image: "Image to Image",
  edit_image: "Image Edit",
  character_image: "Character",
  style_variations: "Style Variations",
  vqa: "Visual Q&A",
  interleave: "Interleaved",
  image_to_video: "Image to Video",
  text_to_video: "Text to Video",
  first_last_frame: "First / Last Frame",
  extend_clip: "Extend Clip",
  video_bridge: "Video Bridge",
  replace_person: "Replace Person",
};

function capabilityLabel(capability) {
  return CAPABILITY_LABELS[capability] ?? String(capability).replaceAll("_", " ");
}

// Curated "getting started" models, flagged `recommended: true` in the catalog
// (config/manifests/builtin.models.jsonc). Within each type section these float to a
// "Recommended" subgroup; the rest collapse under "Additional Supported".
function isRecommendedModel(model) {
  return model.recommended === true;
}

// Group key for the family-organized LoRA list. A LoRA can list several compatible
// families; we group under its primary one and bucket family-less entries under a
// trailing "compatible" group.
function loraGroupKey(lora) {
  return lora.family ?? extractFamilies(lora)[0] ?? "";
}

// The Hugging Face page of a gated model's primary download repo — where the user
// clicks "Agree and access" to be granted access with their token (sc-5999). Derived
// from the first HF download repo (or the mlx repo), so it covers every gated model
// without a per-model manifest field. Falls back to `licenseUrl` when no repo is known.
function gatedRepoUrl(model) {
  const host = model.credentialHost || "huggingface.co";
  const repo =
    (model.downloads ?? []).find((entry) => entry.provider === "huggingface" && entry.repo)?.repo ??
    model.mlx?.repo;
  return repo ? `https://${host}/${repo}` : null;
}

// Gated models (e.g. FLUX.1 [dev]) need an accepted license + a saved credential
// before a download can succeed. The catalog flags these with `gated`/
// `credentialHost`/`licenseUrl` (sc-1898). When the matching credential is already
// present we soften the notice to a ready state; otherwise we point the user at the
// Settings credential screen. `present` is undefined while presence is still
// unknown (e.g. the credential list hasn't loaded) — we still show the link then.
// `repoUrl` links the gated repo so the user can request access (sc-5999); shown
// alongside `licenseUrl` only when the license lives on a different page (e.g.
// Ideogram 4, whose terms are on the source repo but access is on the SceneWorks repo).
function GatedModelNotice({ host, repoUrl, licenseUrl, present, onOpenSettings }) {
  const hostLabel = host || "the required service";
  const showSeparateLicense = licenseUrl && licenseUrl !== repoUrl;
  return (
    <div className={present ? "model-gated-notice ready" : "model-gated-notice"}>
      <p className={present ? "inline-success" : "inline-warning"}>
        {present
          ? `Credential for ${hostLabel} saved — request access on the model page, then download.`
          : `Gated download. Add a ${hostLabel} token, then request access on the model page and accept the license before downloading.`}
      </p>
      <div className="model-gated-actions">
        {present ? null : (
          <button type="button" onClick={onOpenSettings}>
            Add token in Settings
          </button>
        )}
        {repoUrl ? (
          <a href={repoUrl} target="_blank" rel="noreferrer noopener">
            Request access on Hugging Face
          </a>
        ) : null}
        {showSeparateLicense ? (
          <a href={licenseUrl} target="_blank" rel="noreferrer noopener">
            Review license
          </a>
        ) : null}
      </div>
    </div>
  );
}

function referencedPresetNames(recipePresets, kind, id) {
  return recipePresets
    .filter((preset) => {
      if (kind === "model") {
        return preset.model === id;
      }
      return presetLoras(preset).some((lora) => presetLoraId(lora) === id);
    })
    .map((preset) => preset.name ?? preset.id)
    .filter(Boolean);
}

function deleteConfirmation(kind, item, recipePresets) {
  const name = item.name ?? item.id;
  const presetNames = referencedPresetNames(recipePresets, kind, item.id);
  const lines = [
    `Delete ${kind} "${name}"?`,
    "This removes the registry entry and SceneWorks-owned local files when available.",
  ];
  if (presetNames.length) {
    lines.push(`Referenced by presets: ${presetNames.slice(0, 5).join(", ")}.`);
    lines.push("Those presets will keep a broken reference until updated.");
  }
  if (item.scope === "builtin" || item.catalogScope === "builtin") {
    lines.push("Built-in catalog entries stay protected; only local installed files can be removed.");
  }
  return lines.join("\n\n");
}

function deleteResultText(result, name) {
  const removed = result?.removedManifestEntry ? "Removed the registry entry" : "Removed local files";
  const warnings = result?.warnings?.length ? ` ${result.warnings.join(" ")}` : "";
  return `${removed} for ${name}.${warnings}`;
}

export function ModelManagerScreen() {
  const {
    activeProject,
    jobs,
    loras,
    models,
    jobAction,
    setActiveView,
    deleteLora: deleteLoraAction,
    deleteModel: deleteModelAction,
    createModelDownloadJob,
    createLoraDownloadJob,
    createModelConvertJob,
    createLoraImportJob,
    createModelImportJob,
    presets: recipePresets = [],
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  // Third-party LyCORIS now applies on every MLX provider (epic 3641), so the LyCORIS upload is no
  // longer Mac-gated.
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onResumeDownloadJob = (job, payload) => jobAction(job, "retry", { body: payload ?? {} });
  const onFreshDownloadJob = (job, payload) => jobAction(job, "retry", { body: payload ?? {} });
  const onConvertModel = createModelConvertJob;
  const onDeleteLora = deleteLoraAction;
  const onDeleteModel = deleteModelAction;
  const onDownloadModel = createModelDownloadJob;
  const onDownloadLora = createLoraDownloadJob;
  const onImportLora = createLoraImportJob;
  const onImportModel = createModelImportJob;
  const onOpenQueue = () => setActiveView("Queue");
  // LoRA families come from each model's LoRA-compatibility set — NOT its model
  // `family` identity. They usually coincide, but distilled variants differ: e.g.
  // FLUX.2 [klein] has family "flux2-klein" yet accepts "flux2" LoRAs. The import
  // validator + generation-time matcher both key off loraCompatibility.families,
  // so the dropdown must too, or the user picks a family the backend rejects.
  const families = Array.from(new Set(models.flatMap((model) => modelLoraFamilies(model)).filter(Boolean))).sort();
  const familiesKey = families.join("|");
  const [familyFilter, setFamilyFilter] = useState("all");
  const [importingLora, setImportingLora] = useState(false);
  const [importMessage, setImportMessage] = useState({ tone: "neutral", text: "" });
  const [importForm, setImportForm] = useState({
    mode: "url",
    sourceUrl: "",
    file: null,
    secondaryFile: null,
    name: "",
    scope: "global",
    family: "",
    baseModel: "",
  });
  const [fileInputKey, setFileInputKey] = useState(0);
  const [importingModel, setImportingModel] = useState(false);
  const [modelImportMessage, setModelImportMessage] = useState({ tone: "neutral", text: "" });
  const [modelImportForm, setModelImportForm] = useState({
    mode: "url",
    sourceUrl: "",
    file: null,
    name: "",
    type: "image",
    family: "",
  });
  const [modelFileInputKey, setModelFileInputKey] = useState(0);
  const [deletingItem, setDeletingItem] = useState("");
  const [deleteMessage, setDeleteMessage] = useState({ tone: "neutral", text: "" });
  // Read the host's memory so MLX models can be gated against their memory tier.
  // Desktop reads it from the Tauri GPU probe; a remote LAN browser reads the
  // auth-protected REST signal (epic 4484 story 9). `isDesktop`/`tauriInvoke` come
  // from the unified runtime helper (story 6).
  const [unifiedMemoryGb, setUnifiedMemoryGb] = useState(null);
  // GPU memory cap (epic 7819): when the user caps GPU memory, per-model fit must be judged
  // against the *effective* ceiling, not physical RAM. Desktop reads the persisted fraction.
  const [gpuLimitFraction, setGpuLimitFraction] = useState(null);
  // Gated-model credential presence (sc-1898): only fetched when the catalog has a
  // gated model, so non-gated deployments make no extra credential request.
  const [credentials, setCredentials] = useState([]);
  const hasGatedModel = models.some((model) => model.gated);
  const visibleLoras = loras.filter((lora) => matchesFamily(lora, familyFilter));
  // Wan A14B MoE paired upload (sc-1991): when the user targets the wan-video
  // family, let them pick the specific base model and (for two-expert A14B models)
  // upload the low-noise expert half alongside the high-noise primary.
  const wanBaseModelOptions = models.filter((model) => model.family === "wan-video");
  const showBaseModelSelect = importForm.family === "wan-video" && wanBaseModelOptions.length > 0;
  const isMoeBaseModel = WAN_MOE_BASE_MODELS.has(importForm.baseModel);
  const showSecondaryFileSlot = isMoeBaseModel && importForm.mode === "file";
  const moeMissingSecondary = showSecondaryFileSlot && Boolean(importForm.file) && !importForm.secondaryFile;
  // Effective GPU memory available to generations under the user's cap (epic 7819). A capped app
  // can run fewer models than the Mac's physical memory implies, so gate per-model fit on this.
  const memoryIsCapped =
    unifiedMemoryGb != null &&
    typeof gpuLimitFraction === "number" &&
    Number.isFinite(gpuLimitFraction) &&
    gpuLimitFraction > 0 &&
    gpuLimitFraction < 1;
  const effectiveMemoryGb = memoryIsCapped ? unifiedMemoryGb * gpuLimitFraction : unifiedMemoryGb;

  useEffect(() => {
    if (familyFilter !== "all" && !families.includes(familyFilter)) {
      setFamilyFilter("all");
    }
  }, [familiesKey, familyFilter]);

  useEffect(() => {
    setImportForm((current) => (current.family && !families.includes(current.family) ? { ...current, family: "" } : current));
  }, [familiesKey]);

  // The base model + low-noise slot only apply to wan-video imports; clear them
  // when the family changes away so a stale baseModel can't ride along.
  useEffect(() => {
    setImportForm((current) =>
      current.family !== "wan-video" && (current.baseModel || current.secondaryFile)
        ? { ...current, baseModel: "", secondaryFile: null }
        : current,
    );
  }, [importForm.family]);

  useEffect(() => {
    let cancelled = false;
    if (isDesktop) {
      // Desktop: read unified memory straight from the Tauri GPU probe.
      tauriInvoke("get_gpu_info")
        .then((info) => {
          if (!cancelled && info && typeof info.unifiedMemoryMb === "number") {
            setUnifiedMemoryGb(info.unifiedMemoryMb / 1024);
          }
        })
        .catch(() => {});
      // GPU memory cap (epic 7819): the persisted fraction lowers the effective ceiling used for
      // per-model fit, so a capped app flags the models it can no longer hold.
      tauriInvoke("get_app_settings")
        .then((appSettings) => {
          if (!cancelled && appSettings) {
            const fraction = appSettings.gpuMemoryLimitFraction;
            setGpuLimitFraction(typeof fraction === "number" ? fraction : null);
          }
        })
        .catch(() => {});
    } else {
      // Remote LAN browser (epic 4484 story 9): the Tauri probe is unavailable, so
      // read the host's memory from the auth-protected REST signal derived from the
      // registered GPU worker (unified memory on macOS / GPU VRAM on Windows). Without
      // this, memory gating would silently no-op for remote users.
      apiFetch("/api/v1/host-capabilities", serverToken())
        .then((caps) => {
          if (cancelled || !caps) {
            return;
          }
          const gb = caps.unifiedMemoryGb ?? caps.gpuMemoryGb;
          if (typeof gb === "number") {
            setUnifiedMemoryGb(gb);
          }
        })
        .catch(() => {});
    }
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!hasGatedModel) {
      return undefined;
    }
    let cancelled = false;
    loadCredentials()
      .then((list) => {
        if (!cancelled) {
          setCredentials(Array.isArray(list) ? list : []);
        }
      })
      // Presence unknown (e.g. not authenticated yet) — the notice still links to Settings.
      .catch(() => {
        if (!cancelled) {
          setCredentials([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [hasGatedModel]);

  function downloadJobsFor(model) {
    return jobs.filter((job) => job.type === "model_download" && job.payload?.modelId === model.id);
  }

  function convertJobsFor(model) {
    return jobs.filter((job) => job.type === "model_convert" && job.payload?.modelId === model.id);
  }

  function loraDownloadJobsFor(lora) {
    return jobs.filter((job) => job.type === "lora_download" && job.payload?.loraId === lora.id);
  }

  async function importLora(event) {
    event.preventDefault();
    const isFileImport = importForm.mode === "file";
    if ((!isFileImport && !importForm.sourceUrl.trim()) || (isFileImport && !importForm.file) || !onImportLora) {
      return;
    }
    setImportingLora(true);
    setImportMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading LoRA file before queueing import." : "",
    });
    try {
      const familyOverride = importForm.family ? { family: importForm.family } : {};
      // Carry the chosen base model (wan-video) and, for an A14B MoE upload, the
      // low-noise expert half so both land in one record (sc-1991).
      const baseModelOverride = showBaseModelSelect && importForm.baseModel ? { baseModel: importForm.baseModel } : {};
      const secondaryOverride =
        isFileImport && showSecondaryFileSlot && importForm.secondaryFile
          ? { secondaryFile: importForm.secondaryFile }
          : {};
      const job = await onImportLora({
        ...(isFileImport ? { file: importForm.file } : { sourceUrl: importForm.sourceUrl.trim() }),
        name: importForm.name.trim() || undefined,
        scope: importForm.scope,
        ...familyOverride,
        ...baseModelOverride,
        ...secondaryOverride,
      });
      const loraId = job?.payload?.loraId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detectionNote =
        !importForm.family && resolvedFamily ? ` Detected family: ${resolvedFamily}.` : "";
      setImportForm((current) => ({ ...current, sourceUrl: "", file: null, secondaryFile: null, name: "" }));
      // Force a re-mount so choosing the same file again still emits a change event.
      setFileInputKey((current) => current + 1);
      setImportMessage({
        tone: "success",
        text: `${loraId ? `LoRA import queued for ${loraId}.` : "LoRA import queued."}${detectionNote}`,
      });
    } catch (err) {
      setImportMessage({ tone: "error", text: err.message });
    } finally {
      setImportingLora(false);
    }
  }

  async function importModel(event) {
    event.preventDefault();
    const isFileImport = modelImportForm.mode === "file";
    if ((!isFileImport && !modelImportForm.sourceUrl.trim()) || (isFileImport && !modelImportForm.file) || !onImportModel) {
      return;
    }
    setImportingModel(true);
    setModelImportMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading model file before queueing import." : "",
    });
    try {
      const familyOverride = modelImportForm.family ? { family: modelImportForm.family } : {};
      const job = await onImportModel({
        ...(isFileImport ? { file: modelImportForm.file } : { sourceUrl: modelImportForm.sourceUrl.trim() }),
        name: modelImportForm.name.trim() || undefined,
        modelType: modelImportForm.type,
        ...familyOverride,
      });
      const modelId = job?.payload?.modelId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detectionNote =
        !modelImportForm.family && resolvedFamily ? ` Detected family: ${resolvedFamily}.` : "";
      setModelImportForm((current) => ({ ...current, sourceUrl: "", file: null, name: "" }));
      setModelFileInputKey((current) => current + 1);
      setModelImportMessage({
        tone: "success",
        text: `${modelId ? `Model import queued for ${modelId}.` : "Model import queued."}${detectionNote}`,
      });
    } catch (err) {
      setModelImportMessage({ tone: "error", text: err.message });
    } finally {
      setImportingModel(false);
    }
  }

  async function deleteModel(model) {
    if (!onDeleteModel || model.removable === false) {
      return;
    }
    if (typeof window.confirm === "function" && !window.confirm(deleteConfirmation("model", model, recipePresets))) {
      return;
    }
    setDeletingItem(`model:${model.id}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await onDeleteModel(model);
      setDeleteMessage({ tone: "success", text: deleteResultText(result, model.name ?? model.id) });
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  async function deleteLora(lora) {
    if (!onDeleteLora || lora.removable === false) {
      return;
    }
    if (typeof window.confirm === "function" && !window.confirm(deleteConfirmation("lora", lora, recipePresets))) {
      return;
    }
    setDeletingItem(`lora:${lora.scope ?? "global"}:${lora.id}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await onDeleteLora(lora);
      setDeleteMessage({ tone: "success", text: deleteResultText(result, lora.name ?? lora.id) });
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  const completedImportTimes = completedLoraImportTimes(jobs);
  const pendingLoraImportJobs = jobs.filter((job) => job.type === "lora_import" && !isSupersededLoraImport(job, completedImportTimes));
  const localLoraImportJobs = pendingLoraImportJobs.filter((job) => job.status !== "completed" && matchesFamily(job, familyFilter));
  const pendingModelImportJobs = jobs.filter((job) => job.type === "model_import" && job.status !== "completed");
  const isModelFileImport = modelImportForm.mode === "file";
  const modelImportDisabled =
    importingModel ||
    !onImportModel ||
    (isModelFileImport ? !modelImportForm.file : !modelImportForm.sourceUrl.trim());
  const hiddenImportCount =
    familyFilter === "all" ? 0 : pendingLoraImportJobs.filter((job) => job.status !== "completed" && !matchesFamily(job, familyFilter)).length;
  const visibleLoraCount = visibleLoras.length + localLoraImportJobs.length;
  const installedLoraCount = visibleLoras.filter((lora) => lora.installState === "installed").length;
  const unavailableLoraCount = visibleLoras.filter((lora) => lora.installState === "missing").length;
  const pendingLoraCount = visibleLoraCount - installedLoraCount - unavailableLoraCount;
  const loraCountText = [
    installedLoraCount ? `${installedLoraCount} installed` : null,
    unavailableLoraCount ? `${unavailableLoraCount} unavailable` : null,
    pendingLoraCount ? `${pendingLoraCount} pending` : null,
  ].filter(Boolean).join(" · ") || "0 visible";
  const isFileImport = importForm.mode === "file";
  const importDisabled =
    importingLora ||
    !onImportLora ||
    (importForm.scope === "project" && !activeProject) ||
    (isFileImport ? !importForm.file : !importForm.sourceUrl.trim());

  // Models grouped by type for the sectioned layout. Known types keep their fixed
  // order; anything else lands in a trailing "Other" group. Empty groups drop out.
  const knownModelTypes = new Set(MODEL_TYPE_GROUPS.map((group) => group.type));
  const modelGroups = [
    ...MODEL_TYPE_GROUPS.map((group) => ({ ...group, items: models.filter((model) => model.type === group.type) })),
    { type: "other", label: "Other Models", items: models.filter((model) => !knownModelTypes.has(model.type)) },
  ].filter((group) => group.items.length > 0);

  // LoRAs split into Built-In (catalog `scope: "builtin"`) and User (global/project)
  // containers. Built-in entries are a flat list with a Download affordance; user
  // entries keep the family-organized grouping below.
  const builtinLoras = visibleLoras.filter((lora) => lora.scope === "builtin");
  const userLoras = visibleLoras.filter((lora) => lora.scope !== "builtin");

  // Visible user LoRAs grouped by family for the family-organized list. The family
  // dropdown still narrows `visibleLoras` upstream; when a specific family is
  // selected this collapses to a single group.
  const loraGroupMap = new Map();
  userLoras.forEach((lora) => {
    const key = loraGroupKey(lora) || "compatible";
    if (!loraGroupMap.has(key)) {
      loraGroupMap.set(key, []);
    }
    loraGroupMap.get(key).push(lora);
  });
  const loraGroups = [...loraGroupMap.entries()]
    .sort(([a], [b]) => (a === "compatible" ? 1 : b === "compatible" ? -1 : a.localeCompare(b)))
    .map(([family, items]) => ({ family, items }));

  function renderModelCard(model) {
    const downloadJobs = downloadJobsFor(model);
    const downloadJob = downloadJobs.find((job) => !terminalStatuses.has(job.status));
    const installed = model.installState === "installed";
    const incomplete = model.cacheState === "incomplete" || model.repairAvailable;
    const missingRequiredFiles = Array.isArray(model.missingRequiredFiles) ? model.missingRequiredFiles : [];
    const localDownloadJob = installed ? null : downloadJobs.find((job) => job.status !== "completed");
    const failedDownload = localDownloadJob && terminalStatuses.has(localDownloadJob.status);
    const downloadSize = downloadSizeText(model);
    const unassociated = !model.family;
    const capabilities = Array.isArray(model.capabilities) ? model.capabilities : [];
    const deleteKey = `model:${model.id}`;
    const canDelete = Boolean(onDeleteModel) && model.removable !== false;
    // MLX (macOS) variant: only present when the catalog computed mlxConversionState.
    const mlxState = model.mlxConversionState;
    const mlxMinGb = model.mlx?.minMemoryGb ?? null;
    const mlxEnoughMemory = effectiveMemoryGb == null || mlxMinGb == null || effectiveMemoryGb >= mlxMinGb;
    const convertJobs = convertJobsFor(model);
    const convertJob = convertJobs.find((job) => !terminalStatuses.has(job.status));
    const failedConvert = convertJobs.find((job) => terminalStatuses.has(job.status) && job.status !== "completed");
    const showConvertButton = mlxState === "needs_conversion" || mlxState === "converted";
    const gated = Boolean(model.gated);
    const credentialPresent = gated && hasPresentCredential(credentials, model.credentialHost);
    return (
      <article className="model-card" key={model.id}>
        <div>
          <p className="eyebrow">{model.family ?? "unassociated"}</p>
          <h3>{model.name}</h3>
        </div>
        <span className={incomplete ? "status-badge warning" : installed ? "status-badge installed" : "status-badge"}>
          {incomplete ? "incomplete" : installed ? "installed" : "missing"}
        </span>
        {unassociated ? (
          <span className="status-badge warning" title="Set this model's family in user.models.jsonc before using it for generation.">
            needs family
          </span>
        ) : null}
        {macModelBlock(model, macCapabilities) ? (
          <span className="status-badge warning" title={macModelBlock(model, macCapabilities).text}>
            not on Mac
          </span>
        ) : null}
        {capabilities.length ? (
          <ul className="model-capabilities">
            {capabilities.map((capability) => (
              <li className="chip" key={capability}>
                {capabilityLabel(capability)}
              </li>
            ))}
          </ul>
        ) : null}
        <p>{model.ui?.description ?? model.family ?? model.id}</p>
        {gated && !installed ? (
          <GatedModelNotice
            host={model.credentialHost}
            repoUrl={gatedRepoUrl(model) ?? model.licenseUrl ?? null}
            licenseUrl={model.licenseUrl}
            present={credentialPresent}
            onOpenSettings={() => setActiveView("Settings")}
          />
        ) : null}
        {incomplete ? (
          <p className="inline-warning">
            Cached files are incomplete
            {missingRequiredFiles.length ? `: ${missingRequiredFiles.slice(0, 3).join(", ")}${missingRequiredFiles.length > 3 ? "..." : ""}` : ""}.
          </p>
        ) : null}
        <dl>
          <div>
            <dt>Repo</dt>
            <dd>{model.downloads?.[0]?.repo ?? "none"}</dd>
          </div>
          <div>
            <dt>Download size</dt>
            <dd>{downloadSize}</dd>
          </div>
        </dl>
        {localDownloadJob ? (
          <WorkerProgressCard
            job={localDownloadJob}
            onCancel={onCancelJob}
            onRetry={onResumeDownloadJob}
            onFreshRetry={onFreshDownloadJob}
            onOpenQueue={onOpenQueue}
          />
        ) : null}
        {mlxState ? (
          <div className="mlx-status">
            <div className="mlx-status-badges">
              <span className="status-badge">MLX</span>
              {mlxMinGb != null ? (
                <span className={mlxEnoughMemory ? "status-badge" : "status-badge warning"}>needs ≥ {mlxMinGb} GB</span>
              ) : null}
            </div>
            <p>{mlxStatusText(model)}</p>
            {!mlxEnoughMemory ? (
              <p className="inline-warning">
                Needs ≥ {mlxMinGb} GB unified memory;{" "}
                {memoryIsCapped
                  ? `the app is limited to ~${Math.round(effectiveMemoryGb)} GB by your GPU memory cap`
                  : `this Mac has ~${Math.round(effectiveMemoryGb)} GB`}
                . It may run out of memory.
              </p>
            ) : null}
            {convertJob ? <WorkerProgressCard job={convertJob} onCancel={onCancelJob} onOpenQueue={onOpenQueue} /> : null}
            {showConvertButton ? (
              <button
                disabled={mlxState === "converted" || Boolean(convertJob) || !mlxEnoughMemory}
                onClick={() => onConvertModel?.(model)}
                type="button"
              >
                {convertJob
                  ? convertJob.status
                  : mlxState === "converted"
                    ? "MLX ready"
                    : failedConvert
                      ? "Retry MLX Conversion"
                      : "Convert to MLX"}
              </button>
            ) : null}
          </div>
        ) : null}
        <div className="model-card-actions">
          <button
            disabled={(installed && !incomplete) || !model.downloadable || Boolean(downloadJob)}
            onClick={() =>
              failedDownload
                ? onResumeDownloadJob(localDownloadJob, { payloadChanges: { downloadAction: "resume" } })
                : onDownloadModel(model)
            }
            type="button"
          >
            {downloadJob
              ? downloadJob.status
              : failedDownload
                  ? "Resume Download"
                  : incomplete
                    ? "Fix"
                    : installed
                      ? "Ready"
                      : model.downloadSizeLabel
                        ? `Download ${downloadSize}`
                        : "Download"}
          </button>
          <button className="danger-action" disabled={!canDelete || deletingItem === deleteKey} onClick={() => deleteModel(model)} type="button">
            {model.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
          </button>
        </div>
      </article>
    );
  }

  function renderLoraRow(lora) {
    const installed = lora.installState === "installed";
    const missing = lora.installState === "missing";
    const statusText = missing ? "unavailable" : installed ? "installed" : "pending";
    const deleteKey = `lora:${lora.scope ?? "global"}:${lora.id}`;
    // Built-in LoRAs with a Hugging Face source can be fetched on demand (sc-5944) —
    // user LoRAs are installed via the import form, so they get no Download affordance.
    const hfSource = (lora.source?.provider ?? lora.provider) === "huggingface";
    const canDownload = Boolean(onDownloadLora) && lora.scope === "builtin" && !installed && hfSource;
    const downloadJob = loraDownloadJobsFor(lora).find((job) => !terminalStatuses.has(job.status));
    return (
      <article className={missing ? "lora-row warning" : "lora-row"} key={lora.id ?? lora.name}>
        <span>
          <strong>{lora.name ?? lora.id}</strong>
          <small>{[lora.scope, lora.family ?? "compatible"].filter(Boolean).join(" | ")}</small>
        </span>
        <span className={installed ? "status-badge installed" : "status-badge"}>{statusText}</span>
        <span className="lora-row-actions">
          {canDownload ? (
            <button disabled={Boolean(downloadJob)} onClick={() => onDownloadLora(lora)} type="button">
              {downloadJob ? downloadJob.status : "Download"}
            </button>
          ) : null}
          <button
            className="danger-action"
            disabled={!onDeleteLora || lora.removable === false || deletingItem === deleteKey}
            onClick={() => deleteLora(lora)}
            type="button"
          >
            {lora.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
          </button>
        </span>
        {downloadJob ? (
          <div className="lora-row-progress">
            <WorkerProgressCard job={downloadJob} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
          </div>
        ) : null}
      </article>
    );
  }

  return (
    <section className="main-surface models-surface">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Runtime assets</p>
          <h2>Models</h2>
        </div>
        <label>
          LoRA family
          <select onChange={(event) => setFamilyFilter(event.target.value)} value={familyFilter}>
            <option value="all">All families</option>
            {families.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </label>
      </div>

      {modelGroups.map((group) => {
        const recommended = group.items.filter(isRecommendedModel);
        const additional = group.items.filter((model) => !isRecommendedModel(model));
        // Only split into Recommended / Additional Supported when both buckets exist;
        // otherwise show a single grid so we never render an empty or redundant header.
        const split = recommended.length > 0 && additional.length > 0;
        return (
          <details className="model-type-group" key={group.type} open>
            <summary className="model-type-group-heading">
              <h3>{group.label}</h3>
              <span>{group.items.length}</span>
            </summary>
            {split ? (
              <>
                <div className="model-subgroup">
                  <div className="model-subgroup-heading">
                    <h4>Recommended Models</h4>
                    <span>{recommended.length}</span>
                  </div>
                  <div className="model-grid">{recommended.map((model) => renderModelCard(model))}</div>
                </div>
                <details className="model-subgroup model-subgroup-additional">
                  <summary className="model-subgroup-heading">
                    <h4>Additional Supported Models</h4>
                    <span>{additional.length}</span>
                  </summary>
                  <div className="model-grid">{additional.map((model) => renderModelCard(model))}</div>
                </details>
              </>
            ) : (
              <div className="model-grid">{group.items.map((model) => renderModelCard(model))}</div>
            )}
          </details>
        );
      })}
      {deleteMessage.text ? <p className={deleteMessage.tone === "success" ? "inline-success" : "inline-warning"}>{deleteMessage.text}</p> : null}

      <section className="model-import-panel-section">
        {MODEL_IMPORT_ENABLED && (
        <form className="lora-import-panel models-import-panel" aria-label="Import model" onSubmit={importModel}>
          <div>
            <strong>Import model</strong>
            <span>{modelImportForm.family || "auto-detect family"}</span>
          </div>
          <div className="segmented-control compact-segment" aria-label="Model import source">
            <button
              className={modelImportForm.mode === "url" ? "active" : ""}
              disabled={importingModel}
              onClick={() => setModelImportForm((current) => ({ ...current, mode: "url" }))}
              type="button"
            >
              URL
            </button>
            <button
              className={modelImportForm.mode === "file" ? "active" : ""}
              disabled={importingModel}
              onClick={() => setModelImportForm((current) => ({ ...current, mode: "file" }))}
              type="button"
            >
              Upload
            </button>
          </div>
          <div className="models-import-grid">
            <label>
              Type
              <select
                disabled={importingModel}
                onChange={(event) => setModelImportForm((current) => ({ ...current, type: event.target.value }))}
                value={modelImportForm.type}
              >
                {MODEL_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Family
              <select
                disabled={importingModel || !families.length}
                onChange={(event) => setModelImportForm((current) => ({ ...current, family: event.target.value }))}
                value={modelImportForm.family}
              >
                {families.length ? (
                  <>
                    <option value="">Auto-detect</option>
                    {families.map((family) => (
                      <option key={family} value={family}>
                        {family}
                      </option>
                    ))}
                  </>
                ) : (
                  <option value="">No known families</option>
                )}
              </select>
            </label>
            {isModelFileImport ? (
              <label>
                Model File
                <span className="file-picker-row">
                  <span className="file-upload-button">
                    Choose
                    <input
                      accept=".safetensors,.ckpt,.pt,.bin"
                      disabled={importingModel}
                      key={modelFileInputKey}
                      onChange={(event) => setModelImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                      type="file"
                    />
                  </span>
                  <span className="selected-file-name">{modelImportForm.file?.name ?? "No file selected"}</span>
                </span>
              </label>
            ) : (
              <label>
                Source URL
                <input
                  disabled={importingModel}
                  onChange={(event) => setModelImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                  placeholder="https://..."
                  value={modelImportForm.sourceUrl}
                />
              </label>
            )}
            <label>
              Name
              <input
                disabled={importingModel}
                onChange={(event) => setModelImportForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Optional"
                value={modelImportForm.name}
              />
            </label>
            <button disabled={modelImportDisabled} type="submit">
              {importingModel ? (isModelFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
            </button>
          </div>
          {modelImportMessage.text ? <p className={modelImportMessage.tone === "success" ? "inline-success" : "inline-warning"}>{modelImportMessage.text}</p> : null}
        </form>
        )}
        {pendingModelImportJobs.length ? (
          <div className="lora-import-progress">
            <strong>Model imports in progress</strong>
            <div className="local-job-stack">
              {pendingModelImportJobs.map((job) => (
                <WorkerProgressCard job={job} key={job.id} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
              ))}
            </div>
          </div>
        ) : null}
      </section>

      <section className="lora-panel">
        <div className="lora-panel-header">
          <div className="section-heading">
            <p className="eyebrow">LoRAs</p>
            <h2>{familyFilter === "all" ? "All compatible" : familyFilter}</h2>
          </div>
          <span>{loraCountText}</span>
        </div>

        {builtinLoras.length ? (
          <details className="lora-scope-group" open>
            <summary className="lora-scope-group-heading">
              <h3>Built-In LoRAs</h3>
              <span>{builtinLoras.length}</span>
            </summary>
            <div className="lora-list">{builtinLoras.map((lora) => renderLoraRow(lora))}</div>
          </details>
        ) : null}

        <details className="lora-scope-group" open>
        <summary className="lora-scope-group-heading">
          <h3>User LoRAs</h3>
          <span>{userLoras.length}</span>
        </summary>
        <form className="lora-import-panel models-import-panel" aria-label="Import LoRA" onSubmit={importLora}>
          <div>
            <strong>Import LoRA</strong>
            <span>{importForm.family || "auto-detect"}</span>
          </div>
          <div className="segmented-control compact-segment" aria-label="LoRA import source">
            <button
              className={importForm.mode === "url" ? "active" : ""}
              disabled={importingLora}
              onClick={() => setImportForm((current) => ({ ...current, mode: "url" }))}
              type="button"
            >
              URL
            </button>
            <button
              className={importForm.mode === "file" ? "active" : ""}
              disabled={importingLora}
              onClick={() => setImportForm((current) => ({ ...current, mode: "file" }))}
              type="button"
            >
              Upload
            </button>
          </div>
          <div className="models-import-grid">
            <label>
              Scope
              <select
                disabled={importingLora}
                onChange={(event) => setImportForm((current) => ({ ...current, scope: event.target.value }))}
                value={importForm.scope}
              >
                <option value="global">Global</option>
                <option disabled={!activeProject} value="project">
                  Project
                </option>
              </select>
            </label>
            <label>
              Family
              <select
                disabled={importingLora || !families.length}
                onChange={(event) => setImportForm((current) => ({ ...current, family: event.target.value }))}
                value={importForm.family}
              >
                {families.length ? (
                  <>
                    <option value="">Auto-detect</option>
                    {families.map((family) => (
                      <option key={family} value={family}>
                        {family}
                      </option>
                    ))}
                  </>
                ) : (
                  <option value="">No model families</option>
                )}
              </select>
            </label>
            {showBaseModelSelect ? (
              <label>
                Base model
                <select
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, baseModel: event.target.value }))}
                  value={importForm.baseModel}
                >
                  <option value="">Auto / unspecified</option>
                  {wanBaseModelOptions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name ?? model.id}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {isFileImport ? (
              <>
                <label>
                  LoRA File
                  <span className="file-picker-row">
                    <span className="file-upload-button">
                      Choose
                      <input
                        accept=".safetensors,.ckpt,.pt,.bin"
                        disabled={importingLora}
                        key={fileInputKey}
                        onChange={(event) => setImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                        type="file"
                      />
                    </span>
                    <span className="selected-file-name">{importForm.file?.name ?? "No file selected"}</span>
                  </span>
                </label>
                {showSecondaryFileSlot ? (
                  <label>
                    Low-noise expert (Wan A14B MoE)
                    <span className="file-picker-row">
                      <span className="file-upload-button">
                        Choose
                        <input
                          accept=".safetensors,.ckpt,.pt,.bin"
                          disabled={importingLora}
                          key={`secondary-${fileInputKey}`}
                          onChange={(event) => setImportForm((current) => ({ ...current, secondaryFile: event.target.files?.[0] ?? null }))}
                          type="file"
                        />
                      </span>
                      <span className="selected-file-name">{importForm.secondaryFile?.name ?? "No file selected"}</span>
                    </span>
                  </label>
                ) : null}
              </>
            ) : (
              <label>
                Source URL
                <input
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                  placeholder="https://..."
                  value={importForm.sourceUrl}
                />
              </label>
            )}
            <label>
              Name
              <input
                disabled={importingLora}
                onChange={(event) => setImportForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Optional"
                value={importForm.name}
              />
            </label>
            <button disabled={importDisabled} type="submit">
              {importingLora ? (isFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
            </button>
          </div>
          {showSecondaryFileSlot ? (
            <p className="helper-copy">
              Wan A14B is a two-expert model. Upload both the high-noise file and the low-noise expert so each expert
              gets its own weights.
            </p>
          ) : null}
          {moeMissingSecondary ? (
            <p className="inline-warning">
              No low-noise expert selected — this LoRA will load into the high-noise expert only, leaving the
              low-noise expert un-adapted.
            </p>
          ) : null}
          {importForm.scope === "project" && !activeProject ? <p className="helper-copy">Open a project before importing a project LoRA.</p> : null}
          {importMessage.text ? <p className={importMessage.tone === "success" ? "inline-success" : "inline-warning"}>{importMessage.text}</p> : null}
        </form>
        {localLoraImportJobs.length ? (
          <div className="lora-import-progress">
            <strong>LoRA imports in progress</strong>
            <div className="local-job-stack">
              {localLoraImportJobs.map((job) => (
                <WorkerProgressCard job={job} key={job.id} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
              ))}
            </div>
          </div>
        ) : null}
        {hiddenImportCount ? <p className="helper-copy">{hiddenImportCount} LoRA import{hiddenImportCount === 1 ? " is" : "s are"} hidden by this family filter.</p> : null}
        {userLoras.length ? (
          <div className="lora-family-groups">
            {loraGroups.map((group) => (
              <div className="lora-family-group" key={group.family}>
                <div className="lora-family-group-heading">
                  <h3>{group.family === "compatible" ? "Other / compatible" : group.family}</h3>
                  <span>{group.items.length}</span>
                </div>
                <div className="lora-list">{group.items.map((lora) => renderLoraRow(lora))}</div>
              </div>
            ))}
          </div>
        ) : localLoraImportJobs.length ? null : loras.length && familyFilter !== "all" ? (
          <div className="empty-panel compact-panel">No user LoRAs match {familyFilter}</div>
        ) : (
          <div className="empty-panel compact-panel">No user LoRAs yet — import one above.</div>
        )}
        </details>
      </section>
    </section>
  );
}
