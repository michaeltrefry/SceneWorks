import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { Icon } from "../../components/Icons.jsx";
import { datasetPayload } from "../../training/datasetHelpers.js";
import {
  configDraftFromTarget,
  defaultPresetForTarget,
  presetsForTarget,
  trainingConfigSnapshot,
} from "../../training/trainingConfig.js";
import {
  buildJoyCaptionPrompt,
  defaultCaptionSettings,
  joyCaptionExtraOptions,
  joyCaptionModel,
} from "../../training/joyCaptionPrompts.js";
import { errorStatuses, terminalStatuses } from "../../jobTypes.js";
import { JOY_CAPTION_MODEL_ID } from "../../constants.js";
import { DEFAULT_MAC_CAPABILITIES, macTrainingKernelBlocked } from "../../macGating.js";

// Three things a non-technical creator can teach. Each maps to a training preset
// (matched by recommendedFor) and shapes the auto-captions + trigger word.
const KINDS = [
  { id: "person", label: "A person", sub: "A face or character", Glyph: Icon.Character, keywords: ["person", "character", "portrait", "face", "identity"], suffix: "person", useName: true },
  { id: "style", label: "A style", sub: "A look or art style", Glyph: Icon.Sparkle, keywords: ["style", "aesthetic", "look", "art"], suffix: "style", useName: false },
  { id: "object", label: "An object", sub: "A product or thing", Glyph: Icon.Model, keywords: ["object", "product", "thing", "item"], suffix: "object", useName: false },
];

// Plain-language "More options" mapped to safe scalings on proven preset defaults:
// quality nudges step count, strength nudges the LoRA alpha. Neither can produce
// an invalid config — they only scale a known-good baseline.
const QUALITY = { faster: { label: "Faster", steps: 0.6 }, balanced: { label: "Balanced", steps: 1 }, best: { label: "Best quality", steps: 1.5 } };
const STRENGTH = { subtle: { label: "Subtle", alpha: 0.75 }, standard: { label: "Standard", alpha: 1 }, strong: { label: "Strong", alpha: 1.25 } };

// LoRA training works well from a handful of varied examples; below this the
// result rarely holds together, so we ask for a few more first.
const MIN_EXAMPLES = 4;
// On a gated Mac the native captioner has no auto-download, so a missing model
// would just stall the queued caption job. Skip captioning in that case and
// teach from the examples directly (trigger-word training stands on its own).
const useNameOption = joyCaptionExtraOptions[0]?.value ?? "";

function slugify(value) {
  return String(value).toLowerCase().trim().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function captionJobPayload(kind, name) {
  const settings = {
    ...defaultCaptionSettings,
    recaption: true,
    nameInput: kind.useName ? name : "",
    extraOptions: kind.useName ? [useNameOption] : [],
  };
  return {
    captioner: "joy_caption",
    modelNameOrPath: joyCaptionModel,
    recaption: true,
    requestedGpu: "auto",
    options: {
      captionType: settings.captionType,
      captionLength: settings.captionLength,
      extraOptions: settings.extraOptions,
      nameInput: settings.nameInput,
      temperature: 0.6,
      topP: 0.9,
      maxNewTokens: 256,
      captionPrompt: buildJoyCaptionPrompt(settings),
      lowVram: false,
    },
  };
}

export function Teach() {
  const {
    activeProject,
    models = [],
    loras = [],
    jobs = [],
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
    trainingTargets,
    trainingPresets,
    uploadTrainingDatasetItem,
    createTrainingDataset,
    createTrainingDatasetCaptionJob,
    createTrainingJob,
    loadTrainingDataset,
    createModelDownloadJob,
    setActiveView,
  } = useAppContext();

  const [kindId, setKindId] = useState("person");
  const [files, setFiles] = useState([]);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [triggerTouched, setTriggerTouched] = useState(false);
  const [quality, setQuality] = useState("balanced");
  const [strength, setStrength] = useState("standard");
  const [error, setError] = useState("");
  // Pipeline: idle | uploading | describing | teaching | done | error
  const [pipe, setPipe] = useState({ stage: "idle" });
  const advancedRef = useRef(false);
  const fileInputRef = useRef(null);

  const kind = KINDS.find((entry) => entry.id === kindId) ?? KINDS[0];
  const caps = macCapabilities ?? DEFAULT_MAC_CAPABILITIES;

  // Pick the base model to fine-tune: an image LoRA target this machine can
  // actually train, preferring the fast recommended bases, preferring installed.
  const target = useMemo(() => {
    const all = (trainingTargets?.targets ?? []).filter(
      (entry) => entry.modality === "image" && !macTrainingKernelBlocked(caps, entry.kernel),
    );
    const installed = all.filter((entry) => {
      const base = models.find((model) => model.id === entry.baseModel);
      return !base || base.installState !== "missing";
    });
    const preferred = ["z_image_turbo_lora", "sdxl_lora"];
    const pick = (list) => preferred.map((id) => list.find((entry) => entry.id === id)).find(Boolean) ?? list[0] ?? null;
    return pick(installed) ?? pick(all);
  }, [trainingTargets, models, caps]);

  const baseReady = useMemo(() => {
    if (!target) return false;
    const base = models.find((model) => model.id === target.baseModel);
    return !base || base.installState !== "missing";
  }, [target, models]);

  const preset = useMemo(() => {
    const forTarget = presetsForTarget(trainingPresets?.presets ?? [], target?.id);
    const matched = forTarget.find((entry) =>
      (entry.recommendedFor ?? []).some((tag) => kind.keywords.includes(String(tag).toLowerCase())),
    );
    return matched ?? defaultPresetForTarget(trainingPresets?.presets ?? [], target?.id);
  }, [trainingPresets, target?.id, kind]);

  const captionModel = useMemo(() => models.find((model) => model.id === JOY_CAPTION_MODEL_ID), [models]);
  const captionUnavailable = Boolean(caps?.macGatingActive) && captionModel?.installState === "missing";

  // Keep the trigger word in step with the name until the user edits it directly.
  useEffect(() => {
    if (triggerTouched) return;
    const slug = slugify(name);
    setTrigger(slug ? `${slug}_${kind.suffix}` : "");
  }, [name, kind.suffix, triggerTouched]);

  const previews = useMemo(() => files.map((file) => ({ key: `${file.name}:${file.size}`, url: URL.createObjectURL(file) })), [files]);
  useEffect(() => () => previews.forEach((preview) => URL.revokeObjectURL(preview.url)), [previews]);

  const busy = pipe.stage === "uploading" || pipe.stage === "describing" || pipe.stage === "teaching";

  function addFiles(incoming) {
    const images = Array.from(incoming ?? []).filter((file) => file.type.startsWith("image/"));
    if (images.length) setFiles((current) => [...current, ...images]);
  }

  function removeFile(index) {
    setFiles((current) => current.filter((_, position) => position !== index));
  }

  async function startTraining(datasetId, loraName, note) {
    try {
      const fresh = (await loadTrainingDataset(datasetId)) ?? { id: datasetId };
      const draft = configDraftFromTarget(target, { name: loraName }, ["auto"], trigger.trim(), preset);
      draft.outputName = loraName;
      const steps = Number(draft.steps);
      const alpha = Number(draft.alpha);
      if (steps) draft.steps = Math.max(1, Math.round(steps * QUALITY[quality].steps));
      if (alpha) draft.alpha = Math.max(1, Math.round(alpha * STRENGTH[strength].alpha));
      const snapshot = trainingConfigSnapshot({ activeDataset: fresh, configDraft: draft, selectedPreset: preset, selectedTarget: target, dryRun: false });
      const job = await createTrainingJob({
        targetId: snapshot.targetId,
        datasetId: snapshot.datasetId,
        datasetVersion: snapshot.datasetVersion,
        presetId: snapshot.presetId,
        presetVersion: snapshot.presetVersion,
        outputName: snapshot.outputName,
        dryRun: false,
        config: snapshot.config,
      });
      setPipe({ stage: "teaching", datasetId, trainJobId: job?.id, loraName, note: note ?? "" });
    } catch (err) {
      setPipe({ stage: "error", error: err?.message || "Couldn't start teaching.", loraName });
    }
  }
  const startTrainingRef = useRef(startTraining);
  startTrainingRef.current = startTraining;

  // Caption job done → reload the now-captioned dataset and start training.
  // If captioning failed, fall back to teaching from the examples directly.
  useEffect(() => {
    if (pipe.stage !== "describing" || !pipe.captionJobId) return;
    const job = jobs.find((entry) => entry.id === pipe.captionJobId);
    if (!job || !terminalStatuses.has(job.status) || advancedRef.current) return;
    advancedRef.current = true;
    const failed = errorStatuses.has(job.status);
    startTrainingRef.current(pipe.datasetId, pipe.loraName, failed ? "We couldn't auto-describe them, so we taught from the examples directly." : "");
  }, [jobs, pipe.stage, pipe.captionJobId, pipe.datasetId, pipe.loraName]);

  // Training job done → settle the pipeline.
  useEffect(() => {
    if (pipe.stage !== "teaching" || !pipe.trainJobId) return;
    const job = jobs.find((entry) => entry.id === pipe.trainJobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    setPipe((current) =>
      errorStatuses.has(job.status)
        ? { ...current, stage: "error", error: job.error || "Teaching didn't finish — check In progress." }
        : { ...current, stage: "done" },
    );
  }, [jobs, pipe.stage, pipe.trainJobId]);

  async function startTeaching() {
    if (busy) return;
    setError("");
    if (!activeProject) {
      setError("Open or create a workspace first.");
      return;
    }
    if (!target || !baseReady) {
      setError("Add a base model in Settings before teaching.");
      return;
    }
    const cleanName = name.trim();
    const cleanTrigger = trigger.trim();
    if (!cleanName) {
      setError("Give it a name.");
      return;
    }
    if (!cleanTrigger) {
      setError("Add a word to trigger it.");
      return;
    }
    if (files.length < MIN_EXAMPLES) {
      setError(`Add at least ${MIN_EXAMPLES} example photos — 10 to 20 works best.`);
      return;
    }
    advancedRef.current = false;
    setPipe({ stage: "uploading", loraName: cleanName });
    try {
      const uploaded = (await Promise.all(files.map((file) => uploadTrainingDatasetItem(file)))).filter((asset) => asset?.id);
      if (!uploaded.length) throw new Error("Couldn't upload those photos.");
      uploaded.forEach((asset) => {
        asset.datasetOnly = true;
      });
      const assetsById = new Map(uploaded.map((asset) => [asset.id, asset]));
      const payload = datasetPayload({
        activeDataset: null,
        assetsById,
        associatedCharacterId: null,
        captionDraftById: {},
        name: cleanName,
        selectedAssetIds: uploaded.map((asset) => asset.id),
      });
      const dataset = await createTrainingDataset(payload);
      if (!dataset?.id) throw new Error("Couldn't create the example set.");
      if (captionUnavailable) {
        await startTraining(dataset.id, cleanName, "");
        return;
      }
      const captionJob = await createTrainingDatasetCaptionJob(dataset.id, captionJobPayload(kind, cleanName));
      setPipe({ stage: "describing", datasetId: dataset.id, captionJobId: captionJob?.id, loraName: cleanName });
    } catch (err) {
      setPipe({ stage: "error", error: err?.message || "Something went wrong.", loraName: cleanName });
    }
  }

  function reset() {
    advancedRef.current = false;
    setPipe({ stage: "idle" });
    setFiles([]);
    setName("");
    setTrigger("");
    setTriggerTouched(false);
  }

  // Right rail: what's been taught (ready) and what's cooking now.
  const readyLoras = loras.filter((lora) => lora.installState !== "missing");
  const teachingJobs = jobs.filter((job) => job.type === "lora_train" && !terminalStatuses.has(job.status));

  const activeJob = jobs.find((job) => job.id === (pipe.stage === "describing" ? pipe.captionJobId : pipe.trainJobId));
  const pct = Number.isFinite(activeJob?.progress) ? Math.round(activeJob.progress * 100) : null;

  if (!target) {
    return (
      <section className="main-surface sw-make">
        <div className="sw-empty">Teaching needs the training catalog, which isn't loaded yet. Try again in a moment.</div>
      </section>
    );
  }

  if (!baseReady) {
    return (
      <section className="main-surface sw-make">
        <div className="sw-card">
          <h3 className="sw-q">First, add a model to teach from</h3>
          <p className="sw-rendering">Teaching builds on a base model like Z-Image-Turbo or SDXL. Add one in Settings and you're set.</p>
          <button type="button" className="sw-cta" onClick={() => setActiveView("SimpleSettings")}>
            <Icon.Sliders /> Add a model
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="main-surface sw-make">
      <div className="sw-make-grid">
        <div className="sw-card">
          {pipe.stage === "idle" || pipe.stage === "error" ? (
            <>
              <div className="sw-field">
                <h3 className="sw-q">What are you teaching it?</h3>
                <div className="sw-startfrom sw-teach-kinds">
                  {KINDS.map((entry) => {
                    const Glyph = entry.Glyph;
                    return (
                      <button
                        type="button"
                        key={entry.id}
                        className={`sw-sf ${entry.id === kindId ? "on" : ""}`.trim()}
                        onClick={() => setKindId(entry.id)}
                      >
                        <b><Glyph /> {entry.label}</b>
                        <span>{entry.sub}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="sw-field">
                <h3 className="sw-q">Add examples</h3>
                <button
                  type="button"
                  className="sw-drop"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    addFiles(event.dataTransfer?.files);
                  }}
                >
                  <Icon.Folder />
                  <span>Drag photos here, or browse</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(event) => {
                    addFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
                {files.length ? (
                  <>
                    <div className="sw-refs sw-teach-thumbs">
                      {previews.map((preview, index) => (
                        <button type="button" className="sw-ref" key={preview.key} onClick={() => removeFile(index)} title="Remove">
                          <img src={preview.url} alt="" />
                        </button>
                      ))}
                    </div>
                    <p className="sw-meta">{files.length} added · 10–20 clear, varied examples work best. We'll describe them for you automatically.</p>
                  </>
                ) : (
                  <p className="sw-meta">10–20 clear, varied examples work best. We'll describe them for you automatically.</p>
                )}
              </div>

              <div className="sw-field">
                <h3 className="sw-q">Name it</h3>
                <div className="sw-teach-name">
                  <label>
                    Name
                    <input
                      className="sw-input"
                      placeholder={kind.id === "person" ? "e.g. Mara" : kind.id === "style" ? "e.g. Dusk Watercolor" : "e.g. Blue Mug"}
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                    />
                  </label>
                  <label>
                    Word to trigger it
                    <input
                      className="sw-input"
                      placeholder="mara_person"
                      value={trigger}
                      onChange={(event) => {
                        setTriggerTouched(true);
                        setTrigger(event.target.value);
                      }}
                    />
                  </label>
                </div>
              </div>

              <div className="sw-go">
                <button type="button" className="sw-cta" onClick={startTeaching} disabled={busy}>
                  <Icon.Train /> Start teaching
                </button>
                <span className="sw-meta">Runs in the background · about 25–40 minutes</span>
              </div>

              {error ? <p className="sw-notice">{error}</p> : null}
              {pipe.stage === "error" && pipe.error ? <p className="sw-notice">{pipe.error}</p> : null}

              <details className="sw-disclosure">
                <summary>
                  <Icon.ChevDown className="sw-caret" /> More options <span className="sw-meta">— most people never need these</span>
                </summary>
                <div className="sw-adv">
                  <label>
                    Quality vs. speed
                    <select value={quality} onChange={(event) => setQuality(event.target.value)}>
                      {Object.entries(QUALITY).map(([value, option]) => (
                        <option key={value} value={value}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Strength (how strongly it learns)
                    <select value={strength} onChange={(event) => setStrength(event.target.value)}>
                      {Object.entries(STRENGTH).map(([value, option]) => (
                        <option key={value} value={value}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                {captionUnavailable && captionModel ? (
                  <p className="sw-meta sw-teach-capnote">
                    Auto-describe is off until the describer model is added.{" "}
                    <button type="button" className="sw-linkbtn" onClick={() => createModelDownloadJob(captionModel)}>Add it</button> — teaching still works without it.
                  </p>
                ) : null}
              </details>
            </>
          ) : (
            <div className="sw-teach-run">
              <h3 className="sw-q">
                {pipe.stage === "uploading" && "Getting your examples ready…"}
                {pipe.stage === "describing" && "Step 1 of 2 — describing your examples…"}
                {pipe.stage === "teaching" && "Step 2 of 2 — teaching…"}
                {pipe.stage === "done" && `“${pipe.loraName}” is ready`}
              </h3>

              {pipe.stage === "done" ? (
                <>
                  <p className="sw-rendering">Pick it in Make a picture under “Use something you taught,” or with the word <b>{trigger.trim()}</b> in your description.</p>
                  <div className="sw-go">
                    <button type="button" className="sw-cta" onClick={() => setActiveView("MakePicture")}>
                      <Icon.Image /> Make a picture
                    </button>
                    <button type="button" className="sw-act" onClick={reset}>Teach something else</button>
                  </div>
                </>
              ) : (
                <>
                  <p className="sw-rendering">{pipe.note || "You can leave this page — it keeps going in the background. Track it under In progress."}</p>
                  <div className="sw-bar sw-teach-bar"><i style={{ width: `${pct ?? 8}%` }} /></div>
                  <div className="sw-go">
                    <button type="button" className="sw-act" onClick={() => setActiveView("Queue")}>See progress</button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        <aside className="sw-results">
          <h3>What you've taught</h3>
          {teachingJobs.length === 0 && readyLoras.length === 0 ? (
            <p className="sw-rendering">Nothing yet. Teach a person, a style, or an object and it'll show up here, ready to use in any prompt.</p>
          ) : (
            <div className="sw-teach-list">
              {teachingJobs.map((job) => (
                <div className="sw-line" key={job.id}>
                  <span className="sw-teach-dot pending" />
                  <span className="sw-line-label">
                    <b>{job.payload?.outputName || "Teaching…"}</b>
                    <small>learning · {Number.isFinite(job.progress) ? `${Math.round(job.progress * 100)}%` : "in progress"}</small>
                  </span>
                </div>
              ))}
              {readyLoras.map((lora) => (
                <div className="sw-line" key={lora.id}>
                  <span className="sw-teach-dot ready" />
                  <span className="sw-line-label">
                    <b>{lora.name ?? lora.id}</b>
                    <small>ready</small>
                  </span>
                </div>
              ))}
            </div>
          )}
          <p className="sw-meta sw-teach-hint">Once ready, pick it in Make a picture under “Use something you taught.”</p>
        </aside>
      </div>
    </section>
  );
}
