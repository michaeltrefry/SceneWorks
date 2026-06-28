import React from "react";
import { AssetBatchModal, AssetSelectionBar, useAssetBatch } from "../assetBatch.jsx";
import { AssetPickerField, CharacterImportDialog } from "../components/AssetPicker.jsx";
import { AssetCard, emptyTrash } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { terminalStatuses } from "../jobTypes.js";
import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";
import { LoraPickerField, useLoraSelection } from "../components/LoraPickerField.jsx";
import { CharacterAdvancedOptions, useCharacterAdvancedOptions } from "../components/CharacterAdvancedOptions.jsx";
import { KeypointCollectionField } from "../components/KeypointCollectionField.jsx";
import { usePoseLibrary, useUserPoseLoader } from "../poseLibrary.js";
import { extractFamilies } from "../presetUtils.js";

// Curated negative prompts seeded into the advanced panel for each Character
// Studio flow (sc-3857). These are the baseline the editable Negative prompt
// control starts from — the anti-`airbrushed/waxy/plastic/blurry` terms are what
// keep RealVisXL's photoreal output from going shiny/over-contrasty. Angle and
// pose differ: angle guards face framing (hat/hood/occluded face), pose guards
// the body (extra limbs/fingers, cropped/out-of-frame).
const ANGLE_SET_NEGATIVE_PROMPT =
  "hat, hood, hoodie, scarf, holding object, paper, hands near face, covering face, occluded face, cropped, " +
  "multiple people, plastic skin, airbrushed, cgi, 3d render, cartoon, anime, waxy, deformed, blurry";
const POSE_LIBRARY_NEGATIVE_PROMPT =
  "cropped, out of frame, multiple people, extra limbs, deformed hands, extra fingers, " +
  "plastic skin, airbrushed, cgi, 3d render, cartoon, anime, waxy, blurry";

// Resolve a character generation job's produced images from the full asset
// catalog. Using the catalog (not just latestAssets, which is the single most
// recent generation set) is what lets pose / angle previews stream in as each
// image is saved — otherwise a parallel job's newer generation set hides this
// job's partial outputs from latestAssets and nothing surfaces in the card.
function jobImageAssets(job, assets) {
  if (!job?.result || !Array.isArray(assets)) return [];
  const byId = new Map(assets.map((asset) => [asset.id, asset]));
  const ids = job.result.assetIds ?? [];
  if (ids.length) return ids.map((id) => byId.get(id)).filter((asset) => asset?.type === "image");
  if (job.result.generationSetId) {
    return assets.filter((asset) => asset?.type === "image" && asset.generationSetId === job.result.generationSetId);
  }
  return [];
}

// True when the job was started from the Angle Set form (advanced.angleSet)
// rather than the Pose Library (advanced.poses). Keeps the two cards from
// stealing each other's jobs when both are visible for the same character.
function isAngleSetJob(job) {
  return job?.payload?.advanced?.angleSet === true;
}

function isPoseLibraryJob(job) {
  return Array.isArray(job?.payload?.advanced?.poses) && job.payload.advanced.poses.length > 0;
}

// Active jobs for a given character + form-kind predicate, with terminal jobs
// filtered out. Stack order: oldest-first (running run on top, queued runs
// follow in execution order), mirroring buildLocalJobStack in App.jsx so the
// Character Studio matches Image/Video Studio behavior.
function characterFormJobs(imageLocalJobs, characterId, predicate) {
  if (!characterId) return [];
  return (imageLocalJobs ?? [])
    .filter(
      (job) =>
        job?.payload?.characterId === characterId &&
        predicate(job) &&
        !terminalStatuses.has(job.status),
    )
    .sort((left, right) => (left.createdAt ?? "").localeCompare(right.createdAt ?? ""));
}

export function editableLora(link) {
  return {
    name: link?.name ?? "",
    triggerWords: (link?.triggerWords ?? []).join(", "),
    defaultWeight: link?.defaultWeight ?? 0.8,
    families: extractFamilies(link).join(", "),
    scope: link?.scope ?? "project",
  };
}

function summarizeCompatibility(item) {
  const values = extractFamilies(item);
  return values.length ? values.join(", ") : "Unspecified";
}

export function CharacterReferences({
  referenceCandidates,
  onGenerateFromReference,
  onPreview,
  referenceMessage,
  referenceAssetIds,
  removeCharacterReference,
  selectedCharacter,
  setReferenceAssetIds,
  submitReference,
  updateCharacterReference,
}) {
  // Fullscreen preview stays bound to this character's reference images.
  const referenceAssets = (selectedCharacter.references ?? [])
    .map((reference) => reference.asset)
    .filter(Boolean);
  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">References</p>
        <h2>Approved set</h2>
      </div>
      <form className="inline-create asset-reference-create" onSubmit={submitReference}>
        {/* sc-6042: scoped to this character's assets (no category tabs). Import
            project/uploaded media into the character on the Assets tab to grow
            this set. */}
        <AssetPickerField
          assets={referenceCandidates}
          buttonLabel="Add image or frame"
          emptyLabel="No references selected"
          label="Reference assets"
          multiple
          onChange={setReferenceAssetIds}
          showCategories={false}
          values={referenceAssetIds}
        />
        <button disabled={!referenceAssetIds.length} type="submit">
          Add
        </button>
      </form>
      {referenceMessage ? <p className="inline-warning">{referenceMessage}</p> : null}
      <div className="character-reference-grid">
        {(selectedCharacter.references ?? []).map((reference) => (
          <article className={reference.approved ? "reference-card approved" : "reference-card"} key={reference.assetId}>
            <button className="reference-media" onClick={() => reference.asset && onPreview(reference.asset, referenceAssets)} type="button">
              {reference.asset ? <AssetMedia asset={reference.asset} /> : <span>Missing asset</span>}
            </button>
            <div>
              <strong>{reference.asset?.displayName ?? reference.assetId}</strong>
              <span>{reference.role}</span>
            </div>
            <div className="review-actions">
              <button
                className={reference.approved ? "active" : ""}
                onClick={() => updateCharacterReference(selectedCharacter.id, reference.assetId, { approved: !reference.approved })}
                type="button"
              >
                {reference.approved ? "Approved" : "Approve"}
              </button>
              {reference.approved && onGenerateFromReference ? (
                <button onClick={() => onGenerateFromReference(reference.assetId)} type="button">
                  Generate variations
                </button>
              ) : null}
              <button onClick={() => removeCharacterReference(selectedCharacter.id, reference.assetId)} type="button">
                Remove
              </button>
            </div>
          </article>
        ))}
        {selectedCharacter.references?.length ? null : <div className="empty-panel compact-panel">No references</div>}
      </div>
    </section>
  );
}

export function CharacterLooks({
  approvedReferences,
  deleteCharacterLook,
  lookDraft,
  selectedCharacter,
  selectedReferenceIds,
  setLookDraft,
  setSelectedReferenceIds,
  setTestLookId,
  submitLook,
  updateCharacterLook,
}) {
  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">Looks</p>
        <h2>Saved presets</h2>
      </div>
      <form className="look-composer" onSubmit={submitLook}>
        <input
          aria-label="Look name"
          onChange={(event) => setLookDraft((item) => ({ ...item, name: event.target.value }))}
          placeholder="Look name"
          value={lookDraft.name}
        />
        <input
          aria-label="Look notes"
          onChange={(event) => setLookDraft((item) => ({ ...item, description: event.target.value }))}
          placeholder="Notes"
          value={lookDraft.description}
        />
        <button disabled={!lookDraft.name.trim()} type="submit">
          Save Look
        </button>
        <div className="reference-checks">
          {approvedReferences.map((reference) => (
            <label className="checkline" key={reference.assetId}>
              <input
                checked={selectedReferenceIds.includes(reference.assetId)}
                onChange={(event) =>
                  setSelectedReferenceIds((ids) =>
                    event.target.checked ? [...ids, reference.assetId] : ids.filter((id) => id !== reference.assetId),
                  )
                }
                type="checkbox"
              />
              {reference.asset?.displayName ?? reference.assetId}
            </label>
          ))}
        </div>
      </form>
      <div className="look-list">
        {(selectedCharacter.looks ?? []).map((look) => (
          <article className="look-row" key={look.id}>
            <div>
              <strong>{look.name}</strong>
              <span>{look.description || "No notes"}</span>
              <small>{look.approvedReferenceIds?.length ?? 0} approved refs</small>
            </div>
            <div className="review-actions">
              <button onClick={() => setTestLookId(look.id)} type="button">
                Select
              </button>
              <button
                onClick={() =>
                  updateCharacterLook(selectedCharacter.id, look.id, {
                    ...look,
                    recipeSettings: { ...(look.recipeSettings ?? {}), touchedAt: new Date().toISOString() },
                  })
                }
                type="button"
              >
                Refresh
              </button>
              <button onClick={() => deleteCharacterLook(selectedCharacter.id, look.id)} type="button">
                Delete
              </button>
            </div>
          </article>
        ))}
        {selectedCharacter.looks?.length ? null : <div className="empty-panel compact-panel">No looks</div>}
      </div>
    </section>
  );
}

export function CharacterLoras({
  detachCharacterLora,
  loraEdits,
  loraId,
  loras,
  saveLora,
  selectedCharacter,
  setLoraEdit,
  setLoraId,
  submitLora,
}) {
  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">LoRAs</p>
        <h2>Character adapters</h2>
      </div>
      <form className="inline-create" onSubmit={submitLora}>
        <select onChange={(event) => setLoraId(event.target.value)} value={loraId}>
          <option value="">Attach imported LoRA</option>
          {loras.map((lora) => (
            <option key={lora.id} value={lora.id}>
              {lora.name}
            </option>
          ))}
        </select>
        <button disabled={!loraId} type="submit">
          Attach
        </button>
      </form>
      <div className="lora-editor-list">
        {(selectedCharacter.loras ?? []).map((link) => {
          const edit = loraEdits[link.id] ?? editableLora(link);
          return (
            <article className="lora-editor" key={link.id}>
              <div className="lora-editor-head">
                <strong>{link.name}</strong>
                <span>{link.copiedIntoProject ? "Project copy" : link.scope}</span>
              </div>
              <div className="control-grid compact-controls">
                <label>
                  Name
                  <input onChange={(event) => setLoraEdit(link.id, "name", event.target.value)} value={edit.name} />
                </label>
                <label>
                  Families
                  <input onChange={(event) => setLoraEdit(link.id, "families", event.target.value)} value={edit.families} />
                </label>
                <label>
                  Triggers
                  <input onChange={(event) => setLoraEdit(link.id, "triggerWords", event.target.value)} value={edit.triggerWords} />
                </label>
                <label>
                  Weight
                  <input
                    max="2"
                    min="-2"
                    onChange={(event) => setLoraEdit(link.id, "defaultWeight", event.target.value)}
                    step="0.05"
                    type="number"
                    value={edit.defaultWeight}
                  />
                </label>
              </div>
              <small>Compatibility: {summarizeCompatibility(link)}</small>
              <div className="review-actions">
                <button onClick={() => saveLora(link)} type="button">
                  Save
                </button>
                <button onClick={() => detachCharacterLora(selectedCharacter.id, link.id)} type="button">
                  Detach
                </button>
              </div>
            </article>
          );
        })}
        {selectedCharacter.loras?.length ? null : <div className="empty-panel compact-panel">No linked LoRAs</div>}
      </div>
    </section>
  );
}

export function CharacterTest({
  addCharacterReference,
  deleteAsset,
  imageModels,
  latestAssets,
  onPreview,
  purgeAsset,
  selectedCharacter,
  setTestCount,
  setTestLookId,
  setTestModel,
  setTestPrompt,
  setTestResolution,
  submitTest,
  testCount,
  testLookId,
  testModel,
  testPrompt,
  testResolution,
  updateAssetStatus,
}) {
  const [showOutputs, setShowOutputs] = React.useState(false);
  const [viewMode, setViewMode] = React.useState("active");
  // Scope the outputs grid to THIS character (its generated images + approved
  // references) instead of dumping every recent project image, and keep it
  // collapsed by default so it never turns the studio into an endless scroll.
  const scopedAssets = (latestAssets ?? []).filter(
    (asset) =>
      asset.recipe?.normalizedSettings?.characterId === selectedCharacter.id ||
      (asset.metadata?.characterReferences ?? []).some((ref) => ref.characterId === selectedCharacter.id),
  );
  // Discarded (status.trashed) images leave the active grid so they don't keep
  // their slots, but stay reachable through the Trashcan view for restore/purge.
  const activeAssets = scopedAssets.filter((asset) => !asset.status?.trashed);
  const trashedAssets = scopedAssets.filter((asset) => asset.status?.trashed);
  const showingTrash = viewMode === "trashed";
  const characterAssets = showingTrash ? trashedAssets : activeAssets;
  return (
    <section className="character-section test-character-panel">
      <div className="section-heading">
        <p className="eyebrow">Test Character</p>
        <h2>Sample outputs</h2>
      </div>
      <form className="test-character-form" onSubmit={submitTest}>
        <label className="prompt-field">
          Prompt
          <textarea onChange={(event) => setTestPrompt(event.target.value)} value={testPrompt} />
        </label>
        <div className="control-grid">
          <label>
            Look
            <select onChange={(event) => setTestLookId(event.target.value)} value={testLookId}>
              <option value="">Character defaults</option>
              {(selectedCharacter.looks ?? []).map((look) => (
                <option key={look.id} value={look.id}>
                  {look.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select onChange={(event) => setTestModel(event.target.value)} value={testModel}>
              {imageModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Count
            <input min="1" max="8" onChange={(event) => setTestCount(Number(event.target.value))} type="number" value={testCount} />
          </label>
          <label>
            Resolution
            <select onChange={(event) => setTestResolution(event.target.value)} value={testResolution}>
              <option value="768x768">768 x 768</option>
              <option value="1024x1024">1024 x 1024</option>
              <option value="1280x720">1280 x 720</option>
              <option value="720x1280">720 x 1280</option>
            </select>
          </label>
        </div>
        <div className="guidance-strip">
          <strong>Preset-only test</strong>
          <span>Outputs use the prompt and record this character in metadata; image-reference conditioning and LoRA loading are not active yet.</span>
        </div>
        <button className="primary-action" disabled={!testPrompt.trim()} type="submit">
          Test Character
        </button>
      </form>
      <div className="review-panel-head">
        <button className="advanced-toggle" onClick={() => setShowOutputs((value) => !value)} type="button">
          {showOutputs ? "Hide" : "Show"} this character's images ({activeAssets.length})
        </button>
        {showOutputs ? (
          <div className="segmented-control" role="group" aria-label="Character image collection">
            <button className={showingTrash ? "" : "active"} onClick={() => setViewMode("active")} type="button">
              Images ({activeAssets.length})
            </button>
            <button className={showingTrash ? "active" : ""} onClick={() => setViewMode("trashed")} type="button">
              Trashcan ({trashedAssets.length})
            </button>
          </div>
        ) : null}
        {showOutputs && showingTrash ? (
          <button
            className="danger-action empty-trash-button"
            disabled={!trashedAssets.length}
            onClick={() => emptyTrash(trashedAssets, purgeAsset)}
            type="button"
          >
            Empty Trash ({trashedAssets.length})
          </button>
        ) : null}
      </div>
      {showOutputs ? (
        <div className="review-grid">
          {characterAssets.map((asset) => (
            <div className="test-result" key={asset.id}>
              <AssetCard
                asset={asset}
                deleteAsset={deleteAsset}
                onPreview={(previewed) => onPreview(previewed, characterAssets)}
                purgeAsset={purgeAsset}
                updateAssetStatus={updateAssetStatus}
              />
              {showingTrash ? null : (
                <button
                  onClick={() => addCharacterReference(selectedCharacter.id, { assetId: asset.id, approved: true, role: "test-output" })}
                  type="button"
                >
                  Approve as Reference
                </button>
              )}
            </div>
          ))}
          {characterAssets.length ? null : (
            <div className="empty-panel compact-panel">
              {showingTrash
                ? "Trashcan is empty — discarded images for this character will appear here."
                : "No images for this character yet — generate an angle set or a test above."}
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
}

// sc-4195: CharacterAngleSet and CharacterPoseLibrary were ~260-line copy-paste
// twins (backbone picker, reference row + upload, prompt seeding, submit scaffolding,
// active-job stack, character-images strip — sc-3857 negative prompts and sc-2223
// LoRA changes had to be applied to both). The shared backbone now lives in
// CharacterGenerationPanel, parameterized by a `mode` describing the labels, the
// reference role, the job predicate, the prompt seed, and a mode-controller hook that
// owns the mode-specific controls + the advanced payload it contributes.
function CharacterGenerationPanel({
  mode,
  selectedCharacter,
  model,
  models,
  approvedReferences,
  assets = [],
  createImageJob,
  importAsset,
  addCharacterReference,
  latestAssets = [],
  imageLocalJobs = [],
  loras = [],
  rememberLocalGenerationJob,
  onCancel,
  onDuplicate,
  onOpenQueue,
  onPreview,
  onRetry,
}) {
  // sc-2003: multi-backbone picker. `models` is the full list of capable backbones
  // (manifest order); `model` is the resolved default (kept for back-compat with the
  // pre-picker callers). The local state below tracks the user's pick.
  const availableModels = Array.isArray(models) && models.length > 0
    ? models
    : (model ? [model] : []);
  const [selectedModelId, setSelectedModelId] = React.useState(
    model?.id ?? availableModels[0]?.id ?? "",
  );
  const activeModel = availableModels.find((item) => item.id === selectedModelId)
    ?? availableModels[0]
    ?? null;
  // Family-filtered LoRA picker (sc-2223), filtered to the active backbone's family.
  const loraSelection = useLoraSelection(loras, activeModel);
  // Advanced tuning (sc-3857): Guidance, Reference strength, Identity structure,
  // Steps, Sampler/Scheduler, editable Negative prompt, Seed. Defaults track the
  // active backbone; values fold into the job's `advanced` dict at submit.
  const advanced = useCharacterAdvancedOptions(activeModel, {
    defaultNegativePrompt: mode.defaultNegativePrompt,
    identityStructureMode: mode.identityStructureMode,
  });
  const [referenceAssetId, setReferenceAssetId] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [status, setStatus] = React.useState("");
  const fileInputRef = React.useRef(null);
  const characterId = selectedCharacter?.id;
  // Mode-specific controls (JSX) + advanced payload + readiness/labels.
  const controller = mode.useController({ activeModel, loraSelection });

  React.useEffect(() => {
    setReferenceAssetId(approvedReferences[0]?.assetId ?? "");
  }, [characterId, approvedReferences]);
  React.useEffect(() => {
    setPrompt(mode.seedPrompt(selectedCharacter));
    setStatus("");
  }, [characterId, selectedCharacter?.name, selectedCharacter?.description]);
  // Reset the picker selection when the available models change (e.g. a new
  // manifest load) so a stale id doesn't lock the panel.
  React.useEffect(() => {
    if (!availableModels.find((item) => item.id === selectedModelId)) {
      setSelectedModelId(availableModels[0]?.id ?? "");
    }
  }, [availableModels, selectedModelId]);

  if (!activeModel || !selectedCharacter) {
    return null;
  }

  // Active jobs for this character's form, derived from the App-level imageLocalJobs
  // so they persist across navigation and stack when multiple runs are in flight
  // (sc-2092 follow-up). The mode predicate keeps the angle/pose cards from stealing
  // each other's jobs when both are visible for the same character.
  const activeJobs = characterFormJobs(imageLocalJobs, characterId, mode.jobPredicate);
  const characterImages = (latestAssets ?? []).filter(
    (asset) =>
      asset.recipe?.normalizedSettings?.characterId === characterId ||
      (asset.metadata?.characterReferences ?? []).some((ref) => ref.characterId === characterId),
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setStatus("Uploading reference…");
    const asset = await importAsset(file, { throwOnError: false });
    if (asset?.id) {
      setReferenceAssetId(asset.id);
      await addCharacterReference(characterId, { assetId: asset.id, approved: true, role: mode.referenceRole });
      setStatus("");
    } else {
      setStatus("Upload failed — try another image.");
    }
  }

  async function generate() {
    if (!referenceAssetId || !controller.canSubmit || submitting || !activeModel) {
      return;
    }
    setSubmitting(true);
    setStatus("");
    try {
      const job = await createImageJob({
        mode: "character_image",
        model: activeModel.id,
        characterId,
        referenceAssetId,
        prompt: prompt.trim(),
        negativePrompt: advanced.negativePrompt,
        // angleSet/poses make the worker emit one image per angle/pose regardless of
        // count; count must satisfy the API's 1-8 guard, so send 1 (worker overrides).
        count: 1,
        seed: advanced.seedValue,
        width: 1024,
        height: 1024,
        loras: loraSelection.serializedLoras,
        advanced: advanced.buildAdvanced(controller.advancedExtras),
      });
      if (job?.id) {
        rememberLocalGenerationJob?.("image", job);
        setStatus("");
      } else {
        setStatus("Could not start the job — check the error banner.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">{mode.eyebrow}</p>
        <h2>{mode.title(activeModel, selectedCharacter)}</h2>
      </div>
      {mode.renderIntro(activeModel)}
      {availableModels.length > 1 ? (
        <label>
          Backbone
          <select
            onChange={(event) => setSelectedModelId(event.target.value)}
            value={selectedModelId}
          >
            {availableModels.map((backbone) => (
              <option key={backbone.id} value={backbone.id}>
                {backbone.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {approvedReferences.length ? (
        <div className="reference-thumb-row">
          {approvedReferences.map((reference) => (
            <button
              aria-label={`Use ${reference.asset?.displayName ?? reference.assetId} as the ${mode.referenceNoun}`}
              aria-pressed={reference.assetId === referenceAssetId}
              className={reference.assetId === referenceAssetId ? "reference-thumb active" : "reference-thumb"}
              key={reference.assetId}
              onClick={() => setReferenceAssetId(reference.assetId)}
              type="button"
            >
              {reference.asset ? <AssetMedia asset={reference.asset} controls={false} /> : <span>Missing asset</span>}
            </button>
          ))}
        </div>
      ) : (
        <p className="inline-warning">No approved reference yet — upload one below or approve a reference above.</p>
      )}
      {controller.controls}
      <div className="inline-create">
        <button onClick={() => fileInputRef.current?.click()} type="button">
          Upload reference
        </button>
        <input accept="image/*" hidden onChange={onUpload} ref={fileInputRef} type="file" />
        <button disabled={!referenceAssetId || !controller.isReady || submitting} onClick={generate} type="button">
          {submitting ? "Starting…" : controller.submitLabel}
        </button>
      </div>
      <label>
        Prompt
        <textarea onChange={(event) => setPrompt(event.target.value)} rows={2} value={prompt} />
      </label>
      <CharacterAdvancedOptions state={advanced} />
      {activeJobs.length ? (
        <div className="worker-progress-card-stack local-job-stack">
          {activeJobs.map((job) => (
            <WorkerProgressCard
              key={job.id}
              job={{
                ...job,
                payload: {
                  ...job.payload,
                  characterId: selectedCharacter?.id,
                  characterName: selectedCharacter?.name,
                },
              }}
              thumbnailsVariant="image-grid"
              thumbnailAssets={jobImageAssets(job, assets)}
              expectedThumbnailCount={controller.expectedThumbnailCount(job)}
              onThumbnailClick={(previewed) => onPreview?.(previewed, jobImageAssets(job, assets))}
              onCancel={onCancel}
              onRetry={onRetry}
              onDuplicate={onDuplicate}
              onOpenQueue={onOpenQueue}
            />
          ))}
        </div>
      ) : null}
      {!activeJobs.length && status ? <p className="inline-warning">{status}</p> : null}
      {characterImages.length ? (
        <div className="reference-thumb-row">
          {characterImages.map((asset) => (
            <button
              aria-label={`Preview ${asset.displayName ?? asset.id}`}
              className="reference-thumb"
              key={asset.id}
              onClick={() => onPreview?.(asset, characterImages)}
              type="button"
            >
              <AssetMedia asset={asset} controls={false} />
            </button>
          ))}
        </div>
      ) : null}
    </section>
  );
}

// Angle-set mode controller: KeypointCollection override (InstantID only) + the
// angleSet flag. One reference -> all of the backbone's view angles in one batch job.
function useAngleController({ activeModel, loraSelection }) {
  const angleCount = activeModel?.ui?.viewAngles?.length ?? 0;
  // Key Point Library override (sc-4435/sc-4450): only InstantID consumes the angle
  // kps collection (the landmark-ControlNet family — `identityStructure`); the
  // prompt-driven tiers iterate the built-in angle prompts and ignore it. `""` = run
  // the active default collection. `overrideCount` sizes the run to a chosen
  // collection's variable angle count; null falls back to the backbone's nominal count.
  const supportsKpsCollections = Boolean(activeModel?.ui?.identityStructure);
  const [keypointCollectionId, setKeypointCollectionId] = React.useState("");
  const [overrideCount, setOverrideCount] = React.useState(null);
  const onPickCollection = React.useCallback((id, collection) => {
    setKeypointCollectionId(id);
    setOverrideCount(id ? (collection?.orderedPresetIds?.length ?? null) : null);
  }, []);
  // Clear a stale angle-set override when switching to a backbone that can't use it.
  React.useEffect(() => {
    if (!supportsKpsCollections) {
      setKeypointCollectionId("");
      setOverrideCount(null);
    }
  }, [supportsKpsCollections]);
  const effectiveAngleCount = overrideCount ?? angleCount;
  return {
    controls: (
      <>
        <LoraPickerField selection={loraSelection} />
        {supportsKpsCollections ? (
          <KeypointCollectionField value={keypointCollectionId} onChange={onPickCollection} />
        ) : null}
      </>
    ),
    advancedExtras: {
      angleSet: true,
      ...(supportsKpsCollections && keypointCollectionId ? { keypointCollectionId } : {}),
    },
    isReady: true,
    canSubmit: true,
    submitLabel: `Generate angle set (${effectiveAngleCount} views)`,
    expectedThumbnailCount: () => effectiveAngleCount,
  };
}

// One-click multi-angle "turnaround": one reference -> all of the InstantID model's
// view angles in a single batch job (advanced.angleSet). Only rendered when an
// InstantID-style model (one that declares ui.viewAngles) is available.
const ANGLE_MODE = {
  eyebrow: "Angle set",
  title: (activeModel) => `${activeModel.name} turnaround`,
  // The set runs a softer landmark lock by default than single-image (sc-8354). Seed the
  // Identity-structure slider from the backbone's `identityStructure.angleSetDefault` so the
  // surfaced control matches what the worker runs (and the emitted value doesn't pin it back to
  // the single-image 0.80).
  identityStructureMode: "angleSet",
  defaultNegativePrompt: ANGLE_SET_NEGATIVE_PROMPT,
  referenceRole: "angle-set-reference",
  referenceNoun: "angle-set reference",
  jobPredicate: isAngleSetJob,
  // Seed with the character's appearance notes (face-identity engines preserve face
  // but not hair/wardrobe, so describe them here for a consistent turnaround) + tight
  // framing.
  seedPrompt: (character) => {
    const appearance = (character?.description ?? "").trim() || character?.name || "the character";
    return `${appearance}, head and shoulders, neutral expression, consistent outfit, plain grey background, face clearly visible, sharp focus, photorealistic`;
  },
  renderIntro: (activeModel) => (
    <>
      <p>
        Generate {activeModel?.ui?.viewAngles?.length ?? 0} consistent views of this character (front, three-quarters,
        profiles, up/down and the diagonals) from one reference, in a single job. A good starting set for curating a
        character LoRA.
      </p>
      {activeModel?.ui?.identityStructure ? (
        <p className="structured-hint">
          Lower <strong>Identity structure</strong> (under Advanced) to sharpen the off-axis views — it trades a little
          identity for a cleaner image. It can't fully rescue the most extreme up/down angles, and the set stays softer
          than a real photo; for a final pass, run the outputs through the image refine path.
        </p>
      ) : null}
    </>
  ),
  useController: useAngleController,
};

export function CharacterAngleSet({ angleModel, angleModels, ...props }) {
  return <CharacterGenerationPanel mode={ANGLE_MODE} model={angleModel} models={angleModels} {...props} />;
}

// Persistent per-character asset gallery: every image generated in association with the
// character (recipe.normalizedSettings.characterId) or referencing it
// (metadata.characterReferences). Reads the full project asset list, so character outputs
// persist beyond the transient "recent generations" window (sc-2076).
export function CharacterAssets({
  selectedCharacter,
  assets = [],
  projectId,
  importAsset,
  addCharacterReference,
  onPreview,
  deleteAsset,
  purgeAsset,
  updateAssetStatus,
}) {
  const [viewMode, setViewMode] = React.useState("active");
  const [importOpen, setImportOpen] = React.useState(false);
  // Same multi-select + batch toolbar as the Assets page (Batch… / Discard / Move).
  const batch = useAssetBatch();
  const characterId = selectedCharacter?.id;
  // sc-6042: attach project-selected or freshly-uploaded assets to this character.
  // addCharacterReference is the canonical asset↔character link (it writes both the
  // character's references[] and the asset's metadata.characterReferences), so the
  // imports immediately surface in this gallery and the reference picker. Added
  // unapproved so they're library members, not auto-promoted identity references.
  async function importAssetIds(assetIds) {
    if (!characterId || !addCharacterReference) {
      return;
    }
    for (const assetId of assetIds) {
      await addCharacterReference(characterId, { assetId, approved: false, role: "import" });
    }
  }
  if (!selectedCharacter) {
    return null;
  }
  const referenceAssetIds = new Set(
    (selectedCharacter.references ?? [])
      .map((reference) => reference?.assetId ?? reference?.id)
      .filter(Boolean),
  );
  const scopedAssets = (assets ?? [])
    .filter(
      (asset) =>
        // Images, extracted frames, and generated videos all belong in the
        // character's asset library (sc-2296) — the Assets tab is the home for
        // every piece of media made for or referencing this character.
        (asset.type === "image" || asset.type === "frame" || asset.type === "video") &&
        (referenceAssetIds.has(asset.id) ||
          asset.recipe?.normalizedSettings?.characterId === characterId ||
          (asset.metadata?.characterReferences ?? []).some((ref) => ref.characterId === characterId)),
    )
    .sort((left, right) => new Date(right.createdAt ?? 0) - new Date(left.createdAt ?? 0));
  // Discarded (status.trashed) images drop out of the main grid so they no
  // longer hold their slots, but stay reachable through the Trashcan view for
  // restore or permanent purge.
  const activeAssets = scopedAssets.filter((asset) => !asset.status?.trashed);
  const trashedAssets = scopedAssets.filter((asset) => asset.status?.trashed);
  const showingTrash = viewMode === "trashed";
  const visibleAssets = showingTrash ? trashedAssets : activeAssets;
  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">Character assets</p>
        <h2>
          Generated for {selectedCharacter.name}
          {activeAssets.length ? ` (${activeAssets.length})` : ""}
        </h2>
      </div>
      <div className="trash-controls">
        <div className="segmented-control" role="group" aria-label="Character asset collection">
          <button className={showingTrash ? "" : "active"} onClick={() => setViewMode("active")} type="button">
            Media ({activeAssets.length})
          </button>
          <button className={showingTrash ? "active" : ""} onClick={() => setViewMode("trashed")} type="button">
            Trashcan ({trashedAssets.length})
          </button>
        </div>
        {/* sc-6042: bring Project assets or local files into this character's
            library so they become selectable in the Reference picker. */}
        <button
          className="secondary-action character-import-button"
          disabled={!characterId}
          onClick={() => setImportOpen(true)}
          type="button"
        >
          Import
        </button>
        {showingTrash ? (
          <button
            className="danger-action empty-trash-button"
            disabled={!trashedAssets.length}
            onClick={() => emptyTrash(trashedAssets, purgeAsset)}
            type="button"
          >
            Empty Trash ({trashedAssets.length})
          </button>
        ) : null}
      </div>
      {importOpen ? (
        <CharacterImportDialog
          assets={assets}
          character={selectedCharacter}
          characterId={characterId}
          characterName={selectedCharacter.name}
          importAsset={importAsset}
          onClose={() => setImportOpen(false)}
          onImport={importAssetIds}
          projectId={projectId}
        />
      ) : null}
      {/* Hide Discard in the Trashcan view (already discarded); offer the Main Library as a
          Move target so character media can be promoted back out (sc-8341). */}
      <AssetSelectionBar batch={batch} showDiscard={!showingTrash} allowLibraryTarget />
      {visibleAssets.length ? (
        <div className="asset-grid character-asset-grid">
          {visibleAssets.map((asset) => (
            <div
              className={batch.selectedAssetIds.has(asset.id) ? "asset-tile-wrap selected" : "asset-tile-wrap"}
              key={asset.id}
            >
              <label className="asset-tile-check">
                <input
                  aria-label={`Select ${asset.displayName ?? asset.id}`}
                  checked={batch.selectedAssetIds.has(asset.id)}
                  onChange={() => batch.toggleSelect(asset.id)}
                  type="checkbox"
                />
              </label>
              <article className="asset-tile character-asset-card">
                <button
                  aria-label={`Preview ${asset.displayName ?? asset.id}`}
                  className="character-asset-media"
                  onClick={() => onPreview?.(asset, visibleAssets)}
                  type="button"
                >
                  <AssetMedia asset={asset} controls={false} />
                </button>
                <strong>{asset.displayName ?? asset.id}</strong>
                <div className="character-asset-card-actions">
                  {showingTrash ? (
                    <>
                      <button onClick={() => updateAssetStatus?.(asset, { trashed: false })} type="button">
                        Restore
                      </button>
                      <button onClick={() => purgeAsset?.(asset)} type="button">
                        Purge
                      </button>
                    </>
                  ) : (
                    <button onClick={() => deleteAsset?.(asset)} type="button">
                      Discard
                    </button>
                  )}
                </div>
              </article>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">
          {showingTrash
            ? "Trashcan is empty — discarded media for this character will appear here."
            : "No media yet. Angle sets, pose generations, character tests, and any character image or video render collect here automatically."}
        </p>
      )}
      <AssetBatchModal batch={batch} />
    </section>
  );
}

// Training datasets associated with this character (sc-2022). The character is
// the hub for its references, looks, LoRAs, and — here — its LoRA training data.
// This is UI grouping over the shared TrainingDataset backend: "Open" deep-links
// into the Dataset editor, and "Create from images" seeds a new associated
// dataset from the character's generated images. No dataset logic is duplicated.
export function CharacterDatasets({
  selectedCharacter,
  projectId,
  datasets = [],
  imageCount = 0,
  onOpenDataset,
  onCreateDataset,
  creating = false,
}) {
  if (!selectedCharacter) {
    return null;
  }
  const coverAsset = (dataset) =>
    dataset?.coverPath ? { projectId, type: "image", file: { path: dataset.coverPath } } : null;
  return (
    <section className="character-section character-datasets">
      <div className="section-heading">
        <p className="eyebrow">Training datasets</p>
        <h2>
          For {selectedCharacter.name}
          {datasets.length ? ` (${datasets.length})` : ""}
        </h2>
      </div>
      {datasets.length ? (
        <ul className="character-dataset-list">
          {datasets.map((dataset) => {
            const cover = coverAsset(dataset);
            return (
              <li className="character-dataset-row" key={dataset.id}>
                <span className="character-dataset-cover">
                  {cover ? <AssetMedia asset={cover} controls={false} /> : <span className="thumb-placeholder" />}
                </span>
                <span className="character-dataset-meta">
                  <strong>{dataset.name}</strong>
                  <span className="muted">
                    {dataset.itemCount ?? 0} image{(dataset.itemCount ?? 0) === 1 ? "" : "s"} · {dataset.status ?? "draft"}
                  </span>
                </span>
                <button className="secondary-action" onClick={() => onOpenDataset?.(dataset.id)} type="button">
                  Open
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted">No datasets yet for this character. Create one from its images to start a LoRA.</p>
      )}
      <div className="detail-actions">
        <button
          className="primary-action"
          disabled={creating || imageCount === 0}
          onClick={() => onCreateDataset?.()}
          type="button"
        >
          {creating ? "Creating…" : `Create dataset from ${imageCount} image${imageCount === 1 ? "" : "s"}`}
        </button>
      </div>
    </section>
  );
}

// Pose library: pick one or more poses from the bundled OpenPose gallery and generate
// the character in each, in a single batch job (advanced.poses) sharing one seed for
// wardrobe/hair consistency. An OpenPose ControlNet drives the pose; a face-restoration
// pass re-imposes identity at the small full-body face size. Shares
// CharacterGenerationPanel with the angle set (sc-4195).
function usePoseController({ activeModel, loraSelection }) {
  // User-created poses join the built-in library in both the picker and the
  // id→keypoints resolver used to build the job, so saved poses can generate.
  const loadUserPoses = useUserPoseLoader();
  const { byId } = usePoseLibrary({ loadUserPoses });
  const [selectedPoseIds, setSelectedPoseIds] = React.useState([]);
  const [faceRestore, setFaceRestore] = React.useState(false);
  // Strict ControlNet pose-lock strength (sc-2257). Only the strict tier
  // (ui.poseControlScale) honours advanced.controlScale; default 0.9 matches the
  // reference pipeline, 0.65–1.0 is the model-card range, >~1.2 degrades quality.
  const [controlScale, setControlScale] = React.useState(0.9);
  const supportsControlScale = Boolean(activeModel?.ui?.poseControlScale);

  function togglePose(id) {
    setSelectedPoseIds((ids) => (ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id]));
  }

  const poses = selectedPoseIds.map((id) => byId[id]).filter(Boolean).map((pose) => ({
    id: pose.id,
    keypoints: pose.keypoints,
    // Forward DWPose hand/face keypoints when a pose carries them (sc-2257) — the
    // strict Z-Image tier renders them for a firmer, more in-distribution lock.
    ...(pose.hands ? { hands: pose.hands } : {}),
    ...(pose.face ? { face: pose.face } : {}),
  }));

  return {
    controls: (
      <>
        <PoseLibraryPicker
          loadUserPoses={loadUserPoses}
          onClear={() => setSelectedPoseIds([])}
          onToggle={togglePose}
          selectedIds={selectedPoseIds}
        />
        <label className="checkline">
          <input checked={faceRestore} onChange={(event) => setFaceRestore(event.target.checked)} type="checkbox" />
          Restore face (sharper identity; off keeps the raw render — fewer blend artifacts)
        </label>
        {supportsControlScale ? (
          <div className="control-scale-field">
            <div className="lora-weight-row">
              <span>Pose lock strength</span>
              <input
                aria-label="Pose lock strength"
                max="1.5"
                min="0.5"
                onChange={(event) => setControlScale(Number(event.target.value))}
                step="0.05"
                type="range"
                value={controlScale}
              />
              <span className="lora-weight-value">{controlScale.toFixed(2)}</span>
            </div>
            <p className="field-hint">
              How hard the ControlNet locks the pose. 0.9 (the reference default) is a clean lock; lower
              (toward 0.65) loosens it for more natural variation, higher (&gt;1.2) over-constrains and can
              degrade quality.
            </p>
          </div>
        ) : null}
        <LoraPickerField selection={loraSelection} />
      </>
    ),
    advancedExtras: {
      poses,
      faceRestore,
      ...(supportsControlScale ? { controlScale } : {}),
    },
    isReady: selectedPoseIds.length > 0,
    canSubmit: poses.length > 0,
    submitLabel: `Generate ${selectedPoseIds.length || ""} pose${selectedPoseIds.length === 1 ? "" : "s"}`.replace("  ", " "),
    expectedThumbnailCount: (job) =>
      Array.isArray(job.payload?.advanced?.poses) ? job.payload.advanced.poses.length : selectedPoseIds.length,
  };
}

// Only rendered for a model that supports the pose library (ui.poseLibrary).
const POSE_MODE = {
  eyebrow: "Pose library",
  title: (activeModel, character) => `Generate ${character.name} in a pose`,
  defaultNegativePrompt: POSE_LIBRARY_NEGATIVE_PROMPT,
  referenceRole: "pose-set-reference",
  referenceNoun: "pose reference",
  jobPredicate: isPoseLibraryJob,
  // Pose-neutral prompt: the skeleton sets the stance, so describe only appearance +
  // outfit/shoes (face-identity engines preserve the face but NOT hair/wardrobe).
  seedPrompt: (character) => {
    const appearance = (character?.description ?? "").trim() || character?.name || "the character";
    return `${appearance}, consistent outfit and shoes, plain grey background, soft even lighting, sharp focus, photorealistic`;
  },
  renderIntro: () => (
    <p>
      Pick one or more poses and generate this character in each, in a single job. The strict tier (InstantID) uses an
      OpenPose ControlNet to enforce the pose and a face-restoration pass to anchor identity; the best-effort tiers
      (Qwen-Lightning, FLUX.2-klein) approximate the pose via multi-image reference.
    </p>
  ),
  useController: usePoseController,
};

export function CharacterPoseLibrary({ poseModel, poseModels, ...props }) {
  return <CharacterGenerationPanel mode={POSE_MODE} model={poseModel} models={poseModels} {...props} />;
}
