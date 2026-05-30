import React from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard, emptyTrash } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { terminalStatuses } from "../jobTypes.js";
import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";
import { LoraPickerField, useLoraSelection } from "../components/LoraPickerField.jsx";
import { usePoseLibrary, useUserPoseLoader } from "../poseLibrary.js";
import { extractFamilies } from "../presetUtils.js";

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
  imageAssets,
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
  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">References</p>
        <h2>Approved set</h2>
      </div>
      <form className="inline-create asset-reference-create" onSubmit={submitReference}>
        <AssetPickerField
          assets={imageAssets}
          buttonLabel="Add image or frame"
          emptyLabel="No references selected"
          label="Reference assets"
          multiple
          onChange={setReferenceAssetIds}
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
            <button className="reference-media" onClick={() => reference.asset && onPreview(reference.asset)} type="button">
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
  createCharacterLook,
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
  createCharacterTestJob,
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
                onPreview={onPreview}
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

// One-click multi-angle "turnaround": one reference -> all of the InstantID model's
// view angles in a single batch job (advanced.angleSet). Only rendered when an
// InstantID-style model (one that declares ui.viewAngles) is available.
export function CharacterAngleSet({
  selectedCharacter,
  angleModel,
  angleModels,
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
  // sc-2003: multi-backbone picker. angleModels is the full list of viewAngles-
  // capable backbones (manifest order: InstantID first, then prompt-driven
  // tiers). angleModel is the resolved default (kept for back-compat with the
  // pre-picker callers); the local state below tracks the user's pick.
  const availableModels = Array.isArray(angleModels) && angleModels.length > 0
    ? angleModels
    : (angleModel ? [angleModel] : []);
  const [selectedAngleModelId, setSelectedAngleModelId] = React.useState(
    angleModel?.id ?? availableModels[0]?.id ?? "",
  );
  const activeAngleModel = availableModels.find((item) => item.id === selectedAngleModelId)
    ?? availableModels[0]
    ?? null;
  const angleCount = activeAngleModel?.ui?.viewAngles?.length ?? 0;
  // Family-filtered LoRA picker (sc-2223): apply an existing LoRA of this character
  // to its turnaround (the dataset-bootstrapping loop). Filtered to the backbone's family.
  const loraSelection = useLoraSelection(loras, activeAngleModel);
  const [referenceAssetId, setReferenceAssetId] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [status, setStatus] = React.useState("");
  const fileInputRef = React.useRef(null);
  const characterId = selectedCharacter?.id;

  React.useEffect(() => {
    setReferenceAssetId(approvedReferences[0]?.assetId ?? "");
  }, [characterId, approvedReferences]);
  React.useEffect(() => {
    // Seed with the character's appearance notes (face-identity engines preserve
    // face but not hair/wardrobe, so describe them here for a consistent
    // turnaround) + tight framing.
    const appearance = (selectedCharacter?.description ?? "").trim() || selectedCharacter?.name || "the character";
    setPrompt(
      `${appearance}, head and shoulders, neutral expression, consistent outfit, plain grey background, face clearly visible, sharp focus, photorealistic`,
    );
    setStatus("");
  }, [characterId, selectedCharacter?.name, selectedCharacter?.description]);
  // Reset the picker selection when the available models change (e.g. a new
  // manifest load) so a stale id doesn't lock the panel.
  React.useEffect(() => {
    if (!availableModels.find((item) => item.id === selectedAngleModelId)) {
      setSelectedAngleModelId(availableModels[0]?.id ?? "");
    }
  }, [availableModels, selectedAngleModelId]);

  if (!activeAngleModel || !selectedCharacter) {
    return null;
  }

  // Active jobs for this character's angle-set form, derived from the
  // App-level imageLocalJobs so they persist across navigation and stack
  // when multiple runs are in flight (sc-2092 follow-up). The local jobId
  // state is gone — there's nothing to remember per mount.
  const activeJobs = characterFormJobs(imageLocalJobs, characterId, isAngleSetJob);
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
      await addCharacterReference(characterId, { assetId: asset.id, approved: true, role: "angle-set-reference" });
      setStatus("");
    } else {
      setStatus("Upload failed — try another image.");
    }
  }

  async function generate() {
    if (!referenceAssetId || submitting || !activeAngleModel) {
      return;
    }
    setSubmitting(true);
    setStatus("");
    try {
      const job = await createImageJob({
        mode: "character_image",
        model: activeAngleModel.id,
        characterId,
        referenceAssetId,
        prompt: prompt.trim(),
        negativePrompt:
          "hat, hood, hoodie, scarf, holding object, paper, hands near face, covering face, occluded face, cropped, " +
          "multiple people, plastic skin, airbrushed, cgi, 3d render, cartoon, anime, waxy, deformed, blurry",
        // angleSet makes the worker emit one image per pack angle regardless of count;
        // count must satisfy the API's 1-8 guard, so send 1 (the worker overrides it).
        count: 1,
        width: 1024,
        height: 1024,
        loras: loraSelection.serializedLoras,
        advanced: { angleSet: true, ipAdapterScale: 0.8 },
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
        <p className="eyebrow">Angle set</p>
        <h2>{activeAngleModel.name} turnaround</h2>
      </div>
      <p>
        Generate {angleCount} consistent views of this character (front, three-quarters, profiles, up/down and the
        diagonals) from one reference, in a single job. A good starting set for curating a character LoRA.
      </p>
      {availableModels.length > 1 ? (
        <label>
          Backbone
          <select
            onChange={(event) => setSelectedAngleModelId(event.target.value)}
            value={selectedAngleModelId}
          >
            {availableModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {approvedReferences.length ? (
        <div className="reference-thumb-row">
          {approvedReferences.map((reference) => (
            <button
              aria-label={`Use ${reference.asset?.displayName ?? reference.assetId} as the angle-set reference`}
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
      <LoraPickerField selection={loraSelection} />
      <div className="inline-create">
        <button onClick={() => fileInputRef.current?.click()} type="button">
          Upload reference
        </button>
        <input accept="image/*" hidden onChange={onUpload} ref={fileInputRef} type="file" />
        <button disabled={!referenceAssetId || submitting} onClick={generate} type="button">
          {submitting ? "Starting…" : `Generate angle set (${angleCount} views)`}
        </button>
      </div>
      <label>
        Prompt
        <textarea onChange={(event) => setPrompt(event.target.value)} rows={2} value={prompt} />
      </label>
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
              expectedThumbnailCount={angleCount}
              onThumbnailClick={onPreview}
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
              onClick={() => onPreview?.(asset)}
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

// Persistent per-character asset gallery: every image generated in association with the
// character (recipe.normalizedSettings.characterId) or referencing it
// (metadata.characterReferences). Reads the full project asset list, so character outputs
// persist beyond the transient "recent generations" window (sc-2076).
export function CharacterAssets({
  selectedCharacter,
  assets = [],
  onPreview,
  deleteAsset,
  purgeAsset,
  updateAssetStatus,
}) {
  const [viewMode, setViewMode] = React.useState("active");
  const characterId = selectedCharacter?.id;
  if (!selectedCharacter) {
    return null;
  }
  const scopedAssets = (assets ?? [])
    .filter(
      (asset) =>
        // Images, extracted frames, and generated videos all belong in the
        // character's asset library (sc-2296) — the Assets tab is the home for
        // every piece of media made for or referencing this character.
        (asset.type === "image" || asset.type === "frame" || asset.type === "video") &&
        (asset.recipe?.normalizedSettings?.characterId === characterId ||
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
      {visibleAssets.length ? (
        <div className="reference-thumb-row">
          {visibleAssets.map((asset) => (
            <div className="character-asset-thumb" key={asset.id}>
              <button
                aria-label={`Preview ${asset.displayName ?? asset.id}`}
                className="reference-thumb"
                onClick={() => onPreview?.(asset)}
                type="button"
              >
                <AssetMedia asset={asset} controls={false} />
              </button>
              <div className="character-asset-thumb-actions">
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
// pass re-imposes identity at the small full-body face size. Only rendered for a model
// that supports the pose library (ui.poseLibrary).
export function CharacterPoseLibrary({
  selectedCharacter,
  poseModel,
  poseModels,
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
  // sc-2003: multi-backbone picker. poseModels is the full list of poseLibrary-
  // capable backbones (manifest order: InstantID strict tier first, then the
  // multi-image best-effort tiers).
  const availableModels = Array.isArray(poseModels) && poseModels.length > 0
    ? poseModels
    : (poseModel ? [poseModel] : []);
  const [selectedPoseModelId, setSelectedPoseModelId] = React.useState(
    poseModel?.id ?? availableModels[0]?.id ?? "",
  );
  const activePoseModel = availableModels.find((item) => item.id === selectedPoseModelId)
    ?? availableModels[0]
    ?? null;
  // User-created poses join the built-in library in both the picker and the
  // id→keypoints resolver used to build the job, so saved poses can generate.
  const loadUserPoses = useUserPoseLoader();
  const { byId } = usePoseLibrary({ loadUserPoses });
  // Family-filtered LoRA picker (sc-2223), filtered to the active pose backbone's family.
  const loraSelection = useLoraSelection(loras, activePoseModel);
  const [selectedPoseIds, setSelectedPoseIds] = React.useState([]);
  const [faceRestore, setFaceRestore] = React.useState(false);
  // Strict ControlNet pose-lock strength (sc-2257). Only the strict tier
  // (ui.poseControlScale) honours advanced.controlScale; default 0.9 matches the
  // reference pipeline, 0.65–1.0 is the model-card range, >~1.2 degrades quality.
  const [controlScale, setControlScale] = React.useState(0.9);
  const supportsControlScale = Boolean(activePoseModel?.ui?.poseControlScale);
  const [referenceAssetId, setReferenceAssetId] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [status, setStatus] = React.useState("");
  const fileInputRef = React.useRef(null);
  const characterId = selectedCharacter?.id;

  React.useEffect(() => {
    setReferenceAssetId(approvedReferences[0]?.assetId ?? "");
  }, [characterId, approvedReferences]);
  React.useEffect(() => {
    // Pose-neutral prompt: the skeleton sets the stance, so describe only appearance +
    // outfit/shoes (face-identity engines preserve the face but NOT hair/wardrobe).
    const appearance = (selectedCharacter?.description ?? "").trim() || selectedCharacter?.name || "the character";
    setPrompt(`${appearance}, consistent outfit and shoes, plain grey background, soft even lighting, sharp focus, photorealistic`);
    setStatus("");
  }, [characterId, selectedCharacter?.name, selectedCharacter?.description]);
  React.useEffect(() => {
    if (!availableModels.find((item) => item.id === selectedPoseModelId)) {
      setSelectedPoseModelId(availableModels[0]?.id ?? "");
    }
  }, [availableModels, selectedPoseModelId]);

  if (!activePoseModel || !selectedCharacter) {
    return null;
  }

  const activeJobs = characterFormJobs(imageLocalJobs, characterId, isPoseLibraryJob);
  const characterImages = (latestAssets ?? []).filter(
    (asset) =>
      asset.recipe?.normalizedSettings?.characterId === characterId ||
      (asset.metadata?.characterReferences ?? []).some((ref) => ref.characterId === characterId),
  );

  function togglePose(id) {
    setSelectedPoseIds((ids) => (ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id]));
  }

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
      await addCharacterReference(characterId, { assetId: asset.id, approved: true, role: "pose-set-reference" });
      setStatus("");
    } else {
      setStatus("Upload failed — try another image.");
    }
  }

  async function generate() {
    const poses = selectedPoseIds.map((id) => byId[id]).filter(Boolean).map((pose) => ({
      id: pose.id,
      keypoints: pose.keypoints,
      // Forward DWPose hand/face keypoints when a pose carries them (sc-2257) — the
      // strict Z-Image tier renders them for a firmer, more in-distribution lock.
      ...(pose.hands ? { hands: pose.hands } : {}),
      ...(pose.face ? { face: pose.face } : {}),
    }));
    if (!referenceAssetId || !poses.length || submitting || !activePoseModel) {
      return;
    }
    setSubmitting(true);
    setStatus("");
    try {
      const job = await createImageJob({
        mode: "character_image",
        model: activePoseModel.id,
        characterId,
        referenceAssetId,
        prompt: prompt.trim(),
        negativePrompt:
          "cropped, out of frame, multiple people, extra limbs, deformed hands, extra fingers, " +
          "plastic skin, airbrushed, cgi, 3d render, cartoon, anime, waxy, blurry",
        // The worker emits one image per pose in `advanced.poses` regardless of count;
        // count must satisfy the API's 1-8 guard, so send 1.
        count: 1,
        width: 1024,
        height: 1024,
        loras: loraSelection.serializedLoras,
        advanced: {
          poses,
          ipAdapterScale: 0.8,
          faceRestore,
          ...(supportsControlScale ? { controlScale } : {}),
        },
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
        <p className="eyebrow">Pose library</p>
        <h2>Generate {selectedCharacter.name} in a pose</h2>
      </div>
      <p>
        Pick one or more poses and generate this character in each, in a single job. The strict tier (InstantID) uses an
        OpenPose ControlNet to enforce the pose and a face-restoration pass to anchor identity; the best-effort tiers
        (Qwen-Lightning, FLUX.2-klein) approximate the pose via multi-image reference.
      </p>
      {availableModels.length > 1 ? (
        <label>
          Backbone
          <select
            onChange={(event) => setSelectedPoseModelId(event.target.value)}
            value={selectedPoseModelId}
          >
            {availableModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {approvedReferences.length ? (
        <div className="reference-thumb-row">
          {approvedReferences.map((reference) => (
            <button
              aria-label={`Use ${reference.asset?.displayName ?? reference.assetId} as the pose reference`}
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
      <div className="inline-create">
        <button onClick={() => fileInputRef.current?.click()} type="button">
          Upload reference
        </button>
        <input accept="image/*" hidden onChange={onUpload} ref={fileInputRef} type="file" />
        <button disabled={!referenceAssetId || !selectedPoseIds.length || submitting} onClick={generate} type="button">
          {submitting
            ? "Starting…"
            : `Generate ${selectedPoseIds.length || ""} pose${selectedPoseIds.length === 1 ? "" : "s"}`.replace("  ", " ")}
        </button>
      </div>
      <label>
        Prompt
        <textarea onChange={(event) => setPrompt(event.target.value)} rows={2} value={prompt} />
      </label>
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
              expectedThumbnailCount={
                Array.isArray(job.payload?.advanced?.poses) ? job.payload.advanced.poses.length : selectedPoseIds.length
              }
              onThumbnailClick={onPreview}
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
              onClick={() => onPreview?.(asset)}
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
