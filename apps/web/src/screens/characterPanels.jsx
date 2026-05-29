import React from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";
import { usePoseLibrary } from "../poseLibrary.js";
import { extractFamilies } from "../presetUtils.js";

// Resolve a character generation job's produced images from the surrounding
// latestAssets list. The component is propped from above (no AppContext here)
// so this stays a small local helper instead of pulling in a shared util.
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
  // Scope the outputs grid to THIS character (its generated images + approved
  // references) instead of dumping every recent project image, and keep it
  // collapsed by default so it never turns the studio into an endless scroll.
  const characterAssets = (latestAssets ?? []).filter(
    (asset) =>
      asset.recipe?.normalizedSettings?.characterId === selectedCharacter.id ||
      (asset.metadata?.characterReferences ?? []).some((ref) => ref.characterId === selectedCharacter.id),
  );
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
          {showOutputs ? "Hide" : "Show"} this character's images ({characterAssets.length})
        </button>
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
              <button
                onClick={() => addCharacterReference(selectedCharacter.id, { assetId: asset.id, approved: true, role: "test-output" })}
                type="button"
              >
                Approve as Reference
              </button>
            </div>
          ))}
          {characterAssets.length ? null : (
            <div className="empty-panel compact-panel">
              No images for this character yet — generate an angle set or a test above.
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
  createImageJob,
  importAsset,
  addCharacterReference,
  latestAssets = [],
  imageLocalJobs = [],
  rememberLocalGenerationJob,
  onPreview,
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
  const [referenceAssetId, setReferenceAssetId] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [status, setStatus] = React.useState("");
  const [jobId, setJobId] = React.useState(null);
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

  // The just-launched batch job (live progress while it runs) + this character's
  // images (they stream in as the worker writes each angle).
  const activeJob = imageLocalJobs.find((job) => job.id === jobId);
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
        advanced: { angleSet: true, ipAdapterScale: 0.8 },
      });
      if (job?.id) {
        rememberLocalGenerationJob?.("image", job);
        setJobId(job.id);
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
      {activeJob ? (
        <WorkerProgressCard
          job={{
            ...activeJob,
            payload: {
              ...activeJob.payload,
              characterId: selectedCharacter?.id,
              characterName: selectedCharacter?.name,
            },
          }}
          thumbnailsVariant="image-grid"
          thumbnailAssets={jobImageAssets(activeJob, latestAssets)}
          expectedThumbnailCount={angleCount}
          onThumbnailClick={onPreview}
        />
      ) : null}
      {!activeJob && status ? <p className="inline-warning">{status}</p> : null}
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
export function CharacterAssets({ selectedCharacter, assets = [], onPreview }) {
  const characterId = selectedCharacter?.id;
  if (!selectedCharacter) {
    return null;
  }
  const characterAssets = (assets ?? [])
    .filter(
      (asset) =>
        (asset.type === "image" || asset.type === "frame") &&
        (asset.recipe?.normalizedSettings?.characterId === characterId ||
          (asset.metadata?.characterReferences ?? []).some((ref) => ref.characterId === characterId)),
    )
    .sort((left, right) => new Date(right.createdAt ?? 0) - new Date(left.createdAt ?? 0));
  return (
    <section className="character-section">
      <div className="section-heading">
        <p className="eyebrow">Character assets</p>
        <h2>
          Generated for {selectedCharacter.name}
          {characterAssets.length ? ` (${characterAssets.length})` : ""}
        </h2>
      </div>
      {characterAssets.length ? (
        <div className="reference-thumb-row">
          {characterAssets.map((asset) => (
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
      ) : (
        <p className="muted">
          No images yet. Angle sets, pose generations, character tests, and any character-image render collect here automatically.
        </p>
      )}
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
  createImageJob,
  importAsset,
  addCharacterReference,
  latestAssets = [],
  imageLocalJobs = [],
  rememberLocalGenerationJob,
  onPreview,
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
  const { byId } = usePoseLibrary();
  const [selectedPoseIds, setSelectedPoseIds] = React.useState([]);
  const [faceRestore, setFaceRestore] = React.useState(true);
  const [referenceAssetId, setReferenceAssetId] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [status, setStatus] = React.useState("");
  const [jobId, setJobId] = React.useState(null);
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

  const activeJob = imageLocalJobs.find((job) => job.id === jobId);
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
    const poses = selectedPoseIds.map((id) => byId[id]).filter(Boolean).map((pose) => ({ id: pose.id, keypoints: pose.keypoints }));
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
        advanced: { poses, ipAdapterScale: 0.8, faceRestore },
      });
      if (job?.id) {
        rememberLocalGenerationJob?.("image", job);
        setJobId(job.id);
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
        onClear={() => setSelectedPoseIds([])}
        onToggle={togglePose}
        selectedIds={selectedPoseIds}
      />
      <label className="checkline">
        <input checked={faceRestore} onChange={(event) => setFaceRestore(event.target.checked)} type="checkbox" />
        Restore face (sharper identity; off keeps the raw render — fewer blend artifacts)
      </label>
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
      {activeJob ? (
        <WorkerProgressCard
          job={{
            ...activeJob,
            payload: {
              ...activeJob.payload,
              characterId: selectedCharacter?.id,
              characterName: selectedCharacter?.name,
            },
          }}
          thumbnailsVariant="image-grid"
          thumbnailAssets={jobImageAssets(activeJob, latestAssets)}
          expectedThumbnailCount={selectedPoseIds.length}
          onThumbnailClick={onPreview}
        />
      ) : null}
      {!activeJob && status ? <p className="inline-warning">{status}</p> : null}
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
