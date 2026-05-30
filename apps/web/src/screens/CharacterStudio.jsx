import React, { useEffect, useMemo, useState } from "react";
import {
  CharacterAngleSet,
  CharacterAssets,
  CharacterDatasets,
  CharacterLoras,
  CharacterLooks,
  CharacterPoseLibrary,
  CharacterReferences,
  CharacterTest,
  editableLora,
} from "./characterPanels.jsx";
import { CompactSelector } from "../components/CompactSelector.jsx";
import { assetMatchesCharacter } from "../components/DatasetAddDialog.jsx";
import { extractFamilies } from "../presetUtils.js";
import { useAppContext } from "../context/AppContext.js";

const characterTypes = [
  ["person", "Person"],
  ["creature", "Creature"],
  ["object", "Object"],
];

function typeLabel(value) {
  return characterTypes.find(([id]) => id === value)?.[1] ?? "Person";
}

export function CharacterStudio() {
  const {
    activeProject,
    assets,
    characters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    addCharacterReference,
    updateCharacterReference,
    removeCharacterReference,
    createCharacterLook,
    updateCharacterLook,
    deleteCharacterLook,
    attachCharacterLora,
    updateCharacterLora,
    detachCharacterLora,
    createCharacterTestJob,
    createImageJob,
    importAsset,
    imageLocalJobs,
    rememberLocalGenerationJob,
    deleteAsset,
    purgeAsset,
    imageModels,
    latestImageAssets,
    loras,
    setPreviewAsset,
    sendCharacterToImage,
    sendCharacterToVideo,
    updateAssetStatus,
    trainingDatasets = [],
    trainingDatasetsProjectId,
    createTrainingDataset,
    openDatasetInLibrary,
  } = useAppContext();
  const latestAssets = latestImageAssets;
  const onPreview = setPreviewAsset;
  const onSendImage = sendCharacterToImage;
  const onSendVideo = sendCharacterToVideo;
  const [selectedCharacterId, setSelectedCharacterId] = useState(characters[0]?.id ?? "");
  const [draft, setDraft] = useState({ name: "", type: "person", description: "" });
  const [referenceAssetIds, setReferenceAssetIds] = useState([]);
  const [lookDraft, setLookDraft] = useState({ name: "", description: "" });
  const [selectedReferenceIds, setSelectedReferenceIds] = useState([]);
  const [referenceMessage, setReferenceMessage] = useState("");
  const [loraId, setLoraId] = useState("");
  const [loraEdits, setLoraEdits] = useState({});
  const [testPrompt, setTestPrompt] = useState("A clean character reference portrait, consistent identity, studio lighting");
  const [testModel, setTestModel] = useState(imageModels[0]?.id ?? "z_image_turbo");
  const [testLookId, setTestLookId] = useState("");
  const [testCount, setTestCount] = useState(4);
  const [testResolution, setTestResolution] = useState("1024x1024");
  const [creatingDataset, setCreatingDataset] = useState(false);

  const imageAssets = useMemo(
    () => assets.filter((asset) => ["image", "frame", "upload"].includes(asset.type)),
    [assets],
  );
  const selectedCharacter = characters.find((item) => item.id === selectedCharacterId) ?? characters[0] ?? null;
  const approvedReferences = selectedCharacter?.approvedReferences ?? [];
  // sc-2022: datasets the dataset backend reports as owned by this character,
  // and the character's own images (same match the Dataset editor's Character
  // tab uses) that a new dataset would be seeded from.
  const datasetsForProject = trainingDatasetsProjectId === activeProject?.id ? trainingDatasets : [];
  const characterDatasets = useMemo(
    () => datasetsForProject.filter((dataset) => dataset.characterId === selectedCharacter?.id),
    [datasetsForProject, selectedCharacter?.id],
  );
  const characterImageAssetIds = useMemo(
    () =>
      selectedCharacter
        ? imageAssets.filter((asset) => assetMatchesCharacter(asset, selectedCharacter.id)).map((asset) => asset.id)
        : [],
    [imageAssets, selectedCharacter?.id],
  );
  // Thumbnail for the compact selector (sc-2025): a character's first approved
  // reference image, falling back to any reference. Null → placeholder tile.
  const characterThumbAsset = (character) => {
    const references = character?.references ?? [];
    const reference = references.find((item) => item.approved) ?? references[0];
    if (!reference?.assetId) {
      return null;
    }
    return imageAssets.find((asset) => asset.id === reference.assetId) ?? null;
  };
  // Multi-backbone model picker for the angle set + pose library (sc-2003).
  // Each backbone declares ui.viewAngles / ui.poseLibrary in the manifest;
  // the worker dispatch handles per-backbone angle / pose loops (InstantID
  // landmark pack + OpenPose ControlNet for the strict tier; prompt-driven
  // augments + multi-image references for the prompt-driven tiers).
  //
  // Spike-validated angle backbones (sc-2003 follow-up, mean ArcFace cosine):
  //   instantid_realvisxl                 — landmark deterministic, highest
  //   qwen_image_edit_2511_lightning      — 0.62, prompt-driven fast tier
  //   flux2_klein_9b                      — 0.52, holds portrait at profiles
  //   sensenova_u1_8b                     — 0.29, wardrobe-continuity tier
  //
  // Spike-validated pose backbones:
  //   instantid_realvisxl                 — OpenPose ControlNet strict
  //   qwen_image_edit_2511_lightning      — multi-image best-effort
  //   flux2_klein_9b                      — multi-image best-effort
  //
  // SenseNova-U1 is gated out of the pose picker because it2i_generate is
  // single-image only (side-by-side concat is rendered literally, not
  // interpreted as a pose instruction).
  const angleModels = useMemo(
    () => imageModels.filter((item) => Array.isArray(item.ui?.viewAngles) && item.ui.viewAngles.length > 0),
    [imageModels],
  );
  const poseModels = useMemo(
    () => imageModels.filter((item) => item.ui?.poseLibrary),
    [imageModels],
  );
  // Default selection: the first registered backbone (manifest order keeps
  // InstantID first so the existing strict tier remains the default).
  const angleModel = angleModels[0] ?? null;
  const poseModel = poseModels[0] ?? null;

  useEffect(() => {
    if (!selectedCharacter && characters[0]?.id) {
      setSelectedCharacterId(characters[0].id);
    }
  }, [characters, selectedCharacter]);

  useEffect(() => {
    if (!selectedCharacter) {
      setDraft({ name: "", type: "person", description: "" });
      return;
    }
    setDraft({
      name: selectedCharacter.name ?? "",
      type: selectedCharacter.type ?? "person",
      description: selectedCharacter.description ?? "",
    });
    setSelectedReferenceIds((ids) =>
      ids.filter((id) => selectedCharacter.approvedReferences?.some((reference) => reference.assetId === id)),
    );
    setLoraEdits(
      Object.fromEntries((selectedCharacter.loras ?? []).map((link) => [link.id, editableLora(link)])),
    );
    if (testLookId && !selectedCharacter.looks?.some((look) => look.id === testLookId)) {
      setTestLookId("");
    }
  }, [selectedCharacter?.id, selectedCharacter?.updatedAt]);

  useEffect(() => {
    if (!imageModels.some((item) => item.id === testModel)) {
      setTestModel(imageModels[0]?.id ?? "z_image_turbo");
    }
  }, [imageModels, testModel]);

  // Create a draft character straight from the selector's "+ New character"
  // item (sc-2025) — name and type are then edited in the detail form, mirroring
  // the dataset "+ New dataset" flow.
  async function createDraftCharacter() {
    const created = await createCharacter({ name: "New character", type: "person", description: "" });
    if (created) {
      setSelectedCharacterId(created.id);
    }
  }

  async function saveCharacter(event) {
    event.preventDefault();
    if (selectedCharacter) {
      await updateCharacter(selectedCharacter.id, draft);
    }
  }

  // sc-2022: seed a new dataset (associated with this character) from its images
  // and jump into the Dataset editor to caption and train, reusing the shared
  // TrainingDataset engine.
  async function createDatasetFromCharacter() {
    if (!selectedCharacter || !characterImageAssetIds.length || creatingDataset) {
      return;
    }
    setCreatingDataset(true);
    try {
      const created = await createTrainingDataset({
        name: `${selectedCharacter.name} dataset`,
        characterId: selectedCharacter.id,
        items: characterImageAssetIds.map((assetId) => ({ assetId })),
      });
      if (created?.id) {
        openDatasetInLibrary(created.id);
      }
    } finally {
      setCreatingDataset(false);
    }
  }

  async function submitReference(event) {
    event.preventDefault();
    if (selectedCharacter && referenceAssetIds.length) {
      const savedAssetIds = [];
      try {
        for (const assetId of referenceAssetIds) {
          await addCharacterReference(selectedCharacter.id, { assetId, approved: false });
          savedAssetIds.push(assetId);
        }
        setReferenceAssetIds([]);
        setReferenceMessage("");
      } catch (err) {
        const message = err?.message ?? "Unknown error";
        setReferenceAssetIds((ids) => ids.filter((id) => !savedAssetIds.includes(id)));
        setReferenceMessage(
          savedAssetIds.length
            ? `Added ${savedAssetIds.length} reference${savedAssetIds.length === 1 ? "" : "s"}. Could not add the remaining selection: ${message}`
            : `Could not add references: ${message}`,
        );
      }
    }
  }

  async function submitLook(event) {
    event.preventDefault();
    if (!selectedCharacter || !lookDraft.name.trim()) {
      return;
    }
    await createCharacterLook(selectedCharacter.id, {
      name: lookDraft.name,
      description: lookDraft.description,
      approvedReferenceIds: selectedReferenceIds,
      recipeSettings: {},
    });
    setLookDraft({ name: "", description: "" });
    setSelectedReferenceIds([]);
  }

  async function submitLora(event) {
    event.preventDefault();
    if (!selectedCharacter || !loraId) {
      return;
    }
    const lora = loras.find((item) => item.id === loraId);
    if (!lora) {
      setLoraId("");
      return;
    }
    await attachCharacterLora(selectedCharacter.id, {
      loraId: lora.id,
      name: lora.name ?? lora.id,
      sourcePath: lora.installedPath ?? lora.source?.path ?? null,
      triggerWords: lora.triggerWords ?? [],
      defaultWeight: lora.defaultWeight ?? 0.8,
      compatibility: { families: extractFamilies(lora) },
      scope: lora.scope ?? "global",
    });
    setLoraId("");
  }

  async function saveLora(link) {
    const edit = loraEdits[link.id] ?? editableLora(link);
    await updateCharacterLora(selectedCharacter.id, link.id, {
      name: edit.name,
      triggerWords: edit.triggerWords
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      defaultWeight: Number(edit.defaultWeight),
      compatibility: {
        ...(link.compatibility ?? {}),
        families: edit.families
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
      },
      scope: edit.scope,
    });
  }

  function setLoraEdit(linkId, key, value) {
    setLoraEdits((items) => ({
      ...items,
      [linkId]: {
        ...(items[linkId] ?? {}),
        [key]: value,
      },
    }));
  }

  async function submitTest(event) {
    event.preventDefault();
    if (!selectedCharacter) {
      return;
    }
    const [width, height] = testResolution.split("x").map((value) => Number(value));
    await createCharacterTestJob(selectedCharacter.id, {
      prompt: testPrompt,
      model: testModel,
      count: Number(testCount),
      width,
      height,
      lookId: testLookId || null,
    });
  }

  return (
    <section className="main-surface character-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Character Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <CompactSelector
          createLabel="New character"
          disabled={!activeProject}
          getSubtitle={(character) =>
            `${typeLabel(character.type)} · ${character.references?.length ?? 0} ref${(character.references?.length ?? 0) === 1 ? "" : "s"}`
          }
          getThumbAsset={characterThumbAsset}
          items={characters}
          label="Select character"
          onCreate={createDraftCharacter}
          onSelect={(character) => setSelectedCharacterId(character.id)}
          placeholder="Select a character"
          selectedId={selectedCharacter?.id ?? ""}
        />
      </div>

      <div className="character-layout">
        {!selectedCharacter ? (
          <div className="empty-panel">No characters yet — use “New character” to start.</div>
        ) : (
          <section className="character-detail">
            <form className="character-editor" onSubmit={saveCharacter}>
              <div className="control-grid">
                <label>
                  Name
                  <input onChange={(event) => setDraft((item) => ({ ...item, name: event.target.value }))} value={draft.name} />
                </label>
                <label>
                  Type
                  <select onChange={(event) => setDraft((item) => ({ ...item, type: event.target.value }))} value={draft.type}>
                    {characterTypes.map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <label className="prompt-field">
                Notes
                <textarea
                  onChange={(event) => setDraft((item) => ({ ...item, description: event.target.value }))}
                  value={draft.description}
                />
              </label>
              <div className="detail-actions">
                <button className="primary-action" type="submit">
                  Save
                </button>
                <button className="secondary-action" onClick={() => archiveCharacter(selectedCharacter.id)} type="button">
                  Archive
                </button>
                <button
                  className="secondary-action"
                  onClick={() => onSendImage(selectedCharacter, testLookId || null, approvedReferences[0]?.assetId ?? null)}
                  type="button"
                >
                  Image
                </button>
                <button className="secondary-action" onClick={() => onSendVideo(selectedCharacter, testLookId || null)} type="button">
                  Video
                </button>
              </div>
            </form>
            <div className="guidance-strip">
              <strong>Reference identity</strong>
              <span>Approve a reference image, then use Generate variations (or the Image button) to create new images that keep this character's appearance with Kolors IP-Adapter. LoRA conditioning activates in a later runtime slice.</span>
            </div>

            <CharacterReferences
              imageAssets={imageAssets}
              onGenerateFromReference={(assetId) => onSendImage(selectedCharacter, testLookId || null, assetId)}
              onPreview={onPreview}
              referenceMessage={referenceMessage}
              referenceAssetIds={referenceAssetIds}
              removeCharacterReference={removeCharacterReference}
              selectedCharacter={selectedCharacter}
              setReferenceAssetIds={setReferenceAssetIds}
              submitReference={submitReference}
              updateCharacterReference={updateCharacterReference}
            />

            <CharacterAssets assets={assets} onPreview={onPreview} selectedCharacter={selectedCharacter} />

            <CharacterAngleSet
              addCharacterReference={addCharacterReference}
              angleModel={angleModel}
              angleModels={angleModels}
              approvedReferences={approvedReferences}
              createImageJob={createImageJob}
              imageLocalJobs={imageLocalJobs}
              importAsset={importAsset}
              latestAssets={latestAssets}
              loras={loras}
              onPreview={onPreview}
              rememberLocalGenerationJob={rememberLocalGenerationJob}
              selectedCharacter={selectedCharacter}
            />

            <CharacterPoseLibrary
              addCharacterReference={addCharacterReference}
              approvedReferences={approvedReferences}
              createImageJob={createImageJob}
              imageLocalJobs={imageLocalJobs}
              importAsset={importAsset}
              latestAssets={latestAssets}
              loras={loras}
              onPreview={onPreview}
              poseModel={poseModel}
              poseModels={poseModels}
              rememberLocalGenerationJob={rememberLocalGenerationJob}
              selectedCharacter={selectedCharacter}
            />

            <CharacterLooks
              approvedReferences={approvedReferences}
              createCharacterLook={createCharacterLook}
              deleteCharacterLook={deleteCharacterLook}
              lookDraft={lookDraft}
              selectedCharacter={selectedCharacter}
              selectedReferenceIds={selectedReferenceIds}
              setLookDraft={setLookDraft}
              setSelectedReferenceIds={setSelectedReferenceIds}
              setTestLookId={setTestLookId}
              submitLook={submitLook}
              updateCharacterLook={updateCharacterLook}
            />

            <CharacterLoras
              detachCharacterLora={detachCharacterLora}
              loraEdits={loraEdits}
              loraId={loraId}
              loras={loras}
              saveLora={saveLora}
              selectedCharacter={selectedCharacter}
              setLoraEdit={setLoraEdit}
              setLoraId={setLoraId}
              submitLora={submitLora}
            />

            <CharacterDatasets
              creating={creatingDataset}
              datasets={characterDatasets}
              imageCount={characterImageAssetIds.length}
              onCreateDataset={createDatasetFromCharacter}
              onOpenDataset={openDatasetInLibrary}
              projectId={activeProject?.id}
              selectedCharacter={selectedCharacter}
            />

            <CharacterTest
              addCharacterReference={addCharacterReference}
              createCharacterTestJob={createCharacterTestJob}
              deleteAsset={deleteAsset}
              imageModels={imageModels}
              latestAssets={latestAssets}
              onPreview={onPreview}
              purgeAsset={purgeAsset}
              selectedCharacter={selectedCharacter}
              setTestCount={setTestCount}
              setTestLookId={setTestLookId}
              setTestModel={setTestModel}
              setTestPrompt={setTestPrompt}
              setTestResolution={setTestResolution}
              submitTest={submitTest}
              testCount={testCount}
              testLookId={testLookId}
              testModel={testModel}
              testPrompt={testPrompt}
              testResolution={testResolution}
              updateAssetStatus={updateAssetStatus}
            />
          </section>
        )}
      </div>
    </section>
  );
}
