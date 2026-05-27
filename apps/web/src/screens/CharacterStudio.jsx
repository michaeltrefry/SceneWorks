import React, { useEffect, useMemo, useState } from "react";
import {
  CharacterAngleSet,
  CharacterLoras,
  CharacterLooks,
  CharacterReferences,
  CharacterTest,
  editableLora,
} from "./characterPanels.jsx";
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
  } = useAppContext();
  const latestAssets = latestImageAssets;
  const onPreview = setPreviewAsset;
  const onSendImage = sendCharacterToImage;
  const onSendVideo = sendCharacterToVideo;
  const [selectedCharacterId, setSelectedCharacterId] = useState(characters[0]?.id ?? "");
  const [draft, setDraft] = useState({ name: "", type: "person", description: "" });
  const [newCharacter, setNewCharacter] = useState({ name: "", type: "person", description: "" });
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

  const imageAssets = useMemo(
    () => assets.filter((asset) => ["image", "frame", "upload"].includes(asset.type)),
    [assets],
  );
  const selectedCharacter = characters.find((item) => item.id === selectedCharacterId) ?? characters[0] ?? null;
  const approvedReferences = selectedCharacter?.approvedReferences ?? [];
  // The InstantID-style model that can render a one-click angle set (declares view angles).
  const angleModel = useMemo(
    () => imageModels.find((item) => Array.isArray(item.ui?.viewAngles) && item.ui.viewAngles.length > 0) ?? null,
    [imageModels],
  );

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

  async function submitNewCharacter(event) {
    event.preventDefault();
    const created = await createCharacter(newCharacter);
    if (created) {
      setSelectedCharacterId(created.id);
      setNewCharacter({ name: "", type: "person", description: "" });
    }
  }

  async function saveCharacter(event) {
    event.preventDefault();
    if (selectedCharacter) {
      await updateCharacter(selectedCharacter.id, draft);
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
        <form className="inline-create" onSubmit={submitNewCharacter}>
          <input
            aria-label="Character name"
            onChange={(event) => setNewCharacter((item) => ({ ...item, name: event.target.value }))}
            placeholder="New character"
            value={newCharacter.name}
          />
          <select
            aria-label="Character type"
            onChange={(event) => setNewCharacter((item) => ({ ...item, type: event.target.value }))}
            value={newCharacter.type}
          >
            {characterTypes.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <button disabled={!activeProject || !newCharacter.name.trim()} type="submit">
            Create
          </button>
        </form>
      </div>

      {!selectedCharacter ? (
        <div className="empty-panel">No characters yet</div>
      ) : (
        <div className="character-layout">
          <aside className="character-list">
            {characters.map((character) => (
              <button
                className={character.id === selectedCharacter.id ? "character-row active" : "character-row"}
                key={character.id}
                onClick={() => setSelectedCharacterId(character.id)}
                type="button"
              >
                <strong>{character.name}</strong>
                <span>{typeLabel(character.type)}</span>
                <small>{character.references?.length ?? 0} refs</small>
              </button>
            ))}
          </aside>

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
                <button type="submit">Save</button>
                <button onClick={() => archiveCharacter(selectedCharacter.id)} type="button">
                  Archive
                </button>
                <button
                  onClick={() => onSendImage(selectedCharacter, testLookId || null, approvedReferences[0]?.assetId ?? null)}
                  type="button"
                >
                  Image
                </button>
                <button onClick={() => onSendVideo(selectedCharacter, testLookId || null)} type="button">
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

            <CharacterAngleSet
              addCharacterReference={addCharacterReference}
              angleModel={angleModel}
              approvedReferences={approvedReferences}
              createImageJob={createImageJob}
              imageLocalJobs={imageLocalJobs}
              importAsset={importAsset}
              latestAssets={latestAssets}
              onPreview={onPreview}
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
        </div>
      )}
    </section>
  );
}
