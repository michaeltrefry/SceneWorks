import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { angleModelUsable, downloadOffersFor, poseModelUsable } from "../modelEligibility.js";
import { DEFAULT_MAC_CAPABILITIES, macAvailableModels } from "../macGating.js";

const characterTypes = [
  ["person", "Person"],
  ["creature", "Creature"],
  ["object", "Object"],
];

// Tab information architecture (epic 2293): the single stacked column is grouped
// into five focused workspaces. Order is also the keyboard nav order.
const CHARACTER_TABS = [
  ["character", "Character"],
  ["assets", "Assets"],
  ["angles", "Angles"],
  ["poses", "Poses"],
  ["test", "Test"],
];
const CHARACTER_TAB_IDS = CHARACTER_TABS.map(([id]) => id);
const DEFAULT_CHARACTER_TAB = "character";

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
    unarchiveCharacter,
    listArchivedCharacters,
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
    compareFaceLikeness,
    createImageJob,
    createModelDownloadJob,
    importAsset,
    imageLocalJobs,
    jobAction,
    rememberLocalGenerationJob,
    setActiveView,
    deleteAsset,
    purgeAsset,
    imageModels,
    models = [],
    jobs = [],
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
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  const latestAssets = latestImageAssets;
  // Mac UI gating (sc-3486): hide torch-only models from the angle/pose/test pickers.
  const macImageModels = useMemo(
    () => macAvailableModels(imageModels, macCapabilities),
    [imageModels, macCapabilities],
  );
  const onPreview = setPreviewAsset;
  // Model-availability gate (sc-5947): the Angle Set and Pose Library generation tabs each
  // need a model with ui.viewAngles / ui.poseLibrary (e.g. InstantID). Only those two tab
  // panels are gated — character/asset management works without a generation model — so the
  // gate shows recommended downloads in place of the panel, not the whole studio. Offers come
  // from the full catalog (recommended-first); `ready` reads the screen's angle/pose pickers.
  const angleOffers = useMemo(
    () => downloadOffersFor(models, angleModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  const poseOffers = useMemo(
    () => downloadOffersFor(models, poseModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );
  const onOpenModels = () => setActiveView("Models");
  // Job callbacks for character generation cards (Angle Set / Pose Library).
  // jobAction may be missing in test contexts that wrap CharacterStudio with a
  // stub provider; guard so the buttons just no-op there.
  const onCancelCharacterJob = jobAction ? (job) => jobAction(job, "cancel") : undefined;
  const onRetryCharacterJob = jobAction ? (job) => jobAction(job, "retry") : undefined;
  const onDuplicateCharacterJob = jobAction ? (job) => jobAction(job, "duplicate") : undefined;
  const onOpenCharacterJobQueue = setActiveView ? () => setActiveView("Queue") : undefined;
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
  // Archived-characters view (sc-6066). Archived characters aren't in the active
  // `characters` roster (it's fetched without them), so this view lazily fetches
  // them on open and offers a Restore action. Lives here so an archive/restore can
  // refresh it.
  const [archivedOpen, setArchivedOpen] = useState(false);
  const [archivedCharacters, setArchivedCharacters] = useState([]);
  const [archivedLoading, setArchivedLoading] = useState(false);
  const [archivedError, setArchivedError] = useState("");
  const [restoringId, setRestoringId] = useState("");

  // Active tab + per-workspace persistence (epic 2293). The component is keyed by
  // workspace in App.jsx, so this reads the right snapshot per workspace on mount
  // and remounts (re-running the initializer) when the workspace changes — no tab
  // bleed across workspaces. Mirrors the studio-settings localStorage pattern.
  const savedSettings = useMemo(() => loadStudioSettings("character", activeProject?.id ?? null), [activeProject?.id]);
  const [activeTab, setActiveTab] = useState(() =>
    CHARACTER_TAB_IDS.includes(savedSettings.activeTab) ? savedSettings.activeTab : DEFAULT_CHARACTER_TAB,
  );
  useStudioSettingsWriter("character", activeProject?.id ?? null, { activeTab });
  const tabRefs = useRef({});
  const activeTabIndex = CHARACTER_TABS.findIndex(([id]) => id === activeTab);
  // Roving-tabindex keyboard nav, matching the TrainingStudio tablist: arrows wrap,
  // Home/End jump to the ends, and focus follows the selected tab.
  function focusTab(index) {
    const [nextId] = CHARACTER_TABS[(index + CHARACTER_TABS.length) % CHARACTER_TABS.length];
    setActiveTab(nextId);
    window.requestAnimationFrame(() => tabRefs.current[nextId]?.focus());
  }
  function onTabKeyDown(event) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTab(activeTabIndex + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTab(activeTabIndex - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      focusTab(0);
    } else if (event.key === "End") {
      event.preventDefault();
      focusTab(CHARACTER_TABS.length - 1);
    }
  }

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
  // This character's image/frame assets (generated for it, generated referencing
  // it, or attached as references). sc-6042: this is the pool the Reference picker
  // selects from — the character's own assets, not the whole project library.
  const characterReferenceCandidates = useMemo(
    () =>
      selectedCharacter
        ? imageAssets.filter((asset) => assetMatchesCharacter(asset, selectedCharacter.id, selectedCharacter))
        : [],
    [imageAssets, selectedCharacter?.id],
  );
  const characterImageAssetIds = useMemo(
    () => characterReferenceCandidates.map((asset) => asset.id),
    [characterReferenceCandidates],
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
  //   sensenova_u1_8b_fast                — 0.29, wardrobe-continuity tier
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
    () => macImageModels.filter((item) => Array.isArray(item.ui?.viewAngles) && item.ui.viewAngles.length > 0),
    [macImageModels],
  );
  const poseModels = useMemo(
    () => macImageModels.filter((item) => item.ui?.poseLibrary),
    [macImageModels],
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
    if (!macImageModels.some((item) => item.id === testModel)) {
      setTestModel(macImageModels[0]?.id ?? "z_image_turbo");
    }
  }, [macImageModels, testModel]);

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

  // sc-6066: lazily load archived characters for the Archived view. `withCharacterApi`
  // (inside the hook) routes failures to the shared error banner and returns null.
  const loadArchived = useCallback(async () => {
    if (typeof listArchivedCharacters !== "function") {
      return;
    }
    setArchivedLoading(true);
    setArchivedError("");
    try {
      const items = await listArchivedCharacters();
      setArchivedCharacters(items ?? []);
    } catch (err) {
      setArchivedError(err?.message ?? "Could not load archived characters.");
    } finally {
      setArchivedLoading(false);
    }
  }, [listArchivedCharacters]);

  // Fetch when the section is opened (and again if the project changes while open —
  // `loadArchived` identity tracks the active project through the hook).
  useEffect(() => {
    if (archivedOpen) {
      loadArchived();
    }
  }, [archivedOpen, loadArchived]);

  // sc-6066: archiving is destructive-feeling (the character vanishes from the list),
  // so confirm first — a single misclick shouldn't silently hide a character.
  async function handleArchiveSelected() {
    if (!selectedCharacter) {
      return;
    }
    if (
      typeof window !== "undefined" &&
      typeof window.confirm === "function" &&
      !window.confirm(
        `Archive "${selectedCharacter.name || "this character"}"? It will be hidden from the active list. You can restore it later from "Show archived characters".`,
      )
    ) {
      return;
    }
    await archiveCharacter(selectedCharacter.id);
    if (archivedOpen) {
      await loadArchived();
    }
  }

  // sc-6066: restore an archived character back to the active roster and select it.
  async function handleRestoreCharacter(characterId) {
    if (typeof unarchiveCharacter !== "function") {
      return;
    }
    setRestoringId(characterId);
    try {
      const restored = await unarchiveCharacter(characterId);
      if (restored) {
        setArchivedCharacters((items) => items.filter((item) => item.id !== restored.id));
        setSelectedCharacterId(restored.id);
      }
    } finally {
      setRestoringId("");
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
            <div className="segmented-control character-tabs" role="tablist" aria-label="Character workspace">
              {CHARACTER_TABS.map(([id, label]) => (
                <button
                  aria-controls={activeTab === id ? `character-panel-${id}` : undefined}
                  aria-selected={activeTab === id}
                  className={activeTab === id ? "active" : ""}
                  id={`character-tab-${id}`}
                  key={id}
                  onClick={() => setActiveTab(id)}
                  onKeyDown={onTabKeyDown}
                  ref={(node) => {
                    tabRefs.current[id] = node;
                  }}
                  role="tab"
                  tabIndex={activeTab === id ? 0 : -1}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Character tab — identity hub: metadata form + references + saved
                presets (looks) + LoRAs. Every tabpanel stays mounted and is hidden
                when inactive, so each panel's working state (incl. panel-local
                state like the angle/pose prompt and selected pose set) survives
                tab switches. This matches today's all-mounted render cost. */}
            <div
              aria-labelledby="character-tab-character"
              className="character-tabpanel"
              hidden={activeTab !== "character"}
              id="character-panel-character"
              role="tabpanel"
            >
              <form className="character-editor" onSubmit={saveCharacter}>
                <div className="section-heading">
                  <p className="eyebrow">Identity</p>
                  <h2>Name, type &amp; notes</h2>
                </div>
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
                  <button className="secondary-action" onClick={handleArchiveSelected} type="button">
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
                referenceCandidates={characterReferenceCandidates}
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
            </div>

            {/* Assets tab — the character asset library (images + frames) +
                its training datasets. */}
            <div
              aria-labelledby="character-tab-assets"
              className="character-tabpanel"
              hidden={activeTab !== "assets"}
              id="character-panel-assets"
              role="tabpanel"
            >
              <CharacterAssets
                addCharacterReference={addCharacterReference}
                approvedReferences={approvedReferences}
                assets={assets}
                compareFaceLikeness={compareFaceLikeness}
                deleteAsset={deleteAsset}
                importAsset={importAsset}
                onPreview={onPreview}
                projectId={activeProject?.id}
                purgeAsset={purgeAsset}
                selectedCharacter={selectedCharacter}
                updateAssetStatus={updateAssetStatus}
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
            </div>

            {/* Angles tab — Angle Set generation. */}
            <div
              aria-labelledby="character-tab-angles"
              className="character-tabpanel"
              hidden={activeTab !== "angles"}
              id="character-panel-angles"
              role="tabpanel"
            >
              <ModelAvailabilityGate
                ready={angleModels.length > 0}
                title="Angle Set needs an angle-capable model"
                description="Generating character angle sets needs a model like InstantID (RealVisXL). Download one to get started."
                offers={angleOffers}
                downloadJobs={modelDownloadJobs}
                onDownload={createModelDownloadJob}
                onOpenModels={onOpenModels}
                onOpenQueue={onOpenCharacterJobQueue}
                onCancelJob={onCancelCharacterJob}
              >
              <CharacterAngleSet
                addCharacterReference={addCharacterReference}
                angleModel={angleModel}
                angleModels={angleModels}
                approvedReferences={approvedReferences}
                assets={assets}
                catalog={models}
                createImageJob={createImageJob}
                imageLocalJobs={imageLocalJobs}
                importAsset={importAsset}
                latestAssets={latestAssets}
                loras={loras}
                onCancel={onCancelCharacterJob}
                onDuplicate={onDuplicateCharacterJob}
                onOpenQueue={onOpenCharacterJobQueue}
                onPreview={onPreview}
                onRetry={onRetryCharacterJob}
                rememberLocalGenerationJob={rememberLocalGenerationJob}
                selectedCharacter={selectedCharacter}
              />
              </ModelAvailabilityGate>
            </div>

            {/* Poses tab — Pose generation. */}
            <div
              aria-labelledby="character-tab-poses"
              className="character-tabpanel"
              hidden={activeTab !== "poses"}
              id="character-panel-poses"
              role="tabpanel"
            >
              <ModelAvailabilityGate
                ready={poseModels.length > 0}
                title="Pose Library needs a pose-capable model"
                description="Generating character poses needs a model like InstantID (RealVisXL). Download one to get started."
                offers={poseOffers}
                downloadJobs={modelDownloadJobs}
                onDownload={createModelDownloadJob}
                onOpenModels={onOpenModels}
                onOpenQueue={onOpenCharacterJobQueue}
                onCancelJob={onCancelCharacterJob}
              >
              <CharacterPoseLibrary
                addCharacterReference={addCharacterReference}
                approvedReferences={approvedReferences}
                assets={assets}
                catalog={models}
                createImageJob={createImageJob}
                imageLocalJobs={imageLocalJobs}
                importAsset={importAsset}
                latestAssets={latestAssets}
                loras={loras}
                onCancel={onCancelCharacterJob}
                onDuplicate={onDuplicateCharacterJob}
                onOpenQueue={onOpenCharacterJobQueue}
                onPreview={onPreview}
                onRetry={onRetryCharacterJob}
                poseModel={poseModel}
                poseModels={poseModels}
                rememberLocalGenerationJob={rememberLocalGenerationJob}
                selectedCharacter={selectedCharacter}
              />
              </ModelAvailabilityGate>
            </div>

            {/* Test tab — Test Character form. */}
            <div
              aria-labelledby="character-tab-test"
              className="character-tabpanel"
              hidden={activeTab !== "test"}
              id="character-panel-test"
              role="tabpanel"
            >
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
            </div>
          </section>
        )}
      </div>

      {/* Archived characters (sc-6066): archive is a recoverable soft flag, so give it
          a home. Lazily fetched, visually separated from the active roster, with a
          Restore action. Excluded from the active selector and all pickers. */}
      {activeProject ? (
        <section className="archived-characters" aria-label="Archived characters">
          <button
            aria-expanded={archivedOpen}
            className="secondary-action archived-toggle"
            onClick={() => setArchivedOpen((open) => !open)}
            type="button"
          >
            {archivedOpen ? "Hide archived characters" : "Show archived characters"}
          </button>
          {archivedOpen ? (
            <div className="archived-list">
              {archivedLoading ? (
                <p className="muted">Loading archived characters…</p>
              ) : archivedError ? (
                <p className="error-text">{archivedError}</p>
              ) : archivedCharacters.length === 0 ? (
                <p className="muted">No archived characters.</p>
              ) : (
                <ul className="archived-character-list">
                  {archivedCharacters.map((character) => (
                    <li className="archived-character-row" key={character.id}>
                      <span className="archived-character-name">{character.name || "Untitled character"}</span>
                      <span className="archived-character-meta">
                        {typeLabel(character.type)} · {character.references?.length ?? 0} ref
                        {(character.references?.length ?? 0) === 1 ? "" : "s"}
                      </span>
                      <button
                        className="secondary-action"
                        disabled={restoringId === character.id}
                        onClick={() => handleRestoreCharacter(character.id)}
                        type="button"
                      >
                        {restoringId === character.id ? "Restoring…" : "Restore"}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : null}
        </section>
      ) : null}
    </section>
  );
}
