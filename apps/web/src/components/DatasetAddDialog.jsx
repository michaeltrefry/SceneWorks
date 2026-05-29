import React, { useMemo, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage } from "./assetMedia.jsx";
import { Modal } from "./Modal.jsx";

// An asset belongs to a character when it was generated in association with it
// (recipe.normalizedSettings.characterId) or generated referencing it
// (metadata.characterReferences[].characterId). Mirrors the per-character
// gallery filter so the Character tab surfaces the same images.
export function assetMatchesCharacter(asset, characterId) {
  if (!characterId) {
    return false;
  }
  if (asset?.recipe?.normalizedSettings?.characterId === characterId) {
    return true;
  }
  return (asset?.metadata?.characterReferences ?? []).some((reference) => reference?.characterId === characterId);
}

function assetTitle(asset) {
  return asset?.displayName ?? asset?.title ?? asset?.name ?? asset?.id ?? "Untitled asset";
}

const TABS = [
  ["file", "File"],
  ["library", "Asset Library"],
  ["character", "Character"],
];

// Multi-select grid shared by the Library and Character tabs. Selection is
// local to the dialog; nothing is added to the dataset until "Add" is pressed.
function CandidateGrid({ assets, selectedIds, onToggle, emptyLabel }) {
  if (!assets.length) {
    return <div className="empty-panel compact-panel">{emptyLabel}</div>;
  }
  return (
    <div aria-multiselectable className="dataset-add-grid" role="listbox">
      {assets.map((asset) => {
        const selected = selectedIds.includes(asset.id);
        return (
          <button
            aria-selected={selected}
            className={selected ? "dataset-add-card selected" : "dataset-add-card"}
            key={asset.id}
            onClick={() => onToggle(asset.id)}
            role="option"
            type="button"
          >
            <AssetThumbnail asset={asset} />
            <span>{assetTitle(asset)}</span>
          </button>
        );
      })}
    </div>
  );
}

// Add-to-dataset modal with three sources (sc-2026): drag/upload files, pick
// from the (scoped) Asset Library, or pull a character's images. The dataset
// editor owns membership + import; this dialog only collects a selection or
// hands files back via onImport.
export function DatasetAddDialog({ assets = [], memberIds = [], characters = [], importing = false, onImport, onAdd, onClose }) {
  const [tab, setTab] = useState("file");
  const [selectedIds, setSelectedIds] = useState([]);
  const [characterId, setCharacterId] = useState(characters[0]?.id ?? "");
  const [dragActive, setDragActive] = useState(false);

  const memberSet = useMemo(() => new Set(memberIds), [memberIds]);

  // Library tab: studio + uploaded media only — never Character Studio test
  // outputs (origin gating from sc-2024) — and nothing already in the dataset.
  const libraryCandidates = useMemo(
    () =>
      assets.filter(
        (asset) =>
          assetCanRenderAsImage(asset) &&
          asset.origin !== "character_studio" &&
          !memberSet.has(asset.id) &&
          !asset.status?.trashed,
      ),
    [assets, memberSet],
  );

  // Character tab: this character's images (including its Character Studio
  // outputs, which the Library tab hides), minus current members.
  const characterCandidates = useMemo(
    () => assets.filter((asset) => assetMatchesCharacter(asset, characterId) && !memberSet.has(asset.id)),
    [assets, characterId, memberSet],
  );

  function toggle(id) {
    setSelectedIds((ids) => (ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id]));
  }

  function commit() {
    if (selectedIds.length) {
      // sc-2022: adding from the Character tab associates the dataset with that
      // character on the next save; other sources pass no character.
      onAdd(selectedIds, tab === "character" ? characterId : null);
      setSelectedIds([]);
    }
  }

  function switchTab(next) {
    setTab(next);
    setSelectedIds([]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    const files = event.dataTransfer?.files;
    if (files?.length) {
      onImport(files);
    }
  }

  return (
    <Modal className="dataset-add-modal" labelledBy="dataset-add-title" onClose={onClose}>
      <header className="dataset-add-head">
        <div>
          <p className="eyebrow">Add images</p>
          <h2 id="dataset-add-title">Add images to dataset</h2>
        </div>
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
      </header>

      <div className="segmented-control compact-segment" role="tablist" aria-label="Add source">
        {TABS.map(([key, label]) => (
          <button
            aria-selected={tab === key}
            className={tab === key ? "active" : ""}
            key={key}
            onClick={() => switchTab(key)}
            role="tab"
            type="button"
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "file" ? (
        <div
          className={dragActive ? "dataset-add-dropzone active" : "dataset-add-dropzone"}
          onDragLeave={() => setDragActive(false)}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDrop={handleDrop}
        >
          <p>Drag images and caption .txt files here, or</p>
          <label className="file-upload-button">
            <input
              accept="image/*,.txt,text/plain"
              disabled={importing}
              multiple
              onChange={(event) => {
                onImport(event.target.files);
                event.target.value = "";
              }}
              type="file"
            />
            {importing ? "Importing" : "Browse files"}
          </label>
        </div>
      ) : null}

      {tab === "library" ? (
        <CandidateGrid
          assets={libraryCandidates}
          emptyLabel="No library images to add"
          onToggle={toggle}
          selectedIds={selectedIds}
        />
      ) : null}

      {tab === "character" ? (
        <div className="dataset-add-character">
          <label>
            Character
            <select aria-label="Character" onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
              {characters.length ? null : <option value="">No characters yet</option>}
              {characters.map((character) => (
                <option key={character.id} value={character.id}>
                  {character.name ?? character.id}
                </option>
              ))}
            </select>
          </label>
          <CandidateGrid
            assets={characterCandidates}
            emptyLabel={characterId ? "No images for this character yet" : "Select a character"}
            onToggle={toggle}
            selectedIds={selectedIds}
          />
        </div>
      ) : null}

      {tab === "file" ? null : (
        <footer className="dataset-add-footer">
          <span>{selectedIds.length ? `${selectedIds.length} selected` : "No selection"}</span>
          <div className="detail-actions">
            <button onClick={onClose} type="button">
              Done
            </button>
            <button className="primary-action" disabled={!selectedIds.length} onClick={commit} type="button">
              Add {selectedIds.length || ""}
            </button>
          </div>
        </footer>
      )}
    </Modal>
  );
}
