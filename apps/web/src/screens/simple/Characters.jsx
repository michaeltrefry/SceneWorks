import React, { useMemo, useRef, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { Icon } from "../../components/Icons.jsx";
import { AssetThumbnail } from "../../components/assetMedia.jsx";

function approvedRefs(character) {
  const refs = Array.isArray(character?.references) ? character.references : [];
  const approved = refs.filter((ref) => ref.approved);
  return approved.length ? approved : refs;
}

function referenceAssetId(character) {
  return approvedRefs(character)[0]?.assetId ?? null;
}

export function Characters() {
  const {
    activeProject,
    characters = [],
    createCharacter,
    addCharacterReference,
    updateCharacter,
    importAsset,
    imageModels = [],
    createImageJob,
    setUiMode,
    setActiveView,
  } = useAppContext();

  const [selectedId, setSelectedId] = useState(null);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [match, setMatch] = useState(65);
  const [notice, setNotice] = useState("");
  const newFilesRef = useRef(null);
  const addFilesRef = useRef(null);

  const selected = useMemo(
    () => characters.find((character) => character.id === selectedId) ?? characters[0] ?? null,
    [characters, selectedId],
  );

  // character_image needs a model that supports it (z-image does not).
  const characterModel = useMemo(
    () => imageModels.find((model) => (model.capabilities ?? []).includes("character_image")) ?? null,
    [imageModels],
  );

  async function uploadRefs(characterId, files) {
    for (const file of files) {
      const asset = await importAsset(file, { throwOnError: true });
      if (asset?.id) {
        await addCharacterReference(characterId, { assetId: asset.id, approved: true, role: "import" });
      }
    }
  }

  async function handleCreate() {
    const name = newName.trim();
    const files = Array.from(newFilesRef.current?.files ?? []);
    if (!name || busy) return;
    if (!activeProject) {
      setNotice("Open or create a workspace first.");
      return;
    }
    setBusy(true);
    setNotice("");
    try {
      const created = await createCharacter({ name, type: "person", description: "" });
      if (created?.id && files.length) await uploadRefs(created.id, files);
      if (created?.id) setSelectedId(created.id);
      setCreating(false);
      setNewName("");
    } catch (error) {
      setNotice(error?.message || "Couldn't create that character.");
    } finally {
      setBusy(false);
    }
  }

  async function handleAddPhotos(event) {
    const files = Array.from(event.target.files ?? []);
    if (!selected || !files.length) return;
    setBusy(true);
    setNotice("");
    try {
      await uploadRefs(selected.id, files);
    } catch (error) {
      setNotice(error?.message || "Couldn't add those photos.");
    } finally {
      setBusy(false);
      if (addFilesRef.current) addFilesRef.current.value = "";
    }
  }

  async function handleRename() {
    const name = window.prompt("Rename character", selected?.name ?? "");
    const trimmed = name?.trim();
    if (!trimmed || trimmed === selected?.name) return;
    try {
      await updateCharacter(selected.id, { name: trimmed });
    } catch (error) {
      setNotice(error?.message || "Couldn't rename.");
    }
  }

  async function handleGenerate() {
    if (!selected || !prompt.trim() || busy) return;
    if (!characterModel) {
      setNotice("Add a character-capable model (e.g. RealVisXL) in Settings first.");
      return;
    }
    const refId = referenceAssetId(selected);
    if (!refId) {
      setNotice("Add at least one photo of this character first.");
      return;
    }
    setBusy(true);
    setNotice("");
    try {
      const job = await createImageJob({
        mode: "character_image",
        prompt: prompt.trim(),
        negativePrompt: "",
        model: characterModel.id,
        count: 4,
        width: 1024,
        height: 1024,
        characterId: selected.id,
        characterLookId: null,
        referenceAssetId: refId,
        recipePresetId: null,
        loras: [],
        advanced: { resolution: "1024x1024", ipAdapterScale: match / 100 },
      });
      setNotice(job ? "Started — your pictures will appear in My creations." : "Couldn't start that — try again.");
    } finally {
      setBusy(false);
    }
  }

  function openAdvancedCharacters() {
    setUiMode?.("advanced");
    setActiveView?.("Characters");
  }

  return (
    <section className="main-surface sw-make">
      <div className="sw-creations-grid">
        <div>
          <div className="sw-cgrid">
            {characters.map((character) => {
              const ref = approvedRefs(character)[0];
              return (
                <button
                  type="button"
                  key={character.id}
                  className={`sw-ccard ${selected?.id === character.id ? "sel" : ""}`.trim()}
                  onClick={() => setSelectedId(character.id)}
                >
                  <span className="sw-avatar">
                    {ref?.asset ? <AssetThumbnail asset={ref.asset} /> : null}
                  </span>
                  <b>{character.name}</b>
                  <small>{approvedRefs(character).length} photo{approvedRefs(character).length === 1 ? "" : "s"}</small>
                </button>
              );
            })}

            {creating ? (
              <div className="sw-ccard sw-ccard-form">
                <input
                  className="sw-input"
                  placeholder="Name (e.g. Mara)"
                  value={newName}
                  autoFocus
                  onChange={(event) => setNewName(event.target.value)}
                />
                <input ref={newFilesRef} type="file" accept="image/*" multiple className="sw-file" />
                <div className="sw-form-actions">
                  <button type="button" className="sw-btn-primary" onClick={handleCreate} disabled={!newName.trim() || busy}>
                    {busy ? "Creating…" : "Create"}
                  </button>
                  <button type="button" className="sw-act" onClick={() => { setCreating(false); setNewName(""); }}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button type="button" className="sw-ccard sw-ccard-new" onClick={() => setCreating(true)}>
                <span className="sw-plus"><Icon.Plus /></span>
                <b>New character</b>
                <small>Add a few photos</small>
              </button>
            )}
          </div>
        </div>

        {selected ? (
          <aside className="sw-detail">
            <div className="sw-detail-head">
              <h3 className="sw-detail-title">{selected.name}</h3>
              <button type="button" className="sw-rename" onClick={handleRename}>
                <Icon.Wand /> Rename
              </button>
            </div>
            <p className="sw-detail-sub">{approvedRefs(selected).length} reference photo{approvedRefs(selected).length === 1 ? "" : "s"}</p>

            <div className="sw-refs">
              {approvedRefs(selected).slice(0, 6).map((ref) => (
                <span className="sw-ref" key={ref.assetId}>
                  {ref.asset ? <AssetThumbnail asset={ref.asset} /> : null}
                </span>
              ))}
              <button type="button" className="sw-ref sw-ref-add" onClick={() => addFilesRef.current?.click()}>
                <Icon.Plus />
              </button>
              <input ref={addFilesRef} type="file" accept="image/*" multiple hidden onChange={handleAddPhotos} />
            </div>

            <div className="sw-field">
              <h3 className="sw-q">Put them in a scene</h3>
              <textarea
                className="sw-prompt"
                rows={2}
                value={prompt}
                placeholder="walking through a sunlit market, candid"
                onChange={(event) => setPrompt(event.target.value)}
              />
            </div>

            <div className="sw-field">
              <label className="sw-range-label" htmlFor="sw-match">How closely to match</label>
              <input
                id="sw-match"
                className="sw-range"
                type="range"
                min="0"
                max="100"
                value={match}
                onChange={(event) => setMatch(Number(event.target.value))}
              />
            </div>

            <button type="button" className="sw-act primary" onClick={handleGenerate} disabled={busy || !prompt.trim()}>
              <Icon.Image /> Create pictures of {selected.name}
            </button>

            {notice ? <p className="sw-notice">{notice}</p> : null}
            <button
              type="button"
              className="sw-advlink"
              onClick={openAdvancedCharacters}
              title="Angle sheets, pose sets and training live in Advanced"
            >
              Want angle sheets or to train a model? Use Advanced →
            </button>
          </aside>
        ) : (
          <aside className="sw-detail">
            <p className="sw-rendering">Create a character to keep the same face across everything you make.</p>
          </aside>
        )}
      </div>
    </section>
  );
}
