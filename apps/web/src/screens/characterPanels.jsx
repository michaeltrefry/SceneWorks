import React from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";

export function compatibleFamilies(item) {
  const compatibility = item?.compatibility ?? {};
  const values =
    item?.families ??
    item?.compatibleFamilies ??
    item?.modelFamilies ??
    compatibility.families ??
    (item?.family ? [item.family] : []);
  return Array.isArray(values) ? values : [values].filter(Boolean);
}

export function editableLora(link) {
  return {
    name: link?.name ?? "",
    triggerWords: (link?.triggerWords ?? []).join(", "),
    defaultWeight: link?.defaultWeight ?? 0.8,
    families: compatibleFamilies(link).join(", "),
    scope: link?.scope ?? "project",
  };
}

function summarizeCompatibility(item) {
  const values = compatibleFamilies(item);
  return values.length ? values.join(", ") : "Unspecified";
}

export function CharacterReferences({
  imageAssets,
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
      <div className="review-grid">
        {latestAssets.map((asset) => (
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
        {latestAssets.length ? null : <div className="empty-panel compact-panel">No test outputs yet</div>}
      </div>
    </section>
  );
}
