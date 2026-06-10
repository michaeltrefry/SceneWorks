import React from "react";
import { useKeypointCollections } from "../keypointLibrary.js";

// Per-generation angle-set override for Character Studio's InstantID angle panel (epic 4422,
// sc-4435/sc-4450). Lets the user pick which Key Point Library collection "Generate angle set"
// runs, instead of the active default. Value "" = use the active default (no override sent);
// any other value sets advanced.keypointCollectionId, which the worker resolves at generation
// time. onChange(id, collection) also hands back the chosen collection so the caller can size
// the run to its (variable) angle count.
export function KeypointCollectionField({ value = "", onChange }) {
  const { collections, loading, error } = useKeypointCollections();

  function handleChange(id) {
    const collection = collections.find((item) => item.id === id) ?? null;
    onChange?.(id, collection);
  }

  const defaultCollection = collections.find((item) => item.isDefault);

  return (
    <label>
      Angle set
      <select onChange={(event) => handleChange(event.target.value)} value={value}>
        <option value="">
          {defaultCollection
            ? `Default (${defaultCollection.name}, ${defaultCollection.orderedPresetIds?.length ?? 0} angles)`
            : "Default angles"}
        </option>
        {collections.map((collection) => (
          <option key={collection.id} value={collection.id}>
            {collection.name} ({collection.orderedPresetIds?.length ?? 0} angles)
            {collection.isDefault ? " — default" : ""}
          </option>
        ))}
      </select>
      {loading ? <span className="muted"> Loading collections…</span> : null}
      {error ? <span className="inline-warning"> Collections unavailable.</span> : null}
    </label>
  );
}
