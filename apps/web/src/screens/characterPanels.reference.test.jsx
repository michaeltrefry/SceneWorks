import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CharacterGenerationPanel } from "./characterPanels.jsx";

// sc-8851 ([F-049]): the reference-selection effect used to unconditionally snap
// `referenceAssetId` back to approvedReferences[0] on every render where the
// `approvedReferences` identity changed. useCharacters replaces that array (new
// identity, same contents) on every character mutation — including the panel's own
// upload flow (importAsset -> addCharacterReference), which sets the freshly uploaded
// id. The unconditional reset immediately undid that pick, so users generated against
// reference #1 instead of the reference they chose. The fix mirrors ImageStudio's
// guard: keep the current id while it is still an approved reference, and only fall
// back to [0] when the current id is absent/invalid. onUpload also now approves the
// reference before selecting it, so the selection lands on a valid (already-refetched)
// id rather than one the guard would reject.

const MODEL = {
  id: "instantid_realvisxl",
  name: "InstantID RealVisXL",
  family: "instantid",
  ui: { viewAngles: ["front", "left", "right"] },
};

// Minimal mode object exercising every mode.* field the panel reads, with a stub
// controller so we don't drag the real angle/pose controllers (and their fields)
// into a test that is only about reference selection.
const TEST_MODE = {
  eyebrow: "Angle set",
  defaultNegativePrompt: "",
  identityStructureMode: "angleSet",
  referenceRole: "angle-set-reference",
  referenceNoun: "angle-set reference",
  jobPredicate: () => false,
  title: () => "Turnaround",
  seedPrompt: () => "a prompt",
  renderIntro: () => null,
  useController: () => ({
    controls: null,
    advancedExtras: {},
    isReady: true,
    canSubmit: true,
    submitLabel: "Generate",
    expectedThumbnailCount: () => 0,
  }),
};

const character = { id: "char_1", name: "Ada", description: "a scientist" };

// Fresh reference objects each call so the array (and its element) identities differ
// on every "refetch", exactly as useCharacters produces after a mutation.
function refs(...assetIds) {
  return assetIds.map((assetId) => ({ assetId, role: "angle-set-reference", asset: null }));
}

// A stateful host that owns approvedReferences + selectedCharacter and passes an
// addCharacterReference that mutates that state — mirroring the App/useCharacters
// contract where a mutation triggers a refetch and hands the panel a new array. The
// ref exposes setters so a test can drive a refetch / character switch declaratively
// without a nested root.render (which pollutes React act state across tests).
function Host({ initialRefs, initialCharacter, importAsset, onAddCharacterReference, apiRef }) {
  const [approvedReferences, setApprovedReferences] = React.useState(initialRefs);
  const [selectedCharacter, setSelectedCharacter] = React.useState(initialCharacter);

  apiRef.current = {
    setApprovedReferences: (next) => setApprovedReferences(next),
    setSelectedCharacter: (next) => setSelectedCharacter(next),
  };

  const addCharacterReference = React.useCallback(
    async (characterId, payload) => {
      onAddCharacterReference?.(characterId, payload);
      // The refetch: fold the newly approved reference in under a fresh array identity.
      setApprovedReferences((current) => refs(...current.map((r) => r.assetId), payload.assetId));
      return {};
    },
    [onAddCharacterReference],
  );

  return (
    <CharacterGenerationPanel
      mode={TEST_MODE}
      selectedCharacter={selectedCharacter}
      model={MODEL}
      models={[MODEL]}
      catalog={[]}
      approvedReferences={approvedReferences}
      assets={[]}
      createImageJob={vi.fn()}
      importAsset={importAsset ?? vi.fn()}
      addCharacterReference={addCharacterReference}
      latestAssets={[]}
      imageLocalJobs={[]}
      loras={[]}
      rememberLocalGenerationJob={vi.fn()}
    />
  );
}

describe("CharacterGenerationPanel reference selection (sc-8851)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // Some shared children portal to document.body; render into a body-attached node.
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  async function mount(hostProps) {
    const apiRef = React.createRef();
    await act(async () => root.render(<Host {...hostProps} apiRef={apiRef} />));
    return apiRef;
  }

  // The active reference is the one whose thumbnail button is aria-pressed.
  function pickedId() {
    const active = container.querySelector('.reference-thumb[aria-pressed="true"]');
    return active?.getAttribute("aria-label")?.replace(/^Use /, "").replace(/ as the .*$/, "") ?? null;
  }

  function thumbFor(assetId) {
    return [...container.querySelectorAll(".reference-thumb")].find((button) =>
      button.getAttribute("aria-label")?.startsWith(`Use ${assetId} `),
    );
  }

  async function click(element) {
    await act(async () => {
      element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
  }

  it("defaults to the first approved reference on mount", async () => {
    await mount({ initialRefs: refs("ref_a", "ref_b", "ref_c"), initialCharacter: character });
    expect(pickedId()).toBe("ref_a");
  });

  it("preserves the picked id across a same-character refetch (new array identity, same contents)", async () => {
    const api = await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
    });

    // User picks reference #2.
    await click(thumbFor("ref_b"));
    expect(pickedId()).toBe("ref_b");

    // Character mutation -> useCharacters hands a brand-new approvedReferences array
    // with the same contents. Selection must NOT snap back to ref_a.
    await act(async () => api.current.setApprovedReferences(refs("ref_a", "ref_b", "ref_c")));
    expect(pickedId()).toBe("ref_b");
  });

  it("keeps a newly uploaded reference selected after the mutation-triggered refetch", async () => {
    // The panel's own upload flow end to end: importAsset resolves to a new asset, the
    // panel approves it via addCharacterReference (Host folds it into approvedReferences,
    // mimicking the refetch), and only then selects it. The pick must survive.
    const importAsset = vi.fn(async () => ({ id: "ref_new" }));
    const onAdd = vi.fn();
    await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
      importAsset,
      onAddCharacterReference: onAdd,
    });

    const fileInput = container.querySelector('input[type="file"]');
    const file = new File(["x"], "ref.png", { type: "image/png" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    await act(async () => {
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });

    expect(importAsset).toHaveBeenCalled();
    expect(onAdd).toHaveBeenCalledWith(
      "char_1",
      expect.objectContaining({ assetId: "ref_new", approved: true }),
    );
    // The upload-then-select flow survives the refetch: the panel ends up on the reference
    // the user just uploaded, not snapped back to reference #1.
    expect(pickedId()).toBe("ref_new");
  });

  it("falls back to [0] when the picked id disappears from approvedReferences", async () => {
    const api = await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
    });

    await click(thumbFor("ref_c"));
    expect(pickedId()).toBe("ref_c");

    // ref_c is removed (e.g. deleted/unapproved elsewhere). Fall back to the new [0].
    await act(async () => api.current.setApprovedReferences(refs("ref_b", "ref_a")));
    expect(pickedId()).toBe("ref_b");
  });

  it("resets to the new character's first reference on a genuine character switch", async () => {
    const api = await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
    });

    await click(thumbFor("ref_c"));
    expect(pickedId()).toBe("ref_c");

    // Switch to a different character whose references share none of the previous ids.
    await act(async () => {
      api.current.setSelectedCharacter({ id: "char_2", name: "Bly", description: "an artist" });
      api.current.setApprovedReferences(refs("ref_x", "ref_y"));
    });
    expect(pickedId()).toBe("ref_x");
  });
});
