import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  CharacterAdvancedOptions,
  useCharacterAdvancedOptions,
} from "./CharacterAdvancedOptions.jsx";

// The Angles/Poses PiD toggle (epic 7840, sc-8372): it mirrors the Image Studio checkbox and is
// gated on `pidToggleVisible(model, catalog)` — shown only when the active model declares a PiD
// backbone (`ui.pid.checkpointId`) AND that checkpoint is installed in the catalog. When shown and
// checked it folds `usePid: true` into the advanced payload; otherwise the key is absent.

const instantId = { id: "instantid_realvisxl", ui: { pid: { checkpointId: "pid_sdxl" } } };
const nonEligible = { id: "kolors", ui: {} };
const installed = [{ id: "pid_sdxl", installState: "installed" }];
const missing = [{ id: "pid_sdxl", installState: "missing" }];

// Hosts the hook + presentational panel, opens the advanced section, and renders the current
// `buildAdvanced()` output so the test can assert what the job payload would carry.
function Harness({ model, catalog }) {
  const state = useCharacterAdvancedOptions(model, { catalog });
  React.useEffect(() => state.setOpen(true), []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      <CharacterAdvancedOptions state={state} />
      <output data-testid="adv">{JSON.stringify(state.buildAdvanced())}</output>
    </>
  );
}

describe("CharacterAdvancedOptions PiD toggle (sc-8372)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  const pidCheckbox = () => container.querySelector(".pid-decoder-toggle input[type=checkbox]");
  const advanced = () => JSON.parse(container.querySelector('[data-testid="adv"]').textContent);

  it("hides the toggle when the model has no PiD backbone", async () => {
    await act(async () => root.render(<Harness model={nonEligible} catalog={installed} />));
    expect(pidCheckbox()).toBe(null);
    expect(advanced().usePid).toBeUndefined();
  });

  it("hides the toggle when the PiD checkpoint is not installed", async () => {
    await act(async () => root.render(<Harness model={instantId} catalog={missing} />));
    expect(pidCheckbox()).toBe(null);
    expect(advanced().usePid).toBeUndefined();
  });

  it("shows the toggle and emits advanced.usePid when eligible, installed, and checked", async () => {
    await act(async () => root.render(<Harness model={instantId} catalog={installed} />));
    const box = pidCheckbox();
    expect(box).not.toBe(null);
    // Off by default → no usePid in the payload.
    expect(advanced().usePid).toBeUndefined();
    // Checking it folds usePid:true in.
    await act(async () => box.click());
    expect(advanced().usePid).toBe(true);
  });
});
