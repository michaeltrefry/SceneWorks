// sc-4194 regression: the data hooks must return useCallback-stable action
// identities across re-renders that don't change their inputs. Before the fix every
// action was an inline function recreated each render, so appContextValue (which
// spreads them) changed identity on every SSE-driven App re-render and defeated
// memoization across the whole consumer tree. If a hook action is ever reverted to a
// plain inline function, its identity will change on the forced re-render below and
// this test fails.
import React, { useEffect, useRef, useState } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { usePresets } from "./usePresets.js";
import { usePersonTracks } from "./usePersonTracks.js";
import { useCharacters } from "./useCharacters.js";
import { useModelsAndLoras } from "./useModelsAndLoras.js";
import { useTraining } from "./useTraining.js";
import { useTimelines } from "./useTimelines.js";

const ACTION_KEYS = {
  usePresets: ["refreshPresets", "createPreset", "updatePreset", "duplicatePreset", "deletePreset"],
  usePersonTracks: ["refreshPersonTracks", "createPersonDetectionJob", "createPersonTrackJob", "saveTrackCorrections"],
  useCharacters: [
    "refreshCharacters", "createCharacter", "updateCharacter", "archiveCharacter",
    "unarchiveCharacter", "listArchivedCharacters",
    "addCharacterReference", "createCharacterLook", "attachCharacterLora", "createCharacterTestJob",
  ],
  // sc-8811: deleteModel/deleteLora depend on the App-provided refreshData /
  // refreshDataWithLoraOverlay props, which App must pass identity-stable
  // (ref-delegating useCallbacks). With the stable stand-ins below, every action
  // must hold its identity across an unrelated re-render.
  useModelsAndLoras: [
    "refreshLoras", "deleteModel", "deleteLora", "createModelImportJob", "createLoraImportJob",
    "createModelDownloadJob", "createLoraDownloadJob", "createModelConvertJob",
  ],
  useTraining: [
    "refreshTrainingDatasets", "loadTrainingDataset", "loadTrainingDatasetReadiness",
    "setTrainingDatasetItemQualityAck", "createTrainingDataset", "uploadTrainingDatasetItem",
    "updateTrainingDataset", "batchRenameTrainingDataset", "writeTrainingDatasetCaptionSidecars",
    "createTrainingDatasetCaptionJob", "createTrainingDatasetUpscaleJob",
    "createTrainingDatasetAnalysisJob", "createTrainingDatasetFaceAnalysisJob",
    "smartCropTrainingDataset", "stripExifTrainingDataset", "createTrainingJob",
  ],
  useTimelines: [
    "refreshTimelines", "createTimeline", "saveTimeline", "exportTimeline",
    "extractTimelineFrame", "queueTimelineVideoJob",
  ],
};

async function settle() {
  await act(async () => {
    for (let i = 0; i < 4; i += 1) {
      await Promise.resolve();
    }
  });
}

describe("data hook action identity stability (sc-4194)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  // Renders `useHook` with fixed props, forces one re-render via an internal counter,
  // and reports the action identities captured on render #1 and render #2.
  function probe(useHook, actionKeys) {
    const captures = [];
    let bump = () => {};

    // Stable across renders so the only thing changing on the forced re-render is the
    // counter — mirrors how App feeds these hooks identity-stable token/project/setters
    // while jobs churn. If the actions still change identity, it's the hook's fault.
    // A superset of every hook's props: each hook destructures only what it takes.
    const activeProject = { id: "p1", name: "P1" };
    const stableProps = {
      token: "tok",
      activeProject,
      activeProjectRef: { current: activeProject },
      setError: () => {},
      setJobs: () => {},
      requestedGpu: "auto",
      setActiveView: () => {},
      createVideoJob: async () => null,
      refreshData: async () => {},
      refreshDataWithLoraOverlay: async () => {},
    };

    function Harness() {
      const [, setN] = useState(0);
      bump = () => setN((n) => n + 1);
      const api = useHook(stableProps);
      const snapshot = {};
      actionKeys.forEach((key) => {
        snapshot[key] = api[key];
      });
      const ref = useRef(null);
      // Record one snapshot per commit.
      useEffect(() => {
        ref.current = snapshot;
        captures.push(snapshot);
      });
      return null;
    }

    return { Harness, captures, bumpRef: () => bump() };
  }

  it.each(Object.entries(ACTION_KEYS).map(([name]) => name))(
    "%s returns stable action identities across an unrelated re-render",
    async (hookName) => {
      const hooks = { usePresets, usePersonTracks, useCharacters, useModelsAndLoras, useTraining, useTimelines };
      const { Harness, captures, bumpRef } = probe(hooks[hookName], ACTION_KEYS[hookName]);

      root = createRoot(container);
      await act(async () => {
        root.render(<Harness />);
      });
      await settle();

      await act(async () => {
        bumpRef();
      });
      await settle();

      expect(captures.length).toBeGreaterThanOrEqual(2);
      const first = captures[0];
      const last = captures[captures.length - 1];
      for (const key of ACTION_KEYS[hookName]) {
        expect(typeof last[key]).toBe("function");
        expect(last[key]).toBe(first[key]); // identity preserved across re-render
      }
    },
  );
});
