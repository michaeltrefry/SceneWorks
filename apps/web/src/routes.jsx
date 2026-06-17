import React from "react";
import { Icon } from "./components/Icons.jsx";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { PoseLibraryScreen } from "./screens/PoseLibraryScreen.jsx";
import { KeyPointLibraryScreen } from "./screens/KeyPointLibraryScreen.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { DocumentStudio } from "./screens/DocumentStudio.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { TrainingDataSetsLibrary, TrainingStudio } from "./screens/TrainingStudio.jsx";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { EditorScreen } from "./screens/EditorScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { SettingsScreen } from "./screens/SettingsScreen.jsx";
import { LogsScreen } from "./screens/LogsScreen.jsx";
import { LicensesScreen } from "./screens/LicensesScreen.jsx";

// View metadata lives outside App so simple mode can swap or filter routes
// without changing the stateful shell that owns data loading and providers.
const ImageEditor = React.lazy(() =>
  import("./screens/ImageEditor.jsx").then((module) => ({ default: module.ImageEditor })),
);

export const navSections = [
  {
    label: "Workspace",
    items: [
      { id: "Image", icon: Icon.Image },
      { id: "Video", icon: Icon.Video },
      // Character Studio is a generative studio (sc-2300) — it sits with Image/Video,
      // below Video and above Training, not in the Library section.
      { id: "Characters", icon: Icon.Character },
      { id: "Document", icon: Icon.Wand },
      { id: "Train", icon: Icon.Train },
      { id: "ImageEditor", label: "Image Editor", icon: Icon.ImageEditor },
      { id: "Editor", label: "Video Editor", icon: Icon.Editor },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "Library", label: "Assets", icon: Icon.Library },
      { id: "LibraryDataSets", label: "Data Sets", icon: Icon.Train },
      { id: "Poses", label: "Pose Library", icon: Icon.Character },
      { id: "Keypoints", label: "Key Point Library", icon: Icon.Character },
      { id: "Presets", icon: Icon.Preset },
      { id: "Models", icon: Icon.Model },
    ],
  },
  {
    label: "System",
    items: [
      { id: "Queue", icon: Icon.Queue },
      { id: "Logs", icon: Icon.Logs },
      { id: "Settings", icon: Icon.Sliders },
      { id: "Licenses", icon: Icon.Info },
    ],
  },
];

const viewRegistry = {
  Library: {
    title: "Assets",
    blurb: "Browse stills and clips across all your projects.",
    render: () => <LibraryScreen />,
  },
  LibraryDataSets: {
    title: "Data Sets",
    blurb: "Create and caption training datasets.",
    render: () => <TrainingDataSetsLibrary />,
  },
  Poses: {
    title: "Pose Library",
    blurb: "Manage whole-body pose skeletons and create new ones from photos.",
    render: () => <PoseLibraryScreen />,
  },
  Keypoints: {
    title: "Key Point Library",
    blurb: "Capture face-angle framing presets and compose angle-set collections for character turnarounds.",
    render: () => <KeyPointLibraryScreen />,
  },
  Image: {
    title: "Image Studio",
    blurb: "Describe what you want — we'll render variations side by side.",
    render: ({ activeProjectId }) => <ImageStudio key={activeProjectId ?? "default"} />,
  },
  Video: {
    title: "Video Studio",
    blurb: "Bring stills to life, or render new clips from scratch.",
    render: ({ activeProjectId }) => <VideoStudio key={activeProjectId ?? "default"} />,
  },
  Document: {
    title: "Document Studio",
    blurb: "Generate interleaved text-image documents — guides, storyboards, tutorials.",
    render: () => <DocumentStudio />,
  },
  Train: {
    title: "Training Studio",
    blurb: "Build datasets and prepare LoRA training plans.",
    render: () => <TrainingStudio />,
  },
  Editor: {
    title: "Video Editor",
    blurb: "Cut, sequence and export your timeline.",
    render: () => <EditorScreen />,
  },
  ImageEditor: {
    title: "Image Editor",
    blurb: "Crop, upscale and refine a single image on a canvas.",
    render: ({ activeProjectId }) => (
      <React.Suspense fallback={<section className="main-surface">Loading editor…</section>}>
        <ImageEditor key={activeProjectId ?? "default"} />
      </React.Suspense>
    ),
  },
  Characters: {
    title: "Characters",
    blurb: "Keep the same face across every shot.",
    render: ({ activeProjectId }) => <CharacterStudio key={activeProjectId ?? "default"} />,
  },
  Presets: {
    title: "Presets",
    blurb: "Save and share recurring generation setups.",
    render: () => <PresetManagerScreen />,
  },
  Models: {
    title: "Models",
    blurb: "Download, import and manage local checkpoints.",
    render: () => <ModelManagerScreen />,
  },
  Queue: {
    title: "Queue",
    blurb: "All running and recent jobs across workers.",
    render: () => <QueueScreen />,
  },
  Logs: {
    title: "Logs",
    blurb: "This session's activity — routing decisions, worker phases and errors.",
    render: () => <LogsScreen />,
  },
  Settings: {
    title: "Settings",
    blurb: "Paths, service tokens, and detected GPU.",
    render: () => <SettingsScreen />,
  },
  Licenses: {
    title: "Licenses",
    blurb: "Third-party components bundled with SceneWorks and their license notices.",
    render: () => <LicensesScreen />,
  },
};

export function getViewTitle(viewId) {
  return viewRegistry[viewId] ?? { title: viewId, blurb: "" };
}

export function renderActiveView(viewId, options) {
  return viewRegistry[viewId]?.render(options) ?? null;
}
