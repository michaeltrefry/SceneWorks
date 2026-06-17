import React from "react";
import { Icon } from "./components/Icons.jsx";

// View metadata lives outside App so simple mode can swap or filter routes
// without changing the stateful shell that owns data loading and providers.
function lazyScreen(importer, exportName) {
  return React.lazy(() => importer().then((module) => ({ default: module[exportName] })));
}

const loadTrainingScreens = () => import("./screens/TrainingStudio.jsx");

const LibraryScreen = lazyScreen(() => import("./screens/LibraryScreen.jsx"), "LibraryScreen");
const PoseLibraryScreen = lazyScreen(() => import("./screens/PoseLibraryScreen.jsx"), "PoseLibraryScreen");
const KeyPointLibraryScreen = lazyScreen(() => import("./screens/KeyPointLibraryScreen.jsx"), "KeyPointLibraryScreen");
const ModelManagerScreen = lazyScreen(() => import("./screens/ModelManagerScreen.jsx"), "ModelManagerScreen");
const ImageStudio = lazyScreen(() => import("./screens/ImageStudio.jsx"), "ImageStudio");
const DocumentStudio = lazyScreen(() => import("./screens/DocumentStudio.jsx"), "DocumentStudio");
const VideoStudio = lazyScreen(() => import("./screens/VideoStudio.jsx"), "VideoStudio");
const TrainingDataSetsLibrary = lazyScreen(loadTrainingScreens, "TrainingDataSetsLibrary");
const TrainingStudio = lazyScreen(loadTrainingScreens, "TrainingStudio");
const CharacterStudio = lazyScreen(() => import("./screens/CharacterStudio.jsx"), "CharacterStudio");
const EditorScreen = lazyScreen(() => import("./screens/EditorScreen.jsx"), "EditorScreen");
const ImageEditor = lazyScreen(() => import("./screens/ImageEditor.jsx"), "ImageEditor");
const QueueScreen = lazyScreen(() => import("./screens/QueueScreen.jsx"), "QueueScreen");
const PresetManagerScreen = lazyScreen(() => import("./screens/PresetManagerScreen.jsx"), "PresetManagerScreen");
const SettingsScreen = lazyScreen(() => import("./screens/SettingsScreen.jsx"), "SettingsScreen");
const LogsScreen = lazyScreen(() => import("./screens/LogsScreen.jsx"), "LogsScreen");
const LicensesScreen = lazyScreen(() => import("./screens/LicensesScreen.jsx"), "LicensesScreen");
const MakePicture = lazyScreen(() => import("./screens/simple/MakePicture.jsx"), "MakePicture");
const MakeVideo = lazyScreen(() => import("./screens/simple/MakeVideo.jsx"), "MakeVideo");
const MyCreations = lazyScreen(() => import("./screens/simple/MyCreations.jsx"), "MyCreations");
const SimpleCharacters = lazyScreen(() => import("./screens/simple/Characters.jsx"), "Characters");
const SimpleSettings = lazyScreen(() => import("./screens/simple/SimpleSettings.jsx"), "SimpleSettings");

export function RouteFallback({ label = "Loading view…" } = {}) {
  return <section className="main-surface">{label}</section>;
}

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

export const simpleNavSections = [
  {
    label: "Create",
    items: [
      { id: "MakePicture", label: "Make a picture", icon: Icon.Image },
      { id: "MakeVideo", label: "Make a video", icon: Icon.Video },
      { id: "SimpleCharacters", label: "Characters", icon: Icon.Character },
    ],
  },
  {
    label: "Yours",
    items: [
      { id: "MyCreations", label: "My creations", icon: Icon.Library },
      { id: "Queue", label: "In progress", icon: Icon.Queue },
      { id: "SimpleSettings", label: "Settings", icon: Icon.Sliders },
    ],
  },
];

const viewRegistry = {
  MakePicture: {
    title: "Make a picture",
    blurb: "Describe an idea — pick a look if you like — and we'll render a few options.",
    render: ({ activeProjectId } = {}) => <MakePicture key={activeProjectId ?? "default"} />,
  },
  MakeVideo: {
    title: "Make a video",
    blurb: "Start from a description or bring a picture to life.",
    render: ({ activeProjectId } = {}) => <MakeVideo key={activeProjectId ?? "default"} />,
  },
  SimpleCharacters: {
    title: "Characters",
    blurb: "Save a person or character once, then keep the same face across everything.",
    render: ({ activeProjectId } = {}) => <SimpleCharacters key={activeProjectId ?? "default"} />,
  },
  MyCreations: {
    title: "My creations",
    blurb: "Everything you've made in this workspace.",
    render: ({ activeProjectId } = {}) => <MyCreations key={activeProjectId ?? "default"} />,
  },
  SimpleSettings: {
    title: "Settings",
    blurb: "Switch modes and add the models you want to use.",
    render: () => <SimpleSettings />,
  },
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
    render: ({ activeProjectId }) => <ImageEditor key={activeProjectId ?? "default"} />,
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
    render: ({ uiMode, onUiModeChange } = {}) => <SettingsScreen uiMode={uiMode} onUiModeChange={onUiModeChange} />,
  },
  Licenses: {
    title: "Licenses",
    blurb: "Third-party components bundled with SceneWorks and their license notices.",
    render: () => <LicensesScreen />,
  },
};

const simpleVisibleViews = new Set(simpleNavSections.flatMap((section) => section.items.map((item) => item.id)));
const advancedVisibleViews = new Set(navSections.flatMap((section) => section.items.map((item) => item.id)));

export function getNavigationSections(uiMode) {
  return uiMode === "simple" ? simpleNavSections : navSections;
}

export function isViewVisibleInMode(viewId, uiMode) {
  return uiMode === "simple" ? simpleVisibleViews.has(viewId) : advancedVisibleViews.has(viewId);
}

export function getInitialViewForMode(uiMode) {
  return uiMode === "simple" ? "MakePicture" : "Library";
}

export function coerceViewForMode(viewId, uiMode) {
  return isViewVisibleInMode(viewId, uiMode) ? viewId : getInitialViewForMode(uiMode);
}

export function getViewTitle(viewId) {
  return viewRegistry[viewId] ?? { title: viewId, blurb: "" };
}

export function renderActiveView(viewId, options) {
  const view = viewRegistry[viewId];
  if (!view) {
    return null;
  }
  return <React.Suspense fallback={<RouteFallback />}>{view.render(options)}</React.Suspense>;
}
