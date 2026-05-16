import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const navItems = [
  "Library",
  "Image",
  "Video",
  "Characters",
  "Editor",
  "Queue",
];

const placeholders = {
  Library: {
    title: "Library",
    eyebrow: "Project assets",
    body: "Imported and generated media will appear here with recipe metadata, ratings, lineage, and reuse actions.",
  },
  Image: {
    title: "Image Studio",
    eyebrow: "Text, edit, character, variations",
    body: "Prompt-first image workflows land here, with model details tucked into an advanced drawer.",
  },
  Video: {
    title: "Video Studio",
    eyebrow: "Short generated shots",
    body: "Image-to-video, text-to-video, first/last frame, extend, and replacement modes will share this surface.",
  },
  Characters: {
    title: "Character Studio",
    eyebrow: "Reusable identities",
    body: "Characters will gather references, looks, and project LoRAs before full training arrives.",
  },
  Editor: {
    title: "Editor",
    eyebrow: "Timeline assembly",
    body: "Generated and imported clips will be arranged, trimmed, bridged, and exported from here.",
  },
  Queue: {
    title: "Queue",
    eyebrow: "Jobs and GPUs",
    body: "Generation, downloads, exports, tracking, and replacement jobs will show progress here.",
  },
};

async function apiFetch(path, token, options = {}) {
  const headers = new Headers(options.headers ?? {});
  headers.set("Content-Type", "application/json");
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed with ${response.status}`);
  }

  return response.json();
}

function StatusDot({ ok }) {
  return <span className={ok ? "status-dot ok" : "status-dot"} aria-hidden="true" />;
}

function App() {
  const [health, setHealth] = useState(null);
  const [access, setAccess] = useState({ authRequired: false });
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  const [projects, setProjects] = useState([]);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [projectName, setProjectName] = useState("");
  const [error, setError] = useState("");

  const authenticated = useMemo(() => !access.authRequired || token.length > 0, [access, token]);

  useEffect(() => {
    apiFetch("/api/v1/health", "")
      .then(setHealth)
      .catch((err) => setError(err.message));

    apiFetch("/api/v1/access", "")
      .then(setAccess)
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!authenticated) {
      return;
    }

    apiFetch("/api/v1/projects", token)
      .then((items) => {
        setProjects(items);
        setActiveProject((current) => current ?? items[0] ?? null);
      })
      .catch((err) => setError(err.message));
  }, [authenticated, token]);

  function saveToken(event) {
    event.preventDefault();
    window.localStorage.setItem("sceneworks-token", token);
    setError("");
  }

  async function createProject(event) {
    event.preventDefault();
    if (!projectName.trim()) {
      return;
    }

    try {
      const created = await apiFetch("/api/v1/projects", token, {
        method: "POST",
        body: JSON.stringify({ name: projectName }),
      });
      setProjects((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      setActiveProject(created);
      setProjectName("");
      setActiveView("Library");
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  const view = placeholders[activeView];

  return (
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark">SW</span>
          <div>
            <h1>SceneWorks</h1>
            <p>Local creative studio</p>
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              className={activeView === item ? "nav-item active" : "nav-item"}
              key={item}
              onClick={() => setActiveView(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Project</p>
            <strong>{activeProject?.name ?? "No project open"}</strong>
          </div>
          <div className="topbar-status">
            <span>
              <StatusDot ok={health?.status === "ok"} />
              API
            </span>
            <span>GPU auto</span>
            <span>Queue idle</span>
          </div>
        </header>

        {error ? <p className="notice error">{error}</p> : null}

        {access.authRequired && !window.localStorage.getItem("sceneworks-token") ? (
          <section className="auth-band">
            <form onSubmit={saveToken}>
              <label htmlFor="token">Pairing token</label>
              <div className="form-row">
                <input
                  id="token"
                  onChange={(event) => setToken(event.target.value)}
                  placeholder="Enter local token"
                  type="password"
                  value={token}
                />
                <button type="submit">Unlock</button>
              </div>
            </form>
          </section>
        ) : null}

        <section className="project-band">
          <div className="project-list">
            <div className="section-heading">
              <p className="eyebrow">Recent projects</p>
              <h2>Open a workspace</h2>
            </div>
            <div className="project-buttons">
              {projects.length === 0 ? (
                <span className="empty-state">Create the first project to start the Library spine.</span>
              ) : (
                projects.map((project) => (
                  <button
                    className={activeProject?.id === project.id ? "project-pill active" : "project-pill"}
                    key={project.id}
                    onClick={() => setActiveProject(project)}
                    type="button"
                  >
                    {project.name}
                  </button>
                ))
              )}
            </div>
          </div>

          <form className="create-project" onSubmit={createProject}>
            <label htmlFor="project-name">New project</label>
            <div className="form-row">
              <input
                id="project-name"
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="Noir Alley"
                value={projectName}
              />
              <button disabled={!authenticated} type="submit">
                Create
              </button>
            </div>
          </form>
        </section>

        <section className="main-surface">
          <div className="section-heading">
            <p className="eyebrow">{view.eyebrow}</p>
            <h2>{view.title}</h2>
          </div>
          <p className="view-copy">{view.body}</p>

          <div className="media-grid" aria-label={`${view.title} preview placeholders`}>
            <div className="media-tile wide">
              <span>{activeProject ? activeProject.name : "Create a project"}</span>
            </div>
            <div className="media-tile accent">
              <span>Recipes</span>
            </div>
            <div className="media-tile warm">
              <span>Assets</span>
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
