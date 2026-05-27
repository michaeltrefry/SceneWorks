import React, { useEffect, useState } from "react";
import { Modal } from "./Modal.jsx";
import { Markdown } from "./Markdown.jsx";

// Fetches a model's static Markdown prompt guide (served from /public/prompt-guides)
// and renders it inside the shared Modal. `guide` is the resolved metadata
// ({ title, path, sources? }); callers pass a generic fallback when the selected
// model declares none.
export function PromptGuideModal({ guide, modelName, onClose }) {
  const [state, setState] = useState({ status: "loading", content: "" });

  useEffect(() => {
    let active = true;
    setState({ status: "loading", content: "" });
    fetch(guide.path)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((text) => {
        if (active) setState({ status: "ready", content: text });
      })
      .catch(() => {
        if (active) setState({ status: "error", content: "" });
      });
    return () => {
      active = false;
    };
  }, [guide.path]);

  return (
    <Modal className="prompt-guide-modal" labelledBy="prompt-guide-title" onClose={onClose}>
      <header className="prompt-guide-head">
        <div>
          <p className="eyebrow">{modelName ? `${modelName} · Prompt guide` : "Prompt guide"}</p>
          <h2 id="prompt-guide-title">{guide.title}</h2>
        </div>
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
      </header>

      <div className="prompt-guide-body">
        {state.status === "loading" ? <p className="prompt-guide-status">Loading guide…</p> : null}
        {state.status === "error" ? (
          <p className="prompt-guide-status">This guide could not be loaded. Please try again.</p>
        ) : null}
        {state.status === "ready" ? <Markdown content={state.content} /> : null}
      </div>

      {guide.sources?.length ? (
        <footer className="prompt-guide-sources">
          <span>Sources:</span>
          {guide.sources.map((source) => (
            <a key={source.url} href={source.url} target="_blank" rel="noopener noreferrer">
              {source.label}
            </a>
          ))}
        </footer>
      ) : null}
    </Modal>
  );
}
