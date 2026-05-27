import React, { useState } from "react";
import { Icon } from "./Icons.jsx";

// "Refine my prompt" affordance shared by Image and Video Studio (sc-2041).
// Sends the current prompt + the selected model's guide to the refinement worker
// (via the context `refinePrompt`), then shows the rewrite for review. The
// original prompt is never changed until the user clicks Apply.
export function RefinePromptControl({ prompt, guidePath, modelId, workflow, refinePrompt, onApply }) {
  const [status, setStatus] = useState("idle"); // idle | loading | review | error
  const [refined, setRefined] = useState("");
  const [error, setError] = useState("");

  const trimmed = (prompt ?? "").trim();
  const busy = status === "loading";
  const disabled = busy || !trimmed || typeof refinePrompt !== "function";

  async function handleRefine() {
    setStatus("loading");
    setError("");
    try {
      // The guide is first-party context for the rewrite; fetch it best-effort and
      // refine generically if it can't be loaded.
      let guide = "";
      if (guidePath) {
        try {
          const response = await fetch(guidePath);
          if (response.ok) guide = await response.text();
        } catch {
          guide = "";
        }
      }
      const result = await refinePrompt({ prompt: trimmed, modelId, workflow, guide });
      setRefined(result);
      setStatus("review");
    } catch (err) {
      setError(err?.message || "Prompt refinement failed.");
      setStatus("error");
    }
  }

  return (
    <div className="refine-control">
      <button className="hero-link refine-button" disabled={disabled} onClick={handleRefine} type="button">
        <Icon.Wand size={14} /> {busy ? "Refining…" : "Refine my prompt"}
      </button>

      {status === "error" ? (
        <p className="refine-error" role="alert">
          {error}
        </p>
      ) : null}

      {status === "review" ? (
        <div className="refine-review">
          <p className="refine-review-label">Suggested rewrite</p>
          <p className="refine-review-text">{refined}</p>
          <div className="refine-review-actions">
            <button
              className="secondary-action"
              onClick={() => {
                onApply(refined);
                setStatus("idle");
              }}
              type="button"
            >
              Apply
            </button>
            <button className="secondary-action" onClick={() => setStatus("idle")} type="button">
              Keep original
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
