import React from "react";

export function PlaceholderSurface({ activeView, assets, createJob }) {
  return (
    <section className="main-surface">
      <div className="section-heading">
        <p className="eyebrow">{activeView}</p>
        <h2>{activeView}</h2>
      </div>
      <form className="job-composer compact" onSubmit={createJob}>
        <label htmlFor="surface-job-prompt">Prompt</label>
        <input id="surface-job-prompt" defaultValue={`${activeView} placeholder`} />
        <button type="submit">Start job</button>
      </form>
      <div className="media-grid" aria-label={`${activeView} assets`}>
        <div className="media-tile wide">
          <span>{assets.length} assets</span>
        </div>
        <div className="media-tile accent">
          <span>{assets.filter((asset) => asset.status?.favorite).length} favorites</span>
        </div>
        <div className="media-tile warm">
          <span>{assets.filter((asset) => asset.type === "image").length} images</span>
        </div>
      </div>
    </section>
  );
}
