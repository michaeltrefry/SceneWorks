import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import "./styles.css";

const rootElement = typeof document === "undefined" ? null : document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}

export { App } from "./App.jsx";
export { eventUrl } from "./api.js";
