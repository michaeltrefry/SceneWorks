import React from "react";

export function StatusDot({ ok }) {
  return <span className={ok ? "status-dot ok" : "status-dot"} aria-hidden="true" />;
}
