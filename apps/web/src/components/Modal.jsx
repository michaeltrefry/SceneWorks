import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

// Shared modal primitive: a backdrop that closes on outside mousedown, a
// role="dialog" container that closes on Escape, and focus moved into the
// dialog on mount. Extracted so every overlay behaves consistently for
// keyboard and pointer users.
//
// The overlay is portaled to `document.body` rather than rendered inline at the
// call site. A `position: fixed` backdrop only anchors to the viewport when no
// ancestor establishes a containing block; any ancestor with `transform`,
// `filter`, `backdrop-filter`, `contain`, `will-change`, or `perspective` traps
// it inside that ancestor's box, so an inline modal ends up "contained in the
// parent" instead of covering the screen. Portaling to `body` makes the backdrop
// a direct child of `body`, immune to any such ancestor. React portals preserve
// synthetic event bubbling, so the Escape / backdrop-click / focus handlers
// below keep working exactly as before.
export function Modal({ children, onClose, className, labelledBy, label }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  const overlay = (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        aria-label={label}
        aria-labelledby={labelledBy}
        aria-modal="true"
        className={className}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            onClose();
          }
        }}
        onMouseDown={(event) => event.stopPropagation()}
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        {children}
      </div>
    </div>
  );

  // Guard for non-DOM environments; in the browser and jsdom `document.body`
  // always exists by render time.
  if (typeof document === "undefined") return overlay;
  return createPortal(overlay, document.body);
}
