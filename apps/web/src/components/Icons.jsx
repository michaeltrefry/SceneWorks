import React from "react";

function I({ d, fill = false, size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      {fill ? <path d={d} fill="currentColor" stroke="none" /> : <path d={d} />}
    </svg>
  );
}

export const Icon = {
  Library: (p) => <I {...p} d="M4 5h6v14H4zM14 5h6v14h-6zM10 9h4M10 13h4M10 17h4" />,
  Image: (p) => <I {...p} d="M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4" />,
  Video: (p) => <I {...p} d="M3 6h12v12H3zM15 9l6-3v12l-6-3z" />,
  Editor: (p) => <I {...p} d="M3 8h18M3 12h12M3 16h18M16 10l4 2-4 2z" />,
  Character: (p) => <I {...p} d="M12 12a4 4 0 100-8 4 4 0 000 8zM4 20a8 8 0 0116 0" />,
  Preset: (p) => <I {...p} d="M4 7h16M7 12h10M10 17h4" />,
  Model: (p) => <I {...p} d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 3v9M12 12l8-4.5M12 12l-8-4.5" />,
  Queue: (p) => <I {...p} d="M4 7h16M4 12h16M4 17h10" />,
  Search: (p) => <I {...p} d="M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4.3-4.3" />,
  Sparkle: (p) => <I {...p} d="M12 4l1.6 4.4L18 10l-4.4 1.6L12 16l-1.6-4.4L6 10l4.4-1.6zM18 4l.7 1.8L20.5 6.5l-1.8.7L18 9l-.7-1.8L15.5 6.5l1.8-.7z" />,
  Plus: (p) => <I {...p} d="M12 5v14M5 12h14" />,
  Sun: (p) => <I {...p} d="M12 4v2M12 18v2M4 12H2M22 12h-2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4l1.4-1.4M17 7l1.4-1.4M12 8a4 4 0 100 8 4 4 0 000-8z" />,
  Moon: (p) => <I {...p} d="M21 13.5A9 9 0 1110.5 3a7 7 0 0010.5 10.5z" />,
  Bell: (p) => <I {...p} d="M6 16V11a6 6 0 0112 0v5l2 2H4zM10 20a2 2 0 004 0" />,
  Folder: (p) => <I {...p} d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />,
  ChevDown: (p) => <I {...p} d="M6 9l6 6 6-6" />,
  Sliders: (p) => <I {...p} d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h14M18 18h2M14 4v4M6 10v4M16 16v4" />,
  Play: (p) => <I {...p} fill d="M7 4l13 8-13 8z" />,
  ArrowLeft: (p) => <I {...p} d="M19 12H5M11 6l-6 6 6 6" />,
  ArrowRight: (p) => <I {...p} d="M5 12h14M13 6l6 6-6 6" />,
  Wand: (p) => <I {...p} d="M15 4l5 5L9 20l-5-5zM14 5l5 5M3 14l3 3" />,
  Stars: (p) => <I {...p} fill d="M12 2l2.4 7.4H22l-6.2 4.5L18 21l-6-4.4L6 21l2.2-7.1L2 9.4h7.6z" />,
  Trash: (p) => <I {...p} d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />,
};

