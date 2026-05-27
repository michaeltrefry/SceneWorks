import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Markdown } from "./Markdown.jsx";
import { PromptGuideModal } from "./PromptGuideModal.jsx";

async function settle() {
  await act(async () => {
    for (let index = 0; index < 6; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("Markdown renderer", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function render(ui) {
    root = createRoot(container);
    act(() => root.render(ui));
  }

  it("renders headings, lists, bold, inline code, and safe links", () => {
    render(
      <Markdown
        content={[
          "# Title",
          "## Section",
          "A paragraph with **bold**, `code`, and a [link](https://example.com/guide).",
          "",
          "- first item",
          "- second item",
        ].join("\n")}
      />,
    );

    expect(container.querySelector("h1").textContent).toBe("Title");
    expect(container.querySelector("h2").textContent).toBe("Section");
    expect(container.querySelector("strong").textContent).toBe("bold");
    expect(container.querySelector("code").textContent).toBe("code");
    const link = container.querySelector("a");
    expect(link.getAttribute("href")).toBe("https://example.com/guide");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    expect([...container.querySelectorAll("li")].map((li) => li.textContent)).toEqual(["first item", "second item"]);
  });

  it("renders ordered lists and fenced code blocks", () => {
    render(<Markdown content={["1. step one", "2. step two", "", "```", "raw code line", "```"].join("\n")} />);
    expect(container.querySelector("ol")).not.toBeNull();
    expect([...container.querySelectorAll("ol li")].map((li) => li.textContent)).toEqual(["step one", "step two"]);
    expect(container.querySelector("pre code").textContent).toBe("raw code line");
  });

  it("drops unsafe link hrefs instead of rendering a javascript: anchor", () => {
    render(<Markdown content={"Click [here](javascript:alert(1)) now."} />);
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("here");
  });
});

describe("PromptGuideModal", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("fetches the guide path and renders its markdown with the model name", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "# Z-Image Guide\n\nUse short prompts." }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PromptGuideModal
          guide={{ title: "Z-Image-Turbo Prompt Guide", path: "/prompt-guides/z-image-turbo.md" }}
          modelName="Z-Image-Turbo"
          onClose={() => {}}
        />,
      );
    });
    await settle();

    expect(global.fetch).toHaveBeenCalledWith("/prompt-guides/z-image-turbo.md");
    expect(container.querySelector("#prompt-guide-title").textContent).toBe("Z-Image-Turbo Prompt Guide");
    expect(container.textContent).toContain("Z-Image-Turbo · Prompt guide");
    expect(container.querySelector(".markdown-body h1").textContent).toBe("Z-Image Guide");
    expect(container.textContent).toContain("Use short prompts.");
  });

  it("renders metadata source links when provided", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "Body." }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PromptGuideModal
          guide={{
            title: "Guide",
            path: "/prompt-guides/x.md",
            sources: [{ label: "Model card", url: "https://example.com/card" }],
          }}
          onClose={() => {}}
        />,
      );
    });
    await settle();

    const source = container.querySelector(".prompt-guide-sources a");
    expect(source.textContent).toBe("Model card");
    expect(source.getAttribute("href")).toBe("https://example.com/card");
  });

  it("shows an error message when the guide cannot be loaded", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 404, text: async () => "" }));
    root = createRoot(container);
    await act(async () => {
      root.render(<PromptGuideModal guide={{ title: "Guide", path: "/prompt-guides/missing.md" }} onClose={() => {}} />);
    });
    await settle();

    expect(container.textContent).toContain("could not be loaded");
    expect(container.querySelector(".markdown-body")).toBeNull();
  });

  it("closes on Escape", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "Body." }));
    const onClose = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(<PromptGuideModal guide={{ title: "Guide", path: "/prompt-guides/x.md" }} onClose={onClose} />);
    });
    await settle();

    await act(async () => {
      container.querySelector("[role=dialog]").dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(onClose).toHaveBeenCalled();
  });
});
