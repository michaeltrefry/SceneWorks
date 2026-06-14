import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RefinePromptControl } from "./RefinePromptControl.jsx";

async function settle() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
  });
}

function refineButton(container) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Refine my prompt"));
}

function buttonByText(container, text) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === text);
}

describe("RefinePromptControl", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "# Guide\n\nWrite vividly." }));
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

  it("disables the button when the prompt is empty", () => {
    render(<RefinePromptControl prompt="   " guidePath="/prompt-guides/z.md" modelId="z" workflow="image" refinePrompt={vi.fn()} onApply={vi.fn()} />);
    expect(refineButton(container).disabled).toBe(true);
  });

  it("refines, shows the rewrite, and applies it only on Apply", async () => {
    const refinePrompt = vi.fn(async () => "A vivid neon street at midnight, cinematic.");
    const onApply = vi.fn();
    render(
      <RefinePromptControl
        prompt="neon street"
        guidePath="/prompt-guides/z-image-turbo.md"
        modelId="z_image_turbo"
        workflow="image"
        refinePrompt={refinePrompt}
        onApply={onApply}
      />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle();

    // Sent the prompt + model + workflow + fetched guide; original not yet applied.
    expect(refinePrompt).toHaveBeenCalledWith({
      prompt: "neon street",
      modelId: "z_image_turbo",
      workflow: "image",
      guide: "# Guide\n\nWrite vividly.",
    });
    expect(onApply).not.toHaveBeenCalled();
    expect(container.querySelector(".refine-review-text").textContent).toBe("A vivid neon street at midnight, cinematic.");

    await act(async () => {
      buttonByText(container, "Apply").click();
    });
    expect(onApply).toHaveBeenCalledWith("A vivid neon street at midnight, cinematic.");
    expect(container.querySelector(".refine-review")).toBeNull();
  });

  it("keeps the original when the user dismisses the rewrite", async () => {
    const refinePrompt = vi.fn(async () => "rewritten");
    const onApply = vi.fn();
    render(<RefinePromptControl prompt="dog" guidePath="" modelId="z" workflow="image" refinePrompt={refinePrompt} onApply={onApply} />);

    await act(async () => {
      refineButton(container).click();
    });
    await settle();

    await act(async () => {
      buttonByText(container, "Keep original").click();
    });
    expect(onApply).not.toHaveBeenCalled();
    expect(container.querySelector(".refine-review")).toBeNull();
  });

  it("surfaces a failure message and does not apply", async () => {
    const refinePrompt = vi.fn(async () => {
      throw new Error("Prompt refinement runtime is not configured.");
    });
    const onApply = vi.fn();
    render(<RefinePromptControl prompt="dog" guidePath="" modelId="z" workflow="image" refinePrompt={refinePrompt} onApply={onApply} />);

    await act(async () => {
      refineButton(container).click();
    });
    await settle();

    expect(container.querySelector(".refine-error").textContent).toContain("not configured");
    expect(onApply).not.toHaveBeenCalled();
    expect(container.querySelector(".refine-review")).toBeNull();
  });

  it("still refines (generically) when the guide cannot be fetched", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 404, text: async () => "" }));
    const refinePrompt = vi.fn(async () => "rewritten");
    render(
      <RefinePromptControl prompt="dog" guidePath="/prompt-guides/missing.md" modelId="z" workflow="video" refinePrompt={refinePrompt} onApply={vi.fn()} />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle();

    expect(refinePrompt).toHaveBeenCalledWith({ prompt: "dog", modelId: "z", workflow: "video", guide: "" });
    expect(container.querySelector(".refine-review-text").textContent).toBe("rewritten");
  });

  it("offers to download the refinement model when it isn't installed (sc-5605)", async () => {
    const refinePrompt = vi.fn(async () => {
      throw new Error("prompt-refine model path snapshot is not cached for huihui-ai/Llama-3.2-3B-Instruct-abliterated.");
    });
    const onDownloadRefineModel = vi.fn(async () => ({ id: "job-1" }));
    render(
      <RefinePromptControl
        prompt="dog"
        guidePath=""
        modelId="z"
        workflow="image"
        refinePrompt={refinePrompt}
        onApply={vi.fn()}
        refineModel={{ id: "prompt_refine_llama_3_2_3b", name: "Prompt Refiner", installState: "missing", downloadSizeBytes: 7222715642 }}
        onDownloadRefineModel={onDownloadRefineModel}
      />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle();

    // The raw worker error is replaced by a download affordance with a size hint.
    expect(container.querySelector(".refine-missing-model").textContent).toContain("isn’t installed");
    expect(container.querySelector(".refine-missing-model").textContent).toContain("7.2 GB");
    const downloadButton = buttonByText(container, "Download refinement model");
    expect(downloadButton).toBeTruthy();

    await act(async () => {
      downloadButton.click();
    });
    await settle();

    expect(onDownloadRefineModel).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".refine-missing-model").textContent).toContain("Downloading");
  });

  it("shows a plain error (no download CTA) when the refinement model is installed", async () => {
    const refinePrompt = vi.fn(async () => {
      throw new Error("Prompt refinement failed.");
    });
    render(
      <RefinePromptControl
        prompt="dog"
        guidePath=""
        modelId="z"
        workflow="image"
        refinePrompt={refinePrompt}
        onApply={vi.fn()}
        refineModel={{ id: "prompt_refine_llama_3_2_3b", name: "Prompt Refiner", installState: "installed" }}
        onDownloadRefineModel={vi.fn()}
      />,
    );

    await act(async () => {
      refineButton(container).click();
    });
    await settle();

    expect(container.querySelector(".refine-missing-model")).toBeNull();
    expect(container.querySelector(".refine-error").textContent).toContain("Prompt refinement failed.");
    expect(buttonByText(container, "Download refinement model")).toBeUndefined();
  });
});
