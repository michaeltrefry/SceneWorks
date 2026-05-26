import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// SettingsScreen computes `isDesktop` from window.__TAURI__ at module load, so we
// set the Tauri bridge and re-import the module fresh in each test.
async function changeField(input, value) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(
      new window.Event(input.tagName === "SELECT" ? "change" : "input", { bubbles: true }),
    );
  });
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

describe("SettingsScreen service credentials", () => {
  let container;
  let root;
  let invoke;
  let credentials;
  let SettingsScreen;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    credentials = [];
    invoke = vi.fn(async (command) => {
      switch (command) {
        case "get_app_settings":
          return {};
        case "get_gpu_info":
          return { platform: "windows", devices: [] };
        case "list_credentials":
          return credentials;
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(<SettingsScreen />);
    });
    // Flush the initial refresh() Promise.all.
    await act(async () => {});
  }

  it("lists a stored credential by host without exposing the token", async () => {
    credentials = [{ host: "huggingface.co", label: "Hugging Face", scheme: "bearer", present: true }];
    await render();
    expect(invoke).toHaveBeenCalledWith("list_credentials", undefined);
    expect(container.textContent).toContain("Hugging Face");
    expect(container.textContent).toContain("huggingface.co");
  });

  it("flags a recorded credential whose token is missing from the keychain", async () => {
    credentials = [{ host: "civitai.com", label: "Civit.ai", scheme: "query", present: false }];
    await render();
    expect(container.textContent).toContain("token missing");
  });

  it("saves a new credential via set_credential", async () => {
    await render();
    await changeField(container.querySelector('[aria-label="Credential host"]'), "https://Civitai.com");
    await changeField(container.querySelector('[aria-label="Credential label"]'), "Civit.ai");
    await changeField(container.querySelector('[aria-label="Authentication scheme"]'), "query");
    await changeField(container.querySelector('[aria-label="Credential token"]'), "key123");
    await click(container.querySelector(".settings-credential-form button"));
    expect(invoke).toHaveBeenCalledWith("set_credential", {
      host: "https://Civitai.com",
      label: "Civit.ai",
      scheme: "query",
      token: "key123",
    });
  });

  it("removes a credential via delete_credential", async () => {
    credentials = [{ host: "civitai.com", label: "Civit.ai", scheme: "query", present: true }];
    await render();
    await click(container.querySelector(".settings-credential button"));
    expect(invoke).toHaveBeenCalledWith("delete_credential", { host: "civitai.com" });
  });
});
