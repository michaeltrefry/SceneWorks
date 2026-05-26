import React, { useCallback, useEffect, useState } from "react";

// Desktop-only: these settings are backed by Tauri commands in the shell.
const isDesktop = typeof window !== "undefined" && !!window.__TAURI__;
const invoke = (command, args) => window.__TAURI__.core.invoke(command, args);

const SCHEME_LABELS = {
  bearer: "Bearer header",
  query: "Query token",
};

export function SettingsScreen() {
  const [settings, setSettings] = useState(null);
  const [gpu, setGpu] = useState(null);
  const [credentials, setCredentials] = useState([]);
  const [newHost, setNewHost] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [newScheme, setNewScheme] = useState("bearer");
  const [newToken, setNewToken] = useState("");
  const [status, setStatus] = useState("");

  const refresh = useCallback(async () => {
    if (!isDesktop) {
      return;
    }
    try {
      const [loadedSettings, gpuInfo, storedCredentials] = await Promise.all([
        invoke("get_app_settings"),
        invoke("get_gpu_info"),
        invoke("list_credentials"),
      ]);
      setSettings(loadedSettings);
      setGpu(gpuInfo);
      setCredentials(storedCredentials ?? []);
    } catch (error) {
      setStatus(String(error));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (!isDesktop) {
    return (
      <div className="settings-screen">
        <p className="settings-muted">
          Settings are managed by the SceneWorks desktop app.
        </p>
      </div>
    );
  }

  const secretStore = gpu?.platform === "windows" ? "Credential Manager" : "Keychain";
  const dataDirLabel = settings?.dataDir ?? "Default location";

  async function changeDataDir() {
    try {
      const picked = await invoke("choose_data_dir");
      if (picked) {
        const updated = await invoke("set_data_dir", { path: picked });
        setSettings(updated);
        setStatus("Data directory updated — restart SceneWorks to apply.");
      }
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function revealDataDir() {
    if (settings?.dataDir) {
      await invoke("reveal_in_os", { path: settings.dataDir });
    }
  }

  async function addCredential() {
    try {
      await invoke("set_credential", {
        host: newHost,
        label: newLabel,
        scheme: newScheme,
        token: newToken,
      });
      setNewHost("");
      setNewLabel("");
      setNewScheme("bearer");
      setNewToken("");
      await refresh();
      setStatus(`Credential saved to the ${secretStore}.`);
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function removeCredential(host) {
    try {
      await invoke("delete_credential", { host });
      await refresh();
      setStatus(`Removed the credential for ${host}.`);
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function restartWorker() {
    try {
      await invoke("restart_worker");
      setStatus("Restarting the inference worker…");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function rerunSetupWizard() {
    try {
      await invoke("reset_setup");
      window.location.reload();
    } catch (error) {
      setStatus(String(error));
    }
  }

  const canSaveCredential = newHost.trim() && newToken.trim();

  return (
    <div className="settings-screen">
      {status ? <p className="settings-status">{status}</p> : null}

      <section className="settings-card">
        <h3>Data directory</h3>
        <p className="settings-value">{dataDirLabel}</p>
        <div className="settings-actions">
          <button type="button" onClick={changeDataDir}>
            Change…
          </button>
          <button type="button" onClick={revealDataDir} disabled={!settings?.dataDir}>
            Reveal in {gpu?.platform === "windows" ? "Explorer" : "Finder"}
          </button>
        </div>
      </section>

      <section className="settings-card">
        <h3>Service credentials</h3>
        <p className="settings-muted">
          API tokens for model &amp; LoRA downloads (Hugging Face, Civit.ai, and any
          other authenticated source). Stored in the system {secretStore}; tokens
          are never displayed again after saving.
        </p>
        {credentials.length ? (
          <ul className="settings-list">
            {credentials.map((credential) => (
              <li key={credential.host} className="settings-credential">
                <span className="settings-value">
                  {credential.label} — <code>{credential.host}</code>{" "}
                  <span className="settings-muted">
                    ({SCHEME_LABELS[credential.scheme] ?? credential.scheme}
                    {credential.present ? "" : " · token missing"})
                  </span>
                </span>
                <button type="button" onClick={() => removeCredential(credential.host)}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="settings-muted">No credentials saved.</p>
        )}
        <div className="settings-actions settings-credential-form">
          <input
            type="text"
            placeholder="Host (e.g. huggingface.co)"
            value={newHost}
            onChange={(event) => setNewHost(event.target.value)}
            aria-label="Credential host"
          />
          <input
            type="text"
            placeholder="Label (optional)"
            value={newLabel}
            onChange={(event) => setNewLabel(event.target.value)}
            aria-label="Credential label"
          />
          <select
            value={newScheme}
            onChange={(event) => setNewScheme(event.target.value)}
            aria-label="Authentication scheme"
          >
            <option value="bearer">Bearer header</option>
            <option value="query">Query token</option>
          </select>
          <input
            type="password"
            placeholder="Token"
            value={newToken}
            onChange={(event) => setNewToken(event.target.value)}
            aria-label="Credential token"
          />
          <button type="button" onClick={addCredential} disabled={!canSaveCredential}>
            Save
          </button>
        </div>
      </section>

      <section className="settings-card">
        <h3>Detected GPU</h3>
        {gpu?.devices?.length ? (
          <ul className="settings-list">
            {gpu.devices.map((device) => (
              <li key={device}>{device}</li>
            ))}
          </ul>
        ) : (
          <p className="settings-muted">No accelerated GPU detected.</p>
        )}
        {gpu?.unifiedMemoryMb ? (
          <p className="settings-muted">
            Unified memory: {Math.round(gpu.unifiedMemoryMb / 1024)} GB
            {typeof gpu.wiredLimitMb === "number"
              ? ` · GPU cap: ${Math.round(gpu.wiredLimitMb / 1024)} GB`
              : ""}
          </p>
        ) : null}
        {gpu?.platform === "macos" ? (
          <p className="settings-help">
            On 96/128 GB Macs you can raise the GPU memory cap:{" "}
            <code>sudo sysctl iogpu.wired_limit_mb=&lt;bytes&gt;</code>
          </p>
        ) : null}
        {gpu?.platform === "windows" ? (
          <p className="settings-help">
            Requires current NVIDIA drivers with CUDA support.
          </p>
        ) : null}
      </section>

      <section className="settings-card">
        <h3>Inference worker</h3>
        <div className="settings-actions">
          <button type="button" onClick={restartWorker}>
            Restart worker
          </button>
        </div>
      </section>

      <section className="settings-card">
        <h3>Setup wizard</h3>
        <p className="settings-muted">
          Re-open the guided setup to download more models or create another project.
        </p>
        <div className="settings-actions">
          <button type="button" onClick={rerunSetupWizard}>
            Re-run setup wizard
          </button>
        </div>
      </section>
    </div>
  );
}
