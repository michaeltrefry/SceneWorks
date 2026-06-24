import React, { useCallback, useEffect, useState } from "react";
import {
  SCHEME_LABELS,
  loadCredentials,
  removeCredentialRequest,
  saveCredential,
  serverToken,
} from "../credentials.js";
import { apiFetch } from "../api.js";
import { isDesktop, tauriInvoke as invoke } from "../runtime.js";

// The data-dir / GPU / worker / wizard / remote-access controls are desktop-only
// (backed by Tauri commands in the shell) and are gated behind `isDesktop` so a
// remote LAN browser (epic 4484) doesn't render buttons that would call a missing
// Tauri bridge. Service credentials work in both deployments and route through the
// shared ../credentials.js transport (keychain on desktop, authed REST on the
// server / remote browser). `isDesktop`/`invoke` come from the unified runtime
// helper (story 6).

// GPU memory cap (epic 7819). The persisted value is a fraction (0.1–0.99) of total unified
// memory, or null/absent for "no limit". The slider works in whole percent; 100% means Off.
const GPU_LIMIT_MIN_PERCENT = 10;
function fractionToPercent(fraction) {
  if (typeof fraction !== "number" || !Number.isFinite(fraction) || fraction >= 1) return 100;
  return Math.min(100, Math.max(GPU_LIMIT_MIN_PERCENT, Math.round(fraction * 100)));
}

export function SettingsScreen() {
  const [settings, setSettings] = useState(null);
  const [gpu, setGpu] = useState(null);
  const [credentials, setCredentials] = useState([]);
  const [newHost, setNewHost] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [newScheme, setNewScheme] = useState("bearer");
  const [newToken, setNewToken] = useState("");
  const [status, setStatus] = useState("");
  // LAN remote access (epic 4484 stories 4/11). `remote` is the RemoteAccessStatus
  // snapshot from the shell; only fetched/rendered on desktop.
  const [remote, setRemote] = useState(null);
  const [remotePort, setRemotePort] = useState("");
  const [remotePassword, setRemotePassword] = useState("");
  // GPU memory cap (epic 7819). Local slider percent for smooth dragging; committed to the shell
  // on release. 100 = Off (no limit). Synced from persisted settings once they load.
  const [gpuLimitPercent, setGpuLimitPercent] = useState(100);

  const refresh = useCallback(async () => {
    try {
      if (isDesktop) {
        const [loadedSettings, gpuInfo, storedCredentials, remoteAccess] = await Promise.all([
          invoke("get_app_settings"),
          invoke("get_gpu_info"),
          invoke("list_credentials"),
          invoke("get_remote_access"),
        ]);
        setSettings(loadedSettings);
        setGpu(gpuInfo);
        setCredentials(storedCredentials ?? []);
        if (remoteAccess) {
          setRemote(remoteAccess);
          setRemotePort(String(remoteAccess.port));
        }
      } else {
        setCredentials(await loadCredentials());
      }
    } catch (error) {
      setStatus(String(error));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Keep the slider in sync with the persisted cap (also after a commit echoes settings back).
  useEffect(() => {
    if (settings) setGpuLimitPercent(fractionToPercent(settings.gpuMemoryLimitFraction));
  }, [settings]);

  const secretStore = gpu?.platform === "windows" ? "Credential Manager" : "Keychain";
  const credentialLocation = isDesktop
    ? `the system ${secretStore}`
    : "the server's credential store (a restricted file in the config directory)";
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
      const updated = await saveCredential({
        host: newHost,
        label: newLabel,
        scheme: newScheme,
        token: newToken,
      });
      setCredentials(updated ?? []);
      setNewHost("");
      setNewLabel("");
      setNewScheme("bearer");
      setNewToken("");
      setStatus("Credential saved.");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function removeCredential(host) {
    try {
      const updated = await removeCredentialRequest(host);
      setCredentials(updated ?? []);
      setStatus(`Removed the credential for ${host}.`);
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function restartWorker() {
    try {
      // Desktop kills the worker child directly via Tauri; a remote admin restarts it
      // over REST (epic 4484 story 12), which signals the desktop supervisor to respawn.
      if (isDesktop) {
        await invoke("restart_worker");
      } else {
        await apiFetch("/api/v1/worker/restart", serverToken(), { method: "POST" });
      }
      setStatus("Restarting the inference worker…");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function commitGpuMemoryLimit(percent) {
    try {
      // 100% = no limit → send null so the shell clears the cap; otherwise a fraction (0.1–0.99).
      const fraction = percent >= 100 ? null : percent / 100;
      const updated = await invoke("set_gpu_memory_limit", { fraction });
      setSettings(updated);
      setStatus(
        fraction == null
          ? "GPU memory limit removed. Restart the worker to apply."
          : `GPU memory limit set to ${percent}%. Restart the worker to apply.`,
      );
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

  // --- Remote access (LAN) handlers (epic 4484 story 4) ---
  async function saveRemotePassword() {
    try {
      const updated = await invoke("set_remote_access_password", { password: remotePassword });
      setRemote(updated);
      setRemotePassword("");
      setStatus("Remote access password saved.");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function clearRemotePassword() {
    try {
      const updated = await invoke("clear_remote_access_password");
      setRemote(updated);
      setStatus("Password cleared — remote access disabled.");
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function applyRemoteAccess(enabled) {
    const port = Number.parseInt(remotePort, 10);
    if (!Number.isInteger(port) || port < 1024 || port > 65535) {
      setStatus("Choose a port between 1024 and 65535.");
      return;
    }
    try {
      const updated = await invoke("set_remote_access", { enabled, port });
      setRemote(updated);
      setRemotePort(String(updated.port));
      setStatus(
        enabled
          ? "Remote access enabled — restart SceneWorks to apply."
          : "Remote access disabled — restart SceneWorks to apply.",
      );
    } catch (error) {
      setStatus(String(error));
    }
  }

  async function copyRemoteUrl() {
    if (!remote?.url) {
      return;
    }
    try {
      await navigator.clipboard.writeText(remote.url);
      setStatus(`Copied ${remote.url}`);
    } catch {
      setStatus("Couldn't copy to clipboard.");
    }
  }

  const canSaveCredential = newHost.trim() && newToken.trim();

  return (
    <div className="settings-screen">
      {status ? <p className="settings-status">{status}</p> : null}

      {isDesktop ? (
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
      ) : null}

      {isDesktop && remote ? (
        <section className="settings-card">
          <h3>Remote access (LAN)</h3>
          <p className="settings-muted">
            Let another device on your local network use SceneWorks in a browser, with
            generation running on this computer. Off by default; protected by a password
            you set.
          </p>
          <p className="settings-value">
            {remote.enabled ? "Enabled" : "Disabled"}
            {remote.enabled && remote.url ? ` · ${remote.url}` : ""}
          </p>

          <div className="settings-actions settings-credential-form">
            <input
              type="password"
              placeholder={remote.passwordSet ? "Change password" : "Set a password"}
              value={remotePassword}
              onChange={(event) => setRemotePassword(event.target.value)}
              aria-label="Remote access password"
            />
            <button
              type="button"
              onClick={saveRemotePassword}
              disabled={!remotePassword.trim()}
            >
              {remote.passwordSet ? "Change password" : "Set password"}
            </button>
            {remote.passwordSet ? (
              <button type="button" onClick={clearRemotePassword}>
                Clear password
              </button>
            ) : null}
          </div>
          <p className="settings-muted">
            {remote.passwordSet
              ? "A password is set. Remote browsers must enter it to connect."
              : "Set a password before enabling remote access."}
          </p>

          <div className="settings-actions">
            <label htmlFor="remote-port">Port</label>
            <input
              id="remote-port"
              type="number"
              min="1024"
              max="65535"
              value={remotePort}
              onChange={(event) => setRemotePort(event.target.value)}
              aria-label="Remote access port"
            />
            {remote.enabled ? (
              <button type="button" onClick={() => applyRemoteAccess(false)}>
                Disable remote access
              </button>
            ) : (
              <button
                type="button"
                onClick={() => applyRemoteAccess(true)}
                disabled={!remote.passwordSet}
              >
                Enable remote access
              </button>
            )}
            {remote.url ? (
              <button type="button" onClick={copyRemoteUrl}>
                Copy URL
              </button>
            ) : null}
          </div>

          {remote.url ? (
            <p className="settings-muted">
              Open <code>{remote.url}</code> on another device on this network.
              {remote.lanCandidates && remote.lanCandidates.length > 1
                ? ` Other addresses: ${remote.lanCandidates.slice(1).join(", ")}.`
                : ""}
            </p>
          ) : (
            <p className="settings-muted">
              Couldn’t determine this computer’s LAN address — check your network
              connection.
            </p>
          )}

          {/* Security note + platform firewall guidance (story 11). */}
          <p className="settings-help">
            Trusted local networks only. Traffic uses plain HTTP, so the password and
            your content travel unencrypted on the LAN — do not port-forward this or
            expose it to the public internet. The URL only works for devices on the same
            network.
          </p>
          {remote.enabled ? (
            remote.platform === "windows" ? (
              <p className="settings-help">
                The first time SceneWorks binds the network port, Windows shows a
                “Windows Security Alert”. Allow SceneWorks on <strong>Private</strong>{" "}
                networks (not Public), or remote devices can’t connect. To change it
                later: Windows Security → Firewall &amp; network protection → Allow an
                app through firewall.
              </p>
            ) : (
              <p className="settings-help">
                The first time SceneWorks binds the network port, macOS may ask to
                “allow incoming connections” — click Allow. To change it later: System
                Settings → Network → Firewall.
              </p>
            )
          ) : null}
          <p className="settings-muted">
            Changing these settings takes effect after you restart SceneWorks.
          </p>
        </section>
      ) : null}

      <section className="settings-card">
        <h3>Service credentials</h3>
        <p className="settings-muted">
          API tokens for model &amp; LoRA downloads (Hugging Face, Civit.ai, and any
          other authenticated source). Stored in {credentialLocation}; tokens are
          never displayed again after saving. Changing a credential takes effect on
          the next worker restart.
        </p>
        {credentials.length ? (
          <ul className="settings-list">
            {credentials.map((credential) => (
              <li key={credential.host} className="settings-credential">
                <span className="settings-value">
                  {credential.label ? `${credential.label} — ` : ""}
                  <code>{credential.host}</code>{" "}
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
            Save token
          </button>
        </div>
      </section>

      {isDesktop ? (
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
          {gpu?.platform === "macos" && gpu?.unifiedMemoryMb ? (
            <div className="settings-gpu-cap">
              <label htmlFor="gpu-memory-cap">
                GPU memory limit:{" "}
                {gpuLimitPercent >= 100
                  ? "Off — no limit"
                  : `${Math.round((gpu.unifiedMemoryMb / 1024) * (gpuLimitPercent / 100))} GB of ` +
                    `${Math.round(gpu.unifiedMemoryMb / 1024)} GB (${gpuLimitPercent}%)`}
              </label>
              <input
                id="gpu-memory-cap"
                type="range"
                min={GPU_LIMIT_MIN_PERCENT}
                max={100}
                step={5}
                value={gpuLimitPercent}
                aria-label="GPU memory limit (percent of unified memory)"
                onChange={(event) => setGpuLimitPercent(Number(event.target.value))}
                onMouseUp={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                onKeyUp={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                onTouchEnd={(event) => commitGpuMemoryLimit(Number(event.target.value))}
              />
              <p className="settings-help">
                Caps the shared memory generations and LoRA training may use, leaving the rest for
                the system. This is a soft target, not a hard limit — set it too low and large models
                may slow down or fail. Takes effect when you restart the worker.
              </p>
            </div>
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
      ) : null}

      {/* Available in both modes: desktop restarts via Tauri, a remote admin via REST
          (epic 4484 story 12). */}
      <section className="settings-card">
        <h3>Inference worker</h3>
        <div className="settings-actions">
          <button type="button" onClick={restartWorker}>
            Restart worker
          </button>
        </div>
      </section>

      {isDesktop ? (
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
      ) : null}
    </div>
  );
}
