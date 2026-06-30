//! In-app cross-platform auto-updater (sc-1355).
//!
//! On launch the release build asks the GitHub "latest release" pointer
//! (`plugins.updater.endpoints` in `tauri.conf.json`) whether a newer build
//! exists for this platform. The endpoint serves a `latest.json` manifest with a
//! per-target `platforms` map; `tauri-plugin-updater` picks `darwin-aarch64` /
//! `windows-x86_64` for the running build, verifies the minisign signature against
//! `plugins.updater.pubkey`, and — on user accept — downloads, installs, and
//! restarts into the new version.
//!
//! Driven entirely from the Rust shell: the UI is the API-served React app at a
//! loopback `http://127.0.0.1:<port>` origin, so a JS-side updater would need ACL
//! grants at a remote origin. Here we reuse `tauri-plugin-dialog` for the prompt
//! and keep the whole flow out of the webview. The check is fail-soft — offline,
//! no published release, or a parse error just logs and lets the app start.

use tauri::AppHandle;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::{Update, UpdaterExt};

/// Spawn a non-blocking startup update check. No-op in debug builds (`tauri dev`),
/// where the running version is the dev build and no signed bundle exists to swap.
pub fn spawn_startup_check(app: &AppHandle) {
    if cfg!(debug_assertions) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(err) = check_and_prompt(&app).await {
            // Never surface a check failure to the user — a missing network, an
            // unpublished release, or a transient GitHub hiccup must not block use.
            tracing::warn!(error = %err, "auto-update: check failed, continuing without update");
        }
    });
}

/// Query the endpoint and, if a newer release exists, prompt the user. Accepting
/// hands off to [`install_update`]; declining just returns.
async fn check_and_prompt(app: &AppHandle) -> tauri_plugin_updater::Result<()> {
    let Some(update) = app.updater()?.check().await? else {
        tracing::info!("auto-update: already on the latest release");
        return Ok(());
    };

    let current = update.current_version.clone();
    let latest = update.version.clone();
    tracing::info!(%current, %latest, "auto-update: newer release available");

    let app_for_install = app.clone();
    app.dialog()
        .message(format!(
            "SceneWorks {latest} is available (you have {current}).\n\n\
             Download and install it now? SceneWorks will restart to finish."
        ))
        .title("Update available")
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Update now".to_string(),
            "Later".to_string(),
        ))
        .show(move |accepted| {
            if !accepted {
                tracing::info!("auto-update: user deferred the update");
                return;
            }
            // The dialog callback is sync; run the async download/install on the
            // runtime. `update` is moved in (it carries the verified manifest).
            tauri::async_runtime::spawn(async move {
                if let Err(err) = install_update(&app_for_install, update).await {
                    tracing::error!(error = %err, "auto-update: install failed");
                    app_for_install
                        .dialog()
                        .message(
                            "The update could not be installed. You can download the \
                             latest release manually from the SceneWorks releases page.",
                        )
                        .title("Update failed")
                        .kind(MessageDialogKind::Error)
                        .blocking_show();
                }
            });
        });

    Ok(())
}

/// Download + install the verified update, then restart into it. `restart()`
/// diverges (`-> !`), so it is the function's tail and nothing runs after it.
async fn install_update(app: &AppHandle, update: Update) -> tauri_plugin_updater::Result<()> {
    tracing::info!(version = %update.version, "auto-update: downloading");
    update
        .download_and_install(
            |_chunk_len: usize, _content_len: Option<u64>| {},
            || tracing::info!("auto-update: download complete, installing"),
        )
        .await?;
    tracing::info!("auto-update: installed, restarting");
    app.restart()
}
