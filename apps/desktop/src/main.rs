// Hide the extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// On-demand keychain credential socket for the MLX worker (sc-5891). macOS-only:
// it replaces the eager spawn-time keychain reads that prompted at every launch.
#[cfg(target_os = "macos")]
mod cred_ipc;
// First-run CUDA/onnxruntime redist downloader (Windows candle build): the heavy GPU
// runtime DLLs are no longer bundled (NSIS ~2 GB limit) — they're fetched on first
// run into %APPDATA%\SceneWorks\gpu-runtime and resolved from there.
#[cfg(target_os = "windows")]
mod cuda_provision;
// Likely-LAN-address discovery for the remote-access URL (epic 4484, story 5).
mod net;
mod settings;
mod setup;
// In-app cross-platform auto-updater (sc-1355): startup check against the GitHub
// "latest release" pointer, user-prompted download + install + restart.
mod update;

use tauri::RunEvent;

fn main() {
    // Install the tracing backbone for the desktop shell's own logs. The sidecars
    // are separate processes; their stdout is captured into the multi-source ring
    // buffer in `setup.rs` (and re-classified there), independent of this subscriber.
    sceneworks_core::observability::init_logging();

    // Kill any sidecars orphaned by a prior crash/force-quit before spawning
    // fresh ones, so API processes don't accumulate and contend on jobs.db.
    setup::reap_stale_sidecars();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            // Kick off the auto-update check once the app is initialized (no-op in
            // debug builds). Fail-soft: never blocks launch (sc-1355).
            update::spawn_startup_check(app.handle());
            Ok(())
        })
        .manage(setup::Managed::default())
        .invoke_handler(tauri::generate_handler![
            setup::start_setup,
            setup::get_session_logs,
            settings::get_app_settings,
            settings::get_storage_setup,
            settings::save_storage_setup,
            settings::complete_setup,
            settings::reset_setup,
            settings::choose_folder,
            settings::set_data_dir,
            settings::choose_data_dir,
            settings::reveal_in_os,
            // Save an asset to a user-chosen destination + resolve an asset's
            // project-relative path to its absolute on-disk path (sc-8726).
            settings::save_asset_as,
            settings::resolve_asset_path,
            settings::list_credentials,
            settings::set_credential,
            settings::delete_credential,
            settings::restart_worker,
            settings::get_gpu_info,
            // GPU memory cap (epic 7819).
            settings::set_gpu_memory_limit,
            // Live MLX memory telemetry for the Settings readout (epic 7819, sc-7825).
            settings::get_gpu_telemetry,
            // LAN remote access (epic 4484, stories 4/5).
            settings::get_remote_access,
            settings::set_remote_access,
            settings::set_remote_access_password,
            settings::clear_remote_access_password,
            net::get_lan_address,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the SceneWorks desktop shell");

    app.run(|app_handle, event| {
        // Stop the Python worker then the API sidecar gracefully, holding the
        // app open until they exit (or the grace period elapses).
        if let RunEvent::ExitRequested { api, .. } = event {
            if setup::begin_shutdown(app_handle) {
                api.prevent_exit();
            }
        }
    });
}
