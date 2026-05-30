// Hide the extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod settings;
mod setup;

use tauri::RunEvent;

fn main() {
    // Kill any sidecars orphaned by a prior crash/force-quit before spawning
    // fresh ones, so API processes don't accumulate and contend on jobs.db.
    setup::reap_stale_sidecars();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(setup::Managed::default())
        .invoke_handler(tauri::generate_handler![
            setup::start_setup,
            settings::get_app_settings,
            settings::get_storage_setup,
            settings::save_storage_setup,
            settings::complete_setup,
            settings::reset_setup,
            settings::choose_folder,
            settings::set_data_dir,
            settings::choose_data_dir,
            settings::reveal_in_os,
            settings::list_credentials,
            settings::set_credential,
            settings::delete_credential,
            settings::restart_worker,
            settings::get_gpu_info,
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
