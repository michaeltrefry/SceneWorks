// Hide the extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod setup;

use tauri::RunEvent;

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(setup::Managed::default())
        .invoke_handler(tauri::generate_handler![setup::start_setup])
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
