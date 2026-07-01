fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "start_setup",
            "get_session_logs",
            "get_app_settings",
            "get_storage_setup",
            "save_storage_setup",
            "complete_setup",
            "reset_setup",
            "choose_folder",
            "set_data_dir",
            "choose_data_dir",
            "reveal_in_os",
            "resolve_asset_path",
            "save_asset_as",
            "list_credentials",
            "set_credential",
            "delete_credential",
            "restart_worker",
            "get_gpu_info",
            // LAN remote access (epic 4484, stories 4/5).
            "get_remote_access",
            "set_remote_access",
            "set_remote_access_password",
            "clear_remote_access_password",
            "get_lan_address",
        ]),
    ))
    .expect("failed to run Tauri build script");
}
