//! UI preferences (theme, accent, …) persisted as a small JSON file in the
//! config dir.
//! Served over plain HTTP because the bundled desktop UI runs at the API's
//! `http://127.0.0.1:<port>` origin, where both Tauri IPC and origin-keyed
//! `localStorage` are unreliable across launches (the port — and so the origin —
//! changes every launch). Routing through the API, the same channel the rest of
//! the app already uses, makes the choice durable. Non-sensitive, so the routes
//! are public like `/health`.

use super::*;

use std::path::PathBuf;

const PREFERENCES_FILENAME: &str = "ui-preferences.json";

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UiPreferences {
    /// Last-used UI theme (`"light"` or `"dark"`); absent until first set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    theme: Option<String>,
    /// Last-used accent palette id (see `ACCENT_IDS`); absent until first set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accent: Option<String>,
}

/// User-selectable accent palettes. Keep in sync with web/src/accents.js.
const ACCENT_IDS: [&str; 7] = [
    "teal", "indigo", "cobalt", "violet", "coral", "amber", "emerald",
];

fn preferences_path(state: &AppState) -> PathBuf {
    state.settings.config_dir.join(PREFERENCES_FILENAME)
}

fn load_preferences(state: &AppState) -> UiPreferences {
    std::fs::read_to_string(preferences_path(state))
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
        .unwrap_or_default()
}

fn save_preferences(state: &AppState, prefs: &UiPreferences) -> std::io::Result<()> {
    let path = preferences_path(state);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(prefs)?;
    std::fs::write(path, body)
}

/// The valid stored theme for `input`, or `None` if it isn't one we recognize.
fn normalize_theme(input: Option<&str>) -> Option<String> {
    match input.map(str::trim) {
        Some("light") => Some("light".to_owned()),
        Some("dark") => Some("dark".to_owned()),
        _ => None,
    }
}

/// The valid stored accent for `input`, or `None` if it isn't a known palette.
fn normalize_accent(input: Option<&str>) -> Option<String> {
    let value = input.map(str::trim)?;
    ACCENT_IDS
        .iter()
        .find(|id| **id == value)
        .map(|id| (*id).to_owned())
}

/// Current UI preferences (empty object on first run).
pub(crate) async fn get_ui_preferences(
    State(state): State<AppState>,
) -> Result<Json<UiPreferences>, ApiError> {
    Ok(Json(load_preferences(&state)))
}

/// Merge the supplied preferences in and persist. Only recognized fields/values
/// are applied, so an unknown theme leaves the stored one untouched.
pub(crate) async fn set_ui_preferences(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<UiPreferences>,
) -> Result<Json<UiPreferences>, ApiError> {
    let mut prefs = load_preferences(&state);
    if let Some(theme) = normalize_theme(payload.theme.as_deref()) {
        prefs.theme = Some(theme);
    }
    if let Some(accent) = normalize_accent(payload.accent.as_deref()) {
        prefs.accent = Some(accent);
    }
    save_preferences(&state, &prefs)
        .map_err(|error| ApiError::internal(format!("Failed to save UI preferences: {error}")))?;
    Ok(Json(prefs))
}

#[cfg(test)]
mod tests {
    use super::{normalize_accent, normalize_theme};

    #[test]
    fn normalize_theme_accepts_only_known_themes() {
        assert_eq!(normalize_theme(Some(" light ")), Some("light".to_owned()));
        assert_eq!(normalize_theme(Some("dark")), Some("dark".to_owned()));
        assert_eq!(normalize_theme(Some("blue")), None);
        assert_eq!(normalize_theme(None), None);
    }

    #[test]
    fn normalize_accent_accepts_only_known_palettes() {
        assert_eq!(normalize_accent(Some("teal")), Some("teal".to_owned()));
        assert_eq!(
            normalize_accent(Some(" emerald ")),
            Some("emerald".to_owned())
        );
        assert_eq!(normalize_accent(Some("amber")), Some("amber".to_owned()));
        assert_eq!(normalize_accent(Some("fuchsia")), None);
        assert_eq!(normalize_accent(None), None);
    }
}
