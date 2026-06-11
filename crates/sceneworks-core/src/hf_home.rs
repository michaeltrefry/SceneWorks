//! Default Hugging Face cache home for the server/host-mode binaries (sc-1904
//! follow-up).
//!
//! Hugging Face tooling caches under `~/.cache/huggingface` on every platform
//! (huggingface_hub's `HF_HOME` default; mirrored by the Python worker's
//! `hf_cache.huggingface_cache_root` and the desktop's `shared_huggingface_home`).
//! When the rust-api / rust-worker binaries are started with none of the HF cache
//! env vars set (host mode), they otherwise fall back to `<data_dir>/cache/...`,
//! dumping downloads into the app's private data folder. Defaulting `HF_HOME` to
//! the OS Hugging Face home at startup keeps host-mode downloads in the shared
//! per-user cache, deduplicated with every other HF tool. The desktop and Docker
//! Compose already inject `HF_HOME`, so this only changes the env-less case.

use std::path::{Path, PathBuf};

/// The OS Hugging Face home, `~/.cache/huggingface` (the literal `~/.cache`, not
/// the platform cache dir — matching huggingface_hub and `Path.home()/.cache/
/// huggingface` in the Python worker). `None` when no home dir can be resolved.
pub fn os_huggingface_home() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|base| base.home_dir().join(".cache").join("huggingface"))
}

/// Pure decision: the value to default `HF_HOME` to when the process set none of
/// the HF cache env vars. Returns `None` (leave the environment untouched) when
/// any HF cache var is already set, or when no home dir is available. Taking the
/// env values + home as arguments keeps it deterministically testable.
pub fn default_huggingface_home(
    hf_hub_cache: Option<&str>,
    huggingface_hub_cache: Option<&str>,
    hf_home: Option<&str>,
    os_home: Option<PathBuf>,
) -> Option<PathBuf> {
    let is_set = |value: Option<&str>| value.map(str::trim).is_some_and(|value| !value.is_empty());
    if is_set(hf_hub_cache) || is_set(huggingface_hub_cache) || is_set(hf_home) {
        return None;
    }
    os_home
}

/// If no HF cache env var (`HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME`)
/// is set, point `HF_HOME` at the OS Hugging Face home so downloads land in the
/// shared `~/.cache/huggingface` cache rather than the app's data dir. Returns the
/// path it set, or `None` when it left the environment unchanged. Call once at
/// binary startup, before any cache resolution or worker spawn.
pub fn ensure_default_huggingface_home() -> Option<PathBuf> {
    let read = |key: &str| std::env::var(key).ok();
    let chosen = default_huggingface_home(
        read("HF_HUB_CACHE").as_deref(),
        read("HUGGINGFACE_HUB_CACHE").as_deref(),
        read("HF_HOME").as_deref(),
        os_huggingface_home(),
    )?;
    std::env::set_var("HF_HOME", &chosen);
    Some(chosen)
}

/// The Hugging Face hub cache directory: `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE`
/// if set, else `<HF_HOME>/hub`, else `<data_dir>/cache/huggingface/hub`. Shared
/// by the rust-api and rust-worker so cache-path resolution can't drift between
/// them (sc-4279 / F-MLXW-15).
pub fn huggingface_hub_cache_dir(data_dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var("HF_HUB_CACHE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var("HUGGINGFACE_HUB_CACHE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var("HF_HOME")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(path).join("hub");
    }
    data_dir.join("cache").join("huggingface").join("hub")
}

/// The `<X>` in Hugging Face hub's `models--<X>` cache directory name: every
/// character outside `[A-Za-z0-9._-]` becomes `--`, then surrounding `-` are
/// trimmed. `None` when nothing survives. Kept byte-identical to the Python
/// worker (`hf_cache.safe_repo_dir_name`), the rust-api and the rust-worker —
/// pinned by the `repo_slugs.json` cross-language contract (story 1667).
pub fn safe_repo_dir_name(repo: &str) -> Option<String> {
    let safe_repo = repo
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character.to_string()
            } else {
                "--".to_owned()
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    if safe_repo.is_empty() {
        None
    } else {
        Some(safe_repo)
    }
}

/// The `models--<safe_repo>` cache directory for `repo` under the hub cache.
/// `None` when the repo slug sanitizes to nothing.
pub fn huggingface_repo_cache_path(data_dir: &Path, repo: &str) -> Option<PathBuf> {
    let safe_repo = safe_repo_dir_name(repo)?;
    Some(huggingface_hub_cache_dir(data_dir).join(format!("models--{safe_repo}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// sc-4279 / F-MLXW-15: canonical home for the `repo_slugs.json`
    /// cross-language contract now that `safe_repo_dir_name` lives in core (it was
    /// duplicated, with a per-crate test, in both the rust-api and the rust-worker).
    /// The slug must match what the Python worker and HF hub produce for every case.
    #[test]
    fn safe_repo_dir_name_matches_repo_slugs_contract() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rust_migration_contracts/repo_slugs.json");
        let contract: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read repo_slugs.json"))
                .expect("parse repo_slugs.json");
        let cases = contract["cases"].as_array().expect("cases array");
        assert!(!cases.is_empty(), "repo_slugs fixture has no cases");
        for case in cases {
            let repo = case["repo"].as_str().expect("repo string");
            assert_eq!(
                safe_repo_dir_name(repo).as_deref(),
                case["safeRepoDirName"].as_str(),
                "safe_repo_dir_name drift for {repo:?}"
            );
        }
    }

    #[test]
    fn defaults_to_os_home_when_no_hf_env_is_set() {
        let home = Some(PathBuf::from("/home/alice/.cache/huggingface"));
        assert_eq!(
            default_huggingface_home(None, None, None, home.clone()),
            home
        );
        // Blank/whitespace env values count as unset.
        assert_eq!(
            default_huggingface_home(Some(""), Some("   "), None, home.clone()),
            home
        );
    }

    #[test]
    fn leaves_env_untouched_when_any_hf_var_is_set() {
        let home = Some(PathBuf::from("/home/alice/.cache/huggingface"));
        assert_eq!(
            default_huggingface_home(Some("/mnt/hub"), None, None, home.clone()),
            None
        );
        assert_eq!(
            default_huggingface_home(None, Some("/mnt/hub"), None, home.clone()),
            None
        );
        assert_eq!(
            default_huggingface_home(None, None, Some("/srv/hf"), home),
            None
        );
    }

    #[test]
    fn yields_none_without_a_home_dir() {
        assert_eq!(default_huggingface_home(None, None, None, None), None);
    }

    #[test]
    fn os_home_ends_in_cache_huggingface() {
        // Environment-dependent, but every CI/dev runner has a home dir.
        if let Some(home) = os_huggingface_home() {
            assert!(home.ends_with(Path::new(".cache").join("huggingface")));
        }
    }
}
