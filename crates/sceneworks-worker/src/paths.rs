//! Path normalization and app-managed-directory confinement helpers (the worker trust boundary).
use super::*;

pub fn safe_download_dir(value: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            in_replacement = false;
        } else if !in_replacement {
            output.push_str("__");
            in_replacement = true;
        }
    }
    let output = output.trim_matches('_').to_owned();
    if output.is_empty() {
        "download".to_owned()
    } else {
        output
    }
}

pub(crate) fn safe_join(base: &Path, relative: &str) -> WorkerResult<PathBuf> {
    let mut target = base.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            std::path::Component::Normal(value) => target.push(value),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Unsafe snapshot path: {relative}"
                )))
            }
        }
    }
    Ok(target)
}

pub(crate) fn normalize_absolute_path(path: &Path) -> WorkerResult<PathBuf> {
    let mut output = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => output.push(prefix.as_os_str()),
            std::path::Component::RootDir => output.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !output.pop() {
                    return Err(WorkerError::InvalidPayload(format!(
                        "Unsafe absolute path: {}",
                        path.display()
                    )));
                }
            }
            std::path::Component::Normal(value) => output.push(value),
        }
    }
    Ok(output)
}

pub(crate) fn normalized_data_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    normalize_absolute_path(&settings.data_dir)
}

pub(crate) fn ensure_path_under(
    path: PathBuf,
    roots: &[PathBuf],
    label: &str,
) -> WorkerResult<PathBuf> {
    if roots.iter().any(|root| path.starts_with(root)) {
        return Ok(path);
    }
    let allowed = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkerError::InvalidPayload(format!(
        "{label} must be inside an app-managed directory ({allowed})."
    )))
}

pub(crate) fn normalize_app_managed_path(
    settings: &Settings,
    raw_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let data_dir = normalized_data_dir(settings)?;
    let path = normalize_absolute_path(Path::new(raw_path))?;
    ensure_path_under(path, &[data_dir], label)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn normalize_app_managed_cache_path(
    settings: &Settings,
    raw_path: &str,
    cache_dir: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let root = settings.data_dir.join("cache").join(cache_dir);
    let normalized_root = normalize_absolute_path(&root)?;
    let canonical_root = normalize_existing_or_absolute(&root)?;
    let normalized = normalize_absolute_path(Path::new(raw_path))?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(resolved, &[normalized_root, canonical_root], label)
}

/// A model's weights are a read-only source the rust-api resolves (e.g.
/// `resolve_base_model_path`) from either the app data dir *or* the shared
/// Hugging Face hub cache — the default `HF_HOME` the desktop injects points the
/// cache at `~/.cache/huggingface`, outside `data_dir`. Unlike output dirs and
/// dataset roots (write targets, confined to `data_dir`), model weights may
/// legitimately live in that cache, so they are allowed under either root. Used
/// for the training base model and every other read-only model dir (captioner,
/// image/InstantID). Without this, an HF-cache-resident model (e.g. z_image_turbo)
/// fails the data-dir-only check even though the install/resolve gates accepted it.
pub(crate) fn normalize_app_managed_model_path(
    settings: &Settings,
    raw_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let data_dir = normalized_data_dir(settings)?;
    let hf_cache = normalize_absolute_path(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let path = normalize_absolute_path(Path::new(raw_path))?;
    ensure_path_under(path, &[data_dir, hf_cache], label)
}

/// Confine a LoRA adapter path taken from a job payload to an app-managed root
/// (sc-5723 / WKA-002). The path arrives untrusted (`installedPath`/`sourcePath`/
/// `path`/`source.path` on a LoRA spec) and is loaded as adapter weights, so —
/// like every other on-disk model input — it must resolve under the app data dir
/// or the shared Hugging Face hub cache (installed LoRAs live in `<data>/loras` or
/// a project tree under `<data>`; HF-cached adapters live in the hub cache).
/// Without this a crafted payload could point a LoRA at any `.safetensors` on the
/// host, giving the worker an arbitrary-file read primitive across the API boundary.
/// Mirrors `normalize_app_managed_model_path` (model weights share the same roots).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn normalize_app_managed_lora_path(
    settings: &Settings,
    path: &Path,
) -> WorkerResult<PathBuf> {
    let data_dir = normalized_data_dir(settings)?;
    let canonical_data_dir = normalize_existing_or_absolute(&settings.data_dir)?;
    let hf_cache = normalize_absolute_path(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let canonical_hf_cache =
        normalize_existing_or_absolute(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let normalized = normalize_absolute_path(path)?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(
        resolved,
        &[data_dir, canonical_data_dir, hf_cache, canonical_hf_cache],
        "LoRA path",
    )
}

pub(crate) fn normalize_existing_or_absolute(path: &Path) -> WorkerResult<PathBuf> {
    match std::fs::canonicalize(path) {
        Ok(canonical) => normalize_absolute_path(&canonical),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => normalize_absolute_path(path),
        Err(error) => Err(error.into()),
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn looks_like_huggingface_repo(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() || value.contains('\\') || Path::new(value).is_absolute() {
        return false;
    }
    let mut parts = value.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(repo) = parts.next() else {
        return false;
    };
    !owner.is_empty()
        && !repo.is_empty()
        && parts.next().is_none()
        && ![owner, repo]
            .iter()
            .any(|part| *part == "." || *part == "..")
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn resolve_app_managed_model_dir(
    settings: &Settings,
    model_name_or_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let model_name_or_path = model_name_or_path.trim();
    if model_name_or_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, model_name_or_path) {
        return Ok(snapshot);
    }
    if looks_like_huggingface_repo(model_name_or_path) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} snapshot is not cached for {model_name_or_path}."
        )));
    }
    let path = normalize_app_managed_model_path(settings, model_name_or_path, label)?;
    if path.is_dir() {
        return Ok(path);
    }
    if path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} must be a snapshot directory, not a file: {}",
            path.display()
        )));
    }
    Err(WorkerError::InvalidPayload(format!(
        "{label} is not installed at {}.",
        path.display()
    )))
}

pub(crate) fn resolve_training_output_dir(
    settings: &Settings,
    output_dir: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let path = normalize_app_managed_path(settings, output_dir, label)?;
    let data_dir = normalized_data_dir(settings)?;
    // Global-scope outputs land in `<data>/loras` (or `<data>/models` for full
    // fine-tunes); project-scope outputs — the default — land in the owning
    // project's tree, `<data>/projects/<slug>.sceneworks/loras/<lora_id>`, which
    // `resolve_training_output_location` computes API-side from trusted inputs.
    // All three stay inside the app data dir, so allow the projects tree too
    // rather than rejecting every project-scoped run.
    let allowed_roots = [
        data_dir.join("loras"),
        data_dir.join("models"),
        data_dir.join("projects"),
    ];
    ensure_path_under(path, &allowed_roots, label)
}

pub(crate) fn resolve_dataset_item_path(
    settings: &Settings,
    dataset_root: &str,
    image_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let root = normalize_app_managed_path(settings, dataset_root, "Dataset root")?;
    let raw_image = Path::new(image_path.trim());
    if image_path.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let path = if raw_image.is_absolute() {
        normalize_absolute_path(raw_image)?
    } else {
        normalize_absolute_path(&root.join(raw_image))?
    };
    ensure_path_under(path, &[root], label)
}

pub(crate) fn project_path_for_payload(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Option<PathBuf>> {
    let Some(project_id) = optional_payload_string(payload, "projectId") else {
        return Ok(None);
    };
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    Ok(Some(PathBuf::from(project.path)))
}

/// Confine a client-supplied import *source* path (sc-8803 / F-002). LoRA/model
/// import jobs arrive over the unauthenticated local jobs API (LAN-exposed via
/// epic 4484), and the worker copies — or with `uploadedSourcePath: true`, moves —
/// the source into an app-listable target dir, so an unconfined source is an
/// arbitrary-file-read/exfiltration primitive (and move mode deletes the original).
/// The rust-api validates `sourcePath` at job creation, but the worker is the
/// stated trust boundary and must re-confine:
/// - uploaded sources (move mode) must live in the API's staged-upload cache,
///   `<data>/cache/<upload_cache>` (`lora-uploads` / `model-uploads`), matching
///   `cleanup_uploaded_import_source`;
/// - copy-mode sources must live under the app data dir or, for project-scoped
///   imports, the owning project's `loras` tree (resolved from the trusted
///   project store, mirroring `resolve_lora_import_target`).
///
/// Symlinks resolve before the root check, like `normalize_app_managed_lora_path`.
pub(crate) fn resolve_import_source_path(
    settings: &Settings,
    payload: &JsonObject,
    raw_path: &str,
    upload_cache: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let mut roots = Vec::new();
    if payload_bool(payload, "uploadedSourcePath") {
        let upload_root = settings.data_dir.join("cache").join(upload_cache);
        roots.push(normalize_absolute_path(&upload_root)?);
        roots.push(normalize_existing_or_absolute(&upload_root)?);
    } else {
        roots.push(normalized_data_dir(settings)?);
        roots.push(normalize_existing_or_absolute(&settings.data_dir)?);
        if let Some(project_path) = project_path_for_payload(settings, payload)? {
            let project_loras = project_path.join("loras");
            roots.push(normalize_absolute_path(&project_loras)?);
            roots.push(normalize_existing_or_absolute(&project_loras)?);
        }
    }
    let normalized = normalize_absolute_path(Path::new(raw_path))?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(resolved, &roots, label)
}

pub(crate) fn resolve_lora_import_target(
    settings: &Settings,
    payload: &JsonObject,
    fallback_target: PathBuf,
) -> WorkerResult<PathBuf> {
    let target = normalize_absolute_path(
        &optional_payload_string(payload, "targetDir")
            .map(PathBuf::from)
            .unwrap_or(fallback_target),
    )?;
    let mut allowed_roots = vec![normalize_absolute_path(&settings.data_dir.join("loras"))?];
    if let Some(project_path) = project_path_for_payload(settings, payload)? {
        allowed_roots.push(normalize_absolute_path(
            &project_path.join("loras").join("imports"),
        )?);
    }
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "LoRA import targetDir must be inside app-managed data/loras or project/loras/imports"
            .to_owned(),
    ))
}

pub(crate) fn resolve_model_import_target(
    settings: &Settings,
    payload: &JsonObject,
    fallback_target: PathBuf,
) -> WorkerResult<PathBuf> {
    let target = normalize_absolute_path(
        &optional_payload_string(payload, "targetDir")
            .map(PathBuf::from)
            .unwrap_or(fallback_target),
    )?;
    let allowed_roots = [normalize_absolute_path(&settings.data_dir.join("models"))?];
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "Model import targetDir must be inside app-managed data/models".to_owned(),
    ))
}

pub(crate) fn resolve_model_convert_output(
    settings: &Settings,
    output_dir: &str,
) -> WorkerResult<PathBuf> {
    let target = normalize_absolute_path(&PathBuf::from(output_dir))?;
    let allowed_root = normalize_absolute_path(&settings.data_dir.join("models"))?;
    if target.starts_with(&allowed_root) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "Model convert outputDir must be inside app-managed data/models".to_owned(),
    ))
}

pub(crate) fn model_manifest_target(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<PathBuf> {
    let manifest_path = normalize_absolute_path(&PathBuf::from(required_payload_string(
        payload,
        "manifestPath",
    )?))?;
    let allowed = [normalize_absolute_path(
        &settings
            .config_dir
            .join("manifests")
            .join("user.models.jsonc"),
    )?];
    if allowed.iter().any(|path| path == &manifest_path) {
        return Ok(manifest_path);
    }
    Err(WorkerError::InvalidPayload(
        "Model manifestPath must target the global user model manifest".to_owned(),
    ))
}

pub(crate) fn lora_manifest_target(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<PathBuf> {
    let manifest_path = normalize_absolute_path(&PathBuf::from(required_payload_string(
        payload,
        "manifestPath",
    )?))?;
    let mut allowed = vec![normalize_absolute_path(
        &settings
            .config_dir
            .join("manifests")
            .join("user.loras.jsonc"),
    )?];
    if let Some(project_path) = project_path_for_payload(settings, payload)? {
        allowed.push(normalize_absolute_path(
            &project_path.join("loras").join("manifest.jsonc"),
        )?);
    }
    if allowed.iter().any(|path| path == &manifest_path) {
        return Ok(manifest_path);
    }
    Err(WorkerError::InvalidPayload(
        "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest"
            .to_owned(),
    ))
}

pub(crate) fn safe_project_path(project_path: &Path, relative: &str) -> WorkerResult<PathBuf> {
    if relative.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Project-relative path is required.".to_owned(),
        ));
    }
    let mut path = project_path.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Unsafe project-relative path: {relative}"
                )))
            }
        }
    }
    Ok(path)
}

pub(crate) fn relative_path(root: &Path, path: &Path) -> WorkerResult<String> {
    // Project media paths are app-created filenames; keep recipe metadata best-effort
    // if a host path contains non-UTF-8 bytes.
    Ok(path
        .strip_prefix(root)
        .map_err(|_| WorkerError::InvalidPayload("Path is outside project.".to_owned()))?
        .to_string_lossy()
        .replace('\\', "/"))
}
