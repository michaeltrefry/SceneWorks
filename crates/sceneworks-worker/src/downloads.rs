use super::*;
use std::net::SocketAddr;

/// Download `url` to `target` on first use. Existing complete files are reused;
/// partial files resume with HTTP Range when the caller can provide `expected_size`.
/// The transfer shares model-download progress/cancel plumbing instead of buffering
/// the response body in memory.
// Shared by the macOS MLX runtime-weight downloads AND the candle InstantID lane (sc-5491): the
// off-Mac InstantID provider stages its SCRFD/ArcFace/IP-Adapter/ControlNet files via this same
// download-on-first-use path, so it broadened from macOS-only. (All helpers it calls — download_file,
// DownloadProgress, DownloadContext, HuggingFaceSnapshot — already build on every platform.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn ensure_cached_file(
    context: &DownloadContext<'_>,
    url: &str,
    target: &Path,
    label: &str,
    expected_size: Option<u64>,
) -> WorkerResult<PathBuf> {
    let expected_size = match expected_size {
        Some(size) => Some(size),
        None => remote_content_length(context.client, url).await?,
    };
    if let Ok(metadata) = tokio::fs::metadata(target).await {
        if expected_size
            .map(|expected| metadata.len() == expected)
            .unwrap_or(true)
        {
            return Ok(target.to_path_buf());
        }
    }
    if expected_size.is_none() && target.exists() {
        return Ok(target.to_path_buf());
    }
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let started_bytes = existing_download_bytes(target, expected_size).await?;
    let mut progress = DownloadProgress::new(
        label,
        started_bytes,
        expected_size,
        progress_report_interval(context.settings),
    );
    download_file(
        context,
        url,
        target,
        expected_size,
        None,
        label,
        &mut progress,
    )
    .await?;
    Ok(target.to_path_buf())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn remote_content_length(client: &reqwest::Client, url: &str) -> WorkerResult<Option<u64>> {
    // `url` is built from trusted operator/runtime configuration
    // (`Settings::huggingface_base_url`) plus validated HF path pieces. User-provided source URLs
    // use the separate `download_source_url` path with SSRF checks.
    let response = match client.head(url).send().await {
        Ok(response) => response,
        Err(_) => return Ok(None),
    };
    if response.status().is_success() {
        Ok(response.content_length().filter(|value| *value > 0))
    } else {
        Ok(None)
    }
}

/// Resolve a single Hugging Face file and stream it into an app cache target with
/// size-aware resume/progress. This is for first-use runtime weights that are not
/// installed through the full model-download flow.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn ensure_hf_cached_file(
    context: &DownloadContext<'_>,
    repo: &str,
    revision: &str,
    file: &str,
    target: &Path,
) -> WorkerResult<PathBuf> {
    let snapshot = HuggingFaceSnapshot::resolve(
        context.client,
        context.settings,
        repo,
        revision,
        &[file.to_owned()],
    )
    .await?;
    let Some(snapshot_file) = snapshot
        .files
        .into_iter()
        .find(|candidate| candidate.path == file)
    else {
        return Err(WorkerError::InvalidPayload(format!(
            "Hugging Face file {file} not found in {repo}."
        )));
    };
    ensure_cached_file(
        context,
        &snapshot_file.download_url,
        target,
        &snapshot_file.path,
        snapshot_file.size,
    )
    .await
}

#[derive(Debug, Clone)]
pub(crate) struct SnapshotFile {
    pub(crate) path: String,
    pub(crate) size: Option<u64>,
    pub(crate) download_url: String,
    /// SHA-256 of the file content from Hugging Face's `lfs.oid` (tree API
    /// `expand=1`). Present for LFS-tracked files (the weights); `None` for small
    /// non-LFS files (configs/tokenizers), whose integrity rides on the size check.
    pub(crate) sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct HuggingFaceSnapshot {
    pub(crate) files: Vec<SnapshotFile>,
}

impl HuggingFaceSnapshot {
    pub(crate) async fn resolve(
        client: &reqwest::Client,
        settings: &Settings,
        repo: &str,
        revision: &str,
        files: &[String],
    ) -> WorkerResult<Self> {
        let base_url = settings.huggingface_base_url.trim_end_matches('/');
        let tree_url = format!(
            "{base_url}/api/models/{}/tree/{}?recursive=1&expand=1",
            quote_path(repo),
            quote_path(revision)
        );
        let payload = with_hf_auth(settings, client.get(tree_url))
            .await
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        let entries = if let Some(entries) = payload.as_array() {
            entries.clone()
        } else {
            payload
                .get("siblings")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let snapshot_files = entries
            .iter()
            .filter_map(|entry| snapshot_file_from_entry(base_url, repo, revision, entry))
            .filter(|file| allow_pattern_matches(&file.path, files))
            .collect();
        Ok(Self {
            files: snapshot_files,
        })
    }

    pub(crate) fn total_bytes(&self) -> Option<u64> {
        self.files
            .iter()
            .try_fold(0_u64, |total, file| Some(total.saturating_add(file.size?)))
    }
}

pub(crate) fn snapshot_file_from_entry(
    base_url: &str,
    repo: &str,
    revision: &str,
    entry: &Value,
) -> Option<SnapshotFile> {
    let kind = entry.get("type").and_then(Value::as_str);
    if kind.is_some_and(|kind| kind != "file") {
        return None;
    }
    let path = entry
        .get("path")
        .or_else(|| entry.get("rfilename"))
        .and_then(Value::as_str)?;
    Some(SnapshotFile {
        path: path.to_owned(),
        size: entry.get("size").and_then(json_size_to_u64),
        download_url: format!(
            "{base_url}/{}/resolve/{}/{}",
            quote_path(repo),
            // Revisions are pre-validated by `model_jobs::validate_hf_revision`;
            // quote_path is the direct-download path's final URL-segment guard.
            quote_path(revision),
            quote_path(path)
        ),
        sha256: entry
            .get("lfs")
            .and_then(|lfs| lfs.get("oid"))
            .and_then(Value::as_str)
            .and_then(normalize_sha256),
    })
}

/// Normalize a candidate SHA-256 digest (from `lfs.oid` or an HF ETag): strip an
/// optional `sha256:` prefix and surrounding whitespace, lowercase it, and accept it
/// only if it is exactly 64 hex characters. Returns `None` for anything else (e.g. a
/// git blob SHA-1 ETag, a fallback blob name) so callers verify only real content
/// digests.
pub(crate) fn normalize_sha256(value: &str) -> Option<String> {
    let digest = value
        .trim()
        .trim_start_matches("sha256:")
        .trim()
        .to_ascii_lowercase();
    if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Some(digest)
    } else {
        None
    }
}

pub(crate) struct DownloadContext<'a> {
    pub(crate) api: &'a ApiClient,
    pub(crate) client: &'a reqwest::Client,
    pub(crate) settings: &'a Settings,
    pub(crate) job_id: &'a str,
    pub(crate) cancel_message: &'a str,
    pub(crate) fresh_download: bool,
}

const AUTO_RESUME_ATTEMPTS: usize = 1;

/// Download a single file to `dest` (resumable via HTTP Range), rejecting a truncated
/// response (size mismatch) and, when `expected_sha256` is provided, a corrupt one
/// (content-digest mismatch). On a digest mismatch the file is removed so a corrupt
/// artifact is never left behind (sc-6137). `label` names the file in the error.
async fn download_file(
    context: &DownloadContext<'_>,
    url: &str,
    dest: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
    label: &str,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    download_file_inner(context, url, dest, expected_size, label, progress).await?;
    if let Some(expected) = expected_sha256 {
        verify_file_sha256(dest, expected, label).await?;
    }
    Ok(())
}

/// Verify `path`'s SHA-256 equals `expected`; on mismatch, remove the file and return
/// an actionable error. A malformed/absent `expected` (not 64 hex) is treated as "no
/// digest available" and skipped — only a real content digest is enforced.
pub(crate) async fn verify_file_sha256(
    path: &Path,
    expected: &str,
    label: &str,
) -> WorkerResult<()> {
    let Some(expected) = normalize_sha256(expected) else {
        return Ok(());
    };
    let actual = sha256_file(path).await?;
    if actual != expected {
        let _ = tokio::fs::remove_file(path).await;
        return Err(WorkerError::InvalidPayload(format!(
            "{label} failed its integrity check (sha256 {actual}, but the source declares {expected}); \
             the download was corrupted. Re-download the file."
        )));
    }
    Ok(())
}

/// Stream `path` through SHA-256 on the blocking pool (weights are multi-GB; hashing
/// on the async runtime would stall heartbeats/cancel checks).
async fn sha256_file(path: &Path) -> WorkerResult<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> std::io::Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|error| {
        WorkerError::Io(std::io::Error::other(format!(
            "sha256 task failed: {error}"
        )))
    })?
    .map_err(WorkerError::Io)
}

/// Download a single file to `dest` (resumable via HTTP Range), reporting transfer
/// progress and rejecting a truncated response. `label` names the file in the
/// size-mismatch error.
async fn download_file_inner(
    context: &DownloadContext<'_>,
    url: &str,
    dest: &Path,
    expected_size: Option<u64>,
    label: &str,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    if context.fresh_download {
        let removed_bytes = remove_incomplete_download(dest, expected_size).await?;
        if removed_bytes > 0 {
            progress.discard_started_bytes(removed_bytes);
        }
    }
    let mut resume_attempts_remaining = if context.fresh_download {
        0
    } else {
        AUTO_RESUME_ATTEMPTS
    };
    loop {
        let existing_bytes = existing_download_bytes(dest, expected_size).await?;
        if expected_size.is_some_and(|size| existing_bytes == size) {
            return Ok(());
        }
        let mut request = context.client.get(url);
        if existing_bytes > 0 {
            request = request.header(header::RANGE, format!("bytes={existing_bytes}-"));
        }
        let response = with_hf_auth(context.settings, request).await.send().await?;
        let status = response.status();
        if status == StatusCode::RANGE_NOT_SATISFIABLE && existing_bytes > 0 {
            if let Some(expected) = expected_size {
                return Err(WorkerError::InvalidPayload(download_size_mismatch_message(
                    label,
                    existing_bytes,
                    expected,
                )));
            }
        }
        if !status.is_success() {
            return Err(WorkerError::Http(response.error_for_status().unwrap_err()));
        }
        let appending = existing_bytes > 0 && status == StatusCode::PARTIAL_CONTENT;
        if existing_bytes > 0 && !appending {
            progress.discard_started_bytes(existing_bytes);
        }
        let mut response = response;
        let mut output = if appending {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dest)
                .await?
        } else {
            tokio::fs::File::create(dest).await?
        };
        let mut interval = tokio::time::interval(progress.report_interval());
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // A tokio interval's first tick is immediate; consume it so the first chunk
        // doesn't spuriously fire a zero-byte progress report before any transfer.
        interval.tick().await;
        let mut transfer_error = None;
        loop {
            tokio::select! {
                chunk = response.chunk() => {
                    match chunk {
                        Ok(Some(chunk)) => {
                            output.write_all(&chunk).await?;
                            progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                        }
                        Ok(None) => break,
                        Err(error) => {
                            transfer_error = Some(WorkerError::from(error));
                            break;
                        }
                    }
                }
                _ = interval.tick() => {
                    report_download_progress(context, progress).await?;
                }
            }
        }
        output.flush().await?;
        if let Some(error) = transfer_error {
            if let Some(expected) = expected_size {
                let written = tokio::fs::metadata(dest).await?.len();
                if written == expected {
                    return Ok(());
                }
                if written < expected && resume_attempts_remaining > 0 {
                    resume_attempts_remaining -= 1;
                    continue;
                }
            }
            return Err(error);
        }
        // A truncated transfer (e.g. the server closes the stream at what looks like a
        // clean EOF) would otherwise be treated as success and the bad file only surface
        // as an opaque load failure later. When the expected size is known, verify it.
        // Short files are preserved so a later retry can resume them; overlong files are
        // discarded because appending would only move them farther away from the target.
        if let Some(expected) = expected_size {
            let written = tokio::fs::metadata(dest).await?.len();
            if written == expected {
                return Ok(());
            }
            if written < expected && resume_attempts_remaining > 0 {
                resume_attempts_remaining -= 1;
                continue;
            }
            if written > expected {
                let _ = tokio::fs::remove_file(dest).await;
            }
            return Err(WorkerError::InvalidPayload(download_size_mismatch_message(
                label, written, expected,
            )));
        }
        return Ok(());
    }
}

async fn remove_incomplete_download(path: &Path, expected_size: Option<u64>) -> WorkerResult<u64> {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return Ok(0);
    };
    let existing_bytes = metadata.len();
    if expected_size
        .map(|expected| metadata.len() != expected)
        .unwrap_or(true)
    {
        tokio::fs::remove_file(path).await?;
        return Ok(existing_bytes);
    }
    Ok(0)
}

fn format_bytes_with_exact(value: u64) -> String {
    format!("{} ({value} bytes)", format_bytes(value))
}

fn download_size_mismatch_message(label: &str, actual: u64, expected: u64) -> String {
    let delta = actual.abs_diff(expected);
    let direction = if actual < expected {
        "missing"
    } else {
        "extra"
    };
    format!(
        "{label} download ended at {} but expected {}; {} {}.",
        format_bytes_with_exact(actual),
        format_bytes_with_exact(expected),
        format_bytes_with_exact(delta),
        direction
    )
}

/// Download a Hugging Face snapshot as a flat file tree under `target_dir`. Used by
/// the model-import flow, which intentionally populates the app's import store.
pub(crate) async fn download_snapshot(
    context: &DownloadContext<'_>,
    target_dir: &Path,
    snapshot: &HuggingFaceSnapshot,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    for file in &snapshot.files {
        check_download_cancel(context).await?;
        let target_path = safe_join(target_dir, &file.path)?;
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        download_file(
            context,
            &file.download_url,
            &target_path,
            file.size,
            file.sha256.as_deref(),
            &file.path,
            progress,
        )
        .await?;
    }
    Ok(())
}

/// Download a Hugging Face snapshot into the standard hub cache layout under
/// `repo_dir` (`models--<org>--<name>`): content lands in `blobs/<etag>`, the
/// checkpoint is materialized as `snapshots/<commit>/<path>` (a relative symlink to
/// its blob, or a copy where symlinks are unavailable), and `refs/<rev>` records the
/// commit. This matches `huggingface_hub`, so HF-sourced downloads dedupe with other
/// tools and the Python loader instead of duplicating into the private app store
/// (sc-1904).
pub(crate) async fn download_snapshot_into_cache(
    context: &DownloadContext<'_>,
    repo_dir: &Path,
    revision: &str,
    snapshot: &HuggingFaceSnapshot,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    let blobs_dir = repo_dir.join("blobs");
    tokio::fs::create_dir_all(&blobs_dir).await?;
    // A no-redirect client so the metadata HEAD reads huggingface.co's headers
    // (X-Repo-Commit, and X-Linked-Etag for LFS) rather than the CDN's after a
    // redirect — exactly how huggingface_hub resolves an etag/commit.
    let meta_client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut commit: Option<String> = None;
    let mut placements: Vec<(String, String)> = Vec::with_capacity(snapshot.files.len());

    for file in &snapshot.files {
        check_download_cancel(context).await?;
        let head = with_hf_auth(context.settings, meta_client.head(&file.download_url))
            .await
            .send()
            .await?;
        if commit.is_none() {
            commit = header_value(&head, "x-repo-commit");
        }
        let etag = header_value(&head, "x-linked-etag")
            .or_else(|| header_value(&head, "etag"))
            .map(|value| normalize_etag(&value))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| blob_fallback_name(&file.path));
        download_file(
            context,
            &file.download_url,
            &blobs_dir.join(&etag),
            file.size,
            // The blob is named by its etag (= the LFS sha256), and the tree API
            // reports the same digest as `lfs.oid`; verify the content against it.
            file.sha256.as_deref(),
            &file.path,
            progress,
        )
        .await?;
        placements.push((file.path.clone(), etag));
    }

    // Materialize the snapshot once every blob is present: refs/<rev> -> commit and
    // snapshots/<commit>/<path> -> ../../blobs/<etag>.
    let commit = commit.unwrap_or_else(|| revision.to_owned());
    let refs_dir = repo_dir.join("refs");
    tokio::fs::create_dir_all(&refs_dir).await?;
    tokio::fs::write(refs_dir.join(revision), commit.as_bytes()).await?;
    let snapshot_dir = repo_dir.join("snapshots").join(&commit);
    for (relpath, etag) in &placements {
        let link = safe_join(&snapshot_dir, relpath)?;
        if let Some(parent) = link.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        if tokio::fs::symlink_metadata(&link).await.is_ok() {
            let _ = tokio::fs::remove_file(&link).await;
        }
        let depth = link
            .parent()
            .and_then(|parent| parent.strip_prefix(repo_dir).ok())
            .map(|relative| relative.components().count())
            .unwrap_or(2);
        let mut rel_target = PathBuf::new();
        for _ in 0..depth {
            rel_target.push("..");
        }
        rel_target.push("blobs");
        rel_target.push(etag);
        if !link_blob(&rel_target, &link).await {
            tokio::fs::copy(blobs_dir.join(etag), &link).await?;
        }
    }
    Ok(())
}

fn header_value(response: &reqwest::Response, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Strip the surrounding quotes and any weak-validator prefix HTTP/HF put around an
/// ETag, leaving the bare blob name huggingface_hub uses.
fn normalize_etag(raw: &str) -> String {
    raw.trim()
        .trim_start_matches("W/")
        .trim_matches('"')
        .to_owned()
}

/// Blob name when the server returns no etag (a non-HF stub or an endpoint that
/// omits ETag): a filesystem-safe rendering of the repo path. Keeps the download
/// working; only weakens cross-app dedup for that one file.
fn blob_fallback_name(path: &str) -> String {
    path.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// Create a relative symlink from `link` to its blob, returning whether it
/// succeeded. Mirrors huggingface_hub: symlink where supported, and the caller
/// copies instead when this returns false (Windows without privilege).
async fn link_blob(rel_target: &Path, link: &Path) -> bool {
    #[cfg(unix)]
    {
        tokio::fs::symlink(rel_target, link).await.is_ok()
    }
    #[cfg(windows)]
    {
        tokio::fs::symlink_file(rel_target, link).await.is_ok()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (rel_target, link);
        false
    }
}

pub(crate) async fn download_lora_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "LoRA",
        context.settings.max_lora_url_bytes,
    )
    .await
}

pub(crate) async fn download_model_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "Model",
        context.settings.max_model_url_bytes,
    )
    .await
}

pub(crate) async fn download_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
    source_label: &str,
    max_bytes: u64,
) -> WorkerResult<()> {
    let url =
        parse_lora_source_url_with_private(source_url, context.settings.allow_private_lora_urls)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    let file_name = lora_source_url_file_name(source_url)
        .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    tokio::fs::create_dir_all(target_dir).await?;
    let target_path = target_dir.join(file_name);

    // Attach a stored credential matching the source host. Bearer tokens ride an
    // Authorization header (dropped on cross-host redirects below); query tokens
    // are baked into the request URL and never carried onto a redirect target. The
    // secret is resolved lazily (env/file-store, or the macOS desktop socket on
    // first use) so a no-credential install never touches the keychain (sc-5891).
    let credential = crate::credentials_ipc::resolve_credential_for_host(
        context.settings,
        url.host_str().unwrap_or_default(),
    )
    .await;
    let request_url = match &credential {
        Some(cred) if cred.scheme == CredentialScheme::Query => {
            let mut authed = url.clone();
            authed.query_pairs_mut().append_pair("token", &cred.token);
            authed.to_string()
        }
        _ => source_url.to_owned(),
    };
    let bearer = match &credential {
        Some(cred) if cred.scheme == CredentialScheme::Bearer => Some(cred.token.as_str()),
        _ => None,
    };

    let client = source_url_client_for_request(context.settings, &request_url).await?;
    let total_bytes = lora_source_content_length(&client, &request_url, bearer).await?;
    if total_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let existing_bytes = existing_download_bytes(&target_path, total_bytes).await?;
    if total_bytes.is_some_and(|total| total > 0 && existing_bytes == total) {
        return Ok(());
    }
    let range_header = (existing_bytes > 0).then(|| format!("bytes={existing_bytes}-"));
    let mut response = send_source_url_with_redirects(
        context.settings,
        &request_url,
        &client,
        bearer,
        range_header.as_deref(),
    )
    .await?;
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        let range_total = response
            .headers()
            .get(header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(content_range_total);
        if total_bytes
            .or(range_total)
            .is_some_and(|total| total > 0 && existing_bytes == total)
        {
            return Ok(());
        }
    }
    response = response.error_for_status()?;
    let appending = existing_bytes > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    let expected_bytes = total_bytes.or_else(|| {
        response.content_length().map(|remaining| {
            if appending {
                existing_bytes + remaining
            } else {
                remaining
            }
        })
    });
    if expected_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let mut progress = DownloadProgress::new(
        source_url,
        if appending { existing_bytes } else { 0 },
        expected_bytes,
        progress_report_interval(context.settings),
    );
    let mut output = if appending {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target_path)
            .await?
    } else {
        tokio::fs::File::create(&target_path).await?
    };
    let mut interval = tokio::time::interval(progress.report_interval());
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    interval.tick().await;
    loop {
        tokio::select! {
            chunk = response.chunk() => {
                let Some(chunk) = chunk? else {
                    break;
                };
                check_download_cancel(context).await?;
                output.write_all(&chunk).await?;
                progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                if progress.downloaded_bytes() > max_bytes {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{source_label} sourceUrl exceeds the {} limit",
                        format_bytes(max_bytes)
                    )));
                }
            }
            _ = interval.tick() => {
                report_download_progress(context, &progress).await?;
            }
        }
    }
    output.flush().await?;
    if expected_bytes.is_some_and(|expected| progress.downloaded_bytes() != expected) {
        return Err(WorkerError::InvalidPayload(download_size_mismatch_message(
            &format!("{source_label} sourceUrl"),
            progress.downloaded_bytes(),
            expected_bytes.unwrap_or_default(),
        )));
    }
    Ok(())
}

/// Maximum redirect hops to follow on an authenticated source-URL download.
const MAX_SOURCE_URL_REDIRECTS: usize = 5;

/// The stored credential whose host matches `host` (case-insensitive exact match),
/// or `None` when nothing matches.
pub(crate) fn credential_for_host<'a>(
    settings: &'a Settings,
    host: &str,
) -> Option<&'a WorkerCredential> {
    let host = host.trim().to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    settings
        .credentials
        .iter()
        .find(|credential| credential.host == host)
}

/// GET `initial_url`, manually following up to `MAX_SOURCE_URL_REDIRECTS` hops
/// (the download client uses `Policy::none()` so we control each hop). Every
/// redirect target is re-validated for SSRF (scheme + host/DNS), then fetched
/// with a client pinned to the validated socket addresses. The bearer
/// `Authorization` header is dropped on any cross-host hop so a token never
/// leaks to a CDN. Returns the final non-redirect response without
/// `error_for_status`, so the caller can still inspect
/// `RANGE_NOT_SATISFIABLE`.
async fn send_source_url_with_redirects(
    settings: &Settings,
    initial_url: &str,
    initial_client: &reqwest::Client,
    bearer: Option<&str>,
    range_header: Option<&str>,
) -> WorkerResult<reqwest::Response> {
    let mut current_url = initial_url.to_owned();
    let mut current_host = reqwest::Url::parse(&current_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase));
    let mut bearer = bearer.map(str::to_owned);
    let mut client = initial_client.clone();
    for _ in 0..=MAX_SOURCE_URL_REDIRECTS {
        let mut request = client.get(&current_url);
        if let Some(token) = &bearer {
            request = request.bearer_auth(token);
        }
        if let Some(range) = range_header {
            request = request.header(header::RANGE, range);
        }
        let response = request.send().await?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        let location = response
            .headers()
            .get(header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "sourceUrl redirect was missing a Location header".to_owned(),
                )
            })?;
        let base = reqwest::Url::parse(&current_url)
            .map_err(|_| WorkerError::InvalidPayload("sourceUrl was invalid".to_owned()))?;
        let next = base.join(location).map_err(|_| {
            WorkerError::InvalidPayload("sourceUrl redirect target was invalid".to_owned())
        })?;
        if !matches!(next.scheme(), "http" | "https") {
            return Err(WorkerError::InvalidPayload(
                "sourceUrl redirect must use http or https".to_owned(),
            ));
        }
        let next_host = next.host_str().map(str::to_ascii_lowercase);
        if next_host != current_host {
            // Cross-host redirect: never carry the bearer token to a new origin.
            bearer = None;
        }
        current_host = next_host;
        current_url = next.to_string();
        client = source_url_client_for_url(settings, &next).await?;
    }
    Err(WorkerError::InvalidPayload(
        "sourceUrl exceeded the redirect limit".to_owned(),
    ))
}

async fn source_url_client_for_request(
    settings: &Settings,
    request_url: &str,
) -> WorkerResult<reqwest::Client> {
    let url = reqwest::Url::parse(request_url)
        .map_err(|_| WorkerError::InvalidPayload("sourceUrl was invalid".to_owned()))?;
    source_url_client_for_url(settings, &url).await
}

async fn source_url_client_for_url(
    settings: &Settings,
    url: &reqwest::Url,
) -> WorkerResult<reqwest::Client> {
    let validated_addrs = validate_lora_url_dns(settings, url).await?;
    build_source_url_client(url, validated_addrs.as_deref())
}

pub(crate) fn build_source_url_client(
    url: &reqwest::Url,
    validated_addrs: Option<&[SocketAddr]>,
) -> WorkerResult<reqwest::Client> {
    let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
    if let (Some(host), Some(addrs)) = (url.host_str(), validated_addrs) {
        builder = builder.resolve_to_addrs(host, addrs);
    }
    Ok(builder.build()?)
}

pub(crate) async fn lora_source_content_length(
    client: &reqwest::Client,
    request_url: &str,
    bearer: Option<&str>,
) -> WorkerResult<Option<u64>> {
    let mut request = client.head(request_url);
    if let Some(token) = bearer {
        request = request.bearer_auth(token);
    }
    let response = request.send().await?;
    if response.status().is_success() {
        return Ok(response.content_length().filter(|value| *value > 0));
    }
    // A redirecting or auth-gated download endpoint (e.g. Civit.ai) can't report a
    // size via HEAD; fall back to the streamed GET response's content length.
    if response.status().is_redirection() {
        return Ok(None);
    }
    if matches!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED
            | StatusCode::NOT_IMPLEMENTED
            | StatusCode::FORBIDDEN
            | StatusCode::UNAUTHORIZED
    ) {
        return Ok(None);
    }
    response.error_for_status()?;
    Ok(None)
}

pub(crate) fn content_range_total(value: &str) -> Option<u64> {
    value
        .rsplit_once('/')
        .and_then(|(_, total)| total.trim().parse::<u64>().ok())
}

pub(crate) async fn validate_lora_url_dns(
    settings: &Settings,
    url: &reqwest::Url,
) -> WorkerResult<Option<Vec<SocketAddr>>> {
    if settings.allow_private_lora_urls {
        return Ok(None);
    }
    let Some(host) = url.host_str() else {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host is not allowed".to_owned(),
        ));
    };
    let port = url.port_or_known_default().unwrap_or(443);
    if let Ok(address) = host.parse::<IpAddr>() {
        validate_public_ip(address)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
        return Ok(Some(vec![SocketAddr::new(address, port)]));
    }
    let mut addrs = Vec::new();
    for address in tokio::net::lookup_host((host, port)).await? {
        validate_public_ip(address.ip())
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
        addrs.push(address);
    }
    if addrs.is_empty() {
        Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host did not resolve".to_owned(),
        ))
    } else {
        Ok(Some(addrs))
    }
}

pub(crate) async fn report_download_progress(
    context: &DownloadContext<'_>,
    progress: &DownloadProgress<'_>,
) -> WorkerResult<()> {
    heartbeat(
        context.api,
        context.settings,
        WorkerStatus::Busy,
        Some(context.job_id),
    )
    .await?;
    update_job(context.api, context.job_id, progress.payload()).await?;
    check_download_cancel(context).await
}

async fn check_download_cancel(context: &DownloadContext<'_>) -> WorkerResult<()> {
    if cancel_requested_peek(context.api, context.job_id).await {
        mark_job_canceled(context.api, context.job_id, context.cancel_message).await?;
        return Err(WorkerError::Canceled(context.cancel_message.to_owned()));
    }
    Ok(())
}

pub(crate) struct DownloadProgress<'a> {
    repo: &'a str,
    started_bytes: u64,
    transferred_bytes: u64,
    total_bytes: Option<u64>,
    started_at: Instant,
    report_interval: Duration,
}

impl<'a> DownloadProgress<'a> {
    pub(crate) fn new(
        repo: &'a str,
        started_bytes: u64,
        total_bytes: Option<u64>,
        report_interval: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            repo,
            started_bytes,
            transferred_bytes: 0,
            total_bytes,
            started_at: now,
            report_interval,
        }
    }

    fn downloaded_bytes(&self) -> u64 {
        self.started_bytes.saturating_add(self.transferred_bytes)
    }

    fn record_transferred(&mut self, bytes: u64) {
        self.transferred_bytes = self.transferred_bytes.saturating_add(bytes);
    }

    fn discard_started_bytes(&mut self, bytes: u64) {
        self.started_bytes = self.started_bytes.saturating_sub(bytes);
    }

    fn report_interval(&self) -> Duration {
        self.report_interval
    }

    fn payload(&self) -> ProgressRequest {
        download_progress_payload(
            self.repo,
            self.downloaded_bytes(),
            self.total_bytes,
            self.started_bytes,
            self.started_at.elapsed(),
        )
    }
}

pub fn download_progress_payload(
    repo: &str,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    started_bytes: u64,
    elapsed: Duration,
) -> ProgressRequest {
    let transferred_bytes = downloaded_bytes.saturating_sub(started_bytes);
    let elapsed_seconds = elapsed.as_secs_f64().max(0.001);
    let rate = transferred_bytes as f64 / elapsed_seconds;
    let eta_seconds = total_bytes.and_then(|total| {
        if rate > 0.0 {
            let remaining = total.saturating_sub(downloaded_bytes) as f64;
            Some(number_from_f64((remaining / rate).max(0.0)))
        } else {
            None
        }
    });

    let (progress, message) = if let Some(total) = total_bytes {
        let ratio = if total == 0 {
            1.0
        } else {
            (downloaded_bytes as f64 / total as f64).clamp(0.0, 1.0)
        };
        let remaining = total.saturating_sub(downloaded_bytes);
        (
            0.1 + ratio * 0.85,
            format!(
                "Downloading {repo}: {} of {} ({} left).",
                format_bytes(downloaded_bytes),
                format_bytes(total),
                format_bytes(remaining)
            ),
        )
    } else {
        (
            0.1,
            format!(
                "Downloading {repo}: {} written.",
                format_bytes(downloaded_bytes)
            ),
        )
    };

    progress_payload(
        JobStatus::Downloading,
        ProgressStage::Downloading,
        progress,
        &message,
        None,
        None,
        eta_seconds,
    )
}

#[cfg(all(test, target_os = "macos"))]
mod ensure_cached_file_tests {
    use super::{ensure_cached_file, DownloadContext};
    use crate::{
        ApiClient, Settings, DEFAULT_HUGGINGFACE_BASE_URL, DEFAULT_MAX_LORA_URL_BYTES,
        DEFAULT_MAX_MODEL_URL_BYTES,
    };

    /// sc-4283 / F-MLXW-22: when the target already exists, `ensure_cached_file`
    /// returns it without any network access (the cache-hit short-circuit shared
    /// by all the download-on-first-use weight fetchers). A bogus URL proves no
    /// request is made.
    #[tokio::test]
    async fn returns_existing_target_without_downloading() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("weights").join("model.safetensors");
        tokio::fs::create_dir_all(target.parent().unwrap())
            .await
            .expect("parent dir");
        tokio::fs::write(&target, b"already here")
            .await
            .expect("seed target");

        let client = reqwest::Client::new();
        let settings = Settings {
            api_url: "http://127.0.0.1:1".to_owned(),
            access_token: None,
            data_dir: dir.path().join("data"),
            config_dir: dir.path().join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 5,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
        };
        let api = ApiClient::new(&settings);
        let context = DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
            fresh_download: false,
        };
        let resolved = ensure_cached_file(
            &context,
            "http://invalid.invalid/should-not-fetch",
            &target,
            "test weights",
            Some(12),
        )
        .await
        .expect("cache hit returns without downloading");
        assert_eq!(resolved, target);
        // Content untouched (no overwrite).
        assert_eq!(
            tokio::fs::read(&target).await.expect("read"),
            b"already here"
        );
    }
}
