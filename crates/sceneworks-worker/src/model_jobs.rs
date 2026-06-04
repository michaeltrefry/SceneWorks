use super::*;

pub(crate) async fn run_model_download_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = match required_payload_string(&job.payload, "repo") {
        Ok(repo) => repo,
        Err(error) => {
            fail_job(
                api,
                &job.id,
                "Model download is missing a repository.",
                Some(error.to_string()),
            )
            .await?;
            return Ok(());
        }
    };
    let files = payload_string_array(&job.payload, "files");
    let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
    let fresh_download = optional_payload_string(&job.payload, "downloadAction") == Some("fresh");
    // The worker is the trust boundary (jobs API is unauthenticated for local use), so a
    // client-supplied targetDir must be constrained to app-managed data/models the same way
    // import jobs are, not used verbatim.
    let target_dir = resolve_model_import_target(
        settings,
        &job.payload,
        settings
            .data_dir
            .join("models")
            .join(safe_download_dir(repo)),
    )?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.1,
            &format!("Downloading {repo}: estimating size."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Model download canceled before transfer started.",
    )
    .await?;

    if let Some(cache_path) =
        download_model_with_hf_cli(api, settings, job, repo, revision, &files, &target_dir).await?
    {
        if !reconcile_downloaded_model_family(api, job, &cache_path).await? {
            return Ok(());
        }
        let mut result = JsonObject::new();
        result.insert(
            "modelId".to_owned(),
            job.payload.get("modelId").cloned().unwrap_or(Value::Null),
        );
        result.insert("repo".to_owned(), Value::String(repo.to_owned()));
        result.insert(
            "path".to_owned(),
            Value::String(cache_path.display().to_string()),
        );
        result.insert(
            "storage".to_owned(),
            Value::String("huggingface_cache".to_owned()),
        );
        result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                "Model download completed in the Hugging Face cache.",
                None,
                Some(result),
                None,
            ),
        )
        .await?;
        return Ok(());
    }

    // Download into the standard Hugging Face hub cache (models--<org>--<name>),
    // not the private app store, so HF-sourced weights dedupe with other tools and
    // the Python loader instead of being duplicated under data/models (sc-1904).
    let repo_dir = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    let snapshot =
        HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
    if let Some(total_bytes) = snapshot.total_bytes() {
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Downloading,
                ProgressStage::Downloading,
                0.1,
                &format!("Downloading {repo}: 0 B of {}.", format_bytes(total_bytes)),
                None,
                None,
                None,
            ),
        )
        .await?;
    }

    let mut progress = DownloadProgress::new(
        repo,
        directory_size(&repo_dir.join("blobs")).await,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    download_snapshot_into_cache(
        &DownloadContext {
            api,
            client: http_client,
            settings,
            job_id: &job.id,
            cancel_message: "Model download canceled by user.",
            fresh_download,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let cache_path = huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(repo_dir);
    // A lightweight install marker stays in the app store (parity with the CLI
    // path's marker_dir) so the catalog's data/models pointer and bookkeeping
    // remain intact; the weights themselves live only in the shared HF cache.
    write_model_install_marker(&target_dir, &job.payload, repo, &job.id).await?;

    if !reconcile_downloaded_model_family(api, job, &cache_path).await? {
        return Ok(());
    }

    let mut result = JsonObject::new();
    result.insert(
        "modelId".to_owned(),
        job.payload.get("modelId").cloned().unwrap_or(Value::Null),
    );
    result.insert("repo".to_owned(), Value::String(repo.to_owned()));
    result.insert(
        "path".to_owned(),
        Value::String(cache_path.display().to_string()),
    );
    result.insert(
        "storage".to_owned(),
        Value::String("huggingface_cache".to_owned()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Model download completed in the Hugging Face cache.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// Convert a model's native (diffusers) checkpoint into the local MLX format on
/// macOS/Apple Silicon. The native checkpoint must already be downloaded into the
/// Hugging Face cache (via a model_download job); this shells out to the venv's
/// Python `mlx_video.convert_wan` tool, which is where MLX/torch live. The desktop
/// shell points `SCENEWORKS_PYTHON` at the bundled interpreter (mirrors
/// `SCENEWORKS_FFMPEG`); dev/server fall back to `python3` on PATH.
///
/// Real conversion is exercised on Mac hardware in sc-1509; this wires the tracked
/// job, progress, cancellation, and failure surfacing.
pub(crate) async fn run_model_convert_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let model_id = required_payload_string(&job.payload, "modelId")?.to_owned();
    let source_repo = required_payload_string(&job.payload, "sourceRepo")?.to_owned();
    let output_dir = required_payload_string(&job.payload, "outputDir")?.to_owned();
    let dtype = optional_payload_string(&job.payload, "dtype")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("bfloat16")
        .to_owned();
    // Optional MLX quantization. `quantizeOnly` quantizes an already-converted bf16
    // MLX dir (turnkey models); otherwise quantization rides on the native->MLX
    // conversion. `bits`/`group-size` are validated by the convert tool's choices.
    let quantize_only = payload_bool(&job.payload, "quantizeOnly");
    let quantize_bits = job.payload.get("quantizeBits").and_then(Value::as_u64);
    let quantize_group_size = job.payload.get("quantizeGroupSize").and_then(Value::as_u64);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            &format!("Preparing MLX conversion for {model_id}."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "MLX conversion canceled before it started.").await?;

    let Some(checkpoint_dir) = huggingface_snapshot_dir(&settings.data_dir, &source_repo) else {
        fail_job(
            api,
            &job.id,
            "Native checkpoint is not downloaded.",
            Some(format!(
                "Download {source_repo} before converting it to MLX."
            )),
        )
        .await?;
        return Ok(());
    };

    // Converter discriminator (sc-2235). Absent => the default mlx-video Wan
    // converter (existing behavior). "flux2_klein_diffusers" => convert a
    // FLUX.2-klein single-file fine-tune into a diffusers dir via the mlx-flux
    // sidecar venv, borrowing VAE/text-encoder/tokenizer from an installed base.
    let converter = optional_payload_string(&job.payload, "converter")
        .map(str::to_owned)
        .unwrap_or_default();
    let is_flux2_klein = converter == "flux2_klein_diffusers";

    let (python, flux2_source_file, flux2_base_dir, flux2_script) = if is_flux2_klein {
        let source_file_name = required_payload_string(&job.payload, "sourceFile")?.to_owned();
        let base_repo = required_payload_string(&job.payload, "baseRepo")?.to_owned();
        let source_file = checkpoint_dir.join(&source_file_name);
        if !source_file.is_file() {
            fail_job(
                api,
                &job.id,
                "Converted-model source file is missing.",
                Some(format!("Expected {source_file_name} in {source_repo}.")),
            )
            .await?;
            return Ok(());
        }
        let Some(base_dir) = huggingface_snapshot_dir(&settings.data_dir, &base_repo) else {
            fail_job(
                api,
                &job.id,
                "Base FLUX.2-klein model is not installed.",
                Some(format!(
                    "Install {base_repo} before converting {model_id} — its VAE, text encoder, \
                     and tokenizer are reused."
                )),
            )
            .await?;
            return Ok(());
        };
        let py = std::env::var("SCENEWORKS_MLX_FLUX_PYTHON")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "SCENEWORKS_MLX_FLUX_PYTHON is not set; the mlx-flux sidecar venv is required \
                     to convert FLUX.2-klein fine-tunes."
                        .to_owned(),
                )
            })?;
        let script = std::env::var("SCENEWORKS_MLX_FLUX_CONVERT")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "SCENEWORKS_MLX_FLUX_CONVERT is not set; cannot locate mlx_flux_convert.py."
                        .to_owned(),
                )
            })?;
        (py, Some(source_file), Some(base_dir), Some(script))
    } else {
        let py = std::env::var("SCENEWORKS_PYTHON")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "python3".to_owned());
        (py, None, None, None)
    };

    // Convert into a unique temp sibling and only promote it on success, so a
    // canceled/failed conversion never leaves a partial directory that the catalog
    // and adapter would treat as a ready model (convert tools write config.json
    // before all weight shards).
    // Constrain the client-supplied outputDir to app-managed data/models the same way import
    // jobs constrain targetDir; the worker is the trust boundary, so never create/rename a
    // converted model tree to an arbitrary location.
    let final_dir = resolve_model_convert_output(settings, &output_dir)?;
    let parent = final_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tokio::fs::create_dir_all(&parent).await?;
    let temp_dir = parent.join(format!(
        ".{}.converting-{}",
        final_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("mlx"),
        job.id
    ));
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.2,
            &format!("Converting {model_id} to MLX ({dtype}). This can take several minutes."),
            None,
            None,
            None,
        ),
    )
    .await?;

    let mut command = Command::new(&python);
    if is_flux2_klein {
        // Self-contained converter script in the scene_worker package, run by the
        // mlx-flux sidecar python: --source-file <single-file> --base-dir <base
        // klein snapshot> --out-dir <temp>. It validates the converted transformer
        // against the base diffusers layout and assembles the borrowed components.
        command
            .arg(flux2_script.as_ref().expect("flux2 script resolved"))
            .arg("--source-file")
            .arg(flux2_source_file.as_ref().expect("flux2 source resolved"))
            .arg("--base-dir")
            .arg(flux2_base_dir.as_ref().expect("flux2 base resolved"))
            .arg("--out-dir")
            .arg(&temp_dir);
    } else {
        command
            .arg("-m")
            .arg("mlx_video.convert_wan")
            .arg("--checkpoint-dir")
            .arg(&checkpoint_dir)
            .arg("--output-dir")
            .arg(&temp_dir)
            .arg("--dtype")
            .arg(&dtype)
            .arg("--model-version")
            .arg("auto");
        if quantize_only {
            command.arg("--quantize-only");
        } else if quantize_bits.is_some() {
            command.arg("--quantize");
        }
        if let Some(bits) = quantize_bits {
            command.arg("--bits").arg(bits.to_string());
        }
        if let Some(group_size) = quantize_group_size {
            command.arg("--group-size").arg(group_size.to_string());
        }
    }
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Failed to start MLX conversion ({python}). Ensure the worker venv has \
                 mlx-video-with-audio installed: {error}"
            ))
        })?;

    let mut stderr = child.stderr.take();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        if let Some(stderr) = stderr.as_mut() {
            let _ = stderr.read_to_end(&mut bytes).await;
        }
        bytes
    });

    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let status = loop {
        tokio::select! {
            status = child.wait() => break status?,
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                if let Err(error) =
                    check_cancel(api, &job.id, "MLX conversion canceled by user.").await
                {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
                    return Err(error);
                }
            }
        }
    };
    let stderr_bytes = stderr_task.await.unwrap_or_default();

    if !status.success() {
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        let stderr_text = String::from_utf8_lossy(&stderr_bytes);
        // Keep the tail (char-safe) so the job error surfaces the real failure.
        let tail: String = stderr_text.trim().chars().rev().take(1200).collect();
        let detail: String = tail.chars().rev().collect();
        return Err(WorkerError::InvalidPayload(format!(
            "MLX conversion failed (exit {}). {detail}",
            status.code().unwrap_or(-1),
        )));
    }

    // Promote the completed conversion atomically; on any rename failure the partial
    // temp dir is removed so it can't be picked up later.
    if let Err(error) = finalize_converted_dir(&temp_dir, &final_dir).await {
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        return Err(error);
    }

    let mut result = JsonObject::new();
    result.insert("modelId".to_owned(), Value::String(model_id));
    result.insert("sourceRepo".to_owned(), Value::String(source_repo));
    result.insert(
        "path".to_owned(),
        Value::String(final_dir.display().to_string()),
    );
    result.insert("storage".to_owned(), Value::String("mlx_local".to_owned()));
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "MLX conversion completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// Resolve the local Hugging Face snapshot directory for a cached repo (the dir that
/// actually holds the checkpoint files). Prefers the commit referenced by
/// `refs/main`, else the first snapshot directory. Returns `None` when the repo is
/// not present in the cache.
pub(crate) fn huggingface_snapshot_dir(data_dir: &Path, repo: &str) -> Option<PathBuf> {
    let repo_dir = huggingface_repo_cache_path(data_dir, repo)?;
    let snapshots = repo_dir.join("snapshots");
    if let Ok(rev) = std::fs::read_to_string(repo_dir.join("refs").join("main")) {
        let candidate = snapshots.join(rev.trim());
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    std::fs::read_dir(&snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
}

/// Atomically promote a freshly converted temp directory to its final location,
/// replacing any stale directory there. On error the final location is left
/// untouched (the caller removes the temp dir), so a complete `final_dir` only ever
/// appears after a fully successful conversion.
pub(crate) async fn finalize_converted_dir(temp_dir: &Path, final_dir: &Path) -> WorkerResult<()> {
    if final_dir.exists() {
        tokio::fs::remove_dir_all(final_dir).await?;
    }
    if let Some(parent) = final_dir.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::rename(temp_dir, final_dir).await?;
    Ok(())
}

pub(crate) async fn download_model_with_hf_cli(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    revision: &str,
    files: &[String],
    marker_dir: &Path,
) -> WorkerResult<Option<PathBuf>> {
    let Some(program) = hf_cli_program().await else {
        return Ok(None);
    };
    if settings.huggingface_base_url.trim_end_matches('/') != DEFAULT_HUGGINGFACE_BASE_URL {
        return Ok(None);
    }
    let cache_dir = huggingface_hub_cache_dir(&settings.data_dir);
    tokio::fs::create_dir_all(&cache_dir).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.12,
            &format!("Downloading {repo} into the Hugging Face cache."),
            None,
            None,
            None,
        ),
    )
    .await?;

    let mut command = Command::new(program);
    command
        .arg("download")
        .arg(repo)
        .arg("--repo-type")
        .arg("model")
        .arg("--revision")
        .arg(revision)
        .arg("--cache-dir")
        .arg(&cache_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    configure_hf_cli_environment(&mut command);
    if let Some(token) = &settings.huggingface_token {
        command.env("HF_TOKEN", token);
    }
    for pattern in files {
        command.arg("--include").arg(pattern);
    }
    let fresh_download = optional_payload_string(&job.payload, "downloadAction") == Some("fresh");
    if fresh_download {
        command.arg("--force-download");
    }

    let mut child = command.spawn().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "Failed to start Hugging Face CLI. Falling back to direct downloads is only possible when the CLI is absent, not when it fails to launch: {error}"
        ))
    })?;
    let mut stderr = child.stderr.take();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        if let Some(stderr) = stderr.as_mut() {
            let _ = stderr.read_to_end(&mut bytes).await;
        }
        bytes
    });
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let status = loop {
        tokio::select! {
            status = child.wait() => break status?,
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                if let Err(error) = check_cancel(api, &job.id, "Model download canceled by user.").await {
                    let _ = child.kill().await;
                    return Err(error);
                }
            }
        }
    };
    let stderr = stderr_task.await.unwrap_or_default();
    let cache_path = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr);
        // Some Windows installs run the Python-based HF CLI with a legacy stdio
        // codepage. The download can complete, then the process exits non-zero
        // while printing a Unicode checkmark/progress footer. If the cache now has
        // a snapshot, keep the completed transfer instead of failing the job.
        if hf_cli_encoding_failure(&stderr)
            && huggingface_snapshot_dir(&settings.data_dir, repo).is_some()
        {
            write_model_install_marker(marker_dir, &job.payload, repo, &job.id).await?;
            return Ok(Some(cache_path));
        }
        let detail = bounded_tail(&stderr, 10, 2000);
        let message = if detail.trim().is_empty() {
            "Hugging Face CLI download failed without stderr output.".to_owned()
        } else {
            format!("Hugging Face CLI download failed:\n{detail}")
        };
        return Err(WorkerError::InvalidPayload(message));
    }

    write_model_install_marker(marker_dir, &job.payload, repo, &job.id).await?;
    Ok(Some(cache_path))
}

pub(crate) const HF_CLI_UTF8_ENV: [(&str, &str); 3] = [
    ("PYTHONUTF8", "1"),
    ("PYTHONIOENCODING", "utf-8"),
    ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
];

pub(crate) fn configure_hf_cli_environment(command: &mut Command) {
    for (key, value) in HF_CLI_UTF8_ENV {
        command.env(key, value);
    }
}

pub(crate) fn hf_cli_encoding_failure(stderr: &str) -> bool {
    let normalized = stderr.to_ascii_lowercase();
    normalized.contains("charmap")
        && (normalized.contains("codec can't encode")
            || normalized.contains("unicodeencodeerror")
            || normalized.contains("character maps to <undefined>"))
}

pub(crate) async fn hf_cli_program() -> Option<&'static str> {
    for program in ["hf", "huggingface-cli"] {
        let status = Command::new(program)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        if status.is_ok_and(|status| status.success()) {
            return Some(program);
        }
    }
    None
}

/// Locates the first `.safetensors` under `dir`, reads its header, and
/// runs the architecture detector. Returns `Ok(None)` when no header is
/// available or the signature is inconclusive. Returns `Err(message)`
/// only when a file was found but its header is unreadable or malformed —
/// the caller surfaces that message via `fail_job`.
pub(crate) fn detect_family_in_target_dir(dir: &Path) -> Result<Option<String>, String> {
    let Some(safetensors_path) = first_safetensors_path(dir) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect downloaded LoRA file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Downloaded LoRA file has an invalid safetensors header.".to_owned()
        }
    })?;
    Ok(detect_lora_family(&header))
}

pub(crate) async fn run_lora_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = optional_payload_string(&job.payload, "repo");
    let source_url = optional_payload_string(&job.payload, "sourceUrl");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "loraId")
        .or_else(|| optional_payload_string(&job.payload, "name"))
        .map(str::to_owned)
        .or_else(|| repo.map(str::to_owned))
        .or_else(|| source_url.and_then(|value| lora_source_url_file_stem(value).ok()))
        .map(|value| safe_download_dir(&value))
        .unwrap_or_else(|| {
            source_path
                .and_then(|path| {
                    Path::new(path)
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .map(safe_download_dir)
                })
                .unwrap_or_else(|| "lora".to_owned())
        });
    let target_dir = resolve_lora_import_target(
        settings,
        &job.payload,
        settings.data_dir.join("loras").join(&target_name),
    )?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Importing,
            0.1,
            "Importing LoRA.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "LoRA import canceled before transfer started.",
    )
    .await?;

    if let Some(repo) = repo {
        let files = payload_string_array(&job.payload, "files");
        let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
        let snapshot =
            HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
        let mut progress = DownloadProgress::new(
            repo,
            directory_size(&target_dir).await,
            snapshot.total_bytes(),
            progress_report_interval(settings),
        );
        // LoRA HF imports intentionally skip the model install marker for parity with the Python worker.
        download_snapshot(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "LoRA import canceled by user.",
                fresh_download: false,
            },
            &target_dir,
            &snapshot,
            &mut progress,
        )
        .await?;
    } else if let Some(source_path) = source_path {
        let prefer_move = payload_bool(&job.payload, "uploadedSourcePath");
        if let Some(secondary_source_path) =
            optional_payload_string(&job.payload, "secondarySourcePath")
        {
            // Paired Wan A14B MoE upload (sc-1991): write both halves into one
            // record under the high/low_noise convention so the high half resolves
            // as the primary (transformer) and the low half as the transformer_2
            // sibling, regardless of the user's original upload filenames.
            let (high_name, low_name) = wan_moe_pair_filenames(&target_name);
            import_lora_source_file_as(
                Path::new(source_path),
                &target_dir,
                &high_name,
                prefer_move,
            )
            .await?;
            import_lora_source_file_as(
                Path::new(secondary_source_path),
                &target_dir,
                &low_name,
                prefer_move,
            )
            .await?;
        } else {
            import_lora_source_path(Path::new(source_path), &target_dir, prefer_move).await?;
        }
    } else if let Some(source_url) = source_url {
        download_lora_source_url(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "LoRA import canceled by user.",
                fresh_download: false,
            },
            source_url,
            &target_dir,
        )
        .await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "LoRA import failed.",
            Some("Provide repo, sourceUrl, or sourcePath for LoRA import".to_owned()),
        )
        .await;
    }

    let detected_family = match detect_family_in_target_dir(&target_dir) {
        Ok(detected) => detected,
        Err(detail) => {
            return fail_job(api, &job.id, "LoRA import failed.", Some(detail)).await;
        }
    };
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    let resolved_family = match (supplied_family, detected_family) {
        (Some(supplied), Some(detected)) => {
            if supplied != detected {
                return fail_job(
                    api,
                    &job.id,
                    "LoRA import failed.",
                    Some(format!(
                        "LoRA file appears to be a {detected} model, but family was declared as {supplied}. Re-import with family {detected} or pick a different file."
                    )),
                )
                .await;
            }
            Some(supplied)
        }
        (None, Some(detected)) => Some(detected),
        (Some(supplied), None) => {
            eprintln!(
                "LoRA import job {}: architecture detection inconclusive; accepting supplied family {supplied}",
                job.id
            );
            Some(supplied)
        }
        (None, None) => None,
    };

    write_lora_install_marker(&target_dir, &job.payload, &job.id).await?;
    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let mut manifest_entry = manifest_entry;
        if let Some(family) = resolved_family {
            manifest_entry
                .entry("family")
                .or_insert(Value::String(family));
        }
        let manifest_path = lora_manifest_target(settings, &job.payload)?;
        upsert_lora_manifest_entry(&manifest_path, manifest_entry).await?;
    }

    let mut result = JsonObject::new();
    result.insert(
        "repo".to_owned(),
        repo.map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "sourceUrl".to_owned(),
        source_url
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "LoRA import completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) fn model_family_detection_error(error: SafetensorsHeaderError) -> String {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect imported model file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Imported model file has an invalid safetensors header.".to_owned()
        }
    }
}

/// Outcome of re-checking a downloaded model's architecture family against the
/// catalog-declared family (sc-1663). Kept pure (no API I/O) so the decision is
/// unit-testable; [`reconcile_downloaded_model_family`] maps it to a job failure.
#[derive(Debug)]
pub(crate) enum DownloadFamilyCheck {
    /// Detection agrees, is inconclusive, or no family was declared — proceed.
    Proceed,
    /// The catalog declared one family but the weights are confidently another.
    Mismatch(FamilyMismatch),
    /// A safetensors file was found but its header could not be read.
    DetectionFailed(SafetensorsHeaderError),
}

/// Re-detect the architecture family of the downloaded weights and reconcile it
/// against the catalog-declared `supplied` family. A missing declaration or an
/// inconclusive detector result proceeds — the curated catalog is trusted when there
/// is no confident contradicting signal — so this never blocks a legitimate
/// download; only a confident conflict is a mismatch.
pub(crate) fn check_downloaded_model_family(
    supplied: Option<String>,
    model_dir: &Path,
) -> DownloadFamilyCheck {
    let detected = match detect_model_family(model_dir) {
        Ok(detected) => detected,
        Err(error) => return DownloadFamilyCheck::DetectionFailed(error),
    };
    match reconcile_detected_family(supplied, detected) {
        Ok(_) => DownloadFamilyCheck::Proceed,
        Err(mismatch) => DownloadFamilyCheck::Mismatch(mismatch),
    }
}

/// Enforce family parity with model import on a completed download: verify the
/// downloaded weights match the catalog-declared family and fail the job on a
/// confident mismatch (or an unreadable header). Returns `Ok(true)` when the
/// download may complete, `Ok(false)` when the job was already failed and the
/// caller should return.
pub(crate) async fn reconcile_downloaded_model_family(
    api: &ApiClient,
    job: &JobSnapshot,
    model_dir: &Path,
) -> WorkerResult<bool> {
    let supplied = optional_payload_string(&job.payload, "family").map(str::to_owned);
    match check_downloaded_model_family(supplied, model_dir) {
        DownloadFamilyCheck::Proceed => Ok(true),
        DownloadFamilyCheck::DetectionFailed(error) => {
            let detail = match error {
                SafetensorsHeaderError::Io(io_error) => {
                    format!("Unable to inspect downloaded model file: {io_error}")
                }
                SafetensorsHeaderError::InvalidHeader => {
                    "Downloaded model file has an invalid safetensors header.".to_owned()
                }
            };
            fail_job(api, &job.id, "Model download failed.", Some(detail)).await?;
            Ok(false)
        }
        DownloadFamilyCheck::Mismatch(mismatch) => {
            fail_job(
                api,
                &job.id,
                "Model download failed.",
                Some(format!(
                    "Downloaded model files appear to be {}, but the catalog declared family {}. Fix the catalog entry to family {} or correct the download source.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await?;
            Ok(false)
        }
    }
}

pub(crate) async fn run_model_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = optional_payload_string(&job.payload, "repo");
    let source_url = optional_payload_string(&job.payload, "sourceUrl");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "modelId")
        .map(safe_download_dir)
        .unwrap_or_else(|| "model".to_owned());
    let target_dir = resolve_model_import_target(
        settings,
        &job.payload,
        settings
            .data_dir
            .join("models")
            .join("imports")
            .join(target_name),
    )?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Importing,
            0.1,
            "Importing model.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Model import canceled before transfer started.",
    )
    .await?;

    if let Some(repo) = repo {
        let files = payload_string_array(&job.payload, "files");
        let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
        let snapshot =
            HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
        let mut progress = DownloadProgress::new(
            repo,
            directory_size(&target_dir).await,
            snapshot.total_bytes(),
            progress_report_interval(settings),
        );
        download_snapshot(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "Model import canceled by user.",
                fresh_download: false,
            },
            &target_dir,
            &snapshot,
            &mut progress,
        )
        .await?;
    } else if let Some(source_path) = source_path {
        import_lora_source_path(
            Path::new(source_path),
            &target_dir,
            payload_bool(&job.payload, "uploadedSourcePath"),
        )
        .await?;
    } else if let Some(source_url) = source_url {
        download_model_source_url(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "Model import canceled by user.",
                fresh_download: false,
            },
            source_url,
            &target_dir,
        )
        .await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "Model import failed.",
            Some("Provide repo, sourceUrl, or sourcePath for model import".to_owned()),
        )
        .await;
    }

    let detected_family = match detect_model_family(&target_dir) {
        Ok(detected) => detected,
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(model_family_detection_error(error)),
            )
            .await;
        }
    };
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    let resolved_family = match reconcile_detected_family(supplied_family, detected_family) {
        Ok(family) => family,
        Err(mismatch) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(format!(
                    "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await;
        }
    };

    write_model_install_marker(&target_dir, &job.payload, repo.unwrap_or(""), &job.id).await?;
    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let mut manifest_entry = manifest_entry;
        if let Some(family) = resolved_family.clone() {
            manifest_entry
                .entry("family")
                .or_insert(Value::String(family));
        }
        let model_type = manifest_entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("image")
            .to_owned();
        let family = manifest_entry
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_owned);
        apply_model_manifest_defaults(&mut manifest_entry, &model_type, family.as_deref());
        if let Some(paths) = manifest_entry
            .entry("paths")
            .or_insert_with(|| json!({}))
            .as_object_mut()
        {
            paths.insert(
                "model".to_owned(),
                Value::String(target_dir.display().to_string()),
            );
        }
        let manifest_path = model_manifest_target(settings, &job.payload)?;
        upsert_model_manifest_entry(&manifest_path, manifest_entry).await?;
    }

    let mut result = JsonObject::new();
    result.insert(
        "modelId".to_owned(),
        job.payload.get("modelId").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "repo".to_owned(),
        repo.map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "sourceUrl".to_owned(),
        source_url
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert(
        "family".to_owned(),
        resolved_family.map(Value::String).unwrap_or(Value::Null),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Model import completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}
