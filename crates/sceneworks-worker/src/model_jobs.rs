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
        overlay_derived_tokenizer(api, settings, http_client, &job.id, repo, &cache_path).await?;
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
    // Some upstreams (Kolors sc-4764, Qwen-Image sc-6570) ship no fast `tokenizer.json`; overlay the
    // derived one so the in-process Rust generator/trainer can construct. No-op for every other repo.
    overlay_derived_tokenizer(api, settings, http_client, &job.id, repo, &cache_path).await?;
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

/// Download a built-in catalog LoRA's Hugging Face repo/file into the shared HF cache
/// (sc-5944). Mirrors the `run_model_download_job` cache path but skips the model-only
/// steps (family reconciliation, tokenizer overlay, install marker): a LoRA is a single
/// adapter file. Landing it in the HF cache is exactly what the API's install-state probe
/// (`lora_huggingface_cached_file`) and the engine's on-demand generation-time fetch read,
/// so the catalog entry flips to "installed" on the next `/api/v1/loras` refresh.
pub(crate) async fn run_lora_download_job(
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
                "LoRA download is missing a repository.",
                Some(error.to_string()),
            )
            .await?;
            return Ok(());
        }
    };
    let files = payload_string_array(&job.payload, "files");
    let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.1,
            &format!("Downloading LoRA {repo}: estimating size."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "LoRA download canceled before transfer started.",
    )
    .await?;

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
                &format!(
                    "Downloading LoRA {repo}: 0 B of {}.",
                    format_bytes(total_bytes)
                ),
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
            cancel_message: "LoRA download canceled by user.",
            fresh_download: false,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let cache_path = huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(repo_dir);

    let mut result = JsonObject::new();
    result.insert(
        "loraId".to_owned(),
        job.payload.get("loraId").cloned().unwrap_or(Value::Null),
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
            "LoRA download completed in the Hugging Face cache.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// A base model whose upstream snapshot omits the HF **fast** `tokenizer.json` the in-process Rust
/// generators/trainers require, paired with the small public `SceneWorks/*` repo that hosts the
/// derived `tokenizer.json` to overlay at install. Materializing once + hosting (rather than running
/// a slow→fast conversion locally) keeps a Python-free desktop working out-of-box.
struct DerivedTokenizerOverlay {
    /// Upstream base-model repo whose snapshot ships only a *slow* tokenizer.
    base_repo: &'static str,
    /// SceneWorks-hosted repo holding the materialized fast `tokenizer.json`.
    tokenizer_repo: &'static str,
}

/// The single overlaid file (in every case the HF fast serialization) and its in-snapshot subdir.
const DERIVED_TOKENIZER_FILE: &str = "tokenizer.json";

/// Base models that need a derived fast-tokenizer overlay. Keep entries narrowly scoped to the exact
/// base repo — sibling repos (IP-Adapter, Edit variants that ship their own `tokenizer.json`, etc.)
/// must NOT match.
const DERIVED_TOKENIZER_OVERLAYS: &[DerivedTokenizerOverlay] = &[
    // Kolors (sc-4764): conditions on ChatGLM3-6B, which ships only the **slow** SentencePiece
    // tokenizer (`tokenizer.model`). The Rust Kolors generator (sc-3875) + LoRA/LoKr trainer
    // (sc-4732) construct via `KolorsTokenizer::from_dir`, which reads only `tokenizer/tokenizer.json`.
    // Derived via mlx-gen `tools/build_kolors_tokenizer.py` (content-id-parity validated).
    DerivedTokenizerOverlay {
        base_repo: "Kwai-Kolors/Kolors-diffusers",
        tokenizer_repo: "SceneWorks/kolors-chatglm3-tokenizer",
    },
    // Qwen-Image (sc-6570): ships the Qwen2 BPE tokenizer as `vocab.json` + `merges.txt` only. The MLX
    // `qwen-image` provider's `load_tokenizer` reads `tokenizer/tokenizer.json`. Derived via mlx-gen
    // `tools/build_qwen_tokenizer.py` (byte-identical to the fork's runtime fast tokenizer). Note:
    // `Qwen/Qwen-Image-Edit-2511` already ships its own `tokenizer.json`, so it needs no overlay.
    DerivedTokenizerOverlay {
        base_repo: "Qwen/Qwen-Image",
        tokenizer_repo: "SceneWorks/qwen-image-tokenizer",
    },
    // Qwen-Image-2512 (sc-8271): the Dec-2025 base refresh is architecturally identical to
    // `Qwen/Qwen-Image` and reuses the same Qwen2 BPE tokenizer (unchanged across the line),
    // so it points at the same hosted overlay repo.
    DerivedTokenizerOverlay {
        base_repo: "Qwen/Qwen-Image-2512",
        tokenizer_repo: "SceneWorks/qwen-image-tokenizer",
    },
];

/// The hosted tokenizer repo + overlay dest for a just-downloaded `repo`, or `None` when `repo` is not
/// a base model that needs the overlay (a no-op for every other model). Pure so the repo guard +
/// target path are unit-testable without a download.
pub(crate) fn derived_tokenizer_overlay(
    repo: &str,
    snapshot_dir: &Path,
) -> Option<(&'static str, PathBuf)> {
    let trimmed = repo.trim();
    DERIVED_TOKENIZER_OVERLAYS
        .iter()
        .find(|overlay| overlay.base_repo == trimmed)
        .map(|overlay| {
            (
                overlay.tokenizer_repo,
                snapshot_dir.join("tokenizer").join(DERIVED_TOKENIZER_FILE),
            )
        })
}

/// After a base-model download, overlay the derived `tokenizer.json` from its SceneWorks tokenizer
/// repo into the snapshot's `tokenizer/` dir, so the in-process generator/trainer can construct (they
/// read only `tokenizer/tokenizer.json`, which these upstreams omit). No-op for any other repo;
/// idempotent — skips when the file is already present (a re-install, or a snapshot that already
/// shipped it). Reuses the standard HF resolve + download path, so the SceneWorks repo's
/// auth/size/resume handling matches every other download.
pub(crate) async fn overlay_derived_tokenizer(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job_id: &str,
    repo: &str,
    snapshot_dir: &Path,
) -> WorkerResult<()> {
    let Some((tokenizer_repo, dest)) = derived_tokenizer_overlay(repo, snapshot_dir) else {
        return Ok(());
    };
    if dest.exists() {
        return Ok(());
    }
    let tokenizer_dir = dest
        .parent()
        .expect("overlay dest always has a tokenizer/ parent");
    let snapshot = HuggingFaceSnapshot::resolve(
        http_client,
        settings,
        tokenizer_repo,
        "main",
        &[DERIVED_TOKENIZER_FILE.to_owned()],
    )
    .await?;
    let mut progress = DownloadProgress::new(
        tokenizer_repo,
        0,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    download_snapshot(
        &DownloadContext {
            api,
            client: http_client,
            settings,
            job_id,
            cancel_message: "Derived tokenizer overlay canceled by user.",
            fresh_download: false,
        },
        tokenizer_dir,
        &snapshot,
        &mut progress,
    )
    .await
}

/// Native Rust/MLX FLUX.2-klein single-file → diffusers convert (sc-3136), replacing the
/// retired Python `mlx_flux_convert.py` (sc-3032). Loads the wikeeyang single-file
/// transformer, remaps it to the diffusers layout (fused-qkv 1→3 split + the load-bearing
/// `norm_out` scale/shift swap), validates against the base, and assembles a local diffusers
/// dir whose borrowed vae/text-encoder/tokenizer are absolute symlinks (so they survive the
/// worker's temp→final atomic rename). Runs MLX on macOS; the candle (Windows/CUDA) lane runs the
/// byte-equivalent candle twin (sc-7459) so `flux2_klein_9b_true_v2` installs + converts off-Mac too.
#[cfg(target_os = "macos")]
fn convert_flux2_klein_diffusers(
    source_file: &Path,
    base_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    mlx_gen_flux2::convert_and_assemble(source_file, base_dir, out_dir)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// Candle (Windows/CUDA) twin of the macOS converter (sc-7459): the same single-file → diffusers key
/// remap (fused-qkv split + the `norm_out` scale/shift swap), implemented on candle CPU tensors, so the
/// `flux2_klein_9b_true_v2` weight variant has a real install-time convert lane on the candle worker
/// (the borrowed components are hardlinked, not symlinked — Windows symlinks fail to read; see the
/// crate's `convert` module). Reached only on the `backend-candle` build.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn convert_flux2_klein_diffusers(
    source_file: &Path,
    base_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720 / sc-7459): backend-specific weight converter; not a registry contract.
    candle_gen_flux2::convert_and_assemble(source_file, base_dir, out_dir)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
fn convert_flux2_klein_diffusers(
    _source_file: &Path,
    _base_dir: &Path,
    _out_dir: &Path,
) -> Result<(), String> {
    Err(
        "FLUX.2-klein conversion requires the candle backend (Windows/CUDA, --features \
         backend-candle) or macOS (mlx-gen-flux2)."
            .to_owned(),
    )
}

/// Native Rust/MLX LTX-2.3 weight converter (mlx-gen-ltx, sc-3224 engine + sc-3240 cutover). The LTX
/// path was never actually routed through the Rust job before — only `convert_wan` was ever shelled.
/// Splits the single-file checkpoint (`source_file`, e.g. eros `10Eros_v1_bf16.safetensors`),
/// sanitizes + Q4/Q8-quantizes the transformer, merges the latent upsampler from `upscaler_dir` (the
/// base `Lightricks/LTX-2.3` snapshot — the loader hard-requires `upsampler.safetensors`), and emits
/// the split MLX dir. `bits` is the reference `python -m mlx_video.convert --quantize --q-bits <bits>`
/// recipe (audio-inclusive). Runs MLX, so macOS-only.
#[cfg(target_os = "macos")]
fn convert_ltx_native(
    source_file: &Path,
    upscaler_dir: &Path,
    out_dir: &Path,
    bits: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    let opts = mlx_gen_ltx::LtxConvertOpts::audio_quant(bits);
    mlx_gen_ltx::convert_and_assemble(source_file, Some(upscaler_dir), out_dir, &opts)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
fn convert_ltx_native(
    _source_file: &Path,
    _upscaler_dir: &Path,
    _out_dir: &Path,
    _bits: i32,
) -> Result<(), String> {
    Err("LTX-2.3 MLX conversion requires macOS (mlx-gen-ltx); the candle path serves other platforms."
        .to_owned())
}

/// Native Rust/MLX FLUX.2-dev pre-quantization convert (sc-5921, epic 5914). FLUX.2-dev is too large
/// to quantize in memory at load — the dense bf16 snapshot (60 GB DiT + 45 GB Mistral TE) peaks
/// ~105 GB > the 128 GB ceiling — so the install-time convert pre-quantizes the DiT + text encoder to
/// Q`bits` **on disk** (offline peak ~34.5 GB; MLX mmap-streams the source, sc-5917) and assembles a
/// packed snapshot: real packed `transformer/` + `text_encoder/` dirs (each one safetensors + a
/// `quantization` manifest in config.json) plus the unchanged VAE + tokenizer + `model_index.json`
/// symlinked from the gated source (the VAE is byte-identical to klein's). The worker then loads the
/// packed dir via the `modelPath` seam; the load-time `.quantize()` is a no-op on already-packed
/// weights. Absolute symlinks survive the worker's temp→final atomic rename, and `model_index.json`
/// doubles as the catalog's "converted" marker (`mlx_catalog_status` looks for a top-level
/// config.json/model_index.json). Runs MLX, so macOS-only.
#[cfg(target_os = "macos")]
fn convert_flux2_dev_prequant(
    source_dir: &Path,
    out_dir: &Path,
    bits: i32,
    group_size: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    mlx_gen_flux2::quantize_flux2_dit(
        &source_dir.join("transformer"),
        &out_dir.join("transformer"),
        bits,
        group_size,
    )
    .map_err(|error| format!("quantize FLUX.2-dev transformer: {error}"))?;
    mlx_gen_flux2::quantize_flux2_text_encoder_dir(
        &source_dir.join("text_encoder"),
        &out_dir.join("text_encoder"),
        bits,
        group_size,
    )
    .map_err(|error| format!("quantize FLUX.2-dev text encoder: {error}"))?;
    for sub in ["vae", "tokenizer", "model_index.json"] {
        let src = source_dir.join(sub);
        if !src.exists() {
            return Err(format!(
                "FLUX.2-dev source is missing `{sub}` — expected the gated diffusers snapshot of \
                 black-forest-labs/FLUX.2-dev (transformer/ text_encoder/ vae/ tokenizer/ \
                 model_index.json)."
            ));
        }
        let canonical = std::fs::canonicalize(&src).map_err(|error| error.to_string())?;
        std::os::unix::fs::symlink(&canonical, out_dir.join(sub))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn convert_flux2_dev_prequant(
    _source_dir: &Path,
    _out_dir: &Path,
    _bits: i32,
    _group_size: i32,
) -> Result<(), String> {
    Err(
        "FLUX.2-dev conversion requires macOS (mlx-gen-flux2); this model is macOS-only."
            .to_owned(),
    )
}

/// SD3.5 gated diffusers snapshot → packed Q`bits` MLX dir (sc-7871). SD3.5's snapshot is
/// self-contained (its own transformer + triple text encoder + tokenizer + VAE), and the engine
/// loader reads the diffusers layout directly, reusing the SDXL CLIP encoders, the FLUX T5 encoder,
/// and the Z-Image 16-ch VAE — all loaded DENSE. So conversion pre-quantizes ONLY the MMDiT
/// `transformer/` on disk (the bulk: an 8B DiT) and symlinks the unchanged text encoders / tokenizers
/// / VAE / scheduler / model_index.json through. Absolute symlinks survive the worker's temp→final
/// atomic rename, and `model_index.json` doubles as the catalog's "converted" marker. `variant`
/// selects the MMDiT arch (Large/Turbo share one layout; Medium is the MMDiT-X dual-attention layout).
/// Runs MLX, so macOS-only.
#[cfg(target_os = "macos")]
fn convert_sd3_prequant(
    source_dir: &Path,
    out_dir: &Path,
    variant: Sd3Variant,
    bits: i32,
    group_size: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    let arch = match variant {
        Sd3Variant::Large | Sd3Variant::LargeTurbo => mlx_gen_sd3::Sd3Arch::large(),
        Sd3Variant::Medium => mlx_gen_sd3::Sd3Arch::medium(),
    };
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let src_transformer = source_dir.join("transformer");
    if !src_transformer.is_dir() {
        return Err(
            "SD3.5 source is missing `transformer/` — expected the gated diffusers snapshot of \
             stabilityai/stable-diffusion-3.5-* (transformer/ text_encoder/ text_encoder_2/ \
             text_encoder_3/ tokenizer/ tokenizer_2/ tokenizer_3/ vae/ model_index.json)."
                .to_owned(),
        );
    }
    // Quantize the MMDiT transformer/ on disk (validates the arch first, writes a packed dir).
    mlx_gen_sd3::quantize_sd3_dir(
        &arch,
        &src_transformer,
        &out_dir.join("transformer"),
        bits,
        group_size,
    )
    .map_err(|error| format!("quantize SD3.5 transformer: {error}"))?;
    // Symlink the dense (reused-as-is) text encoders / tokenizers / VAE / scheduler / model_index.json.
    for sub in [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "vae",
        "scheduler",
        "model_index.json",
    ] {
        let src = source_dir.join(sub);
        if !src.exists() {
            // scheduler/ is optional metadata for the engine (the pipeline uses a static flow-match
            // Euler schedule), so only the load-bearing modules are hard-required.
            if sub == "scheduler" {
                continue;
            }
            return Err(format!(
                "SD3.5 source is missing `{sub}` — expected the full gated diffusers snapshot of \
                 stabilityai/stable-diffusion-3.5-* (the triple text encoder + tokenizers + 16-ch VAE \
                 are reused dense)."
            ));
        }
        let canonical = std::fs::canonicalize(&src).map_err(|error| error.to_string())?;
        std::os::unix::fs::symlink(&canonical, out_dir.join(sub))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn convert_sd3_prequant(
    _source_dir: &Path,
    _out_dir: &Path,
    _variant: Sd3Variant,
    _bits: i32,
    _group_size: i32,
) -> Result<(), String> {
    Err("SD3.5 conversion requires macOS (mlx-gen-sd3); this model is macOS-only.".to_owned())
}

/// The base `Lightricks/LTX-2.3` latent upsampler the LTX loader hard-requires (emitted as
/// `upsampler.safetensors` in the converted dir). Neither the eros nor the base single-file
/// checkpoint bundles it, so the converter merges it from the base repo at convert time.
const LTX_SPATIAL_UPSCALER_FILE: &str = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors";

/// Ensure the LTX-2.3 spatial upsampler `file` from `repo` is in the Hugging Face cache — fetching
/// just that file on demand if missing — and return the repo's snapshot dir (the converter's
/// `upscaler_dir`). Mirrors how the torch adapter pulls its spatial upscaler at generation time, so
/// converting eros does not require a full ~157 GB base-LTX install. Returns `None` if the file
/// cannot be obtained (HF CLI unavailable and not already cached) so the caller surfaces a clear
/// failure. The fetch uses a scratch marker dir so its install marker never lands in the model tree.
async fn ensure_ltx_upscaler_cached(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    file: &str,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, repo) {
        if snapshot.join(file).exists() {
            return Ok(Some(snapshot));
        }
    }
    let scratch = settings
        .data_dir
        .join("cache")
        .join(format!(".ltx-upscaler-fetch-{}", job.id));
    tokio::fs::create_dir_all(&scratch).await?;
    let files = vec![file.to_owned()];
    let fetched =
        download_model_with_hf_cli(api, settings, job, repo, "main", &files, &scratch).await?;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    let snapshot = match fetched {
        Some(dir) => Some(dir),
        None => huggingface_snapshot_dir(&settings.data_dir, repo),
    };
    Ok(snapshot.filter(|dir| dir.join(file).exists()))
}

/// Resolved native conversion plan for [`run_model_convert_job`] (sc-3240). The Python
/// `mlx_video.convert_wan` subprocess (and its mlx-video venv) is gone — every convert-required
/// model maps to exactly one native converter, keyed by the manifest `mlx.converter` discriminator.
enum ConvertPlan {
    /// FLUX.2-klein single-file fine-tune → diffusers dir (sc-3136), borrowing VAE/TE/tokenizer.
    Flux2 {
        source_file: PathBuf,
        base_dir: PathBuf,
    },
    /// Single-file LTX-2.3 → split MLX dir; `upscaler_dir` carries the loader-required upsampler.
    Ltx {
        source_file: PathBuf,
        upscaler_dir: PathBuf,
        bits: i32,
    },
    /// FLUX.2-dev gated diffusers snapshot → packed Q`bits` dir (sc-5921): pre-quantize the DiT +
    /// Mistral text encoder on disk, symlink the unchanged VAE/tokenizer/model_index.json.
    Flux2Dev {
        source_dir: PathBuf,
        bits: i32,
        group_size: i32,
    },
    /// SD3.5 gated diffusers snapshot → packed Q`bits` dir (sc-7871): pre-quantize ONLY the MMDiT
    /// `transformer/` on disk (the triple text encoder + VAE are reused dense and symlinked through
    /// unchanged). `variant` selects the MMDiT arch (Large/Turbo share one layout; Medium is MMDiT-X).
    Sd3 {
        source_dir: PathBuf,
        variant: Sd3Variant,
        bits: i32,
        group_size: i32,
    },
}

/// The SD3.5 MMDiT variant a `sd3_5_*_quant` conversion targets — a target-neutral mirror of
/// `mlx_gen_sd3::Sd3Variant` (which is macOS-only), so [`ConvertPlan`] stays buildable on every
/// target. The macOS converter maps it back to the engine variant in [`convert_sd3_prequant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sd3Variant {
    /// `sd3_5_large_quant` — SD3.5 Large (8B MMDiT, true-CFG).
    Large,
    /// `sd3_5_large_turbo_quant` — SD3.5 Large Turbo (same MMDiT layout, ADD-distilled).
    LargeTurbo,
    /// `sd3_5_medium_quant` — SD3.5 Medium (2.5B MMDiT-X, dual-attention first 13 blocks).
    Medium,
}

/// Convert a model's native checkpoint into the local MLX format on macOS/Apple Silicon, fully
/// in-process via the linked `mlx-gen-*` converters (epic 2337). The native checkpoint must already
/// be downloaded into the Hugging Face cache (via a model_download job). The converter is selected by
/// the manifest `mlx.converter` discriminator: `flux2_klein_diffusers` (sc-3136), `ltx_video`
/// (mlx-gen-ltx), or `flux2_dev_quant` (FLUX.2-dev pre-quantization, sc-5921) — the models that
/// install via in-app conversion, plus `sd3_5_large_quant` / `sd3_5_large_turbo_quant` /
/// `sd3_5_medium_quant` (SD3.5 transformer pre-quantization, sc-7871). The Python
/// `mlx_video.convert_wan` subprocess + `SCENEWORKS_PYTHON` wiring were retired here (sc-3240); the
/// Wan2.2 converters were decommissioned once those models flipped to pre-converted SceneWorks
/// downloads (sc-5603, epic 5594).
///
/// Real conversion is exercised on Mac hardware via the `#[ignore]` real-weight tests below; this
/// wires the tracked job, progress, cancellation, and failure surfacing.
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
    // conversion. `bits` is validated by the convert tool's choices (LTX honors it; the
    // FLUX.2-klein converter is bf16-only and ignores it).
    let quantize_only = payload_bool(&job.payload, "quantizeOnly");
    let quantize_bits = job.payload.get("quantizeBits").and_then(Value::as_u64);

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

    // Converter discriminator (sc-2235 / sc-3224 / sc-3240). Every convert-required model now
    // declares one in its manifest `mlx.converter`; there is NO Python fallback — the
    // `mlx_video.convert_wan` subprocess and its mlx-video venv were retired at this cutover.
    //   flux2_klein_diffusers          -> FLUX.2-klein single-file → diffusers dir (sc-3136)
    //   ltx_video                      -> single-file LTX-2.3 → split MLX dir (mlx-gen-ltx)
    //   flux2_dev_quant                -> FLUX.2-dev diffusers snapshot → packed Q4 dir (sc-5921)
    let converter = optional_payload_string(&job.payload, "converter")
        .map(str::to_owned)
        .unwrap_or_default();

    // Quantize-only (re-quantize a pre-converted turnkey bf16 MLX dir) was a capability of the
    // Python `convert_wan --quantize-only` with no native equivalent; it is unreachable from the UI
    // and superseded by native-conversion-with-quant. Surface it explicitly rather than silently
    // promoting an unconverted dir.
    if quantize_only {
        fail_job(
            api,
            &job.id,
            "Quantize-only MLX conversion is no longer supported.",
            Some(
                "In-place re-quantization of a pre-converted MLX model was removed with the legacy \
                 mlx-video converter (sc-3240). Convert the native checkpoint with quantization \
                 instead."
                    .to_owned(),
            ),
        )
        .await?;
        return Ok(());
    }

    let plan = match converter.as_str() {
        "flux2_klein_diffusers" => {
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
            ConvertPlan::Flux2 {
                source_file,
                base_dir,
            }
        }
        "ltx_video" => {
            let source_file_name = required_payload_string(&job.payload, "sourceFile")?.to_owned();
            let source_file = checkpoint_dir.join(&source_file_name);
            if !source_file.is_file() {
                fail_job(
                    api,
                    &job.id,
                    "LTX-2.3 source checkpoint file is missing.",
                    Some(format!("Expected {source_file_name} in {source_repo}.")),
                )
                .await?;
                return Ok(());
            }
            let upscaler_repo = required_payload_string(&job.payload, "baseRepo")?.to_owned();
            let upscaler_file = optional_payload_string(&job.payload, "upscalerFile")
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(LTX_SPATIAL_UPSCALER_FILE)
                .to_owned();
            let Some(upscaler_dir) =
                ensure_ltx_upscaler_cached(api, settings, job, &upscaler_repo, &upscaler_file)
                    .await?
            else {
                fail_job(
                    api,
                    &job.id,
                    "LTX-2.3 spatial upscaler is unavailable.",
                    Some(format!(
                        "Could not obtain {upscaler_file} from {upscaler_repo}; install the base \
                         LTX-2.3 model or check connectivity before converting {model_id}."
                    )),
                )
                .await?;
                return Ok(());
            };
            // Default to the reference Q4 recipe when the manifest/request specifies no bits.
            let bits = quantize_bits.map_or(4, |bits| bits as i32);
            ConvertPlan::Ltx {
                source_file,
                upscaler_dir,
                bits,
            }
        }
        "flux2_dev_quant" => {
            // FLUX.2-dev is self-contained (its own VAE/tokenizer/TE in the snapshot), so the whole
            // gated diffusers snapshot dir is the convert source — no single source file, no base
            // repo. Q4 is mandatory (in-app Q4 load of the dense bf16 would peak ~105 GB), so default
            // to Q4 / group-size 64 when the request omits them.
            let bits = quantize_bits.map_or(4, |bits| bits as i32);
            let group_size = job
                .payload
                .get("quantizeGroupSize")
                .and_then(Value::as_u64)
                .map_or(64, |group| group as i32);
            ConvertPlan::Flux2Dev {
                source_dir: checkpoint_dir.clone(),
                bits,
                group_size,
            }
        }
        // SD3.5 (epic 7841 / sc-7871) — the gated diffusers snapshot is self-contained (its own
        // transformer/ + triple text encoder + tokenizer + VAE), so the whole snapshot dir is the
        // convert source: pre-quantize ONLY the MMDiT transformer/ on disk and symlink the dense TE /
        // VAE / tokenizer / scheduler / model_index.json through. Default Q8 / group-size 64 (the
        // manifest `quantize: 8`, reference group size) when the request omits them. The converter id
        // selects the MMDiT arch variant.
        converter @ ("sd3_5_large_quant" | "sd3_5_large_turbo_quant" | "sd3_5_medium_quant") => {
            let bits = quantize_bits.map_or(8, |bits| bits as i32);
            let group_size = job
                .payload
                .get("quantizeGroupSize")
                .and_then(Value::as_u64)
                .map_or(64, |group| group as i32);
            let variant = match converter {
                "sd3_5_large_quant" => Sd3Variant::Large,
                "sd3_5_large_turbo_quant" => Sd3Variant::LargeTurbo,
                _ => Sd3Variant::Medium,
            };
            ConvertPlan::Sd3 {
                source_dir: checkpoint_dir.clone(),
                variant,
                bits,
                group_size,
            }
        }
        "" => {
            fail_job(
                api,
                &job.id,
                "No MLX converter is configured for this model.",
                Some(format!(
                    "{model_id} sets mlx.requiresConversion but no mlx.converter; the legacy \
                     converter was retired (sc-3240)."
                )),
            )
            .await?;
            return Ok(());
        }
        other => {
            fail_job(
                api,
                &job.id,
                "Unknown MLX converter.",
                Some(format!(
                    "Unrecognized mlx.converter '{other}' for {model_id}."
                )),
            )
            .await?;
            return Ok(());
        }
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

    // The native MLX converters are blocking and not interruptible mid-run (minutes on real
    // weights), so honor cancel up front and run on a blocking thread. On any failure the partial
    // temp dir is removed so it can never be promoted by the atomic rename below. The Flux2 path's
    // borrowed vae/text-encoder/tokenizer are absolute symlinks that survive the rename.
    check_cancel(api, &job.id, "MLX conversion canceled by user.").await?;
    let temp = temp_dir.clone();
    let outcome = match plan {
        ConvertPlan::Flux2 {
            source_file,
            base_dir,
        } => {
            tokio::task::spawn_blocking(move || {
                convert_flux2_klein_diffusers(&source_file, &base_dir, &temp)
            })
            .await
        }
        ConvertPlan::Ltx {
            source_file,
            upscaler_dir,
            bits,
        } => {
            tokio::task::spawn_blocking(move || {
                convert_ltx_native(&source_file, &upscaler_dir, &temp, bits)
            })
            .await
        }
        ConvertPlan::Flux2Dev {
            source_dir,
            bits,
            group_size,
        } => {
            tokio::task::spawn_blocking(move || {
                convert_flux2_dev_prequant(&source_dir, &temp, bits, group_size)
            })
            .await
        }
        ConvertPlan::Sd3 {
            source_dir,
            variant,
            bits,
            group_size,
        } => {
            tokio::task::spawn_blocking(move || {
                convert_sd3_prequant(&source_dir, &temp, variant, bits, group_size)
            })
            .await
        }
    };
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(detail)) => {
            let _ = tokio::fs::remove_dir_all(&temp_dir).await;
            return Err(WorkerError::Engine(format!(
                "MLX conversion failed. {detail}"
            )));
        }
        Err(join_error) => {
            let _ = tokio::fs::remove_dir_all(&temp_dir).await;
            return Err(task_join_error("MLX conversion task", join_error));
        }
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
///
/// On Windows the resolved snapshot is repaired in place
/// ([`materialize_snapshot_hardlinks`]): the candle worker process cannot traverse
/// the relative `snapshots/<rev>/… -> ../blobs/<etag>` symlinks that `huggingface_hub`
/// (and, before this fix, our own downloader) create — the open fails with
/// `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448), so every model load died at the first
/// file read. The repair replaces those symlinks with hardlinks to the same blob (no
/// reparse point, same volume, still deduped); it is idempotent and best-effort, and a
/// no-op on every other platform.
pub(crate) fn huggingface_snapshot_dir(data_dir: &Path, repo: &str) -> Option<PathBuf> {
    let dir = resolve_huggingface_snapshot_dir(data_dir, repo)?;
    // Windows-only in production (the symlink-traversal defect is Windows-specific); the
    // function itself is still compiled under `test` so the conversion is unit-testable
    // off-Windows, but it must not run against a developer's real HF cache during an
    // unrelated `cargo test` on Unix — hence the call, not just the body, is gated.
    #[cfg(windows)]
    materialize_snapshot_hardlinks(&dir);
    Some(dir)
}

fn resolve_huggingface_snapshot_dir(data_dir: &Path, repo: &str) -> Option<PathBuf> {
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

/// Replace the relative `snapshots/<rev>/… -> ../blobs/<etag>` symlinks under a resolved
/// HF snapshot dir with hardlinks to the same blob. **Windows-only** (a no-op
/// elsewhere): the candle worker process cannot traverse those reparse points — the
/// open fails with `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448, see
/// [`downloaded_model_detection_io_error_is_inconclusive`]) — but a hardlink is a plain
/// directory entry to the blob data (no reparse, same volume, still deduped) and reads
/// fine. Caches our downloader writes now hardlink from the start
/// ([`crate::downloads`]); this repairs caches already materialized as symlinks (e.g. by
/// `huggingface_hub`). Best-effort and idempotent: once converted there is nothing left
/// to do, and any per-file failure leaves that entry untouched for the loader to
/// surface. Compiled on `test` too so the conversion is unit-testable off-Windows.
#[cfg(any(windows, test))]
fn materialize_snapshot_hardlinks(dir: &Path) {
    fn repair(dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_symlink() {
                relink_to_blob(&path);
            } else if file_type.is_dir() {
                repair(&path);
            }
        }
    }
    repair(dir);
}

/// Replace a single snapshot symlink with a hardlink (or, failing that, a copy) to its
/// blob, so the entry is always present afterward. The symlink target is read **without
/// traversing it** (`read_link`), so resolving the blob never hits the os-448 path; the
/// symlink is removed only once its blob is confirmed to exist.
#[cfg(any(windows, test))]
fn relink_to_blob(link: &Path) {
    let Ok(target) = std::fs::read_link(link) else {
        return;
    };
    let Some(parent) = link.parent() else {
        return;
    };
    // `canonicalize` resolves the `..` segments and confirms the blob exists; it opens
    // the blob (a plain file, not a reparse point) so it does not hit the os-448 path.
    let Ok(blob) = std::fs::canonicalize(parent.join(target)) else {
        return;
    };
    if std::fs::remove_file(link).is_err() {
        return;
    }
    if std::fs::hard_link(&blob, link).is_err() {
        // Cross-volume or a filesystem without hardlinks: fall back to a real copy so
        // the entry is never left missing.
        let _ = std::fs::copy(&blob, link);
    }
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
    validate_hf_cli_download_inputs(repo, revision, files)?;
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
    // Resolve the HF token lazily (env, or a one-time pull of the recorded
    // `huggingface.co` keychain credential from the desktop socket on macOS) so the
    // keychain is read only when a download actually needs it (sc-5891).
    if let Some(token) = crate::credentials_ipc::resolve_hf_token(settings).await {
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
        WorkerError::Engine(format!(
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
    let stderr = stderr_task
        .await
        .map_err(|error| task_join_error("Hugging Face CLI stderr reader task", error))?;
    let repo_cache_path =
        huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
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
            let cache_path =
                huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(repo_cache_path);
            write_model_install_marker(marker_dir, &job.payload, repo, &job.id).await?;
            return Ok(Some(cache_path));
        }
        let detail = bounded_tail(&stderr, 10, 2000);
        let message = if detail.trim().is_empty() {
            "Hugging Face CLI download failed without stderr output.".to_owned()
        } else {
            format!("Hugging Face CLI download failed:\n{detail}")
        };
        return Err(WorkerError::Engine(message));
    }

    let cache_path = huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(repo_cache_path);
    write_model_install_marker(marker_dir, &job.payload, repo, &job.id).await?;
    Ok(Some(cache_path))
}

pub(crate) fn validate_hf_cli_download_inputs(
    repo: &str,
    revision: &str,
    files: &[String],
) -> WorkerResult<()> {
    validate_hf_repo_id(repo)?;
    validate_hf_revision(revision)?;
    for pattern in files {
        validate_hf_include_pattern(pattern)?;
    }
    Ok(())
}

fn validate_hf_repo_id(repo: &str) -> WorkerResult<()> {
    let parts = safe_hf_path_parts(repo, "Hugging Face repo")?;
    if parts.len() > 2 {
        return Err(WorkerError::InvalidPayload(
            "Hugging Face repo must be `name` or `namespace/name`.".to_owned(),
        ));
    }
    for part in parts {
        if !part.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        }) {
            return Err(WorkerError::InvalidPayload(format!(
                "Invalid Hugging Face repo `{repo}`: only letters, numbers, `.`, `_`, `-`, and one `/` separator are allowed."
            )));
        }
        if part.starts_with('-')
            || part.starts_with('.')
            || part.ends_with('-')
            || part.ends_with('.')
        {
            return Err(WorkerError::InvalidPayload(format!(
                "Invalid Hugging Face repo `{repo}`: path components cannot start or end with `-` or `.`."
            )));
        }
        if part.contains("--") || part.contains("..") {
            return Err(WorkerError::InvalidPayload(format!(
                "Invalid Hugging Face repo `{repo}`: path components cannot contain `--` or `..`."
            )));
        }
    }
    Ok(())
}

fn validate_hf_revision(revision: &str) -> WorkerResult<()> {
    // Shared guard for both HF CLI (`--revision` is arg-isolated) and direct HTTP
    // download (`downloads::quote_path` percent-encodes the revision in URLs).
    // Keep leading slash / traversal rejection here so both paths fail closed.
    let parts = safe_hf_path_parts(revision, "Hugging Face revision")?;
    if !revision.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '/')
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face revision `{revision}`: only letters, numbers, `.`, `_`, `-`, and `/` are allowed."
        )));
    }
    for part in parts {
        if part == "." || part == ".." {
            return Err(WorkerError::InvalidPayload(format!(
                "Invalid Hugging Face revision `{revision}`: traversal components are not allowed."
            )));
        }
    }
    Ok(())
}

fn validate_hf_include_pattern(pattern: &str) -> WorkerResult<()> {
    let _parts = safe_hf_path_parts(pattern, "Hugging Face include pattern")?;
    if pattern.starts_with('/') || pattern.contains('\\') {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face include pattern `{pattern}`: absolute paths and backslashes are not allowed."
        )));
    }
    if pattern.split('/').any(|part| part == "." || part == "..") {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face include pattern `{pattern}`: traversal components are not allowed."
        )));
    }
    if !pattern.chars().all(|character| {
        character.is_ascii_alphanumeric()
            || matches!(
                character,
                '_' | '-' | '.' | '/' | '*' | '?' | '[' | ']' | '{' | '}' | ',' | '!'
            )
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face include pattern `{pattern}`: unsupported characters are not allowed."
        )));
    }
    Ok(())
}

fn safe_hf_path_parts<'a>(value: &'a str, label: &str) -> WorkerResult<Vec<&'a str>> {
    if value.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot be empty."
        )));
    }
    if value.starts_with('-') {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot start with `-`."
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot contain control characters."
        )));
    }
    if value.starts_with('/') || value.contains('\\') {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} must be a relative Hugging Face identifier."
        )));
    }
    let parts: Vec<_> = value.split('/').collect();
    if parts.iter().any(|part| part.is_empty()) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot contain empty path components."
        )));
    }
    Ok(parts)
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
/// available or the signature is inconclusive. Returns the structured
/// [`SafetensorsHeaderError`] when a file was found but its header is
/// unreadable, malformed, or fronts a truncated/incomplete file — the caller
/// turns it into a message via [`lora_family_detection_error`] and discards the
/// bad file.
pub(crate) fn detect_family_in_target_dir(
    dir: &Path,
) -> Result<Option<String>, SafetensorsHeaderError> {
    let Some(safetensors_path) = first_safetensors_path(dir) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path)?;
    Ok(detect_lora_family(&header))
}

/// Actionable failure message for a LoRA-import safetensors inspection error.
pub(crate) fn lora_family_detection_error(error: &SafetensorsHeaderError) -> String {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect imported LoRA file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Imported LoRA file has an invalid safetensors header.".to_owned()
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => format!(
            "Imported LoRA file is incomplete or corrupt ({actual} bytes on disk, but its header \
             declares at least {declared} bytes of tensor data) — the download was likely \
             interrupted. Re-import the complete file."
        ),
    }
}

/// Whether a safetensors inspection error means the file on disk is itself bad
/// (a valid-but-truncated or malformed file) rather than a transient read
/// failure — i.e. it should be discarded so it can't be picked up as installed.
pub(crate) fn safetensors_file_is_unusable(error: &SafetensorsHeaderError) -> bool {
    matches!(
        error,
        SafetensorsHeaderError::InvalidHeader | SafetensorsHeaderError::IncompleteData { .. }
    )
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

    // sc-6137: verify a caller-supplied content digest. HF repo files are already
    // per-file verified during download; this covers source-URL/uploaded imports where
    // a SHA-256 is supplied out-of-band. `verify_file_sha256` removes the file on
    // mismatch, so a corrupt artifact is never left behind.
    if let Some(expected_sha256) = optional_payload_string(&job.payload, "expectedSha256") {
        if let Some(file) = first_safetensors_path(&target_dir) {
            if let Err(error) = verify_file_sha256(&file, expected_sha256, "Imported LoRA").await {
                return fail_job(api, &job.id, "LoRA import failed.", Some(error.to_string()))
                    .await;
            }
        }
    }

    let detected_family = match detect_family_in_target_dir(&target_dir) {
        Ok(detected) => detected,
        Err(error) => {
            // A truncated/corrupt file must not linger: `lora_is_installed` (API side)
            // treats any `*.safetensors` under the dir as installed, so a rejected
            // partial would still look selectable. Discard it before failing (sc-6072).
            if safetensors_file_is_unusable(&error) {
                if let Some(bad_file) = first_safetensors_path(&target_dir) {
                    let _ = tokio::fs::remove_file(&bad_file).await;
                }
            }
            let detail = lora_family_detection_error(&error);
            return fail_job(api, &job.id, "LoRA import failed.", Some(detail)).await;
        }
    };
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    if let (Some(supplied), None) = (&supplied_family, &detected_family) {
        tracing::info!(
            event = "lora_import_architecture_inconclusive",
            jobId = %job.id,
            family = %supplied,
            "LoRA import: architecture detection inconclusive; accepting supplied family"
        );
    }
    // Shared reconcile policy + canonicalization (krea2/krea-2/krea_2 → krea_2),
    // matching the API pre-flight so a manual family override agrees with detection.
    let resolved_family = match reconcile_detected_family(supplied_family, detected_family) {
        Ok(family) => family,
        Err(mismatch) => {
            return fail_job(
                api,
                &job.id,
                "LoRA import failed.",
                Some(format!(
                    "LoRA file appears to be a {} model, but family was declared as {}. Re-import with family {} or pick a different file.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await;
        }
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
        upsert_manifest_entry(&manifest_path, "loras", manifest_entry).await?;
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
        SafetensorsHeaderError::IncompleteData { declared, actual } => format!(
            "Imported model file is incomplete or corrupt ({actual} bytes on disk, but its header \
             declares at least {declared} bytes of tensor data) — the download was likely \
             interrupted. Re-download the complete file."
        ),
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
        Err(error) if downloaded_model_detection_io_error_is_inconclusive(&error) => {
            return DownloadFamilyCheck::Proceed;
        }
        Err(error) => return DownloadFamilyCheck::DetectionFailed(error),
    };
    match reconcile_detected_family(supplied, detected) {
        Ok(_) => DownloadFamilyCheck::Proceed,
        Err(mismatch) => DownloadFamilyCheck::Mismatch(mismatch),
    }
}

pub(crate) fn downloaded_model_detection_io_error_is_inconclusive(
    error: &SafetensorsHeaderError,
) -> bool {
    let SafetensorsHeaderError::Io(io_error) = error else {
        return false;
    };
    // Hugging Face snapshots may contain symlinks/reparse points into `blobs`.
    // On some Windows machines, opening those links fails with ERROR_UNTRUSTED_
    // MOUNT_POINT (448). The download is already complete; this optional family
    // check should become inconclusive rather than failing the model install.
    io_error.raw_os_error() == Some(448)
        || io_error
            .to_string()
            .to_ascii_lowercase()
            .contains("untrusted mount point")
}

/// Enforce family parity with model import on a completed download: verify the
/// downloaded weights match the catalog-declared family and fail the job on a
/// confident mismatch or a clearly invalid header. Windows cache-link traversal
/// errors are treated as inconclusive because the download itself has completed.
/// Returns `Ok(true)` when the download may complete, `Ok(false)` when the job was
/// already failed and the caller should return.
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
                SafetensorsHeaderError::IncompleteData { declared, actual } => format!(
                    "Downloaded model file is incomplete or corrupt ({actual} bytes on disk, but \
                     its header declares at least {declared} bytes of tensor data) — the download \
                     was likely interrupted. Re-download the complete file."
                ),
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

    // sc-6137: verify a caller-supplied content digest for source-URL/uploaded model
    // imports (HF repo files are per-file verified during download). On mismatch the
    // file is removed and the job fails with an actionable message.
    if let Some(expected_sha256) = optional_payload_string(&job.payload, "expectedSha256") {
        if let Some(file) = first_safetensors_path(&target_dir) {
            if let Err(error) = verify_file_sha256(&file, expected_sha256, "Imported model").await {
                return fail_job(
                    api,
                    &job.id,
                    "Model import failed.",
                    Some(error.to_string()),
                )
                .await;
            }
        }
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
        upsert_manifest_entry(&manifest_path, "models", manifest_entry).await?;
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// sc-6072: `safetensors_file_is_unusable` selects exactly the cases where the
    /// file on disk is itself bad (so the import path discards it) versus a transient
    /// read failure (kept for a retry).
    #[test]
    fn safetensors_file_is_unusable_classifies_variants() {
        assert!(!safetensors_file_is_unusable(&SafetensorsHeaderError::Io(
            std::io::Error::other("transient")
        )));
        assert!(safetensors_file_is_unusable(
            &SafetensorsHeaderError::InvalidHeader
        ));
        assert!(safetensors_file_is_unusable(
            &SafetensorsHeaderError::IncompleteData {
                declared: 1024,
                actual: 16,
            }
        ));
    }

    /// sc-6072: a truncated `.safetensors` in the import target dir is rejected by the
    /// import-time detector with an actionable, file-incompleteness message.
    #[test]
    fn detect_family_in_target_dir_rejects_truncated_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let header = json!({
            "__metadata__": { "format": "pt" },
            "lora.down.weight": { "dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512] },
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize");
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        // Declares 512 bytes of tensor data but writes none — truncated download.
        std::fs::write(dir.path().join("lora.safetensors"), buffer).expect("write");

        let error = detect_family_in_target_dir(dir.path())
            .expect_err("a truncated file must not be accepted");
        assert!(
            matches!(error, SafetensorsHeaderError::IncompleteData { .. }),
            "expected IncompleteData, got {error:?}"
        );
        assert!(safetensors_file_is_unusable(&error));
        let message = lora_family_detection_error(&error).to_ascii_lowercase();
        assert!(
            message.contains("incomplete") && message.contains("re-import"),
            "message should be actionable, got: {message}"
        );
    }

    /// Resolve a HuggingFace cache snapshot dir for `models--<dir>` (test helper).
    fn hf_snapshot(model_dir: &str) -> PathBuf {
        let home = std::env::var("HOME").expect("HOME");
        std::fs::read_dir(
            Path::new(&home).join(format!(".cache/huggingface/hub/{model_dir}/snapshots")),
        )
        .expect("HF cache snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a snapshot dir")
    }

    /// Real-weights smoke for the native Rust/MLX FLUX.2-klein true_v2 converter
    /// (sc-3136 consumer cutover, sc-3032): runs the actual `convert_flux2_klein_diffusers`
    /// wrapper used by `run_model_convert_job` on the cached wikeeyang bf16 single-file +
    /// base klein-9B, and asserts the assembled diffusers dir is complete (real transformer
    /// weights + config + model_index) with the borrowed components as resolvable symlinks.
    /// Mirrors mlx-gen's `convert_real_weights` but exercises the SceneWorks call site.
    /// Needs both repos in the HF cache (writes ~18 GB to the temp dir). Run with:
    ///   cargo test -p sceneworks-worker --lib -- --ignored flux2_true_v2_rust_convert
    #[test]
    #[ignore]
    fn flux2_true_v2_rust_convert_real_weights() {
        let source = hf_snapshot("models--wikeeyang--Flux2-Klein-9B-True-V2")
            .join("Flux2-Klein-9B-True-v2-bf16.safetensors");
        let base = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9B");
        assert!(
            source.is_file(),
            "missing wikeeyang bf16 single-file: {}",
            source.display()
        );
        assert!(
            base.join("transformer").is_dir(),
            "missing base klein transformer: {}",
            base.display()
        );

        let out = std::env::temp_dir().join(format!("sw_true_v2_convert_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);

        convert_flux2_klein_diffusers(&source, &base, &out).expect("native true_v2 convert");

        // Transformer weights + config are written as real files (the remapped, base-validated
        // diffusers transformer); model_index.json is copied.
        assert!(out
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file());
        assert!(out.join("transformer/config.json").is_file());
        assert!(out.join("model_index.json").is_file());
        // vae / text_encoder / tokenizer / scheduler are absolute symlinks borrowed from the
        // base install; `.exists()` follows the link, so a broken symlink fails here.
        for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            assert!(
                out.join(sub).exists(),
                "borrowed `{sub}` missing or broken symlink in {}",
                out.display()
            );
        }

        let _ = std::fs::remove_dir_all(&out);
    }

    /// Real-weights smoke for the native Rust/MLX LTX-2.3 converter (sc-3224 engine + sc-3240
    /// cutover) — the LTX path was never routed through the Rust job before. Runs the actual
    /// `convert_ltx_native` used by `run_model_convert_job` on the cached eros single-file + the base
    /// LTX-2.3 upscaler, and asserts the split MLX dir is complete, including the
    /// `upsampler.safetensors` the loader hard-requires (merged from the base repo) and the
    /// `split_model.json` Q4 quant manifest. Needs both repos in the HF cache. Run with:
    ///   cargo test -p sceneworks-worker --lib -- --ignored ltx_eros_rust_convert
    #[test]
    #[ignore]
    fn ltx_eros_rust_convert_real_weights() {
        let source =
            hf_snapshot("models--TenStrip--LTX2.3-10Eros").join("10Eros_v1_bf16.safetensors");
        let upscaler_dir = hf_snapshot("models--Lightricks--LTX-2.3");
        assert!(
            source.is_file(),
            "missing eros single-file checkpoint: {}",
            source.display()
        );
        assert!(
            upscaler_dir.join(LTX_SPATIAL_UPSCALER_FILE).is_file(),
            "missing base LTX-2.3 spatial upscaler: {}",
            upscaler_dir.display()
        );

        let out = std::env::temp_dir().join(format!("sw_ltx_eros_convert_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);

        convert_ltx_native(&source, &upscaler_dir, &out, 4).expect("native LTX-2.3 eros convert");

        for file in [
            "transformer.safetensors",
            "connector.safetensors",
            // The loader-required latent upsampler, merged from the base LTX-2.3 repo (sc-3240).
            "upsampler.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "config.json",
            "embedded_config.json",
            // The Q4 quant geometry the loader reads back (sc-2686).
            "split_model.json",
        ] {
            assert!(
                out.join(file).is_file(),
                "converted LTX dir missing `{file}` in {}",
                out.display()
            );
        }

        let _ = std::fs::remove_dir_all(&out);
    }
}

// Cross-platform (not macOS-gated like the module above): exercises the Windows
// symlink-repair logic, which is compiled under `test` on every platform.
#[cfg(test)]
mod hardlink_tests {
    use super::*;

    /// The HF-cache symlink-traversal fix: a `snapshots/<rev>/<sub>/file ->
    /// ../blobs/<etag>` symlink is replaced in place by a non-reparse entry (hardlink,
    /// or a copy fallback) pointing at the same bytes, and the blob is left untouched.
    /// Recurses into component sub-dirs. Skips when the platform won't let the test
    /// create a symlink (e.g. Windows without Developer Mode).
    #[test]
    fn materialize_snapshot_hardlinks_replaces_symlinks_with_real_files() {
        let root = tempfile::tempdir().expect("tempdir");
        let blobs = root.path().join("blobs");
        std::fs::create_dir_all(&blobs).expect("blobs dir");
        let blob = blobs.join("deadbeef");
        std::fs::write(&blob, b"tokenizer-bytes").expect("write blob");

        let snapshot = root.path().join("snapshots").join("rev");
        let tokenizer_dir = snapshot.join("tokenizer");
        std::fs::create_dir_all(&tokenizer_dir).expect("tokenizer dir");
        let link = tokenizer_dir.join("tokenizer.json");
        // Relative, like huggingface_hub: snapshots/rev/tokenizer/ -> ../../../blobs.
        let rel_target: PathBuf = ["..", "..", "..", "blobs", "deadbeef"].iter().collect();

        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&rel_target, &link).is_ok();
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_file(&rel_target, &link).is_ok();
        #[cfg(not(any(unix, windows)))]
        let made = false;
        if !made {
            return; // no privilege to create a symlink — nothing to convert
        }
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "precondition: entry is a symlink before repair"
        );

        materialize_snapshot_hardlinks(&snapshot);

        let meta = std::fs::symlink_metadata(&link).expect("metadata after repair");
        assert!(
            !meta.file_type().is_symlink(),
            "the symlink must be replaced by a non-reparse entry"
        );
        assert_eq!(
            std::fs::read(&link).expect("read repaired entry"),
            b"tokenizer-bytes"
        );
        assert_eq!(
            std::fs::read(&blob).expect("read blob"),
            b"tokenizer-bytes",
            "the blob itself must be untouched"
        );
    }
}

/// Offline quant-matrix tier builder (sc-8513, epic 8506) — the generalization of the FLUX.2-dev
/// pilot's one-off builder into a per-converter dispatch over the SAME production `ConvertPlan`
/// converters, so each hosted tier is byte-equivalent to the install-time output.
///
/// Produces ONE complete, HF-uploadable turnkey subdir (`q4/`, `q8/`, or `bf16/`) from a cached
/// dense diffusers source:
/// - `q4`/`q8`: run the production converter (packs `transformer/`, symlinks the dense TE/VAE/
///   tokenizer through), then dereference those symlinks into real files (HF upload does not follow
///   cache blob symlinks across a move).
/// - `bf16`: mirror the dense diffusers tree as-is (no quantize) — the dense backbone IS the bf16
///   tier.
///
/// `#[ignore]`: needs the cached dense source snapshot + Metal. Run ONE tier at a time (bounds the
/// transformer quantize disk peak), e.g.:
/// ```text
/// SC8513_CONVERTER=sd3_5_large_quant SC8513_BITS=4 \
///   SC8513_SRC="$HOME/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots/<rev>" \
///   SC8513_OUT="$HOME/sc8513/sd3.5-large-mlx" \
///   cargo test -p sceneworks-worker --release tier_builder::build_tier -- --ignored --nocapture
/// ```
/// `SC8513_BITS<=0` builds the bf16 tier. New architectures: add a `SC8513_CONVERTER` arm mapping to
/// the model's production converter + its dense-reuse subdir list.
#[cfg(all(test, target_os = "macos"))]
mod tier_builder {
    use super::*;
    use std::path::{Path, PathBuf};

    fn env_path(key: &str) -> PathBuf {
        PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
    }

    /// Recursively copy `src` → `dst`, resolving any symlink (incl. HF-cache blob symlinks) to its
    /// real bytes so the output is a self-contained tree of real files.
    fn copy_tree_deref(src: &Path, dst: &Path) {
        if src.is_dir() {
            std::fs::create_dir_all(dst).unwrap();
            for entry in std::fs::read_dir(src).unwrap() {
                let entry = entry.unwrap();
                copy_tree_deref(&entry.path(), &dst.join(entry.file_name()));
            }
        } else {
            let real = std::fs::canonicalize(src).unwrap();
            std::fs::copy(&real, dst).unwrap();
        }
    }

    /// Replace every symlink under `dir` (the dense TE/VAE/tokenizer the converter borrowed from the
    /// source) with a real copy of its canonical target, in place.
    fn deref_symlinks(dir: &Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let ty = std::fs::symlink_metadata(&path).unwrap().file_type();
            if ty.is_symlink() {
                let real = std::fs::canonicalize(&path).unwrap();
                std::fs::remove_file(&path).unwrap();
                if real.is_dir() {
                    copy_tree_deref(&real, &path);
                } else {
                    std::fs::copy(&real, &path).unwrap();
                }
            } else if ty.is_dir() {
                deref_symlinks(&path);
            }
        }
    }

    /// The diffusers subdirs a converter reuses DENSE (also the extra bf16-mirror set beyond
    /// `transformer/`). `scheduler` is optional metadata and skipped when absent.
    fn dense_subs(converter: &str) -> &'static [&'static str] {
        match converter {
            "sd3_5_large_quant" | "sd3_5_large_turbo_quant" | "sd3_5_medium_quant" => &[
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
                "tokenizer",
                "tokenizer_2",
                "tokenizer_3",
                "vae",
                "scheduler",
                "model_index.json",
            ],
            "flux2_dev_quant" => &["text_encoder", "vae", "tokenizer", "model_index.json"],
            other => panic!("unknown SC8513_CONVERTER {other}"),
        }
    }

    #[test]
    #[ignore = "real-weight: cached dense source snapshot + Metal"]
    fn build_tier() {
        let converter = std::env::var("SC8513_CONVERTER").expect("set SC8513_CONVERTER");
        let bits: i32 = std::env::var("SC8513_BITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let group_size: i32 = std::env::var("SC8513_GROUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let src = env_path("SC8513_SRC");
        let out_root = env_path("SC8513_OUT");
        let tier = if bits <= 0 {
            "bf16".to_owned()
        } else {
            format!("q{bits}")
        };
        let out = out_root.join(&tier);
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        eprintln!(
            "[sc-8513] {converter} {tier} (bits {bits}, group {group_size}) {} -> {}",
            src.display(),
            out.display()
        );

        if bits <= 0 {
            // bf16: the dense backbone IS the tier — mirror transformer + the dense-reuse subdirs.
            copy_tree_deref(&src.join("transformer"), &out.join("transformer"));
            eprintln!("[sc-8513]   bf16 transformer mirrored");
            for sub in dense_subs(&converter) {
                let s = src.join(sub);
                if !s.exists() {
                    assert_eq!(*sub, "scheduler", "bf16 source missing required `{sub}`");
                    continue;
                }
                copy_tree_deref(&s, &out.join(sub));
                eprintln!("[sc-8513]   bf16 {sub} mirrored");
            }
        } else {
            // q4/q8: production converter packs transformer/ + symlinks dense parts; then realize.
            let result = match converter.as_str() {
                "sd3_5_large_quant" => {
                    convert_sd3_prequant(&src, &out, Sd3Variant::Large, bits, group_size)
                }
                "sd3_5_large_turbo_quant" => {
                    convert_sd3_prequant(&src, &out, Sd3Variant::LargeTurbo, bits, group_size)
                }
                "sd3_5_medium_quant" => {
                    convert_sd3_prequant(&src, &out, Sd3Variant::Medium, bits, group_size)
                }
                "flux2_dev_quant" => convert_flux2_dev_prequant(&src, &out, bits, group_size),
                other => panic!("unknown SC8513_CONVERTER {other}"),
            };
            result.unwrap_or_else(|error| panic!("production convert failed: {error}"));
            eprintln!("[sc-8513]   transformer packed; dereferencing dense symlinks");
            deref_symlinks(&out);
        }
        eprintln!("[sc-8513] DONE {tier} at {}", out.display());
    }
}
