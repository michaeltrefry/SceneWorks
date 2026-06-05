use super::*;

pub(crate) async fn discover_gpu(requested_gpu_id: &str) -> DiscoveredGpu {
    if requested_gpu_id == "cpu" {
        return cpu_gpu();
    }
    // Apple-Silicon native MLX GPU worker (epic 3018): an explicit `mlx` gpu id
    // selects the in-process mlx-gen engine. Kept off the `auto`/nvidia paths so
    // the existing macOS CPU utility worker is unaffected; the desktop launch +
    // routing wiring that selects `mlx` lands in sc-3021/sc-3032.
    #[cfg(target_os = "macos")]
    if requested_gpu_id == "mlx" {
        return mlx_gpu();
    }
    let gpus = discover_gpus().await;
    if requested_gpu_id.is_empty() || requested_gpu_id == "auto" {
        return gpus.into_iter().next().unwrap_or_else(cpu_gpu);
    }
    gpus.into_iter()
        .find(|gpu| gpu.id == requested_gpu_id)
        .unwrap_or_else(|| fallback_gpu(requested_gpu_id))
}

pub(crate) async fn discover_gpus() -> Vec<DiscoveredGpu> {
    let visible_ids = visible_gpu_ids_from_env();
    if visible_ids.as_ref().is_some_and(Vec::is_empty) {
        return Vec::new();
    }
    let gpus = query_nvidia_gpus().await;
    if let Some(ids) = visible_ids {
        let by_id = gpus
            .into_iter()
            .map(|gpu| (gpu.id.clone(), gpu))
            .collect::<BTreeMap<_, _>>();
        return ids
            .into_iter()
            .map(|gpu_id| {
                by_id
                    .get(&gpu_id)
                    .cloned()
                    .unwrap_or_else(|| fallback_gpu(&gpu_id))
            })
            .collect();
    }
    gpus
}

pub(crate) async fn query_nvidia_gpus() -> Vec<DiscoveredGpu> {
    let output = tokio::time::timeout(
        Duration::from_secs(3),
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output(),
    )
    .await;
    match output {
        Ok(Ok(output)) if output.status.success() => {
            parse_nvidia_smi_gpus(&String::from_utf8_lossy(&output.stdout))
        }
        _ => Vec::new(),
    }
}

pub(crate) fn parse_nvidia_smi_gpus(output: &str) -> Vec<DiscoveredGpu> {
    output
        .trim()
        .lines()
        .filter_map(|line| {
            let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
            if parts.len() < 3 {
                return None;
            }
            let index = parts[0];
            let name = parts[1];
            let memory_mb = parts[2];
            Some(DiscoveredGpu {
                id: index.to_owned(),
                name: format!("{name} ({memory_mb} MB)"),
                capabilities: vec![
                    WorkerCapability::Placeholder,
                    WorkerCapability::Gpu,
                    WorkerCapability::Unknown("nvidia".to_owned()),
                ],
                utilization: utilization_from_parts(&parts),
            })
        })
        .collect()
}

pub(crate) fn utilization_from_parts(parts: &[&str]) -> Option<WorkerUtilizationSnapshot> {
    if parts.len() < 6 {
        return None;
    }
    Some(WorkerUtilizationSnapshot {
        memory_total_mb: parse_u64(parts[2]),
        memory_used_mb: parse_u64(parts[3]),
        memory_free_mb: parse_u64(parts[4]),
        gpu_load_percent: parse_f64(parts[5]),
    })
}

pub(crate) fn parse_u64(value: &str) -> Option<u64> {
    value.parse().ok()
}

pub(crate) fn parse_f64(value: &str) -> Option<f64> {
    value.parse().ok()
}

pub(crate) async fn gpu_utilization(gpu_id: &str) -> Option<WorkerUtilizationSnapshot> {
    if gpu_id == "cpu" {
        return None;
    }
    query_nvidia_gpus()
        .await
        .into_iter()
        .find(|gpu| gpu.id == gpu_id)
        .and_then(|gpu| gpu.utilization)
}

pub(crate) fn visible_gpu_ids_from_env() -> Option<Vec<String>> {
    visible_gpu_ids(std::env::var("NVIDIA_VISIBLE_DEVICES").ok().as_deref())
}

pub(crate) fn visible_gpu_ids(value: Option<&str>) -> Option<Vec<String>> {
    let value = value.map(str::trim).filter(|value| !value.is_empty())?;
    match value {
        "all" => None,
        "void" | "none" => Some(Vec::new()),
        _ => Some(
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect(),
        ),
    }
}

pub(crate) fn cpu_gpu() -> DiscoveredGpu {
    DiscoveredGpu {
        id: "cpu".to_owned(),
        name: "Rust CPU utility worker".to_owned(),
        capabilities: vec![WorkerCapability::Placeholder, WorkerCapability::Cpu],
        utilization: None,
    }
}

/// The Apple-Silicon native MLX GPU worker (epic 3018): advertises `image_generate`,
/// served in-process by the linked mlx-gen engine. `Gpu` (not `Cpu`) so the API's
/// `worker_supports_job` lets GPU jobs route here; it deliberately does NOT carry the
/// CPU utility capabilities, so downloads/imports/etc. still go to the CPU worker.
/// Video + the remaining image capabilities are added as their stories land
/// (sc-3022.. image, sc-3034.. video).
#[cfg(target_os = "macos")]
pub(crate) fn mlx_gpu() -> DiscoveredGpu {
    DiscoveredGpu {
        id: "mlx".to_owned(),
        name: "Apple Silicon (MLX)".to_owned(),
        capabilities: vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate],
        utilization: None,
    }
}

pub(crate) fn fallback_gpu(gpu_id: &str) -> DiscoveredGpu {
    DiscoveredGpu {
        id: gpu_id.to_owned(),
        name: format!("GPU {gpu_id}"),
        capabilities: vec![WorkerCapability::Placeholder, WorkerCapability::Gpu],
        utilization: None,
    }
}

pub(crate) fn worker_capabilities(gpu: &DiscoveredGpu) -> Vec<WorkerCapability> {
    let utility_jobs_enabled =
        std::env::var("SCENEWORKS_UTILITY_JOBS").map_or(true, |value| value.trim() != "0");
    worker_capabilities_with_utility(gpu, utility_jobs_enabled)
}

pub(crate) fn worker_capabilities_with_utility(
    gpu: &DiscoveredGpu,
    utility_jobs_enabled: bool,
) -> Vec<WorkerCapability> {
    let mut capabilities = gpu.capabilities.clone();
    let is_cpu = capabilities.contains(&WorkerCapability::Cpu);
    if is_cpu && utility_jobs_enabled {
        capabilities.extend([
            WorkerCapability::FrameExtract,
            WorkerCapability::TimelineExport,
            WorkerCapability::ModelDownload,
            WorkerCapability::ModelImport,
            WorkerCapability::ModelConvert,
            WorkerCapability::LoraImport,
            // Procedural detection/tracking is a preview only. Real, model-backed
            // PersonDetect/PersonTrack run on the Python GPU worker; advertising the
            // preview capabilities here keeps the CPU placeholder claimable solely
            // for explicit `preview: true` jobs (jobs_store::worker_supports_job).
            WorkerCapability::PersonDetectPreview,
            WorkerCapability::PersonTrackPreview,
        ]);
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

pub(crate) fn gpu_worker_id(base_worker_id: &str, gpu_id: &str) -> String {
    let safe_gpu_id = slugify_worker_id_part(gpu_id, "gpu");
    if safe_gpu_id == "0" && base_worker_id.ends_with("-0") {
        return base_worker_id.to_owned();
    }
    if base_worker_id.ends_with("-0") && safe_gpu_id.chars().all(|value| value.is_ascii_digit()) {
        return format!(
            "{}{}",
            &base_worker_id[..base_worker_id.len() - 1],
            safe_gpu_id
        );
    }
    format!("{base_worker_id}-gpu-{safe_gpu_id}")
}

pub(crate) fn cpu_worker_id(base_worker_id: &str) -> String {
    let base = base_worker_id.strip_suffix("-0").unwrap_or(base_worker_id);
    format!("{base}-cpu")
}

pub(crate) fn slugify_worker_id_part(value: &str, fallback: &str) -> String {
    let mut output = String::new();
    let mut previous_dash = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            previous_dash = false;
        } else if !previous_dash && !output.is_empty() {
            output.push('-');
            previous_dash = true;
        }
    }
    while output.ends_with('-') {
        output.pop();
    }
    if output.is_empty() {
        fallback.to_owned()
    } else {
        output
    }
}
