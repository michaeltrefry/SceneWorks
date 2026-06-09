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
        let mut gpu = mlx_gpu();
        // Seed the registration with a live snapshot so the worker shows
        // memory/load immediately, mirroring the nvidia path (heartbeats refresh
        // it). `query_mlx_utilization` reads Apple-Silicon unified-memory + GPU
        // load from the same unprivileged probes the Python mps worker uses.
        gpu.utilization = query_mlx_utilization().await;
        return gpu;
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
    // The Apple-Silicon `mlx` worker has no nvidia-smi to query; read its
    // unified-memory + GPU load from IOKit instead (epic 3018) so the dashboard
    // shows memory/load for it like it does for the Python `mps` worker.
    #[cfg(target_os = "macos")]
    if gpu_id == "mlx" {
        return query_mlx_utilization().await;
    }
    query_nvidia_gpus()
        .await
        .into_iter()
        .find(|gpu| gpu.id == gpu_id)
        .and_then(|gpu| gpu.utilization)
}

/// Apple-Silicon unified-memory + GPU-load snapshot for the `mlx` worker, shaped
/// like the nvidia path. Total = the machine's unified RAM (`sysctl hw.memsize`).
/// Used = **system-wide** memory pressure from `vm_stat` (App + Wired + Compressed,
/// the same figure Activity Monitor shows as "Memory Used") — on unified memory the
/// GPU draws from this whole pool, so the IOAccelerator "In use system memory" stat
/// (GPU-resident only) badly under-reports it (it stays ~5-7% while the machine is
/// half-full). Load = IOKit's `IOAccelerator` "Device Utilization %" (`ioreg`,
/// unprivileged). Returns `None` only when no probe yields anything.
#[cfg(target_os = "macos")]
pub(crate) async fn query_mlx_utilization() -> Option<WorkerUtilizationSnapshot> {
    let total_mb = sysctl_memsize_mb().await;
    let used_mb = vm_stat_used_mb().await;
    let gpu_load_percent = ioreg_gpu_load().await;
    mlx_utilization_from(total_mb, used_mb, gpu_load_percent)
}

/// Total unified memory (MB) via `sysctl -n hw.memsize`.
#[cfg(target_os = "macos")]
async fn sysctl_memsize_mb() -> Option<u64> {
    let output = tokio::time::timeout(
        Duration::from_secs(2),
        Command::new("sysctl").args(["-n", "hw.memsize"]).output(),
    )
    .await;
    match output {
        Ok(Ok(output)) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
            .ok()
            .map(|bytes| bytes / (1024 * 1024)),
        _ => None,
    }
}

/// Numeric `PerformanceStatistics` from the `IOAccelerator` IOKit node, via
/// `ioreg -r -c IOAccelerator -d 1` (the integrated GPU's live stats, no elevated
/// privileges). Empty when ioreg is unavailable.
#[cfg(target_os = "macos")]
async fn ioreg_accelerator_stats() -> BTreeMap<String, u64> {
    let output = tokio::time::timeout(
        Duration::from_secs(3),
        Command::new("ioreg")
            .args(["-r", "-c", "IOAccelerator", "-d", "1"])
            .output(),
    )
    .await;
    match output {
        Ok(Ok(output)) if output.status.success() => {
            parse_ioreg_accelerator_stats(&String::from_utf8_lossy(&output.stdout))
        }
        _ => BTreeMap::new(),
    }
}

/// Parse every `"key" = <integer>` pair from `ioreg` output (last value wins per
/// key), mirroring the Python `re.findall(r'"([^"]+)"\s*=\s*(\d+)', ...)`.
#[cfg(target_os = "macos")]
fn parse_ioreg_accelerator_stats(output: &str) -> BTreeMap<String, u64> {
    let mut stats = BTreeMap::new();
    let bytes = output.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let Some(relative_close) = output[index + 1..].find('"') else {
            break;
        };
        let key_end = index + 1 + relative_close;
        let key = &output[index + 1..key_end];
        let mut cursor = key_end + 1;
        while bytes
            .get(cursor)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            cursor += 1;
        }
        if bytes.get(cursor) == Some(&b'=') {
            cursor += 1;
            while bytes
                .get(cursor)
                .is_some_and(|byte| byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            let digits_start = cursor;
            while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
                cursor += 1;
            }
            if cursor > digits_start {
                if let Ok(value) = output[digits_start..cursor].parse::<u64>() {
                    stats.insert(key.to_owned(), value);
                }
            }
        }
        index = key_end + 1;
    }
    stats
}

/// Integrated-GPU load percent from the IOAccelerator stats: "Device Utilization %"
/// (preferred), else "GPU Activity(%)".
#[cfg(target_os = "macos")]
async fn ioreg_gpu_load() -> Option<f64> {
    let stats = ioreg_accelerator_stats().await;
    stats
        .get("Device Utilization %")
        .or_else(|| stats.get("GPU Activity(%)"))
        .map(|value| *value as f64)
}

/// System-wide used memory (MB) from `vm_stat`, i.e. Activity Monitor's
/// "Memory Used" = App Memory + Wired + Compressed. Returns `None` when the
/// page size or the required counters can't be read.
#[cfg(target_os = "macos")]
async fn vm_stat_used_mb() -> Option<u64> {
    let output =
        tokio::time::timeout(Duration::from_secs(2), Command::new("vm_stat").output()).await;
    match output {
        Ok(Ok(output)) if output.status.success() => {
            system_used_mb_from_vm_stat(&String::from_utf8_lossy(&output.stdout))
        }
        _ => None,
    }
}

/// The page size (bytes) from the `vm_stat` header `(page size of N bytes)`.
#[cfg(target_os = "macos")]
fn parse_vm_stat_page_size(output: &str) -> Option<u64> {
    const MARKER: &str = "page size of ";
    let start = output.find(MARKER)? + MARKER.len();
    output[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Each `Label: <count>.` line of `vm_stat` as a page-count map.
#[cfg(target_os = "macos")]
fn parse_vm_stat_counts(output: &str) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for line in output.lines() {
        let Some((label, value)) = line.split_once(':') else {
            continue;
        };
        if let Ok(pages) = value.trim().trim_end_matches('.').parse::<u64>() {
            counts.insert(label.trim().to_owned(), pages);
        }
    }
    counts
}

/// Compute Activity Monitor's "Memory Used" (App + Wired + Compressed) in MB from
/// raw `vm_stat` output. App Memory = anonymous pages minus the purgeable (cache)
/// pages that can be reclaimed. `None` if the page size or any required counter is
/// absent.
#[cfg(target_os = "macos")]
fn system_used_mb_from_vm_stat(output: &str) -> Option<u64> {
    let page_size = parse_vm_stat_page_size(output)?;
    let counts = parse_vm_stat_counts(output);
    let anonymous = counts.get("Anonymous pages").copied()?;
    let wired = counts.get("Pages wired down").copied()?;
    let compressed = counts.get("Pages occupied by compressor").copied()?;
    let purgeable = counts.get("Pages purgeable").copied().unwrap_or(0);
    let app_pages = anonymous.saturating_sub(purgeable);
    let used_pages = app_pages + wired + compressed;
    Some(used_pages.saturating_mul(page_size) / (1024 * 1024))
}

/// Build the snapshot from the unified-memory total, system-wide used memory, and
/// GPU load. Free = total − used (saturating). Returns `None` only when no field
/// is available (non-Apple-Silicon, sandboxed, etc.).
#[cfg(target_os = "macos")]
fn mlx_utilization_from(
    total_mb: Option<u64>,
    used_mb: Option<u64>,
    gpu_load_percent: Option<f64>,
) -> Option<WorkerUtilizationSnapshot> {
    let free_mb = match (total_mb, used_mb) {
        (Some(total), Some(used)) => Some(total.saturating_sub(used)),
        _ => None,
    };
    if total_mb.is_none() && used_mb.is_none() && gpu_load_percent.is_none() {
        return None;
    }
    Some(WorkerUtilizationSnapshot {
        memory_total_mb: total_mb,
        memory_used_mb: used_mb,
        memory_free_mb: free_mb,
        gpu_load_percent,
    })
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
/// `image_edit` (plain Image Edit, sc-3513), `image_detail` (tile-ControlNet refine,
/// sc-3060), `video_generate`, and LoRA/LoKr
/// `lora_train` + `lora_train_execute` (epic 3039), served in-process by the linked
/// mlx-gen engine. `Gpu` (not
/// `Cpu`) so the API's `worker_supports_job` lets GPU jobs route here; it deliberately
/// does NOT carry the CPU utility capabilities, so downloads/imports/etc. still go to
/// the CPU worker. `video_generate` is claimed from the video runtime onward (sc-3033);
/// the procedural stub backs models whose real MLX path is not yet linked (Wan sc-3034,
/// LTX+audio sc-3035), and the API-side MLX-vs-Python routing is sc-3036.
///
/// Training (epic 3039, sc-3043/3049): the engine is always linked on macOS, so this
/// worker can both validate plans (`lora_train`) and run real training
/// (`lora_train_execute`) — unlike the Python worker, which advertises execute only
/// when its torch backend is present. The API gates which `lora_train` jobs reach
/// here to the MLX-native families (`jobs_store::training_job_is_mlx_eligible`);
/// `kolors`/`lens` and LoKr-on-Wan stay on the Python torch worker.
///
/// Dataset captioning (sc-3556): JoyCaption is linked through mlx-gen's captioner
/// registry and runs in-process on this worker; the Python captioner remains the
/// Windows/Linux and explicit non-MLX fallback.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_gpu() -> DiscoveredGpu {
    DiscoveredGpu {
        id: "mlx".to_owned(),
        name: "Apple Silicon (MLX)".to_owned(),
        capabilities: vec![
            WorkerCapability::Gpu,
            WorkerCapability::ImageGenerate,
            // Plain Image Edit (sc-3513): the `image_edit` job type
            // (`mode=edit_image` + `sourceAssetId`, epic 2427) runs the same engine
            // edit paths as the `character_image` reference flow — qwen/flux2/sdxl
            // edit dispatched by payload model+mode in `run_image_generate_job`. The
            // API only routes MLX-eligible edit models here (`image_job_is_mlx_eligible`);
            // torch-only edit models stay on the Python worker.
            WorkerCapability::ImageEdit,
            // Tile-ControlNet detail refine (epic 3041, sc-3060) — the SDXL-family
            // `image_detail` job runs in-process on the engine here too.
            WorkerCapability::ImageDetail,
            WorkerCapability::VideoGenerate,
            // Native Rust LoRA/LoKr training (epic 3039): plan validation +
            // real execution, both served in-process by `mlx_gen::load_trainer`.
            WorkerCapability::LoraTrain,
            WorkerCapability::LoraTrainExecute,
            WorkerCapability::TrainingCaption,
            // DWPose whole-body pose detection (epic 3482, sc-3487): RTMW via
            // onnxruntime/CoreML, served in-process by `pose_jobs::run_pose_detect_job`.
            // Replaces the Python rtmlib `pose_detect` path so the Pose Library
            // "create from photo" flow + InstantID pose conditioning keep working on a
            // Python-free Mac. The detector auto-provisions its onnx weights on first
            // use (download-on-first-use parity with rtmlib).
            WorkerCapability::PoseDetect,
            // Real-ESRGAN image upscaling (epic 3482, sc-3489): RRDBNet x2/x4 via
            // onnxruntime/CoreML, served in-process by `upscale_jobs::run_image_upscale_job`.
            // Replaces the Python torch Real-ESRGAN path so the Image Editor upscale tool
            // works on a Python-free Mac. Only `engine=real-esrgan` (the default) is
            // served here; `aura-sr` stays on the Python worker (routing oracle).
            WorkerCapability::ImageUpscale,
            // Real, model-backed person detection + tracking (epic 3482, sc-3488 /
            // sc-3633/3634/3709): the native-MLX YOLO11 detector + SORT/ByteTrack tracker +
            // SAM2 segmenter run in-process (`person_jobs` / `person_track` /
            // `person_segment`), replacing the Python onnxruntime/torch path so the
            // Replace-Person detect → track → mask flow works on a Python-free Mac. A
            // `preview: true` job still routes to the CPU placeholder's procedural preview
            // capability (`required_capability` in jobs_store).
            WorkerCapability::PersonDetect,
            WorkerCapability::PersonTrack,
        ],
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

#[cfg(all(test, target_os = "macos"))]
mod mlx_utilization_tests {
    use super::*;

    // Trimmed, realistic `ioreg -r -c IOAccelerator -d 1` excerpt: the load stat
    // lives in a nested `PerformanceStatistics = {...}` dict and string values must
    // be ignored.
    const IOREG_SAMPLE: &str = r#"
+-o IOAccelerator  <class IOAccelerator>
    {
      "IOPlatformUUID" = "ABCD-1234-NOT-A-NUMBER"
      "PerformanceStatistics" = {"Device Utilization %"=37,"GPU Activity(%)"=12}
      "IOGVAStatistics" = {"someText"="skip me"}
    }
"#;

    // Trimmed real `vm_stat` output (16 KB pages).
    const VM_STAT_SAMPLE: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    38555.
Pages active:                                2048579.
Pages inactive:                              2046703.
Pages speculative:                               609.
Pages wired down:                            1655232.
Pages purgeable:                               18353.
Anonymous pages:                             3033035.
Pages stored in compressor:                  3915548.
Pages occupied by compressor:                2483449.
";

    #[test]
    fn ioreg_parser_extracts_load_and_skips_string_values() {
        let stats = parse_ioreg_accelerator_stats(IOREG_SAMPLE);
        assert_eq!(stats.get("Device Utilization %"), Some(&37));
        assert_eq!(stats.get("GPU Activity(%)"), Some(&12));
        // String-valued keys are never recorded as numbers.
        assert_eq!(stats.get("IOPlatformUUID"), None);
        assert_eq!(stats.get("someText"), None);
    }

    #[test]
    fn vm_stat_page_size_parsed_from_header() {
        assert_eq!(parse_vm_stat_page_size(VM_STAT_SAMPLE), Some(16_384));
        assert_eq!(parse_vm_stat_page_size("no header here"), None);
    }

    #[test]
    fn vm_stat_counts_parsed_and_header_skipped() {
        let counts = parse_vm_stat_counts(VM_STAT_SAMPLE);
        assert_eq!(counts.get("Anonymous pages"), Some(&3_033_035));
        assert_eq!(counts.get("Pages wired down"), Some(&1_655_232));
        assert_eq!(counts.get("Pages occupied by compressor"), Some(&2_483_449));
        assert_eq!(counts.get("Pages purgeable"), Some(&18_353));
        // The header line's value is non-numeric and must not become a count.
        assert_eq!(counts.get("Mach Virtual Memory Statistics"), None);
    }

    #[test]
    fn vm_stat_used_is_app_plus_wired_plus_compressed() {
        let used = system_used_mb_from_vm_stat(VM_STAT_SAMPLE).expect("used computed");
        // App (anonymous - purgeable) + wired + compressed, in 16 KB pages.
        let expected = (3_033_035u64 - 18_353 + 1_655_232 + 2_483_449) * 16_384 / (1024 * 1024);
        assert_eq!(used, expected);
        // ~109 GB on this sample — system-wide, not the ~few-GB GPU-resident figure.
        assert!(used > 100_000, "expected system-wide used, got {used} MB");
    }

    #[test]
    fn vm_stat_used_is_none_without_required_counters() {
        assert!(system_used_mb_from_vm_stat("Pages free: 10.\n").is_none());
        assert!(system_used_mb_from_vm_stat("garbage").is_none());
    }

    #[test]
    fn builds_snapshot_with_total_used_and_load() {
        let snapshot = mlx_utilization_from(Some(131_072), Some(111_771), Some(37.0))
            .expect("snapshot built from real data");
        assert_eq!(snapshot.memory_total_mb, Some(131_072));
        assert_eq!(snapshot.memory_used_mb, Some(111_771));
        assert_eq!(snapshot.memory_free_mb, Some(131_072 - 111_771));
        assert_eq!(snapshot.gpu_load_percent, Some(37.0));
    }

    #[test]
    fn free_is_clamped_when_used_exceeds_total() {
        let snapshot =
            mlx_utilization_from(Some(8_192), Some(9_216), None).expect("snapshot built");
        assert_eq!(snapshot.memory_free_mb, Some(0));
    }

    #[test]
    fn returns_none_when_nothing_is_available() {
        assert!(mlx_utilization_from(None, None, None).is_none());
    }

    #[test]
    fn load_only_snapshot_leaves_memory_unset() {
        let snapshot =
            mlx_utilization_from(None, None, Some(55.0)).expect("snapshot built from load only");
        assert_eq!(snapshot.gpu_load_percent, Some(55.0));
        assert_eq!(snapshot.memory_total_mb, None);
        assert_eq!(snapshot.memory_free_mb, None);
    }

    // Live end-to-end probe against this machine's real `sysctl`/`vm_stat`/`ioreg`.
    // Ignored by default (machine-dependent); run with `--ignored` on Apple Silicon.
    #[tokio::test]
    #[ignore]
    async fn live_mlx_utilization_probe_reports_system_memory() {
        let snapshot = query_mlx_utilization()
            .await
            .expect("Apple Silicon should report a utilization snapshot");
        let total = snapshot
            .memory_total_mb
            .expect("total unified memory present");
        let used = snapshot.memory_used_mb.expect("system used memory present");
        assert!(used <= total, "used {used} must not exceed total {total}");
        assert_eq!(snapshot.memory_free_mb, Some(total - used));
        eprintln!("live mlx utilization snapshot: {snapshot:?}");
    }
}
