use super::*;

pub(crate) async fn discover_gpu(settings: &Settings) -> DiscoveredGpu {
    let requested_gpu_id = settings.gpu_id.as_str();
    if requested_gpu_id == "cpu" {
        return cpu_gpu();
    }
    // Apple-Silicon native MLX GPU worker (epic 3018): an explicit `mlx` gpu id
    // selects the in-process mlx-gen engine. Kept off the `auto`/nvidia paths so
    // the existing macOS CPU utility worker is unaffected; the desktop launch +
    // routing wiring that selects `mlx` lands in sc-3021/sc-3032.
    #[cfg(target_os = "macos")]
    if requested_gpu_id == "mlx" {
        // The worker's advertised capabilities are derived from the linked engine
        // registry + the enabled tensor backends in `settings` (sc-3723).
        let mut gpu = mlx_gpu(settings);
        // Seed the registration with a live snapshot so the worker shows
        // memory/load immediately, mirroring the nvidia path (heartbeats refresh
        // it). `query_mlx_utilization` reads Apple-Silicon unified-memory + GPU
        // load from the same unprivileged probes the Python mps worker uses.
        gpu.utilization = query_mlx_utilization().await;
        return gpu;
    }
    let gpus = discover_gpus().await;
    let gpu = if requested_gpu_id.is_empty() || requested_gpu_id == "auto" {
        gpus.into_iter().next().unwrap_or_else(cpu_gpu)
    } else {
        gpus.into_iter()
            .find(|gpu| gpu.id == requested_gpu_id)
            .unwrap_or_else(|| fallback_gpu(requested_gpu_id))
    };
    with_candle_capabilities(gpu, settings)
}

/// Light up the Windows/CUDA candle SDXL lane on the discovered nvidia GPU (epic 3672, sc-3678).
/// When the candle backend is enabled, extend the GPU's advertised capabilities with the
/// registry-DERIVED core (`engines::registry_capabilities` → `image_generate` / `video_generate`
/// from the linked candle generator descriptors, plus `lora_train` + `lora_train_execute` from the
/// linked candle trainers — sc-7817) plus a `candle` marker capability. The marker lets the API
/// routing gate (`jobs_store::worker_supports_job`) recognize this worker and confine each lane to
/// the candle-served shapes (txt2img/txt2video, the candle-trainable kernels, …) — every other shape
/// falls back to the Python torch worker. Mirrors the macOS `mlx_gpu` derivation, but bolted onto
/// the real nvidia GPU descriptor rather than a sentinel id.
///
/// All-targets signature so `discover_gpu` is uniform; a no-op everywhere except the Windows candle
/// build with `backend_candle_enabled`, so production routing is unchanged until the lane is on.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(unused_mut, unused_variables)
)]
fn with_candle_capabilities(mut gpu: DiscoveredGpu, settings: &Settings) -> DiscoveredGpu {
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if settings.backend_candle_enabled && gpu.capabilities.contains(&WorkerCapability::Gpu) {
        let derived = crate::engines::registry_capabilities(settings);
        if !derived.is_empty() {
            for capability in derived {
                if !gpu.capabilities.contains(&capability) {
                    gpu.capabilities.push(capability);
                }
            }
            // Plain Image Edit (sc-5487, epic 5480): the distinct `image_edit` job type
            // (`mode == "edit_image"` + `sourceAssetId`, epic 2427) runs the bespoke candle edit lanes
            // (SdxlEdit / Flux2Edit / QwenEdit) via `run_image_generate_job`, which dispatches by payload
            // model+mode, not job type. Like the MLX carve-out (`mlx_gpu`), `image_edit` is NOT derivable
            // from a single generator descriptor (`registry_capabilities` only emits `image_generate`),
            // so advertise it explicitly — otherwise the candle worker never claims an edit job and the
            // API enforce-fails it `candle_unsupported`. The routing gate confines the claim to the
            // candle-eligible edit models (`image_job_is_candle_eligible`); torch-only edit models stay on
            // the co-resident Python worker.
            if !gpu.capabilities.contains(&WorkerCapability::ImageEdit) {
                gpu.capabilities.push(WorkerCapability::ImageEdit);
            }
            // SenseNova-U1 VQA + Document-Studio interleave (sc-5501) run off the `Generator`
            // registry via the concrete candle `T2iModel::{vqa, interleave_gen}` (their text /
            // text+image output the neutral contract can't express), so they are NOT in
            // `registry_capabilities`. Advertise them explicitly — the candle SenseNova provider is
            // force-linked under `backend-candle`, and the routing gate confines them to the
            // SenseNova-U1 ids (`understanding_job_is_mlx_eligible`).
            for capability in [
                WorkerCapability::ImageVqa,
                WorkerCapability::ImageInterleave,
            ] {
                if !gpu.capabilities.contains(&capability) {
                    gpu.capabilities.push(capability);
                }
            }
            // Image + video upscaling (sc-5928 SeedVR2 + sc-5499 Real-ESRGAN, epic 4811 / epic 5482):
            // off-Mac the candle worker serves `image_upscale` for BOTH Real-ESRGAN (`ort`/CUDA in
            // `upscale_jobs`, the off-Mac sibling of the Mac CoreML path — sc-5499) and SeedVR2
            // (`candle-gen-seedvr2`, force-linked under backend-candle, via `gen_core::load("seedvr2")`),
            // AND the net-new SeedVR2 `video_upscale`. These are job-type capabilities (not a generation
            // modality), so they aren't in `registry_capabilities`; advertise them explicitly. The
            // routing gate (`upscale_job_is_candle_eligible` / `video_upscale_job_is_candle_eligible`)
            // admits Real-ESRGAN + SeedVR2; only `aura-sr` has no candle path (dropped as an offered
            // engine, sc-3668 / sc-5499) and runs on the Python torch worker until Phase 7.
            for capability in [
                WorkerCapability::ImageUpscale,
                WorkerCapability::VideoUpscale,
                WorkerCapability::DatasetUpscale,
            ] {
                if !gpu.capabilities.contains(&capability) {
                    gpu.capabilities.push(capability);
                }
            }
            // SCRFD 5-point face-landmark extraction (sc-5497, epic 5482): the candle SCRFD/ArcFace face
            // stack (`candle-gen-face`, the InstantID/PuLID detector reused directly from kps_jobs.rs)
            // serves `kps_extract` for the Key Point Library "extract kps from this image" flow — the
            // off-Mac sibling of the native-MLX path. A job-type capability (not a generation modality),
            // so it isn't in `registry_capabilities`; advertise it explicitly. Unlike SeedVR2, the Python
            // InsightFace path CAN serve kps_extract, so there's NO torch-refusal gate — the candle worker
            // claims it Python-free, with the co-resident torch worker as fallback (the Python kps path is
            // retired wholesale in Phase 7, epic 5483).
            if !gpu.capabilities.contains(&WorkerCapability::KpsExtract) {
                gpu.capabilities.push(WorkerCapability::KpsExtract);
            }
            // Dataset Doctor face pass (sc-6538): the off-Mac sibling of the macOS face stack — the
            // candle SCRFD/ArcFace stack (`candle-gen-face`, reused from kps_jobs.rs) embeds the largest
            // face of each Person-dataset image. A job-type capability (not a generation modality), so
            // it isn't in `registry_capabilities`; advertise it explicitly, like kps_extract.
            if !gpu
                .capabilities
                .contains(&WorkerCapability::DatasetFaceAnalysis)
            {
                gpu.capabilities.push(WorkerCapability::DatasetFaceAnalysis);
            }
            // On-demand face-likeness compare (sc-4415): the off-Mac sibling of the macOS face stack —
            // the candle SCRFD/ArcFace stack scores a candidate asset against a source identity
            // reference. A job-type capability (not a generation modality), so it isn't in
            // `registry_capabilities`; advertise it explicitly, like dataset_face_analysis.
            if !gpu
                .capabilities
                .contains(&WorkerCapability::FaceLikenessCompare)
            {
                gpu.capabilities.push(WorkerCapability::FaceLikenessCompare);
            }
            // DWPose whole-body pose detection (sc-5496, epic 5482): the off-Mac sibling of the macOS
            // `ort`/CoreML path (sc-3487) — the same RTMW detector via `pose_jobs::run_pose_detect_job`
            // with the CUDA execution provider, serving `pose_detect` for the Pose Library "create from
            // photo" flow + InstantID pose conditioning. A job-type capability (not a generation modality),
            // so it isn't in `registry_capabilities`; advertise it explicitly. Like kps_extract — and
            // unlike SeedVR2 — the Python rtmlib path CAN serve pose_detect, so there's NO torch-refusal
            // gate: the candle worker claims it Python-free, with the co-resident torch worker as a
            // fallback (the Python rtmlib path is retired wholesale in Phase 7, epic 5483).
            if !gpu.capabilities.contains(&WorkerCapability::PoseDetect) {
                gpu.capabilities.push(WorkerCapability::PoseDetect);
            }
            // YOLO11 person detection + selected-person ByteTrack tracking (sc-5498, epic 5482):
            // the off-Mac sibling of the macOS native-MLX path (sc-3633/sc-3634) — `yolo11m.onnx`
            // via `ort`/CUDA in `person_jobs` + the pure-Rust SORT/ByteTrack in `person_track`,
            // serving `person_detect` (Replace-Person candidate boxes) + `person_track` (the
            // reusable selected-person track). Job-type capabilities (not generation modalities),
            // so they aren't in `registry_capabilities`; advertise them explicitly. Like
            // kps_extract / pose_detect — and unlike SeedVR2 — the Python Ultralytics path CAN
            // serve them, so there's NO torch-refusal gate: the candle worker claims them
            // Python-free, with the co-resident torch worker as a fallback (retired wholesale in
            // Phase 7, epic 5483). Person *segmentation* (SAM masks) is NOT ported off-Mac yet
            // (epic 3792, sc-5062), so candle person tracks are box-only (`maskState = "missing"`).
            for capability in [
                WorkerCapability::PersonDetect,
                WorkerCapability::PersonTrack,
            ] {
                if !gpu.capabilities.contains(&capability) {
                    gpu.capabilities.push(capability);
                }
            }
            // The lane marker the routing gate keys off (mirrors the existing `nvidia` marker).
            gpu.capabilities
                .push(WorkerCapability::Unknown("candle".to_owned()));
        }
    }
    gpu
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

/// Total unified memory in GB (`sysctl hw.memsize`), for per-job memory-budget guards such as the
/// FLUX.2-dev multi-reference edit footprint check (sc-6124). `None` if the probe fails (the guard
/// then no-ops rather than blocking a possibly-fine job). macOS-only — its sole caller, the FLUX.2
/// edit path (`image_jobs/flux2.rs`), is itself `#[cfg(target_os = "macos")]`.
#[cfg(target_os = "macos")]
pub(crate) async fn total_unified_memory_gb() -> Option<f64> {
    sysctl_memsize_mb().await.map(|mb| mb as f64 / 1024.0)
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
/// sc-3060), `video_generate` (+ the `video_extend` / `video_bridge` clip-conditioning
/// modes on the LTX IC-LoRA path, sc-3522), and LoRA/LoKr
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
pub(crate) fn mlx_gpu(settings: &Settings) -> DiscoveredGpu {
    let mut capabilities = vec![WorkerCapability::Gpu];
    // Registry-DERIVED core (sc-3723): exactly the capabilities backed by a linked engine
    // descriptor whose tensor backend is enabled in `settings` — `image_generate`,
    // `video_generate`, `lora_train`, `lora_train_execute`, `training_caption`. Adding a
    // family (or a whole new candle backend) lights these up with no change here; see
    // `engines::registry_capabilities`.
    capabilities.extend(crate::engines::registry_capabilities(settings));
    // Hardcoded CARVE-OUTS: capabilities the worker serves that are NOT expressible as a
    // single registered generator descriptor (bespoke understanding/edit modes, advanced
    // video modes on the LTX IC-LoRA path, and the onnx / native-MLX side stacks). These stay
    // explicit until each grows its own registry surface.
    capabilities.extend([
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
        // SenseNova-U1 understanding + Document Studio (epic 3180, sc-3905): visual
        // question answering (`image_vqa`) and interleaved text-image generation
        // (`image_interleave`) run in-process via the concrete `T2iModel` (the modes the
        // `Generator` contract can't express). The API routes these here only for the
        // SenseNova-U1 ids (`jobs_store::understanding_job_is_mlx_eligible`); off macOS the
        // capabilities are never advertised, so the Python torch worker serves them on
        // Windows/Linux.
        WorkerCapability::ImageVqa,
        WorkerCapability::ImageInterleave,
        // Clip-conditioning advanced video modes (epic 3040, sc-3522): extend_clip /
        // video_bridge run the LTX IC-LoRA keyframe-append path in-process. The API gates
        // these `video_extend` / `video_bridge` jobs to the LTX engines
        // (`jobs_store::video_job_is_mlx_eligible`); a Wan extend/bridge has no IC-LoRA
        // path and stays on the Python torch worker.
        WorkerCapability::VideoExtend,
        WorkerCapability::VideoBridge,
        // replace_person → native Wan-VACE (epic 3040, sc-3521): the `PersonReplace`
        // job builds the masked control inputs (source clip + onnx-track mask + character
        // refs) and runs the engine `wan_vace` provider in-process — the native
        // equivalent of the torch `WanVACEPipeline` path. The API routes only
        // MLX-eligible replace_person jobs here; non-VACE replacement + Windows/Linux
        // keep the Python torch path.
        WorkerCapability::PersonReplace,
        // DWPose whole-body pose detection (epic 3482, sc-3487): RTMW via
        // onnxruntime/CoreML, served in-process by `pose_jobs::run_pose_detect_job`.
        // Replaces the Python rtmlib `pose_detect` path so the Pose Library
        // "create from photo" flow + InstantID pose conditioning keep working on a
        // Python-free Mac. The detector auto-provisions its onnx weights on first
        // use (download-on-first-use parity with rtmlib).
        WorkerCapability::PoseDetect,
        // SCRFD 5-point face-landmark extraction (epic 4422, sc-4433): native-MLX
        // SCRFD in-process (`kps_jobs::run_kps_extract_job`, the InstantID face-stack
        // detector) for the Key Point Library "extract kps from this image" flow,
        // Python-free on Mac. Reuses the InstantID `scrfd_10g` bundle (cached on
        // first InstantID/extraction use).
        WorkerCapability::KpsExtract,
        // Dataset Doctor face pass (epic 6529 P3, sc-6538): the native SCRFD+ArcFace stack
        // (`mlx-gen-face`, the same detector+recognizer the InstantID/kps paths use) embeds the
        // largest face of each Person-dataset image for the identity / subject-prominence checks
        // (`face_analysis_jobs::run_dataset_face_analysis_job`). Hardcoded like KpsExtract — the
        // `FaceEmbedder` stack has no gen-core registry, so it isn't in `registry_capabilities`;
        // advertised here so a `dataset_face_analysis` job routes to the Mac worker by construction.
        WorkerCapability::DatasetFaceAnalysis,
        // On-demand face-likeness compare (epic 4406, sc-4415): scores a CANDIDATE asset against a
        // SOURCE identity reference through the same native SCRFD+ArcFace stack
        // (`face_likeness_compare_jobs::run_face_likeness_compare_job`). Hardcoded like
        // DatasetFaceAnalysis — the `FaceEmbedder` stack has no gen-core registry, so it isn't in
        // `registry_capabilities`; advertised here so the Character Studio Assets compare tool routes
        // to the Mac worker by construction.
        WorkerCapability::FaceLikenessCompare,
        // Real-ESRGAN image upscaling (epic 3482, sc-3489): RRDBNet x2/x4 via
        // onnxruntime/CoreML, served in-process by `upscale_jobs::run_image_upscale_job`.
        // Replaces the Python torch Real-ESRGAN path so the Image Editor upscale tool
        // works on a Python-free Mac. Only `engine=real-esrgan` (the default) is
        // served here; `aura-sr` stays on the Python worker (routing oracle).
        WorkerCapability::ImageUpscale,
        // Dataset Doctor one-tap upscale (sc-6539): reuses the Real-ESRGAN engine to upscale flagged
        // low-resolution training items, then re-points each. Advertised wherever image_upscale is.
        WorkerCapability::DatasetUpscale,
        // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt
        // segmentation, served in-process by `segment_jobs::run_image_segment_job` (the
        // box-PVS path of the sc-4926 SAM3 stack). The Image Editor smart-select tool's
        // backend: a box prompt → a binary inpaint mask asset. Advertised ONLY here (no
        // torch/candle SAM3 image path), so a segment job routes to the Mac worker by
        // construction.
        WorkerCapability::ImageSegment,
        // SeedVR2 video upscaling (epic 4811, sc-4816): native-MLX one-step super-resolution
        // (`mlx-gen-seedvr2`), served in-process by `video_jobs::run_video_upscale_job` —
        // SceneWorks' first video upscaler. Decodes the source clip, runs the temporal-chunked
        // 5D upscale, re-encodes, and passes the source audio through. Mac-only (no torch path).
        WorkerCapability::VideoUpscale,
        // Real, model-backed person detection + tracking (epic 3482, sc-3488 /
        // sc-3633/3634/3709): the native-MLX YOLO11 detector + SORT/ByteTrack tracker +
        // SAM2 segmenter run in-process (`person_jobs` / `person_track` /
        // `person_segment`), replacing the Python onnxruntime/torch path so the
        // Replace-Person detect → track → mask flow works on a Python-free Mac. A
        // `preview: true` job still routes to the CPU placeholder's procedural preview
        // capability (`required_capability` in jobs_store).
        WorkerCapability::PersonDetect,
        WorkerCapability::PersonTrack,
    ]);
    DiscoveredGpu {
        id: "mlx".to_owned(),
        name: "Apple Silicon (MLX)".to_owned(),
        capabilities,
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
            // Explicit built-in LoRA download (sc-5944): fetches the catalog LoRA's HF
            // repo/file into the shared HF cache, same as model_download.
            WorkerCapability::LoraDownload,
            // Procedural detection/tracking is a preview only. Real, model-backed
            // PersonDetect/PersonTrack run on the macOS MLX worker (native
            // YOLO11/SAM2, epic 3482) or the Python GPU worker on Windows/Linux;
            // advertising the preview capabilities here keeps the CPU placeholder
            // claimable solely for explicit `preview: true` jobs
            // (jobs_store::worker_supports_job).
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
