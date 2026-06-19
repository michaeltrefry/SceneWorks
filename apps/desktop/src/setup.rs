//! First-run Python venv bootstrap + startup orchestration (sc-1348).
//!
//! The frontend setup screen calls the `start_setup` command once it is ready to
//! receive events; this provisions the uv-managed venv (streaming progress),
//! then spawns the API sidecar, health-gates it, and navigates the window to the
//! local API. `start_setup` is also the retry entry point.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sceneworks_core::session_log::{LogEntry, LogQuery, SessionLog};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Process-global in-app session log (sc-3451). Every captured sidecar line
/// (api/worker/mlx-worker) is mirrored here as it's appended to disk, so the
/// `get_session_logs` command can serve the current session's activity — the
/// MLX routing decisions, claim contention and worker phases — without parsing
/// the append-only files in `~/Library/Logs/SceneWorks/`. "Current session" =
/// this desktop process's lifetime (the buffer is created on first capture).
static SESSION_LOG: OnceLock<SessionLog> = OnceLock::new();

pub fn session_log() -> &'static SessionLog {
    SESSION_LOG.get_or_init(SessionLog::default)
}

/// Read back the current session's log entries for the in-app Logs screen
/// (sc-3452). `after_seq` tails only new lines; the rest are filters.
#[tauri::command]
pub fn get_session_logs(
    after_seq: Option<u64>,
    limit: Option<usize>,
    source: Option<String>,
    level: Option<String>,
    search: Option<String>,
) -> Vec<LogEntry> {
    session_log().query(&LogQuery {
        after_seq,
        limit,
        source,
        level,
        search,
    })
}

const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Process handles + run guards shared across the app.
#[derive(Default)]
pub struct Managed {
    pub api: Mutex<Option<CommandChild>>,
    /// The Apple-Silicon MLX GPU worker (sc-3289): the same `sceneworks-api`
    /// binary re-launched in worker mode (`SCENEWORKS_WORKER_ONLY=1`,
    /// `SCENEWORKS_GPU_ID=mlx`). Only populated on macOS.
    pub mlx_worker: Mutex<Option<CommandChild>>,
    /// The Windows candle (CUDA) GPU worker (sc-5561): the same `sceneworks-api`
    /// binary re-launched in worker mode (`SCENEWORKS_WORKER_ONLY=1`,
    /// `SCENEWORKS_GPU_ID=0`, `SCENEWORKS_BACKEND_CANDLE_ENABLED=true`). Runs
    /// alongside the Python torch worker (Wave A). Only populated on the Windows
    /// candle build.
    pub candle_worker: Mutex<Option<CommandChild>>,
    /// On-demand keychain credential socket served to the MLX worker (sc-5891).
    /// Started once before the worker spawns; the worker pulls a host's secret from
    /// it the first time a download needs auth, so the keychain is read lazily
    /// instead of eagerly at launch. macOS-only.
    #[cfg(target_os = "macos")]
    pub cred_ipc: Mutex<Option<crate::cred_ipc::CredIpc>>,
    /// OS-assigned API port, discovered from the sidecar's startup line.
    api_port: Mutex<Option<u16>>,
    /// PIDs of the spawned sidecars, persisted to disk so an unclean exit
    /// (crash/force-quit) doesn't leave them orphaned — the next launch reaps
    /// any survivors before spawning fresh ones.
    pids: Mutex<SidecarPids>,
    running: AtomicBool,
    pub shutting_down: AtomicBool,
}

/// PIDs of the API + Python worker + MLX worker sidecars owned by this launch.
#[derive(Default, Clone, Serialize, Deserialize)]
struct SidecarPids {
    api: Option<u32>,
    /// The MLX GPU worker (sc-3289). `#[serde(default)]` so a pidfile written by
    /// an older build (no such field) still deserializes for reaping.
    #[serde(default)]
    mlx_worker: Option<u32>,
    /// The Windows candle GPU worker (sc-5561). `#[serde(default)]` so an older
    /// pidfile (no such field) still deserializes for reaping.
    #[serde(default)]
    candle_worker: Option<u32>,
}

#[derive(Clone, Serialize)]
struct SetupStatus {
    phase: String,
    message: String,
    error: bool,
}

pub(crate) fn emit(app: &AppHandle, phase: &str, message: impl Into<String>, error: bool) {
    let _ = app.emit(
        "setup-status",
        SetupStatus {
            phase: phase.to_owned(),
            message: message.into(),
            error,
        },
    );
}

/// Per-OS application support root: `~/Library/Application Support/SceneWorks`
/// (macOS), `%APPDATA%\SceneWorks` (Windows), `$XDG_DATA_HOME/sceneworks` or
/// `~/.local/share/sceneworks` (Linux). Mirrors the API's path resolver.
pub fn app_support_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("SceneWorks");
    }
    #[cfg(target_os = "windows")]
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata).join("SceneWorks");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Ok(data) = std::env::var("XDG_DATA_HOME") {
            return PathBuf::from(data).join("sceneworks");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("sceneworks");
        }
    }
    std::env::temp_dir().join("SceneWorks")
}

/// Platform-appropriate logs directory (also used for the API/worker logs).
pub fn logs_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Logs")
            .join("SceneWorks");
    }
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("SceneWorks").join("logs");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Ok(state) = std::env::var("XDG_STATE_HOME") {
            return PathBuf::from(state).join("sceneworks").join("logs");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("state")
                .join("sceneworks")
                .join("logs");
        }
    }
    std::env::temp_dir().join("SceneWorks").join("logs")
}

/// Platform default workspace data directory, used when the user hasn't picked a
/// custom location in the first-run splash / Settings.
pub fn default_data_dir() -> PathBuf {
    app_support_dir().join("data")
}

fn config_dir() -> PathBuf {
    app_support_dir().join("config")
}

/// Shared per-user Hugging Face cache (`~/.cache/huggingface`) — the default
/// `HF_HOME` when the user hasn't chosen a custom location. Dedups with other
/// HF-based tools on the machine and reuses anything already downloaded.
pub fn shared_huggingface_home() -> PathBuf {
    if let Some(home) = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return PathBuf::from(home).join(".cache").join("huggingface");
    }
    app_support_dir().join("cache").join("huggingface")
}

/// Hugging Face cache home injected into both sidecars so the rust-api model
/// catalog and the Python inference worker resolve weights from the same root.
/// Without this the API falls back to `<data_dir>/cache/huggingface` while the
/// worker uses huggingface_hub's default `~/.cache/huggingface`, so they
/// disagree and every catalog entry shows "missing" (sc-1473 Step 1 gap).
/// Resolution order: an explicit `HF_HOME` in the environment (useful under
/// `tauri dev`), then the user's persisted choice from the first-run splash, then
/// the shared per-user cache. Because the splash persists this *before* the
/// sidecars spawn, the chosen location takes effect with no app restart.
fn huggingface_home() -> PathBuf {
    if let Ok(value) = std::env::var("HF_HOME") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Some(dir) = crate::settings::load_settings()
        .hf_home
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(dir);
    }
    shared_huggingface_home()
}

/// Seed the builtin model/LoRA/recipe-preset catalogs into the desktop's
/// `config_dir/manifests`, overwriting on every launch so they track the app
/// version. The server stack ships these in the repo's `config/`, but the desktop
/// must provide them itself or Model Manager is empty and the native LTX/Wan
/// adapters can't map model resources to files. User customizations live in the
/// separate `user.*.jsonc` files, which seeding never touches. Delegates to the
/// shared `sceneworks_core` seeder (same embedded copies the rust-api uses);
/// returns an error if any required manifest can't be installed so the caller
/// aborts setup rather than starting with missing model mappings.
fn seed_builtin_manifests() -> Result<(), String> {
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &config_dir(),
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .map_err(|error| error.to_string())
}

/// Data directory: the settings override if set, otherwise the platform default.
fn resolved_data_dir() -> PathBuf {
    crate::settings::load_settings()
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir)
}

fn append_log(path: &Path, line: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
    // Mirror into the in-app session buffer (sc-3451), tagged by the log's file stem
    // ("worker.log" -> "worker", "mlx-worker.log" -> "mlx-worker").
    let source = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("app");
    session_log().push_line(source, line);
}

/// Extract the port from the API's `listening on http://127.0.0.1:PORT` line.
fn parse_listening_port(line: &str) -> Option<u16> {
    const MARKER: &str = "127.0.0.1:";
    let start = line.find(MARKER)? + MARKER.len();
    line[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Health check that also confirms the responder is genuinely the SceneWorks API
/// (HTTP 200 with the expected service/runtime in the JSON body) before we
/// navigate the privileged Tauri window to it — a foreign service that grabbed
/// the port must not be trusted.
fn health_is_sceneworks(port: u16) -> bool {
    use std::io::Read;
    use std::net::TcpStream;
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /api/v1/health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    let ok_status = response
        .lines()
        .next()
        .is_some_and(|status_line| status_line.contains(" 200"));
    ok_status
        && response.contains("\"service\":\"sceneworks-api\"")
        && response.contains("\"runtime\":\"rust\"")
}

/// Resolve the ffmpeg binary the Rust worker shells out to (frame sampling,
/// frame extract, timeline export, video-gen audio mux — all via
/// `media_jobs::run_ffmpeg`, which honors `SCENEWORKS_FFMPEG`). The desktop ships
/// no system ffmpeg, so without this those jobs fail. Prefers the static ffmpeg
/// bundled next to the app (staged by build-sidecar.mjs into the `ffmpeg` resource
/// dir, so a packaged Python-free Mac still works — epic 3482, sc-3767). Returns
/// None when it isn't bundled (pre-bundle / dev — caller then leaves
/// `SCENEWORKS_FFMPEG` unset → PATH ffmpeg).
fn resolve_bundled_ffmpeg(app: &AppHandle) -> Option<String> {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("ffmpeg").join(if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        });
        if bundled.exists() {
            return Some(bundled.to_string_lossy().to_string());
        }
    }
    None
}

/// Resolve the onnxruntime dynamic library the Rust worker's DWPose pose detector
/// (`ort`, sc-3487) dlopens at runtime via `ORT_DYLIB_PATH` (the `load-dynamic`
/// feature). Prefers the dylib bundled next to the app (staged by build-sidecar.mjs
/// into the `onnxruntime` resource dir, so a packaged Python-free Mac still detects
/// poses), the same CoreML-enabled build. Returns None when it isn't bundled
/// (pre-bundle / dev). macOS-only — pose detection on the Rust worker is macOS-only,
/// so this returns None elsewhere.
#[cfg(target_os = "macos")]
fn resolve_bundled_onnxruntime(app: &AppHandle) -> Option<String> {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("onnxruntime").join("libonnxruntime.dylib");
        if bundled.exists() {
            return Some(bundled.to_string_lossy().to_string());
        }
    }
    None
}

/// Resolve the CUDA-enabled onnxruntime DLL the candle worker's `ort` paths (DWPose
/// pose_detect sc-5496, + YOLO/Real-ESRGAN, epic 5482) dlopen at runtime via
/// `ORT_DYLIB_PATH` (the `load-dynamic` feature). The Windows/CUDA analogue of the
/// macOS CoreML resolver above. The onnxruntime-gpu DLLs are no longer bundled (the
/// ~2.7 GB GPU runtime blew past NSIS's datablock limit); they're downloaded on first
/// run into `%APPDATA%\SceneWorks\gpu-runtime\onnxruntime` (cuda_provision.rs) and
/// resolved from there. Returns None until that first-run provisioning completes — the
/// non-candle / dev path never reaches the candle worker that consumes it. Windows-only
/// (the candle GPU worker is Windows-gated here).
#[cfg(target_os = "windows")]
fn resolve_bundled_onnxruntime(_app: &AppHandle) -> Option<String> {
    crate::cuda_provision::onnxruntime_dll_if_present().map(|dll| dll.to_string_lossy().to_string())
}

/// Resolve the CUDA runtime redistributable DLL directory (sc-5560). The candle
/// (Windows/CUDA) worker links cudarc with dynamic-linking, which `LoadLibrary`s
/// cudart/cublas/cublasLt/curand/nvrtc by name at runtime. These DLLs are no longer
/// bundled (the ~2.7 GB GPU runtime exceeded NSIS's ~2 GB datablock limit); they're
/// downloaded on first run into `%APPDATA%\SceneWorks\gpu-runtime\cuda`
/// (cuda_provision.rs) and resolved from there. `spawn_api` /
/// `supervise_candle_worker` prepend this dir to the sidecar's PATH so the loader
/// finds them. Returns None until first-run provisioning has written the DLLs (probes
/// `cudart64_12.dll`); this also gates the candle worker spawn + cuda_preflight, so a
/// pre-provision / failed-provision state leaves the candle lane dormant rather than
/// spawning a worker whose CUDA load would fail. Windows-only (candle is Windows-
/// gated); the driver-side CUDA (nvcuda.dll) is NOT downloaded — it comes with the
/// user's NVIDIA display driver (>= 576.02).
#[cfg(target_os = "windows")]
fn resolve_bundled_cuda_dir(_app: &AppHandle) -> Option<std::path::PathBuf> {
    crate::cuda_provision::cuda_dir_if_present()
}

/// Spawn the API sidecar, pipe its output to api.log, and return the chosen port.
fn spawn_api(app: &AppHandle) -> Result<(), String> {
    let mut command = app
        .shell()
        .sidecar("sceneworks-api")
        .map_err(|error| format!("locate api: {error}"))?
        .env("SCENEWORKS_API_HOST", "127.0.0.1")
        // Let the OS assign a free port (no reserve/release TOCTOU); the actual
        // port is discovered from the API's startup line below.
        .env("SCENEWORKS_API_PORT", "0")
        .env("SCENEWORKS_RUN_UTILITY_INPROCESS", "true")
        // Parent-death watchdog: a force-quit/crash skips `begin_shutdown`, so
        // without this the API orphans to launchd (PPID=1), holding its
        // OS-assigned port + a jobs.db handle. The API self-terminates when this
        // PID disappears; unset (server/Docker) -> the watchdog never fires.
        .env("SCENEWORKS_PARENT_PID", std::process::id().to_string())
        .env(
            "SCENEWORKS_DATA_DIR",
            resolved_data_dir().to_string_lossy().to_string(),
        )
        // Pin the config dir so the API and Python worker share one root on all
        // platforms (Linux otherwise splits XDG data vs config).
        .env(
            "SCENEWORKS_CONFIG_DIR",
            config_dir().to_string_lossy().to_string(),
        )
        // The catalog's install-state detection resolves the HF cache from this;
        // it must match the worker's download root or every model reads "missing".
        .env("HF_HOME", huggingface_home().to_string_lossy().to_string());
    // Epic 3482 (Python Eradication) final cutover (sc-3492) — macOS runs MLX-only.
    // `Settings.mlx_required` ← `SCENEWORKS_MLX_REQUIRED` (sc-3483): the MPS/torch worker
    // never claims an MLX-eligible job, and an MLX-eligible job that no live `mlx` worker
    // takes fails `mlx_unavailable` instead of falling back to MPS. Every Mac Python
    // *inference* surface is now ported to the in-process Rust/MLX worker or UI-gated
    // (sc-3486), and the Python torch worker is no longer spawned on macOS (see
    // `gate_window`), so the flag is enforced here.
    #[cfg(target_os = "macos")]
    {
        command = command.env("SCENEWORKS_MLX_REQUIRED", "1");
    }
    // Off-Mac (epic 5483 Phase 7, sc-5563): candle is the ONLY backend on the desktop —
    // the Python torch worker is no longer spawned (see `gate_window`) and no venv is
    // bundled or bootstrapped. Mirror the Mac MLX-required flip: require candle so a
    // candle-eligible job stranded with no live candle worker fails `candle_unavailable`,
    // and enforce so a shape candle can't serve fails `candle_unsupported` — never a silent
    // torch fallback (there is no torch worker left to fall back to). The candle sweeps are
    // biased to Ok, so only the true generation gaps fail; the CV-aux / segment / training
    // surfaces stay served by the candle worker.
    #[cfg(not(target_os = "macos"))]
    {
        command = command
            .env("SCENEWORKS_CANDLE_REQUIRED", "1")
            .env("SCENEWORKS_CANDLE_UNSUPPORTED_MODE", "enforce");
    }
    // The in-process utility worker shells out to ffmpeg; point it at the bundled
    // static binary (sc-3767) since the desktop has no system ffmpeg on PATH.
    if let Some(ffmpeg) = resolve_bundled_ffmpeg(app) {
        command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
    }
    // DWPose pose detection (sc-3487) loads onnxruntime dynamically; point `ort` at
    // the bundled CoreML-enabled dylib so a packaged Python-free Mac can detect poses.
    #[cfg(target_os = "macos")]
    if let Some(ort_dylib) = resolve_bundled_onnxruntime(app) {
        command = command.env("ORT_DYLIB_PATH", ort_dylib);
    }
    // The candle (Windows/CUDA) worker's cudarc dynamic-linking `LoadLibrary`s the
    // CUDA runtime DLLs by name; prepend the bundled redist dir to the sidecar's
    // PATH so they resolve without a CUDA Toolkit on the machine (sc-5560). No-op on
    // a plain build — the resolver returns None when only the placeholder is staged.
    #[cfg(target_os = "windows")]
    if let Some(cuda_dir) = resolve_bundled_cuda_dir(app) {
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![cuda_dir];
        paths.extend(std::env::split_paths(&existing));
        if let Ok(joined) = std::env::join_paths(paths) {
            command = command.env("PATH", joined);
        }
    }
    // FLUX.2-klein true_v2 single-file conversion is now in-process Rust/MLX
    // (mlx_gen_flux2::convert_and_assemble, sc-3136) — no sidecar venv / converter
    // script, so no SCENEWORKS_MLX_FLUX_* env wiring.
    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("spawn api: {error}"))?;
    record_api_pid(app, child.pid());
    app.state::<Managed>()
        .api
        .lock()
        .expect("api lock")
        .replace(child);

    let log_path = logs_dir().join("api.log");
    let _ = std::fs::create_dir_all(logs_dir());
    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .ok();
        while let Some(event) = events.recv().await {
            let entry = match event {
                CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    // Discover the OS-assigned port from the API's startup line.
                    let state = app_handle.state::<Managed>();
                    let mut port = state.api_port.lock().expect("api port lock");
                    if port.is_none() {
                        if let Some(found) = parse_listening_port(&text) {
                            *port = Some(found);
                        }
                    }
                    text
                }
                CommandEvent::Terminated(payload) => format!(
                    "[desktop] api sidecar terminated: code={:?} signal={:?}\n",
                    payload.code, payload.signal
                ),
                CommandEvent::Error(error) => format!("[desktop] api sidecar error: {error}\n"),
                _ => continue,
            };
            if let Some(file) = file.as_mut() {
                let _ = file.write_all(entry.as_bytes());
                let _ = file.flush();
            }
            // Mirror the API sidecar's output into the in-app session buffer (sc-3451);
            // this loop writes its own file handle so it doesn't go through append_log.
            session_log().push_line("api", &entry);
        }
    });
    Ok(())
}

/// Health-gate the window on a background thread: wait for the API's
/// OS-assigned port, confirm the responder is genuinely SceneWorks, then
/// navigate and start the platform inference worker(s) — the MLX GPU worker on
/// macOS (MLX-only, sc-3492), the Python torch worker elsewhere; show an error
/// after the timeout.
fn gate_window(app: AppHandle) {
    std::thread::spawn(move || {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        loop {
            let port = *app
                .state::<Managed>()
                .api_port
                .lock()
                .expect("api port lock");
            if let Some(port) = port {
                if health_is_sceneworks(port) {
                    if let Ok(url) = format!("http://127.0.0.1:{port}").parse() {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.navigate(url);
                        }
                    }
                    #[cfg(target_os = "macos")]
                    {
                        // Epic 3482 final cutover (sc-3492): macOS is MLX-only — the
                        // Python torch/MPS worker is no longer spawned. Only the
                        // Apple-Silicon MLX GPU worker (sc-3289) runs, executing
                        // MLX-eligible image/video jobs on the in-process Rust mlx-gen
                        // engine. Any MLX-ineligible job fails `mlx_unsupported` /
                        // `mlx_unavailable` (never MPS) per `Settings.mlx_required`.
                        //
                        // Start the on-demand credential socket first (sc-5891) so the
                        // worker can pull a recorded keychain secret lazily at download
                        // time instead of us reading it eagerly here at launch.
                        ensure_cred_ipc(&app);
                        supervise_mlx_worker(app, port);
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        // Epic 5483 Phase 7 (sc-5563): off-Mac is candle-only — the Python
                        // torch worker is no longer spawned (its venv + bundle were dropped),
                        // exactly as macOS went MLX-only in sc-3492. The Windows candle (CUDA)
                        // GPU worker runs the candle-eligible surface; anything candle can't
                        // serve fails loudly (candle_unsupported / candle_unavailable) per
                        // Settings.candle_required (set in spawn_api), never a silent torch
                        // fallback. Spawned only when the candle backend is actually bundled
                        // (a plain build has no CUDA DLLs); without it there is no GPU worker
                        // and GPU jobs fail loudly rather than silently degrading. (Linux
                        // desktop is not a shipped target — the Linux server runs via Docker.)
                        #[cfg(target_os = "windows")]
                        if resolve_bundled_cuda_dir(&app).is_some() {
                            supervise_candle_worker(app, port);
                        }
                    }
                    return;
                }
            }
            if Instant::now() >= deadline {
                emit(&app, "error", "The local API did not start in time.", true);
                return;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    });
}

/// Start the on-demand credential socket (sc-5891) once and stash it in `Managed`.
/// The MLX worker is handed its socket path + token at spawn and pulls a recorded
/// keychain secret from it the first time a download needs auth — so the keychain is
/// read lazily, not eagerly at launch. Idempotent; a start failure is logged and the
/// worker simply gets no credentials (a gated download then fails with an auth error
/// rather than the app prompting at launch).
#[cfg(target_os = "macos")]
fn ensure_cred_ipc(app: &AppHandle) {
    let managed = app.state::<Managed>();
    let mut slot = managed.cred_ipc.lock().expect("cred_ipc lock");
    if slot.is_some() {
        return;
    }
    let socket = app_support_dir().join("cred-ipc.sock");
    match crate::cred_ipc::start(socket) {
        Some(handle) => *slot = Some(handle),
        None => append_log(
            &logs_dir().join("mlx-worker.log"),
            "[desktop] credential socket failed to start; gated downloads will need a re-entered token\n",
        ),
    }
}

/// Drop a host's cached secret from the credential socket (sc-5891) so a later pull
/// re-reads the keychain. Called when the user updates or removes a credential, so a
/// revoked/changed token stops being served without an app restart. No-op off macOS
/// (no socket there).
pub fn invalidate_credential_cache(app: &AppHandle, host: &str) {
    #[cfg(target_os = "macos")]
    {
        if let Some(ipc) = app
            .state::<Managed>()
            .cred_ipc
            .lock()
            .expect("cred_ipc lock")
            .as_ref()
        {
            ipc.invalidate(host);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, host);
    }
}

/// Spawn and supervise the Apple-Silicon MLX GPU worker (sc-3289): the same
/// `sceneworks-api` sidecar binary re-launched in worker mode
/// (`SCENEWORKS_WORKER_ONLY=1`) with `SCENEWORKS_GPU_ID=mlx`, so MLX-eligible
/// image/video jobs run on the in-process Rust mlx-gen engine instead of the
/// Python torch/MPS path. A crash-isolated sibling of the API process; restarted
/// with exponential backoff while the app is open. Output goes to mlx-worker.log.
///
/// Without this worker registered, `jobs_store::should_defer_image_to_mlx_worker`
/// has nowhere to defer and the Python `mps` worker is the fallback — which is
/// why image/video jobs reported MPS before this landed.
#[cfg(target_os = "macos")]
fn supervise_mlx_worker(app: AppHandle, api_port: u16) {
    std::thread::spawn(move || {
        let log_path = logs_dir().join("mlx-worker.log");
        let api_url = format!("http://127.0.0.1:{api_port}");
        // Match the API sidecar's HF cache root so the engine reads the same
        // downloaded weights the catalog tracks.
        let hf_home = huggingface_home().to_string_lossy().to_string();
        // Unique per launch (distinct prefix from the Python `worker-local-*` and
        // the in-process `rust-utility-worker`) so the three workers never collide
        // in the shared jobs.db.
        let worker_id = format!(
            "mlx-worker-local-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or_default()
        );
        let mut backoff = 1u64;
        loop {
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            let sidecar = match app.shell().sidecar("sceneworks-api") {
                Ok(command) => command,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] mlx worker: locate sidecar failed: {error}\n"),
                    );
                    return;
                }
            };
            let mut command = sidecar
                // Dispatches `main` to `run_worker()` (HTTP API never starts).
                .env("SCENEWORKS_WORKER_ONLY", "1")
                .env("SCENEWORKS_GPU_ID", "mlx")
                .env("SCENEWORKS_WORKER_ID", &worker_id)
                .env("SCENEWORKS_API_URL", &api_url)
                .env("HF_HOME", &hf_home)
                // Parent-death watchdog (run_worker() honours this): a force-quit
                // self-terminates the worker so its multi-GB MLX model isn't
                // orphaned to launchd.
                .env("SCENEWORKS_PARENT_PID", std::process::id().to_string())
                .env(
                    "SCENEWORKS_DATA_DIR",
                    resolved_data_dir().to_string_lossy().to_string(),
                )
                .env(
                    "SCENEWORKS_CONFIG_DIR",
                    config_dir().to_string_lossy().to_string(),
                );
            // The worker muxes generated video with ffmpeg; the desktop ships no
            // system ffmpeg, so point it at the bundled binary (as spawn_api does).
            if let Some(ffmpeg) = resolve_bundled_ffmpeg(&app) {
                command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
            }
            // This is the worker that advertises `pose_detect` (epic 3482, sc-3487);
            // point `ort` at the bundled CoreML onnxruntime dylib it dlopens.
            if let Some(ort_dylib) = resolve_bundled_onnxruntime(&app) {
                command = command.env("ORT_DYLIB_PATH", ort_dylib);
            }
            // Lazy credentials (sc-5891): instead of reading the keychain here and
            // injecting HF_TOKEN/SCENEWORKS_CREDENTIALS (which prompted at launch),
            // hand the worker the credential socket + token + the NON-secret list of
            // recorded hosts. The worker pulls a secret only when a download for a
            // recorded host needs it, so nothing-recorded ⇒ no socket call ⇒ no
            // keychain touch. Credential changes still take effect on worker restart.
            {
                let managed = app.state::<Managed>();
                let guard = managed.cred_ipc.lock().expect("cred_ipc lock");
                if let Some(ipc) = guard.as_ref() {
                    command = command
                        .env(
                            "SCENEWORKS_CRED_IPC_SOCKET",
                            ipc.socket.to_string_lossy().to_string(),
                        )
                        .env("SCENEWORKS_CRED_IPC_TOKEN", &ipc.token);
                    let hosts = crate::settings::recorded_credential_hosts().join(",");
                    if !hosts.is_empty() {
                        command = command.env("SCENEWORKS_CREDENTIAL_HOSTS", hosts);
                    }
                }
            }
            let spawned = command.spawn();
            let (mut events, child) = match spawned {
                Ok(pair) => pair,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] mlx worker spawn failed: {error}\n"),
                    );
                    std::thread::sleep(Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(30);
                    continue;
                }
            };
            record_mlx_worker_pid(&app, Some(child.pid()));
            app.state::<Managed>()
                .mlx_worker
                .lock()
                .expect("mlx worker lock")
                .replace(child);
            let started = Instant::now();
            loop {
                match tauri::async_runtime::block_on(events.recv()) {
                    Some(CommandEvent::Stdout(bytes)) | Some(CommandEvent::Stderr(bytes)) => {
                        append_log(&log_path, &String::from_utf8_lossy(&bytes));
                    }
                    Some(CommandEvent::Terminated(payload)) => {
                        append_log(
                            &log_path,
                            &format!(
                                "[desktop] mlx worker terminated: code={:?} signal={:?}\n",
                                payload.code, payload.signal
                            ),
                        );
                        break;
                    }
                    Some(CommandEvent::Error(error)) => {
                        append_log(&log_path, &format!("[desktop] mlx worker error: {error}\n"));
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
            let _ = app
                .state::<Managed>()
                .mlx_worker
                .lock()
                .expect("mlx worker lock")
                .take();
            record_mlx_worker_pid(&app, None);
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            if started.elapsed() > Duration::from_secs(20) {
                backoff = 1;
            }
            append_log(
                &log_path,
                &format!("[desktop] restarting mlx worker in {backoff}s\n"),
            );
            std::thread::sleep(Duration::from_secs(backoff));
            backoff = (backoff * 2).min(30);
        }
    });
}

/// Spawn and supervise the Windows candle (CUDA) GPU worker (sc-5561): the same
/// `sceneworks-api` sidecar re-launched in worker mode (`SCENEWORKS_WORKER_ONLY=1`)
/// with `SCENEWORKS_GPU_ID=0` (the first NVIDIA GPU) and the candle backend enabled
/// (`SCENEWORKS_BACKEND_CANDLE_ENABLED=true`), so candle-eligible image/video/caption
/// jobs run on the in-process candle gen-core engines instead of the Python torch
/// path. A crash-isolated sibling of the API process; restarted with exponential
/// backoff while the app is open. Output goes to candle-worker.log.
///
/// The Windows analogue of `supervise_mlx_worker`. Runs ALONGSIDE the Python torch
/// worker (Wave A): the existing `jobs_store` claim/defer routing
/// (`torch_worker_claims_everything_the_candle_worker_defers`) confines this worker
/// to the candle lane and keeps everything else on Python. Only spawned when the
/// candle redist DLLs are actually bundled (`resolve_bundled_cuda_dir`), so a plain
/// desktop build never starts it.
#[cfg(target_os = "windows")]
fn supervise_candle_worker(app: AppHandle, api_port: u16) {
    std::thread::spawn(move || {
        let log_path = logs_dir().join("candle-worker.log");
        let api_url = format!("http://127.0.0.1:{api_port}");
        // Match the API sidecar's HF cache root so the engine reads the same
        // downloaded weights the catalog tracks.
        let hf_home = huggingface_home().to_string_lossy().to_string();
        // Unique per launch (distinct prefix from the Python `worker-local-*`, the
        // macOS `mlx-worker-local-*`, and the in-process utility worker) so the
        // workers never collide in the shared jobs.db.
        let worker_id = format!(
            "candle-worker-local-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or_default()
        );
        let mut backoff = 1u64;
        loop {
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            let sidecar = match app.shell().sidecar("sceneworks-api") {
                Ok(command) => command,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] candle worker: locate sidecar failed: {error}\n"),
                    );
                    return;
                }
            };
            let mut command = sidecar
                // Dispatches `main` to `run_worker()` (HTTP API never starts).
                .env("SCENEWORKS_WORKER_ONLY", "1")
                // The first NVIDIA GPU (nvidia-smi index 0); a bare index runs the
                // single-GPU `run_worker_loop` (not the `auto` multi-GPU supervisor).
                .env("SCENEWORKS_GPU_ID", "0")
                // Light up the candle lane on the discovered NVIDIA GPU
                // (gpu::with_candle_capabilities + engines::registry_capabilities).
                .env("SCENEWORKS_BACKEND_CANDLE_ENABLED", "true")
                .env("SCENEWORKS_WORKER_ID", &worker_id)
                .env("SCENEWORKS_API_URL", &api_url)
                .env("HF_HOME", &hf_home)
                // Parent-death watchdog (run_worker() honours this): a force-quit
                // self-terminates the worker so its multi-GB model + CUDA context
                // isn't orphaned.
                .env("SCENEWORKS_PARENT_PID", std::process::id().to_string())
                .env(
                    "SCENEWORKS_DATA_DIR",
                    resolved_data_dir().to_string_lossy().to_string(),
                )
                .env(
                    "SCENEWORKS_CONFIG_DIR",
                    config_dir().to_string_lossy().to_string(),
                );
            // cudarc dynamic-linking `LoadLibrary`s the CUDA runtime DLLs by name;
            // prepend the bundled redist dir to this worker's PATH so they resolve
            // without a CUDA Toolkit on the machine (sc-5560).
            if let Some(cuda_dir) = resolve_bundled_cuda_dir(&app) {
                let existing = std::env::var_os("PATH").unwrap_or_default();
                let mut paths = vec![cuda_dir.clone()];
                paths.extend(std::env::split_paths(&existing));
                if let Ok(joined) = std::env::join_paths(paths) {
                    command = command.env("PATH", joined);
                }
                // The candle worker's `ort` (onnxruntime) paths — DWPose pose_detect
                // (sc-5496), then YOLO / Real-ESRGAN (sc-5498/5499, epic 5482) — point
                // `ort` at the bundled CUDA-enabled onnxruntime and tell the worker where
                // the CUDA-12 runtime + cuDNN-9 DLLs live, so its CUDA execution provider
                // engages instead of falling back to CPU. The off-Mac analogue of the
                // macOS CoreML `ORT_DYLIB_PATH` wiring. The `cuda` resource dir holds the
                // version-matched CUDA-12 runtime + cuDNN-9 (staged by build-sidecar.mjs);
                // `ort_cuda::preload_cuda_dylibs` preloads them + puts the dir on the
                // loader search path so cuDNN's lazily-loaded sub-engine DLLs resolve.
                if let Some(ort_dylib) = resolve_bundled_onnxruntime(&app) {
                    let cuda = cuda_dir.to_string_lossy().to_string();
                    command = command
                        .env("ORT_DYLIB_PATH", ort_dylib)
                        .env("SCENEWORKS_ORT_CUDA_DIR", &cuda)
                        .env("SCENEWORKS_ORT_CUDNN_DIR", &cuda);
                }
            }
            // The worker muxes generated video with ffmpeg; point it at the bundled
            // binary when staged (else it falls back to PATH ffmpeg), as spawn_api does.
            if let Some(ffmpeg) = resolve_bundled_ffmpeg(&app) {
                command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
            }
            if let Some(token) = crate::settings::read_hf_token() {
                command = command.env("HF_TOKEN", token);
            }
            if let Some(credentials) = crate::settings::credentials_env_json() {
                command = command.env("SCENEWORKS_CREDENTIALS", credentials);
            }
            let spawned = command.spawn();
            let (mut events, child) = match spawned {
                Ok(pair) => pair,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] candle worker spawn failed: {error}\n"),
                    );
                    std::thread::sleep(Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(30);
                    continue;
                }
            };
            record_candle_worker_pid(&app, Some(child.pid()));
            app.state::<Managed>()
                .candle_worker
                .lock()
                .expect("candle worker lock")
                .replace(child);
            let started = Instant::now();
            loop {
                match tauri::async_runtime::block_on(events.recv()) {
                    Some(CommandEvent::Stdout(bytes)) | Some(CommandEvent::Stderr(bytes)) => {
                        append_log(&log_path, &String::from_utf8_lossy(&bytes));
                    }
                    Some(CommandEvent::Terminated(payload)) => {
                        append_log(
                            &log_path,
                            &format!(
                                "[desktop] candle worker terminated: code={:?} signal={:?}\n",
                                payload.code, payload.signal
                            ),
                        );
                        break;
                    }
                    Some(CommandEvent::Error(error)) => {
                        append_log(
                            &log_path,
                            &format!("[desktop] candle worker error: {error}\n"),
                        );
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
            let _ = app
                .state::<Managed>()
                .candle_worker
                .lock()
                .expect("candle worker lock")
                .take();
            record_candle_worker_pid(&app, None);
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            if started.elapsed() > Duration::from_secs(20) {
                backoff = 1;
            }
            append_log(
                &log_path,
                &format!("[desktop] restarting candle worker in {backoff}s\n"),
            );
            std::thread::sleep(Duration::from_secs(backoff));
            backoff = (backoff * 2).min(30);
        }
    });
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
}

/// File holding this launch's sidecar PIDs, used to reap orphans left by a prior
/// unclean exit. Lives alongside the app's data so it survives across launches.
fn sidecar_pidfile() -> PathBuf {
    app_support_dir().join("desktop-sidecars.json")
}

/// Persist the current sidecar PIDs (best effort, atomic via temp+rename).
fn write_sidecar_pidfile(pids: &SidecarPids) {
    let path = sidecar_pidfile();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(json) = serde_json::to_vec_pretty(pids) else {
        return;
    };
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, &json).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

fn record_api_pid(app: &AppHandle, pid: u32) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.api = Some(pid);
    write_sidecar_pidfile(&pids);
}

#[cfg(target_os = "macos")]
fn record_mlx_worker_pid(app: &AppHandle, pid: Option<u32>) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.mlx_worker = pid;
    write_sidecar_pidfile(&pids);
}

#[cfg(target_os = "windows")]
fn record_candle_worker_pid(app: &AppHandle, pid: Option<u32>) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.candle_worker = pid;
    write_sidecar_pidfile(&pids);
}

/// True when `pid` is one of our sidecars (not a recycled, unrelated PID). The
/// command line must reference the API binary or the Python worker module.
#[cfg(unix)]
fn is_our_sidecar(pid: u32) -> bool {
    let Ok(output) = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let command = String::from_utf8_lossy(&output.stdout);
    command.contains("sceneworks-api") || command.contains("scene_worker")
}

#[cfg(windows)]
fn is_our_sidecar(pid: u32) -> bool {
    // tasklist exposes the image name (sceneworks-api.exe) but not arguments, so
    // the Python worker (python.exe -m scene_worker) can't be matched here; the
    // worker exits on its own when its parent/API is gone, so only the API needs
    // reaping on Windows.
    let Ok(output) = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
    else {
        return false;
    };
    String::from_utf8_lossy(&output.stdout).contains("sceneworks-api")
}

/// SIGTERM then SIGKILL a confirmed-stale sidecar.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    let target = nix::unistd::Pid::from_raw(pid as i32);
    let _ = nix::sys::signal::kill(target, nix::sys::signal::Signal::SIGTERM);
    for _ in 0..20 {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = nix::sys::signal::kill(target, nix::sys::signal::Signal::SIGKILL);
}

#[cfg(windows)]
fn kill_pid(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output();
}

/// Kill sidecars left running by a prior unclean exit before spawning fresh
/// ones. Without this, a crash/force-quit (which skips `begin_shutdown`) leaves
/// orphaned API processes that accumulate, hold ports, and contend on jobs.db.
/// Each recorded PID is identity-checked so a recycled PID is never killed.
pub fn reap_stale_sidecars() {
    let path = sidecar_pidfile();
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let pids: SidecarPids = serde_json::from_slice(&bytes).unwrap_or_default();
    for pid in [pids.api, pids.mlx_worker, pids.candle_worker]
        .into_iter()
        .flatten()
    {
        if is_our_sidecar(pid) {
            kill_pid(pid);
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// Begin graceful shutdown: stop the GPU worker (MLX on macOS, candle on Windows)
/// then the API sidecar.
/// On Unix this sends SIGTERM and waits up to the grace period before
/// force-killing; on Windows it force-kills (CTRL_BREAK handling is a
/// Windows-session refinement). Returns true if shutdown was initiated (caller
/// should prevent the immediate exit), false if it was already in progress.
pub fn begin_shutdown(app: &AppHandle) -> bool {
    let managed = app.state::<Managed>();
    if managed.shutting_down.swap(true, Ordering::SeqCst) {
        return false;
    }
    let mlx_worker = managed.mlx_worker.lock().expect("mlx worker lock").take();
    let candle_worker = managed
        .candle_worker
        .lock()
        .expect("candle worker lock")
        .take();
    let api_child = managed.api.lock().expect("api lock").take();
    let handle = app.clone();
    std::thread::spawn(move || {
        #[cfg(unix)]
        {
            let grace = std::env::var("SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(10)
                .clamp(1, 30);
            let mlx_worker_pid = mlx_worker.as_ref().map(CommandChild::pid);
            let api_pid = api_child.as_ref().map(CommandChild::pid);
            // SIGTERM the workers first, then the API.
            for pid in [mlx_worker_pid, api_pid].into_iter().flatten() {
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(pid as i32),
                    nix::sys::signal::Signal::SIGTERM,
                );
            }
            let deadline = Instant::now() + Duration::from_secs(grace);
            while Instant::now() < deadline {
                if ![mlx_worker_pid, api_pid]
                    .into_iter()
                    .flatten()
                    .any(pid_alive)
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        // Force-kill anything still alive.
        if let Some(child) = mlx_worker {
            let _ = child.kill();
        }
        // Windows-only (candle is Windows); None elsewhere. The Windows shutdown
        // path force-kills (no SIGTERM grace above), and the worker also honours the
        // parent-death watchdog, so this is the belt to that suspenders.
        if let Some(child) = candle_worker {
            let _ = child.kill();
        }
        if let Some(child) = api_child {
            let _ = child.kill();
        }
        // Clean exit: drop the pidfile so the next launch doesn't try to reap
        // PIDs we already terminated.
        let _ = std::fs::remove_file(sidecar_pidfile());
        handle.exit(0);
    });
    true
}

/// Minimum NVIDIA display driver for the bundled CUDA 12.9 runtime (sc-3676 /
/// sc-5560): the floor that supports it and forward-JITs the compute_80 PTX.
#[cfg(target_os = "windows")]
const MIN_NVIDIA_DRIVER: f64 = 576.02;

#[cfg(target_os = "windows")]
const CUDA_REQUIREMENT: &str = "SceneWorks on Windows requires an NVIDIA (CUDA) GPU. \
    No NVIDIA GPU was detected — SceneWorks needs an NVIDIA GPU with driver 576.02 or \
    newer (there is no CPU or AMD fallback).";

/// Decide the preflight verdict from `nvidia-smi --query-gpu=name,driver_version`
/// output (`None` = nvidia-smi missing/failed). Pure so it's unit-testable; the IO
/// lives in `cuda_preflight`. `Ok(())` when a usable GPU is present; `Err(message)`
/// with a clear requirement otherwise (no GPU, or a driver below the floor).
#[cfg(target_os = "windows")]
fn evaluate_nvidia_preflight(smi_output: Option<&str>) -> Result<(), String> {
    let Some(line) =
        smi_output.and_then(|out| out.lines().map(str::trim).find(|line| !line.is_empty()))
    else {
        return Err(CUDA_REQUIREMENT.to_owned());
    };
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next().unwrap_or("");
    let driver = parts.next().unwrap_or("");
    // Block on a too-old driver; if the version is unparseable, don't block on it
    // (the GPU is present — let the worker surface any deeper issue).
    if let Ok(version) = driver.parse::<f64>() {
        if version < MIN_NVIDIA_DRIVER {
            return Err(format!(
                "SceneWorks on Windows requires NVIDIA driver {MIN_NVIDIA_DRIVER} or newer \
                 (found {driver} on {name}). Update your NVIDIA driver to continue."
            ));
        }
    }
    Ok(())
}

/// Windows CUDA preflight (sc-5561). SceneWorks generation off-Mac is CUDA-only —
/// no CPU/AMD fallback — so a machine without an NVIDIA GPU + an adequate driver
/// can run neither candle nor the Python torch worker's cu128 wheels. Probe
/// `nvidia-smi` for a GPU + driver version and return a clear, actionable error so
/// the app says "requires an NVIDIA GPU" up front instead of provisioning a venv and
/// then dead-polling jobs it can never run. `Ok(())` when a usable GPU is present.
#[cfg(target_os = "windows")]
fn cuda_preflight() -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    let mut command = std::process::Command::new("nvidia-smi");
    command.args([
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ]);
    // Don't flash a console window when probing from the GUI app.
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let stdout = match command.output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        // Missing (no NVIDIA driver) or errored → treat as no usable GPU.
        _ => return Err(CUDA_REQUIREMENT.to_owned()),
    };
    evaluate_nvidia_preflight(Some(&stdout))
}

async fn run_startup(app: AppHandle) {
    // Provide the builtin model catalog the rust-api/worker expect before they
    // start, so Model Manager is populated and native video resources resolve.
    // Mandatory: abort (rather than start a half-working app) if it can't be written.
    if let Err(error) = seed_builtin_manifests() {
        emit(&app, "error", format!("Setup failed: {error}"), true);
        return;
    }
    // CUDA-only on Windows (sc-5561): fail fast with a clear requirement message on a
    // machine with no NVIDIA GPU / too-old driver, BEFORE the multi-GB redist download,
    // rather than silently failing or dead-polling a job later. The setup page renders
    // this `error` event (apps/desktop/ui/index.html). The off-Mac desktop is candle/
    // CUDA-only now, so this always runs on Windows (the old "is the redist bundled?"
    // gate can't be used — the redist isn't downloaded yet at this point, and there's no
    // candle feature on the desktop crate to gate on; failing fast before a 2.7 GB
    // download is the whole point).
    #[cfg(target_os = "windows")]
    if let Err(error) = cuda_preflight() {
        emit(&app, "error", error, true);
        return;
    }
    // First-run GPU runtime provisioning (Windows candle build): the CUDA runtime +
    // cuDNN + onnxruntime-gpu DLLs are no longer bundled (they exceeded NSIS's ~2 GB
    // datablock limit), so download them once into %APPDATA%\SceneWorks\gpu-runtime and
    // resolve them from there (cuda_provision.rs). Idempotent + cached on later runs via
    // a version marker; emits per-component progress to the setup screen. On failure,
    // surface it and abort (same slot the removed Python-venv provisioning used).
    #[cfg(target_os = "windows")]
    if let Err(error) = crate::cuda_provision::provision(&app).await {
        emit(
            &app,
            "error",
            format!("GPU runtime download failed: {error}"),
            true,
        );
        return;
    }
    // No Python venv on ANY platform: macOS went MLX-only (epic 3482, sc-3492/sc-3493)
    // and off-Mac went candle-only (epic 5483 Phase 7, sc-5563), so first run starts
    // straight on the native engine with no Python provisioning step.
    // Spawn the API only once across retries.
    if app
        .state::<Managed>()
        .api
        .lock()
        .expect("api lock")
        .is_some()
    {
        return;
    }
    emit(&app, "starting", "Starting the local engine…", false);
    if let Err(error) = spawn_api(&app) {
        emit(&app, "error", error, true);
        return;
    }
    gate_window(app);
}

/// Frontend entry point (called on setup-screen load and on retry). Kicks off
/// provisioning + startup; guarded so concurrent invocations are ignored.
#[tauri::command]
pub async fn start_setup(app: AppHandle) {
    {
        let state = app.state::<Managed>();
        if state.running.swap(true, Ordering::SeqCst) {
            return;
        }
    }
    run_startup(app.clone()).await;
    app.state::<Managed>()
        .running
        .store(false, Ordering::SeqCst);
}

#[cfg(all(test, target_os = "windows"))]
mod preflight_tests {
    use super::{evaluate_nvidia_preflight, MIN_NVIDIA_DRIVER};

    #[test]
    fn no_nvidia_smi_output_requires_an_nvidia_gpu() {
        // nvidia-smi missing/failed (None) or empty output → requirement error.
        assert!(evaluate_nvidia_preflight(None).is_err());
        assert!(evaluate_nvidia_preflight(Some("")).is_err());
        assert!(evaluate_nvidia_preflight(Some("   \n")).is_err());
    }

    #[test]
    fn adequate_driver_passes() {
        assert!(evaluate_nvidia_preflight(Some("NVIDIA RTX PRO 6000, 576.02\n")).is_ok());
        assert!(evaluate_nvidia_preflight(Some("NVIDIA GeForce RTX 4090, 597.36")).is_ok());
    }

    #[test]
    fn too_old_driver_is_rejected_with_the_floor() {
        let verdict = evaluate_nvidia_preflight(Some("NVIDIA GeForce RTX 3090, 560.94"));
        let message = verdict.expect_err("a sub-576.02 driver must fail preflight");
        assert!(message.contains(&MIN_NVIDIA_DRIVER.to_string()));
        assert!(message.contains("560.94"));
    }

    #[test]
    fn unparseable_driver_does_not_block_a_present_gpu() {
        // The GPU is present; an odd version string shouldn't hard-block startup.
        assert!(evaluate_nvidia_preflight(Some("NVIDIA RTX, not-a-version")).is_ok());
    }
}
