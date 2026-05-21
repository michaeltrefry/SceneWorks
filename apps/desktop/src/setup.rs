//! First-run Python venv bootstrap + startup orchestration (sc-1348).
//!
//! The frontend setup screen calls the `start_setup` command once it is ready to
//! receive events; this provisions the uv-managed venv (streaming progress),
//! then spawns the API sidecar, health-gates it, and navigates the window to the
//! local API. `start_setup` is also the retry entry point.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Bump to force a re-provision even if requirements.txt is unchanged.
const SETUP_VERSION: &str = "2";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Process handles + run guards shared across the app.
#[derive(Default)]
pub struct Managed {
    pub api: Mutex<Option<CommandChild>>,
    pub worker: Mutex<Option<CommandChild>>,
    /// OS-assigned API port, discovered from the sidecar's startup line.
    api_port: Mutex<Option<u16>>,
    running: AtomicBool,
    pub shutting_down: AtomicBool,
}

#[derive(Clone, Serialize)]
struct SetupStatus {
    phase: String,
    message: String,
    error: bool,
}

fn emit(app: &AppHandle, phase: &str, message: impl Into<String>, error: bool) {
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

pub fn venv_dir() -> PathBuf {
    app_support_dir().join("python").join("venv")
}

pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

fn marker_path() -> PathBuf {
    app_support_dir().join("python").join(".venv-marker")
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

/// requirements.txt location: an explicit override (testing / custom installs),
/// the bundled resource in a packaged app, or the repo copy during development.
fn requirements_path(app: &AppHandle) -> PathBuf {
    if let Ok(override_path) = std::env::var("SCENEWORKS_DESKTOP_REQUIREMENTS") {
        let trimmed = override_path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("python-src").join("requirements.txt");
        if bundled.exists() {
            return bundled;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("worker")
        .join("requirements.txt")
}

/// requirements-ltx.txt location (native LTX pipelines deps): the bundled
/// resource in a packaged app, or the repo copy during development. Optional —
/// absent in older worker checkouts, in which case LTX video is unavailable.
fn requirements_ltx_path(app: &AppHandle) -> PathBuf {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("python-src").join("requirements-ltx.txt");
        if bundled.exists() {
            return bundled;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("worker")
        .join("requirements-ltx.txt")
}

fn data_dir() -> PathBuf {
    app_support_dir().join("data")
}

fn config_dir() -> PathBuf {
    app_support_dir().join("config")
}

/// Builtin model/LoRA/recipe-preset catalogs the rust-api reads from
/// `config_dir/manifests`. In the server stack these ship in the repo's
/// `config/`, but the desktop must provide them itself or Model Manager is empty
/// and the native LTX/Wan adapters can't map model resources to files. Embedded
/// at compile time from the canonical repo copies.
const BUILTIN_MANIFESTS: &[(&str, &str)] = &[
    (
        "builtin.models.jsonc",
        include_str!("../../../config/manifests/builtin.models.jsonc"),
    ),
    (
        "builtin.loras.jsonc",
        include_str!("../../../config/manifests/builtin.loras.jsonc"),
    ),
    (
        "builtin.recipe-presets.jsonc",
        include_str!("../../../config/manifests/builtin.recipe-presets.jsonc"),
    ),
];

/// Write the builtin manifests into `config_dir/manifests`, overwriting on every
/// launch so they track the app version. User customizations live in the
/// separate `user.*.jsonc` files, which this never touches.
fn seed_builtin_manifests() {
    let dir = config_dir().join("manifests");
    if let Err(error) = std::fs::create_dir_all(&dir) {
        eprintln!("[desktop] seed manifests: create dir failed: {error}");
        return;
    }
    for (name, contents) in BUILTIN_MANIFESTS {
        if let Err(error) = std::fs::write(dir.join(name), contents) {
            eprintln!("[desktop] seed manifests: write {name} failed: {error}");
        }
    }
}

/// Data directory: the settings override if set, otherwise the platform default.
fn resolved_data_dir() -> PathBuf {
    crate::settings::load_settings()
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(data_dir)
}

/// Directory containing the Python `scene_worker` package + requirements: the
/// bundled resource in a packaged app, the repo copy during development.
fn worker_src_dir(app: &AppHandle) -> PathBuf {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("python-src");
        if bundled.join("scene_worker").exists() {
            return bundled;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("worker")
}

/// Directory to add to PYTHONPATH so the worker can import `sceneworks_shared`
/// (scene_worker depends on it at startup). Bundled: it's staged into python-src
/// alongside scene_worker. Development: it lives in the repo's packages/shared.
/// Mirrors the Docker worker's `PYTHONPATH=...:/app/packages/shared`.
fn shared_parent_dir(app: &AppHandle) -> PathBuf {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("python-src");
        if bundled.join("sceneworks_shared").exists() {
            return bundled;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("packages")
        .join("shared")
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

/// Run the bundled `uv` with the given args, streaming output to setup-status
/// log events. Returns Err with a message on a non-zero exit.
async fn run_uv(app: &AppHandle, args: Vec<String>) -> Result<(), String> {
    let (mut events, _child) = app
        .shell()
        .sidecar("uv")
        .map_err(|error| format!("locate uv: {error}"))?
        .args(args)
        .spawn()
        .map_err(|error| format!("spawn uv: {error}"))?;
    let mut exit_code = None;
    while let Some(event) = events.recv().await {
        match event {
            CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                let line = String::from_utf8_lossy(&bytes).trim_end().to_owned();
                if !line.is_empty() {
                    emit(app, "log", line, false);
                }
            }
            CommandEvent::Terminated(payload) => {
                exit_code = payload.code;
                break;
            }
            CommandEvent::Error(error) => return Err(error),
            _ => {}
        }
    }
    match exit_code {
        Some(0) => Ok(()),
        other => Err(format!("uv exited with status {other:?}")),
    }
}

/// Build `uv pip install -r <req>…` args for the given requirement files. On
/// Windows it adds the CUDA extra index so torch/torchaudio resolve to CUDA
/// wheels (other packages still come from PyPI); macOS torch wheels include MPS
/// by default. All requirement files are installed in one pass so torch and its
/// companions (e.g. torchaudio) resolve to a single, ABI-compatible version set.
fn pip_install_args(python: &Path, requirement_files: &[PathBuf]) -> Vec<String> {
    #[cfg_attr(not(target_os = "windows"), allow(unused_mut))]
    let mut args = vec![
        "pip".to_owned(),
        "install".to_owned(),
        "--python".to_owned(),
        python.to_string_lossy().into_owned(),
    ];
    for requirements in requirement_files {
        args.push("-r".to_owned());
        args.push(requirements.to_string_lossy().into_owned());
    }
    #[cfg(target_os = "windows")]
    {
        let index = std::env::var("SCENEWORKS_PYTORCH_INDEX_URL")
            .unwrap_or_else(|_| "https://download.pytorch.org/whl/cu128".to_owned());
        args.push("--extra-index-url".to_owned());
        args.push(index);
    }
    args
}

/// Provision the venv if missing or stale (requirements / setup version changed).
async fn provision_venv(app: &AppHandle) -> Result<(), String> {
    let venv = venv_dir();
    let python = venv_python(&venv);
    let requirements = requirements_path(app);
    let requirements_body = std::fs::read_to_string(&requirements)
        .map_err(|error| format!("read requirements: {error}"))?;
    // Native LTX pipelines deps (ltx-core/ltx-pipelines + a torch-matched
    // torchaudio). Optional: absent in older worker checkouts.
    let requirements_ltx = requirements_ltx_path(app);
    let requirements_ltx_body = std::fs::read_to_string(&requirements_ltx).unwrap_or_default();
    let marker = marker_path();
    let expected = format!("v{SETUP_VERSION}\n{requirements_body}\n# ltx\n{requirements_ltx_body}");

    if python.exists() {
        if let Ok(found) = std::fs::read_to_string(&marker) {
            if found == expected {
                emit(app, "ready", "Python environment ready.", false);
                return Ok(());
            }
        }
    }

    if let Some(parent) = venv.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("create python dir: {error}"))?;
    }

    emit(app, "creating", "Creating the Python environment…", false);
    run_uv(
        app,
        vec![
            "venv".to_owned(),
            "--python".to_owned(),
            "3.12".to_owned(),
            venv.to_string_lossy().into_owned(),
        ],
    )
    .await?;

    emit(
        app,
        "installing",
        "Installing dependencies — this can take several minutes on first run…",
        false,
    );
    // Install the base requirements together with the LTX pipelines deps (when
    // bundled) in a single resolution pass: a separate LTX pass pulls a newer
    // torchaudio whose native extension fails to load against the pinned torch.
    let mut requirement_files = vec![requirements.clone()];
    if requirements_ltx.exists() {
        requirement_files.push(requirements_ltx.clone());
    }
    run_uv(app, pip_install_args(&python, &requirement_files)).await?;

    std::fs::write(&marker, &expected).map_err(|error| format!("write marker: {error}"))?;
    emit(app, "ready", "Python environment ready.", false);
    Ok(())
}

/// Resolve the ffmpeg binary bundled with the venv's imageio-ffmpeg, used by the
/// API's in-process utility worker for timeline export / frame extraction. The
/// desktop ships no system ffmpeg, so without this those jobs fail. Returns None
/// if the venv or imageio-ffmpeg isn't available yet.
fn resolve_bundled_ffmpeg() -> Option<String> {
    let python = venv_python(&venv_dir());
    if !python.exists() {
        return None;
    }
    let mut command = std::process::Command::new(&python);
    command.args([
        "-c",
        "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())",
    ]);
    // Don't flash a console window when probing from the GUI app.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if path.is_empty() || !Path::new(&path).exists() {
        return None;
    }
    Some(path)
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
        .env(
            "SCENEWORKS_DATA_DIR",
            resolved_data_dir().to_string_lossy().to_string(),
        )
        // Pin the config dir so the API and Python worker share one root on all
        // platforms (Linux otherwise splits XDG data vs config).
        .env(
            "SCENEWORKS_CONFIG_DIR",
            config_dir().to_string_lossy().to_string(),
        );
    // The in-process utility worker shells out to ffmpeg; point it at the venv's
    // bundled binary since the desktop has no system ffmpeg on PATH.
    if let Some(ffmpeg) = resolve_bundled_ffmpeg() {
        command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
    }
    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("spawn api: {error}"))?;
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
        }
    });
    Ok(())
}

/// Health-gate the window on a background thread: wait for the API's
/// OS-assigned port, confirm the responder is genuinely SceneWorks, then
/// navigate and start the Python worker; show an error after the timeout.
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
                    supervise_worker(app, port);
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

/// Spawn and supervise the Python inference worker on a background thread,
/// restarting it with exponential backoff if it dies unexpectedly while the app
/// is open. Output is appended to worker.log.
fn supervise_worker(app: AppHandle, api_port: u16) {
    std::thread::spawn(move || {
        let log_path = logs_dir().join("worker.log");
        let python = venv_python(&venv_dir());
        let src = worker_src_dir(&app);
        // Ensure the worker can import sceneworks_shared (staged into python-src
        // when bundled, or packages/shared in dev). Mirrors the Docker worker's
        // PYTHONPATH; `-m` already adds the cwd but we set it explicitly so the
        // dev (unbundled) path resolves too.
        let path_sep = if cfg!(windows) { ";" } else { ":" };
        let pythonpath = format!(
            "{}{}{}",
            src.to_string_lossy(),
            path_sep,
            shared_parent_dir(&app).to_string_lossy()
        );
        let api_url = format!("http://127.0.0.1:{api_port}");
        let mut backoff = 1u64;
        loop {
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            if !python.exists() {
                append_log(
                    &log_path,
                    "[desktop] cannot start worker: venv python missing\n",
                );
                return;
            }
            let mut command = app
                .shell()
                .command(python.to_string_lossy().to_string())
                .args(["-m", "scene_worker"])
                .current_dir(&src)
                .env("PYTHONPATH", &pythonpath)
                .env("SCENEWORKS_API_URL", &api_url)
                .env(
                    "SCENEWORKS_DATA_DIR",
                    resolved_data_dir().to_string_lossy().to_string(),
                )
                .env(
                    "SCENEWORKS_CONFIG_DIR",
                    config_dir().to_string_lossy().to_string(),
                );
            if let Some(token) = crate::settings::read_hf_token() {
                command = command.env("HF_TOKEN", token);
            }
            let spawned = command.spawn();
            let (mut events, child) = match spawned {
                Ok(pair) => pair,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] worker spawn failed: {error}\n"),
                    );
                    std::thread::sleep(Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(30);
                    continue;
                }
            };
            app.state::<Managed>()
                .worker
                .lock()
                .expect("worker lock")
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
                                "[desktop] worker terminated: code={:?} signal={:?}\n",
                                payload.code, payload.signal
                            ),
                        );
                        break;
                    }
                    Some(CommandEvent::Error(error)) => {
                        append_log(&log_path, &format!("[desktop] worker error: {error}\n"));
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
            let _ = app
                .state::<Managed>()
                .worker
                .lock()
                .expect("worker lock")
                .take();
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            // Reset backoff after a stable run; otherwise grow it to avoid a
            // tight crash loop.
            if started.elapsed() > Duration::from_secs(20) {
                backoff = 1;
            }
            append_log(
                &log_path,
                &format!("[desktop] restarting worker in {backoff}s\n"),
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

/// Begin graceful shutdown: stop the Python worker then the API sidecar. On Unix
/// this sends SIGTERM and waits up to the grace period before force-killing; on
/// Windows it force-kills (CTRL_BREAK handling is a Windows-session refinement).
/// Returns true if shutdown was initiated (caller should prevent the immediate
/// exit), false if it was already in progress.
pub fn begin_shutdown(app: &AppHandle) -> bool {
    let managed = app.state::<Managed>();
    if managed.shutting_down.swap(true, Ordering::SeqCst) {
        return false;
    }
    let worker = managed.worker.lock().expect("worker lock").take();
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
            let worker_pid = worker.as_ref().map(CommandChild::pid);
            let api_pid = api_child.as_ref().map(CommandChild::pid);
            // SIGTERM the worker first, then the API.
            for pid in [worker_pid, api_pid].into_iter().flatten() {
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(pid as i32),
                    nix::sys::signal::Signal::SIGTERM,
                );
            }
            let deadline = Instant::now() + Duration::from_secs(grace);
            while Instant::now() < deadline {
                if ![worker_pid, api_pid].into_iter().flatten().any(pid_alive) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        // Force-kill anything still alive.
        if let Some(child) = worker {
            let _ = child.kill();
        }
        if let Some(child) = api_child {
            let _ = child.kill();
        }
        handle.exit(0);
    });
    true
}

async fn run_startup(app: AppHandle) {
    // Provide the builtin model catalog the rust-api/worker expect before they
    // start, so Model Manager is populated and native video resources resolve.
    seed_builtin_manifests();
    if let Err(error) = provision_venv(&app).await {
        emit(&app, "error", format!("Setup failed: {error}"), true);
        return;
    }
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
