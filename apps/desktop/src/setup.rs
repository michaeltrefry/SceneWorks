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

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Bump to force a re-provision even if requirements.txt is unchanged.
const SETUP_VERSION: &str = "3";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Process handles + run guards shared across the app.
#[derive(Default)]
pub struct Managed {
    pub api: Mutex<Option<CommandChild>>,
    pub worker: Mutex<Option<CommandChild>>,
    /// OS-assigned API port, discovered from the sidecar's startup line.
    api_port: Mutex<Option<u16>>,
    /// PIDs of the spawned sidecars, persisted to disk so an unclean exit
    /// (crash/force-quit) doesn't leave them orphaned — the next launch reaps
    /// any survivors before spawning fresh ones.
    pids: Mutex<SidecarPids>,
    running: AtomicBool,
    pub shutting_down: AtomicBool,
}

/// PIDs of the API + Python worker sidecars owned by this launch.
#[derive(Default, Clone, Serialize, Deserialize)]
struct SidecarPids {
    api: Option<u32>,
    worker: Option<u32>,
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

/// requirements-mlx.txt location (Apple Silicon MLX video inference deps): the
/// bundled resource in a packaged app, or the repo copy during development.
/// Optional and macOS-only — installed by `provision_venv` only on darwin so the
/// Windows/Linux PyTorch worker is unaffected.
#[cfg(target_os = "macos")]
fn requirements_mlx_path(app: &AppHandle) -> PathBuf {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("python-src").join("requirements-mlx.txt");
        if bundled.exists() {
            return bundled;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("worker")
        .join("requirements-mlx.txt")
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
/// separate `user.*.jsonc` files, which this never touches. Each file is written
/// atomically (temp + rename) so a crash or partial write can't leave a
/// truncated manifest that parses to an empty/broken catalog. Returns an error
/// if any required manifest can't be installed — the catalog is mandatory, so
/// the caller aborts setup rather than starting with missing model mappings.
fn seed_builtin_manifests() -> Result<(), String> {
    let dir = config_dir().join("manifests");
    std::fs::create_dir_all(&dir).map_err(|error| format!("create manifests dir: {error}"))?;
    for (name, contents) in BUILTIN_MANIFESTS {
        let temp = dir.join(format!("{name}.tmp"));
        std::fs::write(&temp, contents).map_err(|error| format!("write {name}: {error}"))?;
        std::fs::rename(&temp, dir.join(name)).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            format!("install {name}: {error}")
        })?;
    }
    Ok(())
}

/// Data directory: the settings override if set, otherwise the platform default.
fn resolved_data_dir() -> PathBuf {
    crate::settings::load_settings()
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir)
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
    // Apple Silicon MLX video inference deps — macOS-only; empty body elsewhere so
    // the marker stays stable and the Windows/Linux PyTorch worker is untouched.
    #[cfg(target_os = "macos")]
    let requirements_mlx = requirements_mlx_path(app);
    #[cfg(target_os = "macos")]
    let requirements_mlx_body = std::fs::read_to_string(&requirements_mlx).unwrap_or_default();
    #[cfg(not(target_os = "macos"))]
    let requirements_mlx_body = String::new();
    let marker = marker_path();
    let expected = format!(
        "v{SETUP_VERSION}\n{requirements_body}\n# ltx\n{requirements_ltx_body}\n# mlx\n{requirements_mlx_body}"
    );

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

    // Create the venv only when its interpreter is missing. A prior run that
    // created the venv but failed during dependency install (e.g. an
    // unsatisfiable resolution) or was interrupted leaves a venv that `uv venv`
    // refuses to overwrite — without this guard every retry dies on "a virtual
    // environment already exists". When the interpreter is present we skip
    // creation and let the install below reconcile the env. `--clear` wipes any
    // partial/corrupt directory when we do (re)create from scratch.
    if !python.exists() {
        emit(app, "creating", "Creating the Python environment…", false);
        run_uv(
            app,
            vec![
                "venv".to_owned(),
                "--clear".to_owned(),
                "--python".to_owned(),
                "3.12".to_owned(),
                venv.to_string_lossy().into_owned(),
            ],
        )
        .await?;
    }

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
    // MLX deps (Apple Silicon only): the native-MLX LTX/Wan video path. Resolved
    // in the same uv pass so transformers/numpy stay on one ABI-compatible set.
    #[cfg(target_os = "macos")]
    if requirements_mlx.exists() {
        requirement_files.push(requirements_mlx.clone());
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
        )
        // The catalog's install-state detection resolves the HF cache from this;
        // it must match the worker's download root or every model reads "missing".
        .env("HF_HOME", huggingface_home().to_string_lossy().to_string());
    // The in-process utility worker shells out to ffmpeg; point it at the venv's
    // bundled binary since the desktop has no system ffmpeg on PATH.
    if let Some(ffmpeg) = resolve_bundled_ffmpeg() {
        command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
    }
    // MLX model conversion (model_convert jobs) shells out to the venv's Python
    // (mlx_video.convert_wan); point the in-process worker at the bundled interpreter.
    command = command.env(
        "SCENEWORKS_PYTHON",
        venv_python(&venv_dir()).to_string_lossy().to_string(),
    );
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
        // Match the API sidecar's HF cache root so downloaded weights land where
        // the catalog looks for them (and reuse anything already cached there).
        let hf_home = huggingface_home().to_string_lossy().to_string();
        // Unique per app launch (stable across in-launch respawns) so a worker
        // from a prior/overlapping session can't impersonate this one in the
        // shared jobs.db. A fixed "worker-local-0" let two incarnations collide:
        // one claims a job, the other's idle heartbeat instantly interrupts it.
        let worker_id = format!(
            "worker-local-{}-{}",
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
                .env("SCENEWORKS_WORKER_ID", &worker_id)
                .env("SCENEWORKS_API_URL", &api_url)
                .env("HF_HOME", &hf_home)
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
            record_worker_pid(&app, Some(child.pid()));
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
            record_worker_pid(&app, None);
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

fn record_worker_pid(app: &AppHandle, pid: Option<u32>) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.worker = pid;
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
    for pid in [pids.api, pids.worker].into_iter().flatten() {
        if is_our_sidecar(pid) {
            kill_pid(pid);
        }
    }
    let _ = std::fs::remove_file(&path);
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
        // Clean exit: drop the pidfile so the next launch doesn't try to reap
        // PIDs we already terminated.
        let _ = std::fs::remove_file(sidecar_pidfile());
        handle.exit(0);
    });
    true
}

async fn run_startup(app: AppHandle) {
    // Provide the builtin model catalog the rust-api/worker expect before they
    // start, so Model Manager is populated and native video resources resolve.
    // Mandatory: abort (rather than start a half-working app) if it can't be written.
    if let Err(error) = seed_builtin_manifests() {
        emit(&app, "error", format!("Setup failed: {error}"), true);
        return;
    }
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
