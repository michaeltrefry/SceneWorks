//! First-run CUDA / onnxruntime redistributable provisioner (Windows candle build).
//!
//! The candle (Windows/CUDA) desktop needs ~2.7 GB of CUDA runtime + cuDNN +
//! onnxruntime-gpu DLLs at runtime: cudarc dynamic-linking `LoadLibrary`s the CUDA
//! runtime by name, and the worker's `ort` paths dlopen a CUDA-enabled onnxruntime
//! (DWPose / YOLO / Real-ESRGAN, epic 5482). We used to bundle these DLLs into the
//! installer (staged by build-sidecar.mjs into the `cuda` + `onnxruntime` resource
//! dirs), but the set blows past NSIS's ~2 GB datablock limit (`makensis`
//! "mmapping datablock" error). Instead we download them once on first run into
//! `%APPDATA%\SceneWorks\gpu-runtime\{cuda,onnxruntime}` and resolve them from
//! there — the same role the bundled resource dirs used to play, just relocated and
//! fetched lazily.
//!
//! The source is the PyPI `nvidia-*-cu12` + `onnxruntime-gpu` wheels (each a zip):
//! exactly what build-sidecar.mjs / stage-onnxruntime-cuda.py harvested from, so the
//! version-matched CUDA 12.9 runtime + the cuDNN/cuFFT/nvJitLink/nvRTC set
//! onnxruntime-gpu 1.26.0 was validated against (cuDNN 9.23 / cuFFT 11.4, sc-5496).
//! URLs + sha256 are pinned (mirrors the pinned-URL pattern used elsewhere for
//! reproducible downloads): resolved once from the PyPI JSON API and baked in below.
//!
//! Idempotent: a `.redist-marker` written after a full success short-circuits later
//! runs, so the multi-GB fetch happens only on the first launch (or after a version
//! bump that changes the marker).
#![cfg(target_os = "windows")]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tauri::AppHandle;

use crate::setup::{app_support_dir, emit};

/// Bump this when the pinned manifest changes so an existing install re-provisions.
/// Written to `<root>\.redist-marker` after a fully successful provision; a run whose
/// marker already equals this string skips the download entirely.
const REDIST_VERSION: &str = "cuda12.9-ort1.26.0-cudnn9.23-1";

/// Which provisioned subdir a component's DLLs land in. `Cuda` is the cudarc +
/// onnxruntime CUDA-dep dir (cudart/cublas/curand/nvrtc + cuDNN/cuFFT/nvJitLink);
/// `Onnxruntime` holds onnxruntime's own three DLLs.
#[derive(Clone, Copy)]
enum Dest {
    Cuda,
    Onnxruntime,
}

/// A pinned PyPI wheel to fetch + the DLLs to extract from it.
struct Component {
    /// Human label for progress UI ("cuDNN", "cuBLAS", …).
    label: &'static str,
    /// Approximate download size, shown in the progress message.
    approx: &'static str,
    /// Pinned win_amd64 wheel URL (files.pythonhosted.org).
    url: &'static str,
    /// sha256 of the wheel, verified after download.
    sha256: &'static str,
    /// Where the extracted DLLs go.
    dest: Dest,
    /// Specific DLL basenames to extract, or `None` to extract every `*.dll` in the
    /// wheel (used for the single-purpose nvidia-*-cu12 wheels).
    dlls: Option<&'static [&'static str]>,
}

/// The pinned redist set — matches what the build previously bundled (CUDA 12.9 gen
/// libs + the onnxruntime-gpu 1.26.0 CV-aux set). URLs + sha256 were resolved from
/// the PyPI JSON API (`https://pypi.org/pypi/<pkg>/<ver>/json`, the `*-win_amd64.whl`
/// file's `url` + `digests.sha256`). nvidia-*-cu12 wheels put DLLs under
/// `nvidia/<comp>/bin/*.dll`; onnxruntime-gpu under `onnxruntime/capi/*.dll`. The
/// extractor matches by basename, so the internal path is irrelevant.
const COMPONENTS: &[Component] = &[
    // CUDA 12.9 runtime libs cudarc LoadLibrary's by name (the toolkit-redist set
    // build-sidecar.mjs used to copy). Extract every DLL in each single-purpose wheel.
    Component {
        label: "CUDA runtime",
        approx: "≈3 MB",
        url: "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        sha256: "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
        dest: Dest::Cuda,
        dlls: None,
    },
    Component {
        label: "cuBLAS",
        approx: "≈530 MB",
        url: "https://files.pythonhosted.org/packages/45/a1/a17fade6567c57452cfc8f967a40d1035bb9301db52f27808167fbb2be2f/nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl",
        sha256: "1e5fee10662e6e52bd71dec533fbbd4971bb70a5f24f3bc3793e5c2e9dc640bf",
        dest: Dest::Cuda,
        dlls: None,
    },
    Component {
        label: "cuRAND",
        approx: "≈66 MB",
        url: "https://files.pythonhosted.org/packages/e5/98/1bd66fd09cbe1a5920cb36ba87029d511db7cca93979e635fd431ad3b6c0/nvidia_curand_cu12-10.3.10.19-py3-none-win_amd64.whl",
        sha256: "e8129e6ac40dc123bd948e33d3e11b4aa617d87a583fa2f21b3210e90c743cde",
        dest: Dest::Cuda,
        dlls: None,
    },
    Component {
        label: "NVRTC",
        approx: "≈73 MB",
        url: "https://files.pythonhosted.org/packages/52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
        sha256: "72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349",
        dest: Dest::Cuda,
        dlls: None,
    },
    // onnxruntime's CUDA execution provider needs cuDNN-9 (incl. its lazily-loaded
    // sub-engine DLLs), cuFFT, nvJitLink. Extract every DLL.
    Component {
        label: "cuDNN",
        approx: "≈660 MB",
        url: "https://files.pythonhosted.org/packages/b7/ec/d95cc4204dd45f40f2d1512f8ff0d4c3fb1810a893fecc79fcea05dfec0e/nvidia_cudnn_cu12-9.23.0.39-py3-none-win_amd64.whl",
        sha256: "357e5d59a1b79d27eef754aa79b3d9e7adf11baf86dc928dc114df0033c2c912",
        dest: Dest::Cuda,
        dlls: None,
    },
    Component {
        label: "cuFFT",
        approx: "≈190 MB",
        url: "https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
        sha256: "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80",
        dest: Dest::Cuda,
        dlls: None,
    },
    Component {
        label: "nvJitLink",
        approx: "≈34 MB",
        url: "https://files.pythonhosted.org/packages/dd/7e/2eecb277d8a98184d881fb98a738363fd4f14577a4d2d7f8264266e82623/nvidia_nvjitlink_cu12-12.9.86-py3-none-win_amd64.whl",
        sha256: "cc6fcec260ca843c10e34c936921a1c426b351753587fdd638e8cff7b16bb9db",
        dest: Dest::Cuda,
        dlls: None,
    },
    // onnxruntime-gpu's own DLLs. The cp312 wheel: the native DLLs are identical
    // across the cp311/cp312/cp313/cp314 ABI wheels (the cp tag only versions the
    // Python `.pyd` binding we don't ship), so any ABI's DLLs are equivalent. Extract
    // exactly the three the worker dlopens (TensorRT is deliberately not staged).
    Component {
        label: "onnxruntime (GPU)",
        approx: "≈216 MB",
        url: "https://files.pythonhosted.org/packages/a4/e4/9b378a5466ea0bed65e5beb8e09254973c580a6522810a38afbcc45e5105/onnxruntime_gpu-1.26.0-cp312-cp312-win_amd64.whl",
        sha256: "5f49c44689894650990e4c8a857d2edafc276fbd79bba57ceb224bd18d25d491",
        dest: Dest::Onnxruntime,
        dlls: Some(&[
            "onnxruntime.dll",
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_shared.dll",
        ]),
    },
];

/// Root of the provisioned GPU runtime: `%APPDATA%\SceneWorks\gpu-runtime`.
fn root() -> PathBuf {
    app_support_dir().join("gpu-runtime")
}

/// Provisioned CUDA runtime DLL dir (`<root>\cuda`). The candle worker's PATH is
/// prepended with this so cudarc's `LoadLibrary` and onnxruntime's CUDA provider find
/// cudart/cublas/curand/nvrtc/cuDNN/cuFFT/nvJitLink.
pub(crate) fn cuda_dir() -> PathBuf {
    root().join("cuda")
}

/// Provisioned onnxruntime DLL dir (`<root>\onnxruntime`).
fn onnxruntime_dir() -> PathBuf {
    root().join("onnxruntime")
}

/// The provisioned onnxruntime.dll path (set as `ORT_DYLIB_PATH`).
pub(crate) fn onnxruntime_dll() -> PathBuf {
    onnxruntime_dir().join("onnxruntime.dll")
}

/// The provisioned CUDA dir, but only if the redist has actually been downloaded
/// (probes `cudart64_12.dll`, the marker DLL the bundled resolver also probed). The
/// resolvers in setup.rs gate the candle worker / PATH / ORT wiring on this — before
/// first-run provisioning completes it's None, exactly as the empty bundle dir was.
pub(crate) fn cuda_dir_if_present() -> Option<PathBuf> {
    let dir = cuda_dir();
    dir.join("cudart64_12.dll").exists().then_some(dir)
}

/// The provisioned onnxruntime.dll, but only if it has actually been downloaded.
pub(crate) fn onnxruntime_dll_if_present() -> Option<PathBuf> {
    let dll = onnxruntime_dll();
    dll.exists().then_some(dll)
}

/// True when a prior run already provisioned this exact REDIST_VERSION.
fn already_provisioned(root: &Path) -> bool {
    fs::read_to_string(root.join(".redist-marker"))
        .map(|marker| marker.trim() == REDIST_VERSION)
        .unwrap_or(false)
}

/// Hex-encode a sha256 digest for comparison against the pinned value.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Download one wheel into `tmp`, verify its sha256, and extract its DLLs into the
/// destination dir. Network IO is async (reqwest stream); the CPU-bound hash + unzip
/// run on a blocking thread so they don't stall the async runtime.
async fn fetch_component(
    client: &reqwest::Client,
    component: &Component,
    tmp_dir: &Path,
    cuda: &Path,
    ort: &Path,
) -> Result<usize, String> {
    let response = client
        .get(component.url)
        .send()
        .await
        .map_err(|error| format!("download {}: {error}", component.label))?
        .error_for_status()
        .map_err(|error| format!("download {}: {error}", component.label))?;
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("download {}: {error}", component.label))?;

    let wheel_path = tmp_dir.join(format!(
        "{}.whl",
        component.label.replace(['/', ' ', '(', ')'], "_")
    ));
    fs::write(&wheel_path, &bytes)
        .map_err(|error| format!("write {}: {error}", component.label))?;

    let dest = match component.dest {
        Dest::Cuda => cuda.to_path_buf(),
        Dest::Onnxruntime => ort.to_path_buf(),
    };
    let expected = component.sha256.to_owned();
    let label = component.label.to_owned();
    let dlls: Option<Vec<String>> = component
        .dlls
        .map(|names| names.iter().map(|name| name.to_string()).collect());

    // Hashing + unzip are CPU/IO bound — keep them off the async executor.
    tauri::async_runtime::spawn_blocking(move || -> Result<usize, String> {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = hex(&hasher.finalize());
        if digest != expected {
            return Err(format!(
                "{label}: sha256 mismatch (expected {expected}, got {digest})"
            ));
        }
        extract_dlls(&wheel_path, dlls.as_deref(), &dest)
            .map_err(|error| format!("{label}: {error}"))
    })
    .await
    .map_err(|error| format!("{}: extract task failed: {error}", component.label))?
}

/// Extract DLLs from a wheel (zip) into `dest` by basename. `names = None` extracts
/// every `*.dll`; otherwise only the listed basenames. Returns how many were written.
fn extract_dlls(wheel: &Path, names: Option<&[String]>, dest: &Path) -> Result<usize, String> {
    fs::create_dir_all(dest).map_err(|error| format!("create {}: {error}", dest.display()))?;
    let file = fs::File::open(wheel).map_err(|error| format!("open wheel: {error}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| format!("open zip: {error}"))?;
    let mut written = 0usize;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("read zip entry: {error}"))?;
        // Use the sanitized name so a malicious entry can't traverse out of dest.
        let Some(entry_path) = entry.enclosed_name() else {
            continue;
        };
        let Some(base) = entry_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !base.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }
        if let Some(names) = names {
            if !names.iter().any(|name| name == base) {
                continue;
            }
        }
        let out_path = dest.join(base);
        let mut buffer = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut buffer)
            .map_err(|error| format!("extract {base}: {error}"))?;
        fs::write(&out_path, &buffer).map_err(|error| format!("write {base}: {error}"))?;
        written += 1;
    }
    Ok(written)
}

/// Provision the CUDA / onnxruntime redist on first run (idempotent). Emits
/// `setup-status` progress per component while it downloads + verifies + extracts the
/// pinned wheels into `%APPDATA%\SceneWorks\gpu-runtime`. A `.redist-marker` written
/// on full success short-circuits later runs. Returns `Err` with a clear message on
/// any failure so the caller can surface it on the setup screen and abort startup.
///
/// Async so it can be `.await`ed from `run_startup` (a Tauri async command) — driving
/// it via `block_on` from inside the runtime would panic / deadlock. Network IO is
/// async (reqwest stream); the per-component hash + unzip are offloaded to a blocking
/// thread inside `fetch_component`.
pub(crate) async fn provision(app: &AppHandle) -> Result<(), String> {
    let root = root();
    if already_provisioned(&root) {
        return Ok(());
    }

    let cuda = cuda_dir();
    let ort = onnxruntime_dir();
    for dir in [&cuda, &ort] {
        fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    }
    let tmp_dir = root.join(".download-tmp");
    fs::create_dir_all(&tmp_dir).map_err(|error| format!("create temp dir: {error}"))?;

    emit(
        app,
        "provision",
        "Downloading GPU runtime (first run, ~2.7 GB)…",
        false,
    );

    let client = reqwest::Client::builder()
        .build()
        .map_err(|error| format!("http client: {error}"))?;

    let mut outcome: Result<(), String> = Ok(());
    for (index, component) in COMPONENTS.iter().enumerate() {
        emit(
            app,
            "provision",
            format!(
                "Downloading GPU runtime [{}/{}]: {} ({})…",
                index + 1,
                COMPONENTS.len(),
                component.label,
                component.approx
            ),
            false,
        );
        match fetch_component(&client, component, &tmp_dir, &cuda, &ort).await {
            Ok(written) => {
                if let Some(expected) = component.dlls {
                    if written < expected.len() {
                        outcome = Err(format!(
                            "{}: extracted {written}/{} DLLs",
                            component.label,
                            expected.len()
                        ));
                        break;
                    }
                } else if written == 0 {
                    outcome = Err(format!("{}: no DLLs found in wheel", component.label));
                    break;
                }
            }
            Err(error) => {
                outcome = Err(error);
                break;
            }
        }
    }

    // Always clean the temp wheels; ignore failures (best effort).
    let _ = fs::remove_dir_all(&tmp_dir);

    outcome?;

    // Mark success last so a partial/aborted run re-provisions next launch.
    fs::write(root.join(".redist-marker"), REDIST_VERSION)
        .map_err(|error| format!("write marker: {error}"))?;
    emit(app, "provision", "GPU runtime ready.", false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned set must stay internally consistent: every component has a non-empty
    /// pinned URL pointing at the PyPI CDN and a 64-hex-char sha256. Cheap, offline.
    #[test]
    fn manifest_is_well_formed() {
        assert!(!COMPONENTS.is_empty());
        for component in COMPONENTS {
            assert!(
                component.url.starts_with("https://files.pythonhosted.org/"),
                "{}: url must be a pinned PyPI wheel",
                component.label
            );
            assert!(
                component.url.ends_with("-win_amd64.whl"),
                "{}: url must be the win_amd64 wheel",
                component.label
            );
            assert_eq!(
                component.sha256.len(),
                64,
                "{}: sha256 must be 64 hex chars",
                component.label
            );
            assert!(
                component
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}: sha256 must be lowercase hex",
                component.label
            );
        }
    }

    /// End-to-end smoke of the download → sha256 → unzip path on the SMALLEST pinned
    /// wheel (nvidia-cuda-runtime-cu12, ~3 MB): proves the pinned URL/sha256 + the zip
    /// extractor actually produce the expected DLL. Network-gated (`#[ignore]`) so the
    /// normal offline `cargo test` stays fast; run with
    /// `cargo test -p sceneworks-desktop -- --ignored downloader_smoke`.
    #[test]
    #[ignore = "network: downloads ~3 MB from PyPI"]
    fn downloader_smoke() {
        // The smallest component in the manifest (CUDA runtime, ~3 MB).
        let component = COMPONENTS
            .iter()
            .find(|c| c.label == "CUDA runtime")
            .expect("CUDA runtime component present");

        let tmp = std::env::temp_dir().join(format!("sw-cuda-smoke-{}", std::process::id()));
        let dest = tmp.join("cuda");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&dest).expect("create dest");

        // Download (async reqwest) + verify sha256 + unzip — exercising the same code
        // the real provisioner uses (extract_dlls / hex), minus the AppHandle emit.
        let bytes = tauri::async_runtime::block_on(async {
            let client = reqwest::Client::builder().build().expect("client");
            client
                .get(component.url)
                .send()
                .await
                .expect("send")
                .error_for_status()
                .expect("status")
                .bytes()
                .await
                .expect("bytes")
        });

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = hex(&hasher.finalize());
        assert_eq!(
            digest, component.sha256,
            "sha256 must match the pinned value"
        );

        let wheel = tmp.join("runtime.whl");
        fs::write(&wheel, &bytes).expect("write wheel");
        let written = extract_dlls(&wheel, None, &dest).expect("extract");
        assert!(written >= 1, "at least one DLL extracted");
        assert!(
            dest.join("cudart64_12.dll").exists(),
            "cudart64_12.dll must be present after extraction"
        );

        let _ = fs::remove_dir_all(&tmp);
    }
}
