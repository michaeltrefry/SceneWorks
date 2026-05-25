use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio as StdStdio;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode as AxumStatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use sceneworks_core::contracts::WorkerUtilizationSnapshot;
use serde_json::{json, Value};
use tempfile::tempdir;

use super::api_client::ApiClient;
use super::downloads::{
    download_lora_source_url, download_progress_payload, download_snapshot, DownloadContext,
    DownloadProgress, HuggingFaceSnapshot, SnapshotFile,
};
use super::gpu::{
    cpu_gpu, cpu_worker_id, fallback_gpu, gpu_worker_id, parse_nvidia_smi_gpus, visible_gpu_ids,
    worker_capabilities_with_utility,
};
use super::media_jobs::{
    candidate_people, concat_file_contents, crossfade_duration, output_dimensions, plan_segments,
    run_ffmpeg,
};
use super::model_jobs::{
    check_downloaded_model_family, finalize_converted_dir, DownloadFamilyCheck,
};
use super::supervisor::{
    auto_worker_specs, child_environment, restart_exited_children_with_spawner,
    utility_worker_specs, SupervisedChild, WorkerSpec,
};
use super::{
    allow_pattern_matches, bounded_tail, cleanup_uploaded_import_source, copy_lora_source,
    fresh_asset_id, import_lora_source_path, now_rfc3339, resolve_model_convert_output,
    resolve_model_import_target, safe_download_dir, safe_project_path, value_f64,
    write_model_install_marker, JsonObject, Settings, WorkerError, DEFAULT_MAX_LORA_URL_BYTES,
    DEFAULT_MAX_MODEL_URL_BYTES, DEFAULT_TRANSITION_DURATION_SECONDS, INSTALL_MARKER,
};

fn write_safetensors_with_keys(path: &std::path::Path, keys: &[String]) {
    // Minimal valid safetensors: 8-byte little-endian header length + JSON header.
    // The family detector only reads the header, so empty tensor slices are fine.
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
    for key in keys {
        header.insert(
            key.clone(),
            json!({"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}),
        );
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
    let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buffer.extend_from_slice(&header_bytes);
    std::fs::write(path, buffer).expect("write safetensors");
}

fn wan_video_safetensors_keys() -> Vec<String> {
    // Mirrors the Wan2.2 architecture signature the family detector keys on.
    let mut keys = Vec::new();
    for block in 0..30 {
        for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
            keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
            keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
        }
    }
    keys
}

#[test]
fn download_family_check_proceeds_when_no_weights_to_detect() {
    // A curated catalog download with no detectable signal (no safetensors yet, or
    // an inconclusive header) is trusted — the guard must never block a legitimate
    // download, whether or not a family was declared.
    let dir = tempdir().expect("tempdir creates");
    assert!(matches!(
        check_downloaded_model_family(Some("z-image".to_owned()), dir.path()),
        DownloadFamilyCheck::Proceed
    ));
    assert!(matches!(
        check_downloaded_model_family(None, dir.path()),
        DownloadFamilyCheck::Proceed
    ));
}

#[test]
fn download_family_check_flags_confident_mismatch() {
    // Weights that confidently detect as one family while the catalog declared
    // another are rejected (parity with model import).
    let dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &dir.path().join("model.safetensors"),
        &wan_video_safetensors_keys(),
    );
    match check_downloaded_model_family(Some("z-image".to_owned()), dir.path()) {
        DownloadFamilyCheck::Mismatch(mismatch) => {
            assert_eq!(mismatch.supplied, "z-image");
            assert_eq!(mismatch.detected, "wan-video");
        }
        other => panic!("expected a family mismatch, got {other:?}"),
    }
}

#[test]
fn download_family_check_proceeds_when_detection_matches_catalog() {
    let dir = tempdir().expect("tempdir creates");
    write_safetensors_with_keys(
        &dir.path().join("model.safetensors"),
        &wan_video_safetensors_keys(),
    );
    assert!(matches!(
        check_downloaded_model_family(Some("wan-video".to_owned()), dir.path()),
        DownloadFamilyCheck::Proceed
    ));
}

#[tokio::test]
async fn finalize_converted_dir_promotes_atomically_and_replaces_stale() {
    let temp = tempdir().expect("tempdir creates");
    let root = temp.path();
    let final_dir = root.join("mlx").join("wan_2_2");

    // A completed temp conversion sitting in its sibling staging dir.
    let temp_dir = root.join("mlx").join(".wan_2_2.converting-job1");
    std::fs::create_dir_all(&temp_dir).expect("temp dir");
    std::fs::write(temp_dir.join("config.json"), "{}").expect("config");
    std::fs::write(temp_dir.join("model.safetensors"), b"weights").expect("weights");

    // The canonical dir only appears after finalize, so a partial conversion can
    // never be picked up as a ready model.
    assert!(!final_dir.exists());
    finalize_converted_dir(&temp_dir, &final_dir)
        .await
        .expect("finalize");
    assert!(final_dir.join("config.json").is_file());
    assert!(final_dir.join("model.safetensors").is_file());
    assert!(!temp_dir.exists());

    // A re-conversion replaces a stale final dir wholesale.
    let stale_marker = final_dir.join("stale.txt");
    std::fs::write(&stale_marker, "old").expect("stale");
    let temp_dir2 = root.join("mlx").join(".wan_2_2.converting-job2");
    std::fs::create_dir_all(&temp_dir2).expect("temp dir 2");
    std::fs::write(temp_dir2.join("config.json"), "{}").expect("config 2");
    finalize_converted_dir(&temp_dir2, &final_dir)
        .await
        .expect("finalize 2");
    assert!(final_dir.join("config.json").is_file());
    assert!(!stale_marker.exists());
    assert!(!temp_dir2.exists());
}

#[test]
fn download_progress_payload_matches_python_shape() {
    let payload = download_progress_payload(
        "owner/model",
        512 * 1024 * 1024,
        Some(1024 * 1024 * 1024),
        0,
        Duration::from_secs(2),
    );

    assert_eq!(payload.status.as_str(), "downloading");
    assert_eq!(payload.stage.as_str(), "downloading");
    assert_eq!(payload.progress.as_f64(), Some(0.525));
    assert!(payload.message.contains("512.0 MB of 1.0 GB"));
    assert!(payload.eta_seconds.is_some());
}

#[test]
fn pattern_filtering_and_download_dir_match_python_behavior() {
    assert!(allow_pattern_matches(
        "nested/model.safetensors",
        &["*.safetensors".to_owned()]
    ));
    assert!(!allow_pattern_matches(
        "nested/model.ckpt",
        &["*.safetensors".to_owned()]
    ));
    assert_eq!(safe_download_dir("owner/model name"), "owner__model__name");
    assert_eq!(safe_download_dir("///"), "download");
}

#[test]
fn huggingface_cache_paths_follow_hub_layout() {
    let root = tempdir().expect("temp dir creates");
    let path = super::huggingface_repo_cache_path(root.path(), "owner/model-name")
        .expect("cache path resolves");

    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("models--owner--model-name")
    );
}

#[test]
fn repo_slug_functions_match_cross_language_contract() {
    // story 1667: these repo->dir slug ops are duplicated in the Python
    // worker and the Rust API; repo_slugs.json is the shared contract that
    // pins them byte-for-byte across languages.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rust_migration_contracts/repo_slugs.json");
    let contract: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read repo_slugs.json"))
            .expect("parse repo_slugs.json");
    let cases = contract["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "repo_slugs fixture has no cases");
    for case in cases {
        let repo = case["repo"].as_str().expect("repo string");
        assert_eq!(
            super::safe_download_dir(repo),
            case["safeDownloadDir"].as_str().expect("safeDownloadDir"),
            "safe_download_dir drift for {repo:?}"
        );
        assert_eq!(
            super::safe_repo_dir_name(repo).as_deref(),
            case["safeRepoDirName"].as_str(),
            "safe_repo_dir_name drift for {repo:?}"
        );
    }
}

#[test]
fn nvidia_smi_parsing_and_visible_device_filtering_match_python_worker() {
    let gpus = parse_nvidia_smi_gpus(
        "0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 4096, 93791, 12\n\
             1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 8192, 89695, 25\n",
    );

    assert_eq!(
        gpus.iter().map(|gpu| gpu.id.as_str()).collect::<Vec<_>>(),
        ["0", "1"]
    );
    assert_eq!(
        gpus[0].name,
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97887 MB)"
    );
    assert!(gpus[1]
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "nvidia"));
    assert!(gpus[1]
        .capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert_eq!(
        gpus[0].utilization,
        Some(WorkerUtilizationSnapshot {
            memory_total_mb: Some(97887),
            memory_used_mb: Some(4096),
            memory_free_mb: Some(93791),
            gpu_load_percent: Some(12.0),
        })
    );

    assert_eq!(visible_gpu_ids(None), None);
    assert_eq!(visible_gpu_ids(Some("all")), None);
    assert_eq!(visible_gpu_ids(Some("none")), Some(Vec::new()));
    assert_eq!(
        visible_gpu_ids(Some("0, GPU-abcd")),
        Some(vec!["0".to_owned(), "GPU-abcd".to_owned()])
    );
}

#[test]
fn auto_worker_ids_and_child_environment_match_python_supervisor() {
    assert_eq!(gpu_worker_id("worker-gpu-auto-0", "0"), "worker-gpu-auto-0");
    assert_eq!(gpu_worker_id("worker-gpu-auto-0", "1"), "worker-gpu-auto-1");
    assert_eq!(cpu_worker_id("worker-gpu-auto-0"), "worker-gpu-auto-cpu");

    let gpus = vec![fallback_gpu("0"), fallback_gpu("1")];
    let specs = auto_worker_specs("worker-gpu-auto-0", &gpus);
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        [
            "worker-gpu-auto-0",
            "worker-gpu-auto-1",
            "worker-gpu-auto-cpu"
        ]
    );
    assert_eq!(
        specs
            .iter()
            .map(|spec| spec.gpu_id.as_str())
            .collect::<Vec<_>>(),
        ["0", "1", "cpu"]
    );

    let gpu_env = child_environment(&WorkerSpec {
        worker_id: "worker-gpu-auto-1".to_owned(),
        gpu_id: "1".to_owned(),
    });
    assert_eq!(gpu_env["SCENEWORKS_UTILITY_JOBS"], "0");
    assert_eq!(gpu_env["CUDA_VISIBLE_DEVICES"], "1");

    let cpu_env = child_environment(&WorkerSpec {
        worker_id: "worker-gpu-auto-cpu".to_owned(),
        gpu_id: "cpu".to_owned(),
    });
    assert_eq!(cpu_env["SCENEWORKS_UTILITY_JOBS"], "1");
    assert_eq!(cpu_env["CUDA_VISIBLE_DEVICES"], "");
}

#[test]
fn utility_worker_specs_scale_to_requested_count() {
    let single = utility_worker_specs("rust-utility-worker-0", 1);
    assert_eq!(
        single
            .iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        ["rust-utility-worker-cpu"]
    );

    let pool = utility_worker_specs("rust-utility-worker-0", 4);
    assert_eq!(
        pool.iter()
            .map(|spec| spec.worker_id.as_str())
            .collect::<Vec<_>>(),
        [
            "rust-utility-worker-cpu",
            "rust-utility-worker-cpu-1",
            "rust-utility-worker-cpu-2",
            "rust-utility-worker-cpu-3",
        ]
    );
    assert!(pool.iter().all(|spec| spec.gpu_id == "cpu"));

    // A count of 0 must still yield a single worker rather than an empty pool.
    assert_eq!(utility_worker_specs("rust-utility-worker-0", 0).len(), 1);
}

#[test]
fn rust_cpu_capabilities_do_not_claim_gpu_generation_jobs() {
    let cpu_capabilities = worker_capabilities_with_utility(&cpu_gpu(), true);

    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "model_download"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "timeline_export"));
    // The CPU utility worker advertises only the procedural *preview*
    // capabilities; real detection/tracking route to the Python GPU worker.
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect_preview"));
    assert!(cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track_preview"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_detect"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "person_track"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
    assert!(!cpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "video_generate"));

    let gpu_capabilities = worker_capabilities_with_utility(&fallback_gpu("0"), false);
    assert!(gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "gpu"));
    assert!(gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "placeholder"));
    assert!(!gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "model_download"));
    assert!(!gpu_capabilities
        .iter()
        .any(|capability| capability.as_str() == "image_generate"));
}

#[tokio::test]
async fn supervisor_restarts_exited_children_with_backoff_state() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let spec = WorkerSpec {
        worker_id: "worker-gpu-auto-0".to_owned(),
        gpu_id: "0".to_owned(),
    };
    let mut exited = spawn_exit_child();
    for _ in 0..20 {
        if exited.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let mut children = HashMap::from([(
        spec.worker_id.clone(),
        SupervisedChild {
            spec,
            process: exited,
            restart_attempt: 0,
        },
    )]);
    let mut spawns = 0_u32;

    restart_exited_children_with_spawner(&settings, &mut children, |_settings, _spec| {
        spawns += 1;
        Ok(spawn_sleep_child())
    })
    .await
    .expect("child restarts");

    assert_eq!(spawns, 1);
    let child = children
        .get_mut("worker-gpu-auto-0")
        .expect("restarted child is tracked");
    assert_eq!(child.restart_attempt, 1);
    assert!(child
        .process
        .try_wait()
        .expect("child status checks")
        .is_none());
    let _ = child.process.start_kill();
    let _ = child.process.wait().await;
}

#[tokio::test]
async fn writes_model_install_marker_with_expected_keys() {
    let temp = tempdir().expect("tempdir creates");
    let mut payload = serde_json::Map::new();
    payload.insert("modelId".to_owned(), json!("base-model"));
    payload.insert("modelName".to_owned(), json!("Base Model"));

    write_model_install_marker(temp.path(), &payload, "owner/model", "job-1")
        .await
        .expect("marker writes");

    let marker_path = temp.path().join(INSTALL_MARKER);
    let marker: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(marker_path).await.unwrap()).unwrap();
    assert_eq!(marker["repo"], "owner/model");
    assert_eq!(marker["modelId"], "base-model");
    assert_eq!(marker["modelName"], "Base Model");
    assert_eq!(marker["jobId"], "job-1");
    assert!(marker["completedAt"].as_str().is_some());
}

#[tokio::test]
async fn lora_file_and_directory_import_preserve_copy_semantics() {
    let temp = tempdir().expect("tempdir creates");
    let source_file = temp.path().join("mira.safetensors");
    tokio::fs::write(&source_file, b"lora").await.unwrap();
    let file_target = temp.path().join("file-target");

    copy_lora_source(&source_file, &file_target).await.unwrap();

    assert_eq!(
        tokio::fs::read(file_target.join("mira.safetensors"))
            .await
            .unwrap(),
        b"lora"
    );

    let source_dir = temp.path().join("source-dir");
    tokio::fs::create_dir_all(source_dir.join("nested"))
        .await
        .unwrap();
    tokio::fs::write(source_dir.join("nested/adapter.safetensors"), b"adapter")
        .await
        .unwrap();
    let dir_target = temp.path().join("dir-target");

    copy_lora_source(&source_dir, &dir_target).await.unwrap();

    assert_eq!(
        tokio::fs::read(dir_target.join("nested/adapter.safetensors"))
            .await
            .unwrap(),
        b"adapter"
    );
}

#[tokio::test]
async fn uploaded_lora_source_cleanup_removes_staged_file_and_parent() {
    let temp = tempdir().expect("tempdir creates");
    let upload_dir = temp.path().join("upload-1");
    tokio::fs::create_dir_all(&upload_dir).await.unwrap();
    let source_file = upload_dir.join("detail.safetensors");
    tokio::fs::write(&source_file, b"lora").await.unwrap();
    let mut payload = serde_json::Map::new();
    payload.insert(
        "sourcePath".to_owned(),
        json!(source_file.display().to_string()),
    );
    payload.insert("uploadedSourcePath".to_owned(), json!(true));

    cleanup_uploaded_import_source(&payload).await.unwrap();

    assert!(!source_file.exists());
    assert!(!upload_dir.exists());
}

#[tokio::test]
async fn uploaded_lora_file_import_prefers_move_over_copy() {
    let temp = tempdir().expect("tempdir creates");
    let source_file = temp.path().join("uploaded.safetensors");
    tokio::fs::write(&source_file, b"lora").await.unwrap();
    let target_dir = temp.path().join("target");

    import_lora_source_path(&source_file, &target_dir, true)
        .await
        .unwrap();

    assert!(!source_file.exists());
    assert_eq!(
        tokio::fs::read(target_dir.join("uploaded.safetensors"))
            .await
            .unwrap(),
        b"lora"
    );
}

#[tokio::test]
async fn lora_url_import_downloads_to_named_file() {
    let temp = tempdir().expect("tempdir creates");
    let source_url = spawn_binary_stub(b"url-lora".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("url-target");

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
        },
        &format!("{source_url}/loras/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("url LoRA downloads");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"url-lora"
    );
}

#[tokio::test]
async fn lora_url_import_skips_existing_matching_file() {
    let temp = tempdir().expect("tempdir creates");
    let source_url = spawn_binary_stub(b"new-lora".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("url-target");
    tokio::fs::create_dir_all(&target_dir).await.unwrap();
    tokio::fs::write(target_dir.join("style.safetensors"), b"old-lora")
        .await
        .unwrap();

    download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
        },
        &format!("{source_url}/loras/style.safetensors"),
        &target_dir,
    )
    .await
    .expect("existing LoRA is accepted");

    assert_eq!(
        tokio::fs::read(target_dir.join("style.safetensors"))
            .await
            .unwrap(),
        b"old-lora"
    );
}

#[tokio::test]
async fn download_snapshot_rejects_truncated_file() {
    let temp = tempdir().expect("tempdir creates");
    // The stub serves 4 bytes, but the snapshot claims the shard is 64 —
    // a truncated transfer that must not be accepted as complete.
    let base_url = spawn_binary_stub(b"trun".to_vec()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();
    let target_dir = temp.path().join("model");

    let snapshot = HuggingFaceSnapshot {
        files: vec![SnapshotFile {
            path: "shard.safetensors".to_owned(),
            size: Some(64),
            download_url: format!("{base_url}/owner/model/resolve/main/shard.safetensors"),
        }],
    };
    let mut progress = DownloadProgress::new(
        "owner/model",
        0,
        snapshot.total_bytes(),
        Duration::from_secs(3600),
    );

    let error = download_snapshot(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
        },
        &target_dir,
        &snapshot,
        &mut progress,
    )
    .await
    .expect_err("truncated shard is rejected");

    assert!(error.to_string().contains("expected"));
    // The partial file is removed so a retry re-downloads from scratch.
    assert!(!target_dir.join("shard.safetensors").exists());
}

#[tokio::test]
async fn lora_url_import_rejects_failed_and_oversized_downloads() {
    let temp = tempdir().expect("tempdir creates");
    let missing_url =
        spawn_binary_stub_with_options(b"missing".to_vec(), AxumStatusCode::NOT_FOUND, false).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = missing_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
        },
        &format!("{missing_url}/loras/missing.safetensors"),
        &temp.path().join("missing-target"),
    )
    .await
    .expect_err("failed URL returns an error");
    assert!(error.to_string().contains("404"));

    let large_url = spawn_binary_stub(b"too-large".to_vec()).await;
    settings.api_url = large_url.clone();
    settings.max_lora_url_bytes = 4;
    let api = ApiClient::new(&settings);
    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "canceled",
        },
        &format!("{large_url}/loras/large.safetensors"),
        &temp.path().join("large-target"),
    )
    .await
    .expect_err("oversized URL returns an error");
    assert!(error.to_string().contains("exceeds"));
}

#[tokio::test]
async fn lora_url_import_honors_midstream_cancel() {
    let temp = tempdir().expect("tempdir creates");
    let source_url =
        spawn_binary_stub_with_options(b"url-lora".to_vec(), AxumStatusCode::OK, true).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = source_url.clone();
    let api = ApiClient::new(&settings);
    let client = reqwest::Client::new();

    let error = download_lora_source_url(
        &DownloadContext {
            api: &api,
            client: &client,
            settings: &settings,
            job_id: "job-1",
            cancel_message: "LoRA import canceled by user.",
        },
        &format!("{source_url}/loras/style.safetensors"),
        &temp.path().join("cancel-target"),
    )
    .await
    .expect_err("cancel request interrupts the URL import");

    assert!(matches!(error, WorkerError::Canceled(_)));
}

#[test]
fn now_matches_python_second_precision() {
    let value = now_rfc3339();

    assert!(value.ends_with('Z'));
    assert!(!value.trim_end_matches('Z').contains('.'));
}

#[test]
fn ffmpeg_helper_shapes_match_python_timeline_exporter() {
    assert_eq!(output_dimensions("16:9", 720), (1280, 720));
    assert_eq!(output_dimensions("9:16", 720), (720, 1280));
    assert_eq!(output_dimensions("1:1", 721), (722, 722));

    let concat = concat_file_contents(
        [
            PathBuf::from(r"C:\renders\clip one's.mp4"),
            PathBuf::from("nested/two.mp4"),
        ]
        .iter(),
    );
    assert!(concat.contains("C:/renders/clip one'\\''s.mp4"));
    assert!(concat.contains("file 'nested/two.mp4'"));

    let asset_id = fresh_asset_id();
    assert!(asset_id.starts_with("asset_"));
    assert_eq!(asset_id.len(), "asset_".len() + 32);
    assert!(asset_id["asset_".len()..]
        .chars()
        .all(|character| character.is_ascii_hexdigit()));
}

#[test]
fn plan_segments_inserts_gaps_and_totals_duration() {
    let items = vec![
        json!({"assetId": "a", "timelineStart": 1.0, "timelineEnd": 3.0}),
        json!({"assetId": "b", "timelineStart": 3.0, "timelineEnd": 5.0}),
        json!({"assetId": "c", "timelineStart": 6.5, "timelineEnd": 8.0}),
    ];

    let (plan, duration) = plan_segments(&items).expect("plan succeeds");

    assert_eq!(plan.len(), 3);
    // Leading hole before the first item becomes a black gap.
    assert_eq!(plan[0].leading_gap, Some(1.0));
    // Abutting items leave no gap.
    assert_eq!(plan[1].leading_gap, None);
    // Interior hole between items becomes a gap of the missing span.
    assert_eq!(plan[2].leading_gap, Some(1.5));
    // Total duration is the running max of item ends.
    assert_eq!(duration, 8.0);
}

#[test]
fn plan_segments_carries_item_transitions() {
    let items = vec![
        json!({
            "assetId": "a",
            "timelineStart": 0.0,
            "timelineEnd": 2.0,
            "transitionIn": {"type": "crossfade", "duration": 0.8}
        }),
        json!({"assetId": "b", "timelineStart": 2.0, "timelineEnd": 4.0}),
    ];

    let (plan, _) = plan_segments(&items).expect("plan succeeds");

    assert_eq!(plan[0].transition.as_deref(), Some("crossfade"));
    assert_eq!(plan[0].transition_duration, 0.8);
    // Missing transitionIn falls back to the default transition duration.
    assert_eq!(plan[1].transition, None);
    assert_eq!(
        plan[1].transition_duration,
        DEFAULT_TRANSITION_DURATION_SECONDS
    );
}

#[test]
fn plan_segments_rejects_nonpositive_item_span() {
    let items = vec![json!({"assetId": "a", "timelineStart": 2.0, "timelineEnd": 2.0})];

    let error = plan_segments(&items).expect_err("zero-length span rejects");

    assert!(matches!(error, WorkerError::InvalidPayload(_)));
    assert!(error.to_string().contains("timelineEnd must be greater"));
}

#[test]
fn person_detection_jitter_uses_python_sha256_bytes() {
    let detections = candidate_people(1280, 720, "asset_source_clip", 1.25);

    assert_eq!(detections[0]["box"]["x"].as_f64(), Some(0.338));
    assert_eq!(detections[1]["box"]["x"].as_f64(), Some(0.579));
    assert_eq!(detections[2]["box"]["x"].as_f64(), Some(0.134));
}

#[test]
fn missing_crossfade_duration_defaults_to_python_mux_duration() {
    let missing = json!(null);
    assert_eq!(
        value_f64(&missing, DEFAULT_TRANSITION_DURATION_SECONDS),
        0.5
    );
    assert_eq!(crossfade_duration(0.5), 0.5);
    assert_eq!(crossfade_duration(0.0), 0.1);
    assert_eq!(crossfade_duration(2.0), 1.5);
}

#[test]
fn path_and_error_helpers_are_bounded_and_defensive() {
    let temp = tempdir().expect("tempdir creates");
    let error = safe_project_path(temp.path(), "").expect_err("empty relative path rejects");
    assert!(error
        .to_string()
        .contains("Project-relative path is required"));

    let noisy = (0..100)
        .map(|index| format!("line {index} caf\u{e9}"))
        .collect::<Vec<_>>()
        .join("\n");
    let tail = bounded_tail(&noisy, 10, 37);

    assert!(tail.contains("caf\u{e9}"));
    assert!(!tail.contains("line 1 "));
}

#[test]
fn model_destinations_are_constrained_to_data_models() {
    let temp = tempdir().expect("tempdir creates");
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.data_dir = temp.path().to_path_buf();
    let models_root = super::normalize_absolute_path(&temp.path().join("models"))
        .expect("models root normalizes");
    let fallback = temp.path().join("models").join("fallback");

    // model_download/model_import: a targetDir under data/models is accepted.
    let mut payload = JsonObject::new();
    payload.insert(
        "targetDir".to_owned(),
        Value::String(
            temp.path()
                .join("models")
                .join("z_image_turbo")
                .display()
                .to_string(),
        ),
    );
    let resolved = resolve_model_import_target(&settings, &payload, fallback.clone())
        .expect("destination under data/models is accepted");
    assert!(resolved.starts_with(&models_root));

    // No targetDir falls back to the supplied (contained) default.
    let resolved_fallback =
        resolve_model_import_target(&settings, &JsonObject::new(), fallback.clone())
            .expect("fallback under data/models is accepted");
    assert!(resolved_fallback.starts_with(&models_root));

    // A targetDir outside data/models is rejected (arbitrary write blocked).
    let mut escape = JsonObject::new();
    escape.insert(
        "targetDir".to_owned(),
        Value::String(
            temp.path()
                .join("ssh")
                .join("authorized_keys")
                .display()
                .to_string(),
        ),
    );
    let error = resolve_model_import_target(&settings, &escape, fallback)
        .expect_err("destination outside data/models is rejected");
    assert!(error.to_string().contains("data/models"));

    // model_convert: outputDir under data/models is accepted, traversal is rejected.
    let ok = resolve_model_convert_output(
        &settings,
        &temp
            .path()
            .join("models")
            .join("mlx")
            .join("wan")
            .display()
            .to_string(),
    )
    .expect("convert output under data/models is accepted");
    assert!(ok.starts_with(&models_root));

    let traversal = temp
        .path()
        .join("models")
        .join("..")
        .join("escape")
        .display()
        .to_string();
    let convert_error = resolve_model_convert_output(&settings, &traversal)
        .expect_err("convert output escaping data/models is rejected");
    assert!(convert_error.to_string().contains("data/models"));
}

#[tokio::test]
async fn ffmpeg_runner_surfaces_bounded_stderr_from_failing_process() {
    let args = if cfg!(windows) {
        let command = (1..=30)
            .map(|index| format!("echo ffmpeg-line-{index} 1>&2"))
            .collect::<Vec<_>>()
            .join(" & ");
        vec![
            "cmd".to_owned(),
            "/C".to_owned(),
            format!("{command} & exit /B 7"),
        ]
    } else {
        vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "for i in $(seq 1 30); do echo ffmpeg-line-$i >&2; done; exit 7".to_owned(),
        ]
    };

    let error = run_ffmpeg(args, None)
        .await
        .expect_err("non-zero process returns an error");

    match error {
        WorkerError::InvalidPayload(message) => {
            assert!(message.contains("ffmpeg-line-30"));
            assert!(!message.contains("ffmpeg-line-1"));
            assert!(message.len() <= 2000);
        }
        other => panic!("expected InvalidPayload, got {other:?}"),
    }
}

#[tokio::test]
async fn huggingface_snapshot_resolve_accepts_tree_and_sibling_shapes_with_auth() {
    let array_url = spawn_hf_stub(
        json!([
            { "type": "file", "path": "nested/model.safetensors", "size": 7 },
            { "type": "file", "path": "nested/model.ckpt", "size": 9 },
            { "type": "directory", "path": "nested" }
        ]),
        Some("hf_test"),
    )
    .await;
    let client = reqwest::Client::new();
    let array_settings = test_settings(array_url, Some("hf_test"));

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &array_settings,
        "owner/model",
        "main",
        &["*.safetensors".to_owned()],
    )
    .await
    .expect("tree snapshot resolves");

    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "nested/model.safetensors");
    assert_eq!(snapshot.total_bytes(), Some(7));

    let siblings_url = spawn_hf_stub(
        json!({
            "siblings": [
                { "rfilename": "adapter.safetensors", "size": "5" }
            ]
        }),
        None,
    )
    .await;
    let siblings_settings = test_settings(siblings_url, None);

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &siblings_settings,
        "owner/lora",
        "main",
        &["*.safetensors".to_owned()],
    )
    .await
    .expect("siblings snapshot resolves");

    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "adapter.safetensors");
    assert_eq!(snapshot.total_bytes(), Some(5));
}

#[derive(Clone)]
struct HfStubState {
    payload: serde_json::Value,
    token: Option<String>,
}

async fn spawn_hf_stub(payload: serde_json::Value, token: Option<&str>) -> String {
    let state = HfStubState {
        payload,
        token: token.map(str::to_owned),
    };
    let app = Router::new()
        .route("/api/models/:owner/:repo/tree/:revision", get(hf_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn hf_stub(State(state): State<HfStubState>, headers: HeaderMap) -> Response {
    if let Some(token) = &state.token {
        let expected = format!("Bearer {token}");
        let authorized = headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            == Some(expected.as_str());
        if !authorized {
            return (
                AxumStatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing token" })),
            )
                .into_response();
        }
    }
    Json(state.payload).into_response()
}

#[derive(Clone)]
struct BinaryStubState {
    bytes: Vec<u8>,
    status: AxumStatusCode,
    cancel_requested: bool,
}

async fn spawn_binary_stub(bytes: Vec<u8>) -> String {
    spawn_binary_stub_with_options(bytes, AxumStatusCode::OK, false).await
}

async fn spawn_binary_stub_with_options(
    bytes: Vec<u8>,
    status: AxumStatusCode,
    cancel_requested: bool,
) -> String {
    let state = BinaryStubState {
        bytes,
        status,
        cancel_requested,
    };
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_stub))
        .route("/api/v1/jobs/:job_id/progress", post(progress_stub))
        .route("/*path", get(binary_stub).head(binary_head_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn binary_stub(State(state): State<BinaryStubState>, headers: HeaderMap) -> Response {
    let length = state.bytes.len();
    if headers
        .get(axum::http::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == format!("bytes={length}-"))
    {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = AxumStatusCode::RANGE_NOT_SATISFIABLE;
        response.headers_mut().insert(
            axum::http::header::CONTENT_RANGE,
            axum::http::HeaderValue::from_str(&format!("bytes */{length}"))
                .expect("content range header"),
        );
        return response;
    }
    let mut response = state.bytes.into_response();
    *response.status_mut() = state.status;
    response.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&length.to_string()).expect("content length header"),
    );
    response
}

async fn binary_head_stub(State(state): State<BinaryStubState>) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = state.status;
    response.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&state.bytes.len().to_string())
            .expect("content length header"),
    );
    response
}

async fn job_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
}

async fn progress_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
}

fn job_snapshot_json(job_id: &str, cancel_requested: bool) -> Value {
    json!({
        "id": job_id,
        "type": "lora_import",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.1,
        "stage": "importing",
        "message": "running",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": cancel_requested,
        "createdAt": "2026-05-18T00:00:00Z",
        "updatedAt": "2026-05-18T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    })
}

fn test_settings(huggingface_base_url: String, huggingface_token: Option<&str>) -> Settings {
    Settings {
        api_url: "http://127.0.0.1:8000".to_owned(),
        access_token: None,
        data_dir: PathBuf::from("data"),
        config_dir: PathBuf::from("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "cpu".to_owned(),
        is_child_worker: true,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 1,
        huggingface_base_url,
        huggingface_token: huggingface_token.map(str::to_owned),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        allow_private_lora_urls: true,
        utility_workers: 1,
    }
}

fn spawn_exit_child() -> tokio::process::Child {
    let mut command = if cfg!(windows) {
        let mut command = tokio::process::Command::new("cmd");
        command.args(["/C", "exit /B 0"]);
        command
    } else {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "exit 0"]);
        command
    };
    command
        .stdout(StdStdio::null())
        .stderr(StdStdio::null())
        .spawn()
        .expect("test child starts")
}

fn spawn_sleep_child() -> tokio::process::Child {
    let mut command = if cfg!(windows) {
        let mut command = tokio::process::Command::new("cmd");
        command.args(["/C", "ping -n 30 127.0.0.1 >NUL"]);
        command
    } else {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "sleep 30"]);
        command
    };
    command
        .stdout(StdStdio::null())
        .stderr(StdStdio::null())
        .spawn()
        .expect("test child starts")
}
