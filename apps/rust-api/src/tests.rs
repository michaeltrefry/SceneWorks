use super::auth::requires_token;
use super::events::{EventHub, EventMessage};
use super::training::{insufficient_disk_space, resolve_base_model_path};
use super::workers::person_readiness_from_workers;
use super::{
    create_app, create_app_with_state, huggingface_repo_cache_path, inject_converted_model_path,
    inprocess_worker_gpu_id, lora_artifact_paths, merge_model_manifest_entry, mlx_catalog_status,
    open_bind_override_enabled, safe_download_dir, serialize_job_lora, should_warn_open_bind,
    strip_jsonc_comments, sweep_stale_asset_uploads_before, sweep_stale_lora_uploads_before,
    Settings, WorkerCapability, WorkerSnapshot, WorkerStatus, API_MANAGED_MANIFEST_HEADER,
    DEFAULT_API_HOST, EVENT_BUFFER_SIZE, HEARTBEAT_SSE_DATA, HEARTBEAT_SSE_WIRE,
    TEST_MAX_LORA_UPLOAD_BYTES,
};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::time::{Duration, SystemTime};
use tokio_stream::StreamExt;
use tower::ServiceExt;

#[test]
fn default_api_host_is_loopback() {
    // sc-4201 (F-API-1): an out-of-the-box bind must not expose the API to the LAN.
    let ip: std::net::IpAddr = DEFAULT_API_HOST.parse().expect("default host parses");
    assert!(ip.is_loopback(), "default API host must be loopback");
}

#[test]
fn warns_only_on_open_bind_without_token() {
    use std::net::IpAddr;
    let v4 = |s: &str| s.parse::<IpAddr>().unwrap();
    // No token + a wider bind (0.0.0.0 / a concrete LAN IP) → warn.
    assert!(should_warn_open_bind("", v4("0.0.0.0")));
    assert!(should_warn_open_bind("   ", v4("0.0.0.0")));
    assert!(should_warn_open_bind("", v4("192.168.1.5")));
    assert!(should_warn_open_bind("", "::".parse().unwrap()));
    // A token, or a loopback bind, is safe → no warning.
    assert!(!should_warn_open_bind("secret", v4("0.0.0.0")));
    assert!(!should_warn_open_bind("", v4("127.0.0.1")));
    assert!(!should_warn_open_bind("", "::1".parse().unwrap()));
}

#[test]
fn open_bind_override_only_for_explicit_optin() {
    // sc-5720 (API-001): the override that lets an unauthenticated open bind start
    // must require an explicit affirmative value; anything else keeps the refusal.
    for value in ["1", "true", "TRUE", "yes", "YES", " 1 "] {
        assert!(open_bind_override_enabled(value), "{value:?} should opt in");
    }
    for value in ["", "0", "false", "no", "off", "2", "enable"] {
        assert!(
            !open_bind_override_enabled(value),
            "{value:?} must not opt in"
        );
    }
}

#[test]
fn serialize_job_lora_carries_network_type_to_payload() {
    // A trained LoKr adapter records networkType (epic 2193); the generation
    // payload must carry it so the worker can route LoKr off the MLX backend
    // without opening the file.
    let lora = json!({
        "id": "char",
        "family": "sdxl",
        "networkType": "lokr",
        "source": { "provider": "training" },
    });
    let payload = serialize_job_lora(&lora, &json!({}), "char");
    assert_eq!(
        payload.get("networkType").and_then(Value::as_str),
        Some("lokr")
    );

    // A plain LoRA without the field stays absent/null (treated as lora downstream).
    let plain = serialize_job_lora(&json!({ "id": "x", "family": "sdxl" }), &json!({}), "x");
    assert!(plain.get("networkType").map(Value::is_null).unwrap_or(true));
}

fn readiness_worker(
    id: &str,
    status: WorkerStatus,
    capabilities: Vec<WorkerCapability>,
) -> WorkerSnapshot {
    WorkerSnapshot {
        id: id.to_owned(),
        gpu_id: "0".to_owned(),
        gpu_name: None,
        status,
        current_job_id: None,
        capabilities,
        loaded_models: Vec::new(),
        utilization: None,
        registered_at: "2026-05-21T00:00:00Z".to_owned(),
        last_seen_at: "2026-05-21T00:00:00Z".to_owned(),
        extra: Default::default(),
    }
}

#[test]
fn person_readiness_reflects_live_worker_capabilities() {
    let workers = vec![
        readiness_worker(
            "gpu",
            WorkerStatus::Idle,
            vec![
                WorkerCapability::PersonDetect,
                WorkerCapability::PersonTrack,
                WorkerCapability::PersonReplace,
            ],
        ),
        readiness_worker(
            "cpu",
            WorkerStatus::Idle,
            vec![
                WorkerCapability::PersonDetectPreview,
                WorkerCapability::PersonTrackPreview,
            ],
        ),
        // Segment capability exists only on an offline worker -> not ready.
        readiness_worker(
            "dead",
            WorkerStatus::Offline,
            vec![WorkerCapability::PersonSegment],
        ),
    ];
    let readiness = person_readiness_from_workers(&workers);
    assert_eq!(readiness["detect"]["ready"], json!(true));
    assert_eq!(readiness["detect"]["capability"], json!("person_detect"));
    assert_eq!(readiness["track"]["ready"], json!(true));
    assert_eq!(readiness["replace"]["ready"], json!(true));
    assert_eq!(readiness["detectPreview"]["ready"], json!(true));
    assert_eq!(readiness["segment"]["ready"], json!(false));
}

#[test]
fn merge_model_manifest_entry_deep_merges_nested_blocks() {
    // The worker reads the merged manifest entry from the job payload now
    // (story 1653). This pins the behavior-preserving deep merge that the
    // worker's former `ltx_model_manifest_entry` performed: a user entry
    // overrides top-level keys, but builtin's siblings inside a nested block
    // (e.g. resources) survive rather than being replaced wholesale.
    let builtin = json!({
        "id": "ltx_2_3",
        "paths": {"model": "data/models/builtin"},
        "resources": {"checkpoint": {"path": "models/checkpoint.safetensors"}},
    });
    let user = json!({
        "id": "ltx_2_3",
        "paths": {"model": "data/models/user"},
        "resources": {"spatialUpscaler": {"path": "models/spatial.safetensors"}},
    });
    let merged = merge_model_manifest_entry(Some(builtin), Some(user));
    assert_eq!(merged["paths"]["model"], json!("data/models/user"));
    assert_eq!(
        merged["resources"]["checkpoint"]["path"],
        json!("models/checkpoint.safetensors")
    );
    assert_eq!(
        merged["resources"]["spatialUpscaler"]["path"],
        json!("models/spatial.safetensors")
    );
}

#[test]
fn merge_model_manifest_entry_handles_single_or_missing_sources() {
    let builtin = json!({"id": "ltx_2_3", "resources": {"checkpoint": {"path": "a"}}});
    assert_eq!(
        merge_model_manifest_entry(Some(builtin.clone()), None),
        builtin
    );
    let user = json!({"id": "ltx_2_3", "name": "user"});
    assert_eq!(merge_model_manifest_entry(None, Some(user.clone())), user);
    assert_eq!(merge_model_manifest_entry(None, None), json!({}));
}

#[test]
fn model_convert_request_parses_optional_mlx_quant_fields() {
    // The convert endpoint accepts optional camelCase quant knobs (sc-1982); the
    // worker reads the same field names off the job payload, so the contract must
    // hold. Absent fields default to None (unquantized bf16 conversion).
    let bare: super::ModelConvertRequest = serde_json::from_value(json!({})).expect("bare body");
    assert_eq!(bare.quantize_bits, None);
    assert_eq!(bare.quantize_group_size, None);

    let quant: super::ModelConvertRequest =
        serde_json::from_value(json!({"quantizeBits": 4, "quantizeGroupSize": 64}))
            .expect("quant body");
    assert_eq!(quant.quantize_bits, Some(4));
    assert_eq!(quant.quantize_group_size, Some(64));
}

#[test]
fn repo_slug_functions_match_cross_language_contract() {
    // story 1667: safe_download_dir is the api-only repo->dir slug op pinned by
    // the shared repo_slugs.json contract. (safe_repo_dir_name moved to
    // sceneworks-core in sc-4279 and is contract-tested there, so it is no longer
    // re-asserted here.)
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rust_migration_contracts/repo_slugs.json");
    let contract: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read repo_slugs.json"))
            .expect("parse repo_slugs.json");
    let cases = contract["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "repo_slugs fixture has no cases");
    for case in cases {
        let repo = case["repo"].as_str().expect("repo string");
        assert_eq!(
            safe_download_dir(repo),
            case["safeDownloadDir"].as_str().expect("safeDownloadDir"),
            "safe_download_dir drift for {repo:?}"
        );
    }
}

fn test_settings(temp_dir: &tempfile::TempDir) -> Settings {
    Settings {
        app_version: "test".to_owned(),
        host: "127.0.0.1".to_owned(),
        port: 0,
        data_dir: temp_dir.path().join("data"),
        config_dir: temp_dir.path().join("config"),
        access_token: String::new(),
        cors_origins: vec![
            "http://localhost:5173".to_owned(),
            "http://127.0.0.1:5173".to_owned(),
        ],
        worker_timeout_seconds: 90,
        jobs_db_path: temp_dir.path().join("jobs.db"),
        run_utility_inprocess: false,
        mlx_required: false,
        mlx_enforce_unsupported: false,
    }
}

#[test]
fn mlx_catalog_status_reports_turnkey_and_conversion_states() {
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    // No `mlx` block -> None.
    let plain = json!({ "id": "z_image_turbo" });
    assert!(mlx_catalog_status(plain.as_object().unwrap(), &data_dir).is_none());

    // Turnkey model, repo not cached -> missing / ready.
    let ltx = json!({
        "id": "ltx_2_3",
        "mlx": { "minMemoryGb": 31, "repo": "SceneWorks/ltx-2.3-mlx" }
    });
    let status = mlx_catalog_status(ltx.as_object().unwrap(), &data_dir).expect("ltx status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "ready");

    // Turnkey model with the repo cached -> installed / ready.
    let repo_dir =
        huggingface_repo_cache_path(&data_dir, "SceneWorks/ltx-2.3-mlx").expect("repo cache path");
    std::fs::create_dir_all(repo_dir.join("snapshots")).expect("create snapshots");
    let status = mlx_catalog_status(ltx.as_object().unwrap(), &data_dir).expect("ltx status");
    assert_eq!(status.install_state, "installed");
    assert_eq!(status.conversion_state, "ready");

    // Conversion model, native source missing -> missing / needs_source.
    let wan5b = json!({
        "id": "wan_2_2",
        "mlx": {
            "minMemoryGb": 45,
            "requiresConversion": true,
            "convertSourceRepo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        }
    });
    let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
    assert_eq!(status.install_state, "missing");
    assert_eq!(status.conversion_state, "needs_source");

    // Native source cached -> missing / needs_conversion.
    let source_dir = huggingface_repo_cache_path(&data_dir, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        .expect("source cache path");
    std::fs::create_dir_all(source_dir.join("snapshots")).expect("create source snapshots");
    let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
    assert_eq!(status.conversion_state, "needs_conversion");

    // Converted MLX dir present -> installed / converted.
    let converted = data_dir.join("models").join("mlx").join("wan_2_2");
    std::fs::create_dir_all(&converted).expect("create converted dir");
    std::fs::write(converted.join("config.json"), "{}").expect("write config");
    let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
    assert_eq!(status.install_state, "installed");
    assert_eq!(status.conversion_state, "converted");
    assert_eq!(status.converted_path.unwrap(), converted);
}

#[test]
fn inject_converted_model_path_populates_modelpath_seam_once_converted() {
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    // A convert-at-install model (e.g. flux2_klein_9b_true_v2) whose local MLX dir
    // does not exist yet: leave `modelPath` unset so the worker reports the absent
    // conversion rather than silently loading the wrong source repo.
    let make_entry = || {
        json!({
            "id": "flux2_klein_9b_true_v2",
            "mlx": {
                "requiresConversion": true,
                "converter": "flux2_klein_diffusers",
                "convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2"
            }
        })
    };
    let mut entry = make_entry();
    inject_converted_model_path(&mut entry, &data_dir);
    assert!(
        entry.get("modelPath").is_none(),
        "modelPath must stay absent until the conversion has produced a local dir"
    );

    // Once the FLUX.2-klein converter has assembled the diffusers dir (marked by
    // model_index.json), `modelPath` is injected so the worker's resolve_weights_dir
    // loads the converted dir instead of falling back to the single-file source repo.
    let converted = data_dir
        .join("models")
        .join("mlx")
        .join("flux2_klein_9b_true_v2");
    std::fs::create_dir_all(&converted).expect("create converted dir");
    std::fs::write(converted.join("model_index.json"), "{}").expect("write model_index");
    let mut entry = make_entry();
    inject_converted_model_path(&mut entry, &data_dir);
    assert_eq!(
        entry.get("modelPath").and_then(Value::as_str),
        Some(converted.display().to_string().as_str()),
    );

    // An explicit manifest `modelPath` is authoritative and never overwritten.
    let mut pinned = make_entry();
    pinned
        .as_object_mut()
        .unwrap()
        .insert("modelPath".to_owned(), json!("/custom/path"));
    inject_converted_model_path(&mut pinned, &data_dir);
    assert_eq!(
        pinned.get("modelPath").and_then(Value::as_str),
        Some("/custom/path")
    );

    // A non-conversion model is untouched.
    let mut turnkey = json!({
        "id": "ltx_2_3",
        "mlx": { "repo": "SceneWorks/ltx-2.3-mlx" }
    });
    inject_converted_model_path(&mut turnkey, &data_dir);
    assert!(turnkey.get("modelPath").is_none());

    // FLUX.2-dev (sc-5921) converts to a packed Q4 dir whose top level is SUBDIRS
    // (transformer/ + text_encoder/, each with its own config.json) plus a symlinked
    // model_index.json — there is NO top-level config.json. The catalog's "converted"
    // detection keys on the top-level model_index.json, so the modelPath seam is still
    // injected for dev's subdir layout.
    let dev_entry = || {
        json!({
            "id": "flux2_dev",
            "mlx": {
                "requiresConversion": true,
                "converter": "flux2_dev_quant",
                "convertSourceRepo": "black-forest-labs/FLUX.2-dev"
            }
        })
    };
    let dev_converted = data_dir.join("models").join("mlx").join("flux2_dev");
    std::fs::create_dir_all(dev_converted.join("transformer")).expect("create dev transformer");
    std::fs::write(dev_converted.join("transformer").join("config.json"), "{}")
        .expect("write dev transformer config");
    // No top-level config.json — only the model_index.json marker.
    std::fs::write(dev_converted.join("model_index.json"), "{}").expect("write dev model_index");
    let mut entry = dev_entry();
    inject_converted_model_path(&mut entry, &data_dir);
    assert_eq!(
        entry.get("modelPath").and_then(Value::as_str),
        Some(dev_converted.display().to_string().as_str()),
        "dev's subdir layout is detected via its top-level model_index.json marker"
    );
}

fn write_test_safetensors(path: &std::path::Path) {
    std::fs::write(path, test_safetensors_bytes()).expect("test safetensors writes");
}

fn write_test_safetensors_with_keys(path: &std::path::Path, tensor_keys: &[String]) {
    std::fs::write(path, test_safetensors_bytes_with_keys(tensor_keys))
        .expect("test safetensors writes");
}

fn test_safetensors_bytes() -> Vec<u8> {
    test_safetensors_bytes_with_keys(&[])
}

fn test_safetensors_bytes_with_keys(tensor_keys: &[String]) -> Vec<u8> {
    const TENSOR_DATA_END: u64 = 32768;
    let mut object = serde_json::Map::new();
    object.insert("__metadata__".to_owned(), json!({"format": "pt"}));
    let mut data_end = 0_u64;
    for key in tensor_keys {
        object.insert(
            key.clone(),
            json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, TENSOR_DATA_END]}),
        );
        data_end = data_end.max(TENSOR_DATA_END);
    }
    let header = serde_json::to_vec(&Value::Object(object)).expect("header serializes");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    // The data section must hold every tensor the header declares — otherwise the
    // file is rejected as incomplete (sc-6072). Pad to the largest declared
    // `data_offsets` end; empty-key fixtures keep a few bytes so the file is a
    // non-empty but still complete safetensors.
    let data_len = usize::try_from(data_end.max(12)).expect("data length fits usize");
    bytes.resize(bytes.len() + data_len, 0);
    bytes
}

fn z_image_tensor_keys() -> Vec<String> {
    mm_dit_tensor_keys(24)
}

fn qwen_image_tensor_keys() -> Vec<String> {
    mm_dit_tensor_keys(60)
}

fn mm_dit_tensor_keys(block_count: usize) -> Vec<String> {
    let mut keys = Vec::new();
    for block in 0..block_count {
        for module in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "img_mlp.net.0.proj",
            "txt_mlp.net.0.proj",
        ] {
            keys.push(format!(
                "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
            ));
            keys.push(format!(
                "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
            ));
        }
    }
    keys
}

fn wan_video_tensor_keys() -> Vec<String> {
    let mut keys = Vec::new();
    for block in 0..30 {
        for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
            keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
            keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
        }
    }
    keys
}

async fn request(app: axum::Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    request_with_headers(app, method, uri, body, &[]).await
}

async fn request_with_headers(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let request = builder
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("response returns");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers");
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body parses")
    };
    (status, value)
}

async fn request_raw(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: impl Into<Body>,
    headers: &[(&str, &str)],
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let response = app
        .oneshot(builder.body(body.into()).expect("request builds"))
        .await
        .expect("response returns");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers")
        .to_vec();
    (status, headers, bytes)
}

async fn request_multipart_upload(
    app: axum::Router,
    uri: &str,
    filename: &str,
    content_type: &str,
    bytes: &[u8],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_BOUNDARY";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app,
        "POST",
        uri,
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

async fn request_multipart_lora_upload(
    app: axum::Router,
    fields: &[(&str, &str)],
    filename: &str,
    bytes: &[u8],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_LORA_BOUNDARY";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app,
        "POST",
        "/api/v1/loras/import",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

#[test]
fn stale_lora_upload_sweep_removes_only_upload_dirs_before_cutoff() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let upload_root = temp_dir.path().join("data/cache/lora-uploads");
    let expired = upload_root.join("upload-expired");
    let fresh = upload_root.join("upload-fresh");
    let unrelated = upload_root.join("keep-me");
    std::fs::create_dir_all(&expired).expect("expired dir creates");
    std::fs::create_dir_all(&fresh).expect("fresh dir creates");
    std::fs::create_dir_all(&unrelated).expect("unrelated dir creates");

    let removed = sweep_stale_lora_uploads_before(
        &temp_dir.path().join("data"),
        SystemTime::now() + Duration::from_secs(1),
    )
    .expect("stale uploads sweep");

    assert_eq!(removed, 2);
    assert!(!expired.exists());
    assert!(!fresh.exists());
    assert!(unrelated.exists());
}

#[test]
fn stale_asset_upload_sweep_removes_only_upload_tmp_before_cutoff() {
    // sc-4204 (F-API-6): cache/uploads now has a startup-sweep backstop.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let upload_root = temp_dir.path().join("data/cache/uploads");
    std::fs::create_dir_all(&upload_root).expect("upload root creates");
    let expired = upload_root.join("upload-expired.tmp");
    let fresh = upload_root.join("upload-fresh.tmp");
    let unrelated = upload_root.join("keep-me.txt");
    std::fs::write(&expired, b"x").expect("expired writes");
    std::fs::write(&fresh, b"y").expect("fresh writes");
    std::fs::write(&unrelated, b"z").expect("unrelated writes");

    let removed = sweep_stale_asset_uploads_before(
        &temp_dir.path().join("data"),
        SystemTime::now() + Duration::from_secs(1),
    )
    .expect("asset uploads sweep");

    assert_eq!(removed, 2);
    assert!(!expired.exists());
    assert!(!fresh.exists());
    assert!(unrelated.exists(), "non upload-* files are left alone");
}

#[tokio::test]
async fn import_asset_removes_staged_temp_file_when_a_later_field_errors() {
    // sc-4204 (F-API-6): the `file` field is staged to cache/uploads before a later
    // field is parsed; an invalid `provenance` JSON must not leave an orphan tmp.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    let boundary = "SCENEWORKS_IMPORT_BOUNDARY";
    let mut body = Vec::new();
    // `file` first so it is staged, then an invalid `provenance` that errors.
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(b"\x89PNG\r\n\x1a\n payload bytes");
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"provenance\"\r\n\r\n");
    body.extend_from_slice(b"{not valid json");
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let (status, _, response) = request_raw(
        app,
        "POST",
        "/api/v1/projects/project-1/assets",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(&response).expect("json body parses");
    assert!(value["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("Invalid provenance JSON")));

    // No orphaned temp file should remain under cache/uploads.
    let upload_root = data_dir.join("cache").join("uploads");
    let leaked: Vec<_> = std::fs::read_dir(&upload_root)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("upload-"))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leaked.is_empty(),
        "staged upload temp file leaked on error: {leaked:?}"
    );
}

#[tokio::test]
async fn worker_can_register_claim_and_complete_job_through_http() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_generate"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_generate",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist over hills" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"]["id"], created["id"]);
    assert_eq!(claimed["job"]["status"], "preparing");

    let job_id = created["id"].as_str().expect("job id is string");
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Done",
            "workerId": "worker-1",
            "result": { "assetIds": ["asset-1"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["result"], json!({ "assetIds": ["asset-1"] }));

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["completed"], 1);
    assert_eq!(queue["workers"][0]["status"], "idle");
}

#[tokio::test]
async fn progress_ticks_only_republish_queue_on_status_change() {
    // sc-4203 (F-API-5): a pure progress tick (status unchanged) must not trigger the
    // full queue-summary recompute + queue.updated broadcast; a status transition must.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_generate"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_generate",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    // Move the job into `running` (a transition from `preparing`).
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "running", "stage": "running", "progress": 0.2, "message": "step", "workerId": "worker-1" }),
    )
    .await;

    // Subscribe AFTER the transition so we only observe the next ticks' events.
    let mut events = state.events.subscribe();

    // A pure progress tick (running -> running): job.updated, but NOT queue.updated.
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "running", "stage": "running", "progress": 0.6, "message": "step", "workerId": "worker-1" }),
    )
    .await;
    let tick_events = drain_event_names(&mut events).await;
    assert!(
        tick_events.iter().any(|name| name == "job.updated"),
        "a progress tick still emits job.updated: {tick_events:?}"
    );
    assert!(
        !tick_events.iter().any(|name| name == "queue.updated"),
        "a pure progress tick must not republish the queue: {tick_events:?}"
    );

    // A status transition (running -> completed) republishes the queue.
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "completed", "stage": "completed", "progress": 1, "message": "done", "workerId": "worker-1" }),
    )
    .await;
    let done_events = drain_event_names(&mut events).await;
    assert!(
        done_events.iter().any(|name| name == "queue.updated"),
        "a status transition must republish the queue: {done_events:?}"
    );
}

// Collect the event names currently buffered for a subscriber, stopping after a brief
// quiet period (handlers publish synchronously before the request future resolves, so
// everything is already buffered by the time we drain).
async fn drain_event_names(
    events: &mut tokio_stream::wrappers::ReceiverStream<EventMessage>,
) -> Vec<String> {
    let mut names = Vec::new();
    while let Ok(Some(message)) =
        tokio::time::timeout(Duration::from_millis(150), events.next()).await
    {
        names.push(message.event);
    }
    names
}

#[tokio::test]
async fn canceling_queued_job_finishes_without_worker_acknowledgement() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_generate",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist over hills" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let job_id = created["id"].as_str().expect("job id is string");
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");
    assert_eq!(canceled["stage"], "canceled");
    assert_eq!(canceled["progress"], 1.0);
    assert_eq!(canceled["cancelRequested"], true);
    assert_eq!(canceled["message"], "Canceled before a worker started.");
    assert!(canceled["canceledAt"].is_string());
    assert!(canceled["completedAt"].is_string());
    assert_eq!(canceled["workerId"], Value::Null);

    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["canceled"], 1);
    assert_eq!(queue["counts"]["queued"], 0);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_generate"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, claimed) = request(
        app,
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"], Value::Null);
}

#[tokio::test]
async fn project_and_asset_routes_persist_contract_state() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "My Project" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(created["id"]
        .as_str()
        .is_some_and(|value| value.starts_with("project_")));
    assert!(created["path"]
        .as_str()
        .unwrap()
        .ends_with("my-project.sceneworks"));

    let project_id = created["id"].as_str().expect("project id").to_owned();
    let (status, projects) = request(app.clone(), "GET", "/api/v1/projects", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(projects[0]["id"], project_id);

    let (status, uploaded) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Hero Image.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(uploaded["projectId"], project_id);
    assert_eq!(uploaded["type"], "image");
    assert_eq!(uploaded["status"]["trashed"], false);
    assert!(uploaded["url"]
        .as_str()
        .unwrap()
        .contains("/files/assets/uploads/"));

    let (status, heic_upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Plate.HEIC",
        "application/octet-stream",
        b"heic-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(heic_upload["type"], "image");
    assert_eq!(heic_upload["file"]["mimeType"], "image/heic");

    let asset_id = uploaded["id"].as_str().expect("asset id").to_owned();
    let (status, assets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets?includeRejected=true&includeTrashed=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(assets.as_array().unwrap().len(), 2);

    let (status, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["id"], asset_id);

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}/status"),
        json!({ "favorite": true, "rating": 4, "rejected": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["status"]["favorite"], true);
    assert_eq!(updated["status"]["rating"], 4);
    assert_eq!(updated["status"]["rejected"], true);

    let (status, tagged) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
        json!({ "tags": [" Portrait ", "portrait", "Reference"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tagged["tags"], json!(["portrait", "reference"]));

    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted, json!({ "id": asset_id, "status": "trashed" }));

    let (status, reindex) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/reindex"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reindex["assets"], 2);

    let (status, purged) = request(
        app,
        "DELETE",
        &format!("/api/v1/projects/{project_id}/assets/{asset_id}/purge"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(purged, json!({ "id": asset_id, "status": "purged" }));
}

#[tokio::test]
async fn training_targets_route_returns_builtin_registry() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (status, registry) = request(app, "GET", "/api/v1/training/targets", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(registry["schemaVersion"], 1);
    assert_eq!(registry["targets"][0]["id"], "z_image_turbo_lora");
    assert_eq!(registry["targets"][0]["defaults"]["rank"], 16);
    assert_eq!(
        registry["targets"][0]["defaults"]["advanced"]["qualityPreset"],
        "balanced"
    );
}

#[tokio::test]
async fn training_presets_route_returns_builtin_registry() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (status, registry) = request(app, "GET", "/api/v1/training/presets", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(registry["schemaVersion"], 1);
    assert_eq!(
        registry["presets"][0]["id"],
        "z_image_turbo_lora.character.adamw8bit.balanced"
    );
    assert_eq!(registry["presets"][0]["config"]["steps"], 3000);
    assert_eq!(
        registry["presets"][0]["config"]["advanced"]["sampleSteps"],
        8
    );
    let prodigy = registry["presets"]
        .as_array()
        .expect("preset array")
        .iter()
        .find(|preset| preset["id"] == "z_image_turbo_lora.character.prodigyopt.balanced")
        .expect("prodigy preset");
    assert_eq!(prodigy["config"]["optimizer"], "prodigyopt");
    assert_eq!(prodigy["config"]["learningRate"], 1.0);
}

#[tokio::test]
async fn training_dataset_routes_persist_and_validate_project_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Mira LoRA set",
            "items": [{
                "assetId": asset_id,
                "caption": {
                    "text": "miraStyle portrait",
                    "triggerWords": ["miraStyle"]
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(dataset["projectId"], project_id);
    assert_eq!(dataset["version"], 1);
    assert_eq!(dataset["items"][0]["assetId"], asset_id);
    assert_eq!(dataset["items"][0]["path"], "images/item_0001.png");
    assert_eq!(dataset["items"][0]["caption"]["source"], "manual");
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    assert!(project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("item_0001.png")
        .exists());

    let (status, listed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed[0]["id"], dataset_id);
    assert_eq!(listed[0]["itemCount"], 1);
    // The summary carries a cover thumbnail path (sc-2025) for the dataset selector.
    assert_eq!(
        listed[0]["coverPath"],
        format!("training/datasets/{dataset_id}/images/item_0001.png")
    );

    let reloaded_app = create_app(settings).expect("app reloads");
    let (status, detail) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["items"][0]["caption"]["text"], "miraStyle portrait");

    let (status, updated) = request(
        reloaded_app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        json!({
            "items": [{
                "assetId": asset_id,
                "caption": { "text": "miraStyle close portrait" }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["version"], 2);
    assert_eq!(
        updated["items"][0]["caption"]["text"],
        "miraStyle close portrait"
    );
    let dataset_image_path = project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("item_0001.png");
    assert_eq!(
        std::fs::read(&dataset_image_path).expect("dataset image remains"),
        b"png-bytes"
    );

    let (status, error) = request(
        reloaded_app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        json!({
            "items": [
                { "assetId": asset_id },
                { "assetId": "asset_missing" }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["detail"], "Asset not found");
    assert_eq!(
        std::fs::read(&dataset_image_path).expect("old dataset image survives failed update"),
        b"png-bytes"
    );
    let (status, detail_after_failed_update) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail_after_failed_update["version"], 2);

    let (status, sidecars) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-sidecars"),
        json!({
            "items": [{
                "itemId": "item_0001",
                "caption": {
                    "text": "studio portrait",
                    "triggerWords": ["miraStyle"]
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sidecars["dataset"]["version"], 3);
    assert_eq!(
        sidecars["sidecars"][0]["captionPath"],
        format!("training/datasets/{dataset_id}/images/item_0001.txt")
    );
    assert_eq!(
        std::fs::read_to_string(
            project_path
                .join("training")
                .join("datasets")
                .join(&dataset_id)
                .join("images")
                .join("item_0001.txt")
        )
        .expect("caption sidecar writes"),
        "miraStyle, studio portrait\n"
    );

    let (status, caption_job) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-jobs"),
        json!({
            "recaption": true,
            "requestedGpu": "auto",
            "options": {
                "captionType": "Straightforward",
                "captionLength": "40",
                "extraOptions": ["Include information about lighting."],
                "nameInput": "Mira",
                "temperature": 0.5,
                "topP": 0.8,
                "maxNewTokens": 128
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(caption_job["type"], "training_caption");
    assert_eq!(caption_job["payload"]["captioner"], "joy_caption");
    assert_eq!(
        caption_job["payload"]["modelNameOrPath"],
        "fancyfeast/llama-joycaption-beta-one-hf-llava"
    );
    assert_eq!(caption_job["payload"]["items"][0]["itemId"], "item_0001");
    assert_eq!(
        caption_job["payload"]["items"][0]["triggerWords"],
        json!(["miraStyle"])
    );
    let caption_image_path = caption_job["payload"]["items"][0]["imagePath"]
        .as_str()
        .expect("caption image path");
    assert!(caption_image_path.contains(&dataset_id));
    assert!(caption_image_path.ends_with("item_0001.png"));

    // sc-2025: itemIds targets a single image and recaptions it even though it
    // already has a caption (recaption:false would otherwise skip it).
    let (status, single_item_job) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-jobs"),
        json!({ "recaption": false, "itemIds": ["item_0001"] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let single_items = single_item_job["payload"]["items"]
        .as_array()
        .expect("caption items");
    assert_eq!(single_items.len(), 1);
    assert_eq!(single_items[0]["itemId"], "item_0001");

    let (status, renamed) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-rename"),
        json!({
            "items": [{
                "itemId": "item_0001",
                "newItemId": "item_0007",
                "fileStem": "mira_0007",
                "displayName": "mira_0007.png"
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(renamed["version"], 4);
    assert_eq!(renamed["items"][0]["id"], "item_0007");
    assert_eq!(renamed["items"][0]["path"], "images/mira_0007.png");
    assert_eq!(renamed["items"][0]["displayName"], "mira_0007.png");
    let renamed_image_path = project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("mira_0007.png");
    assert_eq!(
        std::fs::read(&renamed_image_path).expect("renamed dataset image remains"),
        b"png-bytes"
    );
    assert!(!dataset_image_path.exists());
    assert_eq!(
        std::fs::read_to_string(renamed_image_path.with_extension("txt"))
            .expect("caption sidecar follows rename"),
        "miraStyle, studio portrait\n"
    );

    let (status, error) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Bad Path",
            "items": [{ "path": "../outside.png" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Invalid dataset item path");

    let (status, error) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Duplicate Items",
            "items": [
                { "id": "same_item", "assetId": asset_id },
                { "id": "same_item", "assetId": asset_id }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Training dataset item IDs must be unique");

    let (_, other_project) = request(
        reloaded_app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Other Training Project" }),
    )
    .await;
    let other_project_id = other_project["id"].as_str().expect("project id").to_owned();
    let (status, other_asset) = request_multipart_upload(
        reloaded_app.clone(),
        &format!("/api/v1/projects/{other_project_id}/assets"),
        "Other.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let other_asset_id = other_asset["id"].as_str().expect("asset id").to_owned();
    let (status, error) = request(
        reloaded_app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Cross Project",
            "items": [{ "assetId": other_asset_id }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["detail"], "Asset not found");

    let (status, deleted) = request(
        reloaded_app.clone(),
        "DELETE",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted, json!({ "id": dataset_id, "status": "deleted" }));
    let (status, listed_after_delete) = request(
        reloaded_app,
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed_after_delete, json!([]));
}

// sc-2022: a dataset can be associated with a character at create time or via a
// later PATCH, and the association surfaces on the detail body and list summary
// so the Character Studio can scope its dataset list client-side. General
// datasets leave `characterId` null.
#[tokio::test]
async fn training_datasets_associate_with_a_character() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Character Datasets" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    // Created with an explicit character association (the "create from a
    // character's images" path).
    let (status, scoped) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Mira identity set",
            "characterId": "character_mira",
            "items": [{ "assetId": asset_id }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(scoped["characterId"], "character_mira");
    let scoped_id = scoped["id"].as_str().expect("dataset id").to_owned();

    // A general dataset leaves the association null.
    let (status, general) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Style set", "items": [{ "assetId": asset_id }] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(general["characterId"], Value::Null);
    let general_id = general["id"].as_str().expect("dataset id").to_owned();

    // The association round-trips through a reload and the list summary.
    let reloaded_app = create_app(settings).expect("app reloads");
    let (status, listed) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let by_id = |id: &str| -> Value {
        listed
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == id)
            .cloned()
            .expect("dataset present in list")
    };
    assert_eq!(by_id(&scoped_id)["characterId"], "character_mira");
    assert_eq!(by_id(&general_id)["characterId"], Value::Null);

    // PATCHing a general dataset associates it (the import-from-character path,
    // which always saves a full item set alongside the new association).
    let (status, associated) = request(
        reloaded_app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/training/datasets/{general_id}"),
        json!({ "characterId": "character_kelsie", "items": [{ "assetId": asset_id }] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(associated["characterId"], "character_kelsie");

    let (status, relisted) = request(
        reloaded_app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let associated_summary = relisted
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == general_id)
        .cloned()
        .expect("associated dataset present");
    assert_eq!(associated_summary["characterId"], "character_kelsie");
}

#[tokio::test]
async fn training_dataset_uploads_are_dataset_owned_not_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dataset Upload Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, upload) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/training/uploads"),
        "DatasetOnly.PNG",
        "image/png",
        b"dataset-only-png",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(upload["datasetOnly"], true);
    let staged_path = upload["file"]["path"]
        .as_str()
        .expect("staged path")
        .to_owned();
    assert!(staged_path.starts_with("training/uploads/"));

    let (status, listed_assets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed_assets.as_array().expect("asset list").len(), 0);

    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Dataset-owned import",
            "items": [{
                "path": staged_path,
                "displayName": "DatasetOnly.PNG",
                "caption": { "text": "dataset only portrait" }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(dataset["items"][0]["assetId"].is_null());
    assert_eq!(dataset["items"][0]["path"], "images/item_0001.png");
    let dataset_id = dataset["id"].as_str().expect("dataset id");
    assert_eq!(
        std::fs::read(
            project_path
                .join("training")
                .join("datasets")
                .join(dataset_id)
                .join("images")
                .join("item_0001.png")
        )
        .expect("dataset-owned image copied"),
        b"dataset-only-png"
    );

    let (status, listed_assets_after_dataset) = request(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        listed_assets_after_dataset
            .as_array()
            .expect("asset list after dataset")
            .len(),
        0
    );
}

#[tokio::test]
async fn asset_library_scope_excludes_character_studio_outputs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Library Scope Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    // Write a normal Image Studio output and a Character Studio test output
    // directly as sidecars (no explicit origin → derived on the first reindex,
    // which the initial /assets call triggers because the table is empty).
    let image_dir = project_path.join("assets/images");
    for (id, mode) in [
        ("img_studio_1", "text_to_image"),
        ("char_test_1", "character_image"),
    ] {
        std::fs::write(image_dir.join(format!("{id}.png")), b"png-bytes").expect("media");
        std::fs::write(
            image_dir.join(format!("{id}.sceneworks.json")),
            serde_json::to_string_pretty(&json!({
                "id": id,
                "type": "image",
                "displayName": id,
                "createdAt": "2026-05-23T00:00:00Z",
                "file": {"path": format!("assets/images/{id}.png")},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false},
                "recipe": {"mode": mode},
            }))
            .expect("json"),
        )
        .expect("sidecar");
    }

    // Default (all) scope returns both, each tagged with a derived origin.
    let (status, all) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let all = all.as_array().expect("all list");
    assert_eq!(all.len(), 2);
    assert!(all
        .iter()
        .any(|asset| asset["id"] == "char_test_1" && asset["origin"] == "character_studio"));

    // Library scope drops the Character Studio output.
    let (status, library) = request(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/assets?scope=library"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let library = library.as_array().expect("library list");
    assert_eq!(library.len(), 1);
    assert_eq!(library[0]["id"], "img_studio_1");
    assert_eq!(library[0]["origin"], "image_studio");
}

#[tokio::test]
async fn create_training_job_resolves_plan_and_queues_lora_train() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target = registry["targets"][0].clone();
    let target_id = target["id"].as_str().expect("target id").to_owned();
    let config = target["defaults"].clone();

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_train");
    assert_eq!(job["status"], "queued");
    assert_eq!(job["requestedGpu"], "auto");
    assert_eq!(job["projectId"], project_id);
    assert_eq!(job["payload"]["dryRun"], true);

    let job_id = job["id"].as_str().expect("job id").to_owned();
    let plan = &job["payload"]["plan"];
    // The resolved plan is self-referential and fully normalized in Rust.
    assert_eq!(plan["jobId"], job_id);
    assert_eq!(plan["provenance"]["sourceJobId"], job_id);
    assert_eq!(plan["target"]["targetId"], target_id);
    assert_eq!(plan["dataset"]["datasetId"], dataset_id);
    assert_eq!(plan["dataset"]["datasetVersion"], 1);
    assert_eq!(plan["dataset"]["items"].as_array().unwrap().len(), 1);
    let lora_id = plan["output"]["loraId"].as_str().expect("lora id");
    assert!(lora_id.starts_with("lora_"));
    assert_eq!(plan["output"]["fileName"], "aurora_style.safetensors");

    // Item image paths resolve under the dataset root on disk.
    let expected_image = project_path
        .join("training")
        .join("datasets")
        .join(&dataset_id)
        .join("images")
        .join("item_0001.png");
    assert_eq!(
        plan["dataset"]["items"][0]["imagePath"]
            .as_str()
            .expect("image path"),
        expected_image.display().to_string()
    );
    // The default target scope is `project`, so the adapter is written into
    // the project's LoRA store (not the shared data dir).
    assert_eq!(
        plan["output"]["outputDir"].as_str().expect("output dir"),
        project_path
            .join("loras")
            .join(lora_id)
            .display()
            .to_string()
    );
    // The submit-time manifest entry carries provenance for the LoRA that
    // registration will recompute and upsert on completion. The manifest path
    // itself is intentionally NOT persisted in the payload — it is recomputed
    // from trusted inputs at completion so a tampered payload cannot redirect
    // the write.
    assert_eq!(job["payload"]["manifestEntry"]["scope"], "project");
    assert_eq!(job["payload"]["manifestEntry"]["family"], "z-image");
    assert_eq!(
        job["payload"]["manifestEntry"]["source"]["path"],
        format!("loras/{lora_id}")
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["provenance"]["datasetId"],
        dataset_id
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["provenance"]["trainingJobId"],
        job_id
    );
    assert!(job["payload"]["manifestPath"].is_null());

    // The job is queued and visible to the queue/worker surface.
    let (status, queued) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs?status=queued",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queued[0]["id"], job_id);
    assert_eq!(queued[0]["type"], "lora_train");

    let (_, preset_registry) =
        request(app.clone(), "GET", "/api/v1/training/presets", Value::Null).await;
    let prodigy_preset = preset_registry["presets"]
        .as_array()
        .expect("preset array")
        .iter()
        .find(|preset| preset["id"] == "z_image_turbo_lora.character.prodigyopt.balanced")
        .expect("prodigy preset")
        .clone();
    let (status, preset_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "presetId": prodigy_preset["id"],
            "presetVersion": prodigy_preset["version"],
            "config": prodigy_preset["config"],
            "outputName": "Aurora Prodigy",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        preset_job["payload"]["plan"]["provenance"]["presetId"],
        "z_image_turbo_lora.character.prodigyopt.balanced"
    );
    assert_eq!(
        preset_job["payload"]["plan"]["provenance"]["presetName"],
        "Prodigy character (experimental)"
    );
    assert_eq!(
        preset_job["payload"]["plan"]["provenance"]["presetConfigSnapshot"]["learningRate"],
        1.0
    );
    assert_eq!(
        preset_job["payload"]["manifestEntry"]["provenance"]["presetId"],
        "z_image_turbo_lora.character.prodigyopt.balanced"
    );

    let (status, error) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "presetId": "z_image_turbo_lora.character.prodigyopt.balanced",
            "presetVersion": 99,
            "config": prodigy_preset["config"],
            "outputName": "Aurora Prodigy",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
            error["detail"],
            "Training preset 'z_image_turbo_lora.character.prodigyopt.balanced' is version 1, but the request pinned version 99."
        );
}

#[tokio::test]
async fn create_training_job_rejects_unknown_target_and_missing_dataset() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let config = registry["targets"][0]["defaults"].clone();

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": "not_a_target",
            "datasetId": "ds_missing",
            "config": config,
            "outputName": "Aurora"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("Unknown training target"));

    let target_id = registry["targets"][0]["id"].as_str().unwrap().to_owned();
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": "ds_missing",
            "config": registry["targets"][0]["defaults"].clone(),
            "outputName": "Aurora"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["detail"], "Training dataset not found");
}

#[tokio::test]
async fn create_training_job_queues_real_run_when_not_dry_run() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    // A real run requires the base model installed (story 1419 guardrail).
    seed_installed_base_model(&settings.data_dir);
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target = registry["targets"][0].clone();
    let target_id = target["id"].as_str().expect("target id").to_owned();
    let config = target["defaults"].clone();

    // Real execution exists (story 1417): a non-dry-run job resolves the same
    // plan and queues for the worker's Z-Image LoRA kernel.
    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_train");
    assert_eq!(job["status"], "queued");
    assert_eq!(job["payload"]["dryRun"], false);
    // The plan is resolved and embedded just like the dry-run path.
    assert_eq!(job["payload"]["plan"]["planVersion"], 1);
    assert_eq!(job["payload"]["plan"]["target"]["kernel"], "z_image_lora");

    let job_id = job["id"].as_str().expect("job id").to_owned();
    let (status, queued) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs?status=queued",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queued[0]["id"], job_id);
    assert_eq!(queued[0]["type"], "lora_train");
}

/// Seeds the Z-Image-Turbo base model as installed (a managed-download
/// marker) so a real training run clears the missing-model guardrail.
fn seed_installed_base_model(data_dir: &std::path::Path) {
    let model_dir = data_dir
        .join("models")
        .join(safe_download_dir("Tongyi-MAI/Z-Image-Turbo"));
    std::fs::create_dir_all(&model_dir).expect("model dir creates");
    std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("model marker writes");
}

/// Drives a project-scoped training job from submission to a completed result
/// and asserts the produced adapter is registered as a normal SceneWorks LoRA.
/// Seeds the base model so the real-run guardrails pass.
async fn submit_real_training_job(
    app: axum::Router,
    project_id: &str,
    data_dir: &std::path::Path,
) -> (String, std::path::PathBuf, std::path::PathBuf) {
    seed_installed_base_model(data_dir);
    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target_id = registry["targets"][0]["id"]
        .as_str()
        .expect("target id")
        .to_owned();
    let mut config = registry["targets"][0]["defaults"].clone();
    // A trigger word flows from the config into the plan and the LoRA entry.
    config["triggerWord"] = json!("auroraStyle");

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = job["id"].as_str().expect("job id").to_owned();
    let output_dir = std::path::PathBuf::from(
        job["payload"]["plan"]["output"]["outputDir"]
            .as_str()
            .unwrap(),
    );
    let file_name = job["payload"]["plan"]["output"]["fileName"]
        .as_str()
        .unwrap()
        .to_owned();
    let adapter_path = output_dir.join(file_name);
    (job_id, output_dir, adapter_path)
}

#[tokio::test]
async fn completed_training_job_registers_lora_with_provenance() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (job_id, output_dir, adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;

    // The worker writes the final adapter into the resolved output dir before
    // it reports completion, alongside step checkpoints it does not clean up.
    // Registration must pick the declared final adapter, not a checkpoint.
    std::fs::create_dir_all(&output_dir).expect("output dir creates");
    write_test_safetensors(&adapter_path);
    let final_name = adapter_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("final adapter name")
        .to_owned();
    let stem = adapter_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("adapter stem");
    write_test_safetensors(&output_dir.join(format!("{stem}-step000250.safetensors")));

    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Trained LoRA saved.",
            "result": { "outputPath": adapter_path.display().to_string() }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    // The registration outcome is folded into the job result so it is
    // observable (rather than silently dropped on failure).
    assert_eq!(completed["result"]["loraRegistered"], true);
    assert!(completed["result"]["loraId"]
        .as_str()
        .is_some_and(|id| id.starts_with("lora_")));
    assert!(completed["result"]["loraManifestPath"]
        .as_str()
        .is_some_and(|path| path.ends_with("manifest.jsonc")));

    // The trained adapter is now a normal, installed, project-scoped LoRA.
    let (status, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let entry = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|item| item["name"] == json!("Aurora Style"))
        .expect("trained LoRA appears in catalog")
        .clone();
    assert_eq!(entry["scope"], "project");
    assert_eq!(entry["family"], "z-image");
    assert_eq!(entry["baseModel"], "z_image_turbo");
    assert_eq!(entry["triggerWords"], json!(["auroraStyle"]));
    assert_eq!(entry["installState"], "installed");
    // The final adapter is registered, not the step checkpoint that shares
    // the output directory.
    assert_eq!(entry["files"], json!([final_name]));
    // installedPath resolves to the trained adapter's directory (the same
    // convention as imported LoRAs), and the adapter file lives under it.
    let lora_id = entry["id"].as_str().expect("lora id");
    let installed_path = entry["installedPath"].as_str().expect("installed path");
    assert!(
        installed_path.contains(lora_id),
        "installed path {installed_path} should point at the trained adapter dir"
    );
    assert!(adapter_path.exists());
    assert_eq!(entry["source"]["provider"], "training");
    assert_eq!(entry["provenance"]["trainingJobId"], job_id);
    assert!(entry["provenance"]["configSnapshot"].is_object());

    // Provenance survives an app restart (manifest is on disk).
    let reloaded = create_app(settings).expect("app reloads");
    let (status, loras) = request(
        reloaded,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(loras
        .as_array()
        .expect("loras array")
        .iter()
        .any(|item| item["name"] == json!("Aurora Style")
            && item["provenance"]["trainingJobId"] == json!(job_id)));
}

#[tokio::test]
async fn failed_or_unwritten_training_job_registers_no_lora() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // A failed job never registers, even though a manifest entry was staged.
    let (failed_job_id, _output_dir, _adapter_path) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{failed_job_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 1,
            "message": "Training failed.",
            "error": "CUDA out of memory"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // A job that reports completed but produced no weights must not leave a
    // broken registry entry either, and the failure is surfaced in the result.
    let (completed_no_weights_id, _, _) =
        submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{completed_no_weights_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Reported complete without weights."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], false);
    assert!(completed["result"]["loraRegistrationError"].is_string());

    let (status, loras) = request(
        app,
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(loras
        .as_array()
        .expect("loras array")
        .iter()
        .all(|item| item["name"] != json!("Aurora Style")));
}

#[tokio::test]
async fn crafted_training_job_cannot_register_outside_canonical_manifest() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    // A `lora_train` job can be crafted directly via the generic job endpoint
    // with an attacker-chosen payload. Stage a real adapter under the canonical
    // project output dir so registration would succeed if (and only if) it uses
    // the recomputed path.
    let crafted_lora_id = "lora_crafted01";
    let adapter_dir = project_path.join("loras").join(crafted_lora_id);
    std::fs::create_dir_all(&adapter_dir).expect("adapter dir creates");
    write_test_safetensors(&adapter_dir.join("crafted.safetensors"));

    // The payload points the manifest write and the source path at locations
    // outside the canonical project manifest. Both must be ignored.
    let evil_manifest = temp_dir.path().join("evil-manifest.jsonc");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "lora_train",
            "projectId": project_id,
            "projectName": "Training Project",
            "requestedGpu": "auto",
            "payload": {
                "dryRun": false,
                "manifestPath": evil_manifest.display().to_string(),
                "manifestEntry": {
                    "id": crafted_lora_id,
                    "name": "Crafted LoRA",
                    "scope": "project",
                    "family": "z-image",
                    "source": { "provider": "evil", "path": "../../../../escape/loras" },
                    "files": ["crafted.safetensors"]
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = job["id"].as_str().expect("job id").to_owned();

    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Crafted completion."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // The write went to the canonical project manifest, not the payload path.
    assert!(
        !evil_manifest.exists(),
        "payload manifestPath must be ignored"
    );
    assert_eq!(completed["result"]["loraRegistered"], true);
    assert_eq!(
        completed["result"]["loraManifestPath"]
            .as_str()
            .expect("manifest path"),
        project_path
            .join("loras")
            .join("manifest.jsonc")
            .display()
            .to_string()
    );

    // The registered entry's source path was recomputed, not taken from the
    // attacker payload.
    let (status, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let entry = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|item| item["id"] == json!(crafted_lora_id))
        .expect("crafted LoRA registered under canonical manifest")
        .clone();
    assert_eq!(entry["scope"], "project");
    assert_eq!(entry["source"]["provider"], "training");
    assert_eq!(entry["source"]["path"], format!("loras/{crafted_lora_id}"));
    // `files` was validated against the recomputed output dir (the declared
    // name exists there), so the registered entry points only inside the
    // canonical LoRA directory.
    assert_eq!(entry["files"], json!(["crafted.safetensors"]));

    // A traversal id is rejected outright: nothing registers and the failure
    // is visible.
    let (_, evil_job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "lora_train",
            "projectId": project_id,
            "projectName": "Training Project",
            "requestedGpu": "auto",
            "payload": {
                "dryRun": false,
                "manifestEntry": {
                    "id": "../../pwned",
                    "name": "Traversal",
                    "scope": "project",
                    "source": { "provider": "evil", "path": "loras/x" }
                }
            }
        }),
    )
    .await;
    let evil_job_id = evil_job["id"].as_str().expect("job id").to_owned();
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{evil_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Traversal completion."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], false);
    assert!(completed["result"]["loraRegistrationError"].is_string());

    // A `..`-traversing files entry is rejected even when a valid adapter
    // exists under the canonical output dir: registration only accepts plain
    // in-tree file names, so generation can never be pointed outside it.
    let traversal_lora_id = "lora_filestrav01";
    let traversal_dir = project_path.join("loras").join(traversal_lora_id);
    std::fs::create_dir_all(&traversal_dir).expect("adapter dir creates");
    write_test_safetensors(&traversal_dir.join("real.safetensors"));
    let (_, files_job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "lora_train",
            "projectId": project_id,
            "projectName": "Training Project",
            "requestedGpu": "auto",
            "payload": {
                "dryRun": false,
                "manifestEntry": {
                    "id": traversal_lora_id,
                    "name": "Files Traversal",
                    "scope": "project",
                    "source": { "provider": "evil", "path": "loras/x" },
                    "files": ["../../../../escape/evil.safetensors"]
                }
            }
        }),
    )
    .await;
    let files_job_id = files_job["id"].as_str().expect("job id").to_owned();
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{files_job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Files traversal completion."
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["result"]["loraRegistered"], false);
    assert!(completed["result"]["loraRegistrationError"].is_string());
}

#[tokio::test]
async fn real_training_job_rejected_when_base_model_missing() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target_id = registry["targets"][0]["id"]
        .as_str()
        .expect("target id")
        .to_owned();
    let config = registry["targets"][0]["defaults"].clone();

    // No base model is installed: a real run is rejected with an actionable
    // message, but a dry run (plan preview) still succeeds.
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("is not installed"));

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn training_job_rejects_cpu_target() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Training Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target_id = registry["targets"][0]["id"]
        .as_str()
        .expect("target id")
        .to_owned();
    let mut config = registry["targets"][0]["defaults"].clone();
    // Targeting a CPU worker for a GPU-only job is rejected with an
    // actionable message (a dry run is GPU-routed too, so this also holds).
    config["advanced"]["requestedGpu"] = json!("cpu");

    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .unwrap()
        .contains("cannot target CPU workers"));
}

#[test]
fn insufficient_disk_space_threshold_is_strict_less_than() {
    assert!(insufficient_disk_space(100, 200));
    assert!(!insufficient_disk_space(200, 200));
    assert!(!insufficient_disk_space(300, 200));
}

#[tokio::test]
async fn timeline_routes_persist_and_create_worker_jobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created_project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Timeline Project" }),
    )
    .await;
    let project_id = created_project["id"]
        .as_str()
        .expect("project id")
        .to_owned();

    let (status, mut timeline) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({ "name": "Main timeline", "aspectRatio": "16:9", "fps": 30 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(timeline["projectId"], project_id);
    assert_eq!(timeline["tracks"].as_array().unwrap().len(), 3);

    let timeline_id = timeline["id"].as_str().expect("timeline id").to_owned();
    timeline["tracks"][0]["items"] = json!([
        {
            "id": "item-1",
            "trackId": "track_main",
            "assetId": "asset-1",
            "type": "video",
            "displayName": "Clip",
            "sourceIn": 2,
            "sourceOut": 6,
            "timelineStart": 10,
            "timelineEnd": 14,
            "speed": 1,
            "fit": "fit",
            "volume": 1
        }
    ]);
    let (status, saved) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(saved["duration"].as_f64(), Some(14.0));
    assert_eq!(
        saved["tracks"][0]["items"][0]["currentVersionAssetId"],
        "asset-1"
    );
    assert_eq!(
        saved["tracks"][0]["items"][0]["versionHistory"][0]["source"],
        "original"
    );

    let (status, timelines) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/timelines"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(timelines[0]["id"], timeline_id);
    assert_eq!(
        timelines[0]["filePath"],
        format!(
            "timelines/main-timeline-{}.sceneworks.timeline.json",
            &timeline_id[timeline_id.len() - 8..]
        )
    );

    let (status, export_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
        json!({ "resolution": 720, "fps": 30, "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(export_job["type"], "timeline_export");
    assert_eq!(export_job["payload"]["timelineId"], timeline_id);

    let (status, frame_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/items/item-1/frames"),
        json!({ "playheadSeconds": 12.5, "intendedUse": "first_frame" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(frame_job["type"], "frame_extract");
    assert_eq!(frame_job["payload"]["sourceAssetId"], "asset-1");
    assert_eq!(frame_job["payload"]["sourceTimestamp"], 4.5);

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 2);
}

#[tokio::test]
async fn models_catalog_carries_mac_support_and_capabilities_endpoint() {
    // sc-3486: the catalog stamps per-model `macSupport`, and the capabilities endpoint
    // carries the master switch (`macGatingActive` = mlx_required) + infra gating.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            { "id": "z_image_turbo", "name": "Z-Image-Turbo", "family": "z-image", "type": "image",
              "adapter": "z_image_diffusers", "capabilities": ["text_to_image"], "downloads": [],
              "paths": {}, "defaults": {}, "limits": {}, "loraCompatibility": { "families": [], "types": [] }, "ui": {} },
            { "id": "unported_image_model", "name": "Unported", "family": "unported", "type": "image",
              "adapter": "procedural_preview", "capabilities": ["text_to_image"], "downloads": [],
              "paths": {}, "defaults": {}, "limits": {}, "loraCompatibility": { "families": [], "types": [] }, "ui": {} },
            { "id": "svd", "name": "SVD", "family": "svd", "type": "video",
              "adapter": "svd_video", "capabilities": ["image_to_video"], "downloads": [],
              "paths": {}, "defaults": {}, "limits": {}, "loraCompatibility": { "families": [], "types": [] }, "ui": {} }
          ]
        }
        "#,
    )
    .expect("builtin models writes");

    let mut settings = test_settings(&temp_dir);
    settings.mlx_required = true;
    let app = create_app(settings).expect("app creates");

    let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let by_id = |id: &str| {
        models
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .cloned()
            .unwrap_or(Value::Null)
    };
    // Unported image model (no Rust/MLX engine) → unsupported on Mac, with a gap reason. No real
    // image model is torch-only anymore: every family was ported to MLX — Kolors (sc-3875),
    // PuLID-FLUX (sc-3344), and finally Lens / Lens-Turbo (epic 3164 / sc-5105, the last one) — so
    // the torch-only gating is demonstrated with a synthetic unported id, which has no dedicated
    // port epic (suggestedEpic absent → "needs an epic", epic 3482 policy).
    let torch_only = by_id("unported_image_model");
    assert_eq!(torch_only["macSupport"]["supported"], false);
    assert!(torch_only["macSupport"]["reason"].is_object());
    assert!(torch_only["macSupport"]["reason"]["suggestedEpic"].is_null());
    // MLX-routed family → supported, stays in the picker.
    assert_eq!(by_id("z_image_turbo")["macSupport"]["supported"], true);
    // SVD is now MLX-routed (sc-3523: `svd`→`svd_xt`, image→video only) → supported.
    assert_eq!(by_id("svd")["macSupport"]["supported"], true);

    // Capabilities endpoint: gating active (mlx_required=true) + infra epics present.
    let (status, caps) = request(app, "GET", "/api/v1/capabilities/mac", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(caps["macGatingActive"], true);
    assert_eq!(
        caps["notAvailableLabel"],
        "Not available on Mac (Rust/MLX only)"
    );
    // Real-ESRGAN upscaling is ported to the Rust worker (sc-3489) → tool supported, no reason.
    assert_eq!(caps["features"]["imageUpscale"]["supported"], true);
    assert_eq!(caps["features"]["imageUpscale"]["reason"], Value::Null);
    // The AuraSR engine is dropped on Mac (sc-3668) AND off-Mac as an offered engine (sc-5499) → its
    // per-engine feature is unsupported on every platform and names the drop.
    assert_eq!(caps["features"]["imageUpscaleAuraSr"]["supported"], false);
    assert_eq!(
        caps["features"]["imageUpscaleAuraSr"]["reason"]["suggestedEpic"],
        "sc-5499"
    );
    // DWPose pose detection is ported to the Rust worker (sc-3487) → supported (sc-4206).
    assert_eq!(caps["features"]["poseFromPhoto"]["supported"], true);
    assert_eq!(caps["features"]["poseFromPhoto"]["reason"], Value::Null);
    // SeedVR2 video upscaling is net-new on Mac (epic 4811 / sc-4816) → supported.
    assert_eq!(caps["features"]["videoUpscale"]["supported"], true);
    assert_eq!(caps["features"]["videoUpscale"]["reason"], Value::Null);
    assert!(caps["training"]["supportedKernels"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "z_image_lora"));
    // Kolors training cut over to the native Rust trainer (sc-4732).
    assert!(caps["training"]["supportedKernels"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "kolors_lora"));
}

#[tokio::test]
async fn capabilities_mac_is_inert_when_mlx_not_required() {
    // The default (observe mode / Windows / Linux) reports gating inactive, so the client
    // applies no gating at all.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, caps) = request(app, "GET", "/api/v1/capabilities/mac", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(caps["macGatingActive"], false);
}

#[tokio::test]
async fn image_job_route_threads_upscale_contract_when_enabled() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "z_image_turbo",
              "name": "Z-Image-Turbo",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image"],
              "downloads": [],
              "paths": {},
              "resources": {
                "imageUpscalers": {
                  "real-esrgan": {
                    "x2": { "repo": "nateraw/real-esrgan", "file": "RealESRGAN_x2plus.pth" },
                    "x4": { "repo": "nateraw/real-esrgan", "file": "RealESRGAN_x4plus.pth" }
                  }
                }
              },
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": [], "types": [] },
              "ui": {}
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let base_request = json!({
        "projectId": "project-1",
        "mode": "text_to_image",
        "prompt": "mist over hills",
        "count": 1,
        "seed": 123
    });
    let (status, base_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        base_request.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(base_job["payload"].get("upscale").is_none());
    assert_eq!(
        base_job["payload"]["modelManifestEntry"]["resources"]["imageUpscalers"]["real-esrgan"]
            ["x4"]["file"],
        json!("RealESRGAN_x4plus.pth")
    );

    let mut disabled_request = base_request.clone();
    disabled_request["upscale"] = json!({ "enabled": false, "factor": 4, "engine": "real-esrgan" });
    let (status, disabled_job) =
        request(app.clone(), "POST", "/api/v1/image/jobs", disabled_request).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(disabled_job["payload"], base_job["payload"]);

    let mut enabled_request = base_request;
    enabled_request["upscale"] = json!({ "enabled": true, "factor": 4, "engine": "real-esrgan" });
    let (status, enabled_job) =
        request(app.clone(), "POST", "/api/v1/image/jobs", enabled_request).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        enabled_job["payload"]["upscale"],
        json!({ "enabled": true, "factor": 4, "engine": "real-esrgan" })
    );

    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "seed": 123,
            "upscale": { "enabled": true, "factor": 3, "engine": "real-esrgan" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "upscale.factor must be 2 or 4");
}

#[tokio::test]
async fn image_and_video_job_routes_normalize_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "projectName": "Project 1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["type"], "image_generate");
    assert_eq!(image_job["projectId"], "project-1");
    assert!(image_job["payload"].get("requestedGpu").is_none());
    assert_eq!(image_job["payload"]["seed"], Value::Null);
    assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 2);

    let (status, edit_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make it dusk",
            "sourceAssetId": "asset-1",
            "seed": 42
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(edit_job["type"], "image_edit");
    assert!(edit_job["payload"].get("seeds").is_none());

    let (status, wide_seed_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": " ",
            "mode": "text_to_image",
            "prompt": "space project id stays Python-compatible",
            "seed": -42
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(wide_seed_job["payload"]["projectId"], " ");
    assert_eq!(wide_seed_job["payload"]["seed"], -42);

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "replace_person",
            "prompt": "hero walks through rain",
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "characterId": "character-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(video_job["type"], "person_replace");
    assert!(video_job["payload"].get("requestedGpu").is_none());

    let (status, integer_duration_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "integer duration stays an integer",
            "duration": 6
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(integer_duration_job["payload"]["duration"], 6);

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 5);
}

#[tokio::test]
async fn bernini_video_modes_validate_required_media() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Bernini" }),
    )
    .await;

    // video_to_video without a source clip is rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "video_to_video",
            "prompt": "make it golden hour"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // reference_to_video without reference images is rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // reference_video_to_video needs BOTH a source clip and references.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "prompt": "swap the subject",
            "referenceAssetIds": ["ref-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Blank reference ids are rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Reference id lists are bounded before the worker has to encode them.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["ref-1", "ref-2", "ref-3", "ref-4", "ref-5", "ref-6", "ref-7", "ref-8", "ref-9"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A complete video_to_video request creates a base video_generate job that
    // carries the source clip.
    let (status, v2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "video_to_video",
            "prompt": "make it golden hour",
            "sourceClipAssetId": "clip-a"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(v2v_job["type"], "video_generate");
    assert_eq!(v2v_job["payload"]["mode"], "video_to_video");
    assert_eq!(v2v_job["payload"]["sourceClipAssetId"], "clip-a");

    // A complete reference_video_to_video request carries both the clip and the refs.
    let (status, rv2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "prompt": "swap the subject",
            "sourceClipAssetId": "clip-a",
            "referenceAssetIds": ["ref-1", "ref-2"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(rv2v_job["type"], "video_generate");
    assert_eq!(rv2v_job["payload"]["referenceAssetIds"][0], "ref-1");
    assert_eq!(rv2v_job["payload"]["referenceAssetIds"][1], "ref-2");

    // multi_video_to_video (sc-5425) needs at least two source clips.
    for clips in [json!([]), json!(["clip-a"])] {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "model": "bernini",
                "mode": "multi_video_to_video",
                "prompt": "blend the takes",
                "sourceClipAssetIds": clips
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // Blank source-clip ids are rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-a", "  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Source clip lists are bounded before worker-side video conditioning.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-1", "clip-2", "clip-3", "clip-4", "clip-5", "clip-6", "clip-7", "clip-8", "clip-9"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A complete multi_video_to_video request carries the clip array.
    let (status, mv2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-a", "clip-b"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(mv2v_job["type"], "video_generate");
    assert_eq!(mv2v_job["payload"]["sourceClipAssetIds"][0], "clip-a");
    assert_eq!(mv2v_job["payload"]["sourceClipAssetIds"][1], "clip-b");

    // ads2v (sc-5425) needs a source clip, a reference video, AND >=1 reference image.
    let ads2v_incomplete = [
        json!({ "referenceClipAssetId": "clip-ref", "referenceAssetIds": ["ref-1"] }),
        json!({ "sourceClipAssetId": "clip-src", "referenceAssetIds": ["ref-1"] }),
        json!({ "sourceClipAssetId": "clip-src", "referenceClipAssetId": "clip-ref" }),
    ];
    for extra in ads2v_incomplete {
        let mut body = json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit with the reference clip"
        });
        let object = body.as_object_mut().unwrap();
        for (key, value) in extra.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        let (status, _) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // A complete ads2v request carries the source clip, reference video, and references.
    let (status, ads2v_job) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit with the reference clip",
            "sourceClipAssetId": "clip-src",
            "referenceClipAssetId": "clip-ref",
            "referenceAssetIds": ["ref-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(ads2v_job["type"], "video_generate");
    assert_eq!(ads2v_job["payload"]["sourceClipAssetId"], "clip-src");
    assert_eq!(ads2v_job["payload"]["referenceClipAssetId"], "clip-ref");
    assert_eq!(ads2v_job["payload"]["referenceAssetIds"][0], "ref-1");
}

#[tokio::test]
async fn person_tracking_routes_match_contracts() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Tracking Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
    std::fs::write(
        project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": project_id,
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [],
            "status": {}
        }))
        .expect("json"),
    )
    .expect("track sidecar writes");

    let (status, tracks) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/person-tracks"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tracks[0]["id"], "track_1");
    assert_eq!(
        tracks[0]["path"],
        "person-tracks/track_1.sceneworks.person-track.json"
    );

    let (status, track) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/person-tracks/track_1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(track["name"], "Hero");

    let (status, detection_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/person-tracks/detections"),
        json!({ "sourceAssetId": "asset-video", "sourceTimestamp": 1.25 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(detection_job["type"], "person_detect");
    assert_eq!(detection_job["payload"]["sourceTimestamp"], 1.25);
    assert!(detection_job["projectName"]
        .as_str()
        .is_some_and(|value| value.starts_with("tracking")));

    let detection = json!({
        "id": "person_1",
        "box": { "x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6 }
    });
    let (status, track_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/person-tracks/jobs"),
        json!({
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "detection": detection,
            "trackName": "Hero"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(track_job["type"], "person_track");
    assert_eq!(track_job["payload"]["trackName"], "Hero");

    for invalid_path in [
        format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E"),
        format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E%2Fescape"),
        format!("/api/v1/projects/{project_id}/person-tracks/track~bad"),
    ] {
        let (status, error) = request(app.clone(), "GET", &invalid_path, Value::Null).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Invalid person track ID");
    }

    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 2);
}

#[tokio::test]
async fn generation_job_routes_reject_incompatible_loras() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                },
                {
                  "id": "ltx_2_3",
                  "name": "LTX",
                  "family": "ltx-video",
                  "type": "video",
                  "adapter": "ltx_video",
                  "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["ltx-video"] },
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "bad_qwen",
                  "name": "Bad Qwen",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "loras": [{ "id": "qwen_style" }]
                }
              ]
            }
            "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("qwen.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Compatibility" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, image_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "z_image_turbo",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        image_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    let (status, unknown_model_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "missing_model",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        unknown_model_error["detail"],
        "Model missing_model not found; cannot verify LoRA compatibility"
    );

    let (status, preset_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "z_image_turbo",
            "recipePresetId": "bad_qwen"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        preset_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    for (mode, extra) in [
        ("image_to_video", json!({ "sourceAssetId": "asset-image" })),
        ("text_to_video", json!({})),
        (
            "first_last_frame",
            json!({ "sourceAssetId": "asset-first", "lastFrameAssetId": "asset-last" }),
        ),
        ("extend_clip", json!({ "sourceClipAssetId": "asset-video" })),
        (
            "video_bridge",
            json!({ "sourceClipAssetId": "asset-left", "bridgeRightClipAssetId": "asset-right" }),
        ),
        (
            "replace_person",
            json!({ "sourceClipAssetId": "asset-video", "personTrackId": "track-1", "characterId": "character-1" }),
        ),
    ] {
        let mut payload = json!({
            "projectId": project_id,
            "mode": mode,
            "prompt": "motion",
            "model": "ltx_2_3",
            "loras": [{ "id": "qwen_style" }]
        });
        payload
            .as_object_mut()
            .expect("video payload object")
            .extend(extra.as_object().expect("extra payload object").clone());
        let (status, video_error) =
            request(app.clone(), "POST", "/api/v1/video/jobs", payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mode}");
        assert_eq!(
            video_error["detail"],
            "LoRA qwen_style is not compatible with model ltx_2_3"
        );
    }

    let (_, character) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters"),
        json!({ "name": "Mira", "type": "person" }),
    )
    .await;
    let character_id = character["id"].as_str().expect("character id");
    let character_lora = temp_dir
        .path()
        .join("data/loras/character-qwen.safetensors");
    write_test_safetensors(&character_lora);
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
        json!({
            "name": "Character Qwen",
            "sourcePath": character_lora.display().to_string(),
            "compatibility": { "families": ["qwen-image"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, character_error) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
        json!({ "prompt": "portrait", "model": "z_image_turbo" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(character_error["detail"]
        .as_str()
        .unwrap()
        .contains("is not compatible with model z_image_turbo"));
}

#[tokio::test]
async fn video_jobs_expand_recipe_presets_server_side() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "vid-model",
              "name": "Vid Model",
              "family": "wan-video",
              "type": "video",
              "adapter": "wan_video",
              "capabilities": ["text_to_video", "image_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["wan-video"] },
              "ui": { "label": "Vid" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "motion-lora",
              "name": "Motion LoRA",
              "family": "wan-video",
              "triggerWords": ["motion"],
              "compatibility": { "families": ["wan-video"] },
              "source": { "provider": "local", "path": "loras/motion.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_motion",
              "name": "Dream Motion",
              "workflow": "text_to_video",
              "model": "vid-model",
              "defaults": { "duration": 8, "fps": 30, "resolution": "1280x720", "quality": "best", "negativePrompt": "jitter" },
              "prompt": { "prefix": "cinematic", "suffix": "smooth camera motion" },
              "loras": [{ "id": "motion-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("motion.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Video Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "vid-model",
            // Client render settings that DIFFER from the preset's declared
            // defaults — the studio seeds the form from the preset but the user
            // is free to override, so these submitted values must win.
            "duration": 10,
            "fps": 24,
            "width": 640,
            "height": 640,
            "quality": "fast",
            "negativePrompt": "client jitter",
            "recipePresetId": "dream_motion"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // Prompt prefix/suffix are folded in server-side around the raw client
    // prompt — the regression that motivated this path.
    assert_eq!(
        video_job["payload"]["prompt"],
        "cinematic, a fox runs, smooth camera motion"
    );
    // Render settings are client-owned and overrideable: the submitted values
    // win over the preset's declared defaults (8 / 30 / 1280x720 / best / jitter).
    assert_eq!(video_job["payload"]["duration"], 10);
    assert_eq!(video_job["payload"]["fps"], 24);
    assert_eq!(video_job["payload"]["width"], 640);
    assert_eq!(video_job["payload"]["height"], 640);
    assert_eq!(video_job["payload"]["quality"], "fast");
    assert_eq!(video_job["payload"]["negativePrompt"], "client jitter");
    // Preset LoRA merged in (client sent none) and stamped under advanced.
    assert_eq!(video_job["payload"]["loras"][0]["id"], "motion-lora");
    assert_eq!(
        video_job["payload"]["advanced"]["recipePresetId"],
        "dream_motion"
    );
}

#[tokio::test]
async fn lora_download_endpoint_queues_hf_download_for_builtin_lora() {
    // sc-5944: built-in LoRAs gain an explicit Download (mirrors model download). A
    // catalog LoRA with a Hugging Face source queues a `lora_download` job carrying the
    // repo/file the worker fetches into the HF cache; a non-HF source or already-installed
    // LoRA is rejected, and an unknown id is 404.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "ltx_ic_union",
                  "name": "LTX IC Union",
                  "family": "ltx-video",
                  "compatibility": { "families": ["ltx-video"] },
                  "source": {
                    "provider": "huggingface",
                    "repo": "Lightricks/LTX-2.3-IC",
                    "file": "ic-union.safetensors"
                  }
                },
                {
                  "id": "local_only",
                  "name": "Local Only",
                  "family": "z-image",
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/local.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("builtin loras writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/loras/ltx_ic_union/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_download");
    assert_eq!(job["payload"]["loraId"], "ltx_ic_union");
    assert_eq!(job["payload"]["loraName"], "LTX IC Union");
    assert_eq!(job["payload"]["provider"], "huggingface");
    assert_eq!(job["payload"]["repo"], "Lightricks/LTX-2.3-IC");
    assert_eq!(job["payload"]["files"][0], "ic-union.safetensors");
    assert_eq!(job["payload"]["family"], "ltx-video");

    // A LoRA whose source is not a Hugging Face repo can't be fetched this way.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/loras/local_only/download",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // An unknown LoRA id is a 404.
    let (status, _) = request(app, "POST", "/api/v1/loras/missing/download", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn model_and_lora_routes_match_manifest_behavior() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "base-model",
                  "name": "Base Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [
                    { "provider": "huggingface", "repo": "owner/alternate-model", "files": ["*.bin"], "estimatedSizeBytes": 536870912 },
                    { "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"], "default": true, "estimatedSizeBytes": 12884901888 }
                  ],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": { "label": "Base" }
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                { "id": "base-model", "name": "User Model", "ui": { "label": "User" } }
              ]
            }
            "#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style-lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": ["style"],
                  "compatibility": { "families": ["z-image", "wan-video"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "cinematic",
                  "name": "Cinematic",
                  "workflow": "text_to_image",
                  "model": "base-model",
                  "defaults": { "count": 4, "resolution": "1280x720", "negativePrompt": "flat lighting" },
                  "prompt": { "suffix": "cinematic lighting" },
                  "loras": [{ "id": "style-lora", "weight": 0.5 }]
                }
              ]
            }
            "#,
        )
        .expect("builtin recipe presets writes");
    std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                { "id": "cinematic", "name": "My Cinematic", "defaults": { "count": 2, "resolution": "1280x720", "negativePrompt": "flat lighting" } },
                { "id": "legacy_edit", "name": "Legacy Edit", "modes": ["edit_image"], "builtInLoras": [{ "id": "style-lora", "weight": 0.25 }] }
              ]
            }
            "#,
        )
        .expect("user recipe presets writes");
    let marker_dir = temp_dir.path().join("data/models/owner__model");
    std::fs::create_dir_all(&marker_dir).expect("model dir creates");
    std::fs::write(marker_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("marker writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["name"], "User Model");
    assert_eq!(models[0]["adapter"], "z_image_diffusers");
    assert_eq!(models[0]["downloadable"], true);
    assert_eq!(models[0]["downloadSizeBytes"], 12884901888_u64);
    assert_eq!(models[0]["downloadSizeLabel"], "12.0 GB");
    assert_eq!(models[0]["downloadSizeEstimated"], true);
    assert_eq!(models[0]["installState"], "installed");
    assert!(models[0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.ends_with("owner__model")));

    let (status, loras) = request(
        app.clone(),
        "GET",
        "/api/v1/loras?modelFamily=wan-video",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(loras.as_array().unwrap().len(), 1);
    assert_eq!(loras[0]["id"], "style-lora");

    let (status, presets) =
        request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let presets = presets.as_array().unwrap();
    assert_eq!(presets.len(), 2);
    let cinematic = presets
        .iter()
        .find(|preset| preset["id"] == "cinematic")
        .expect("cinematic preset");
    assert_eq!(cinematic["name"], "My Cinematic");
    assert_eq!(cinematic["scope"], "global");
    assert_eq!(cinematic["workflow"], "text_to_image");
    assert_eq!(cinematic["defaults"]["count"], 2);
    assert_eq!(cinematic["loras"][0]["id"], "style-lora");
    assert_eq!(cinematic["builtInLoras"][0]["id"], "style-lora");
    let legacy_edit = presets
        .iter()
        .find(|preset| preset["id"] == "legacy_edit")
        .expect("legacy edit preset");
    assert_eq!(legacy_edit["workflow"], "edit_image");
    assert_eq!(legacy_edit["model"], "base-model");
    assert_eq!(legacy_edit["loras"][0]["id"], "style-lora");
    assert_eq!(
        legacy_edit["appliedDefaults"]["notes"][0],
        "workflow inferred from legacy modes as edit_image"
    );
    assert_eq!(
        legacy_edit["appliedDefaults"]["notes"][1],
        "model defaulted to base-model for legacy preset"
    );
    assert_eq!(
        legacy_edit["appliedDefaults"]["notes"][2],
        "builtInLoras migrated to loras"
    );

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "city at night",
            "model": "base-model",
            // Client render settings that DIFFER from the preset's declared
            // defaults (count 2 / 1280x720 / "flat lighting") — the studio seeds
            // the form from the preset but the user can override, so these
            // submitted values must win.
            "count": 1,
            "width": 512,
            "height": 512,
            "negativePrompt": "client negative prompt",
            "recipePresetId": "cinematic",
            "advanced": { "resolution": "512x512" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        image_job["payload"]["prompt"],
        "city at night, cinematic lighting"
    );
    assert_eq!(image_job["payload"]["loras"][0]["id"], "style-lora");
    assert!(image_job["payload"]["loras"][0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.ends_with("data/loras/style.safetensors")
            || value.ends_with("data\\loras\\style.safetensors")
            || value.ends_with("loras/style.safetensors")
            || value.ends_with("loras\\style.safetensors")));
    assert_eq!(
        image_job["payload"]["loras"][0]["source"]["path"],
        "loras/style.safetensors"
    );
    assert_eq!(image_job["payload"]["model"], "base-model");
    assert_eq!(image_job["payload"]["loras"][0]["family"], "z-image");
    assert_eq!(
        image_job["payload"]["loras"][0]["compatibility"]["families"][0],
        "z-image"
    );
    // Render settings are client-owned and overrideable: the submitted values
    // win over the preset's declared defaults.
    assert_eq!(image_job["payload"]["count"], 1);
    assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 1);
    assert_eq!(image_job["payload"]["width"], 512);
    assert_eq!(image_job["payload"]["height"], 512);
    assert_eq!(
        image_job["payload"]["negativePrompt"],
        "client negative prompt"
    );
    assert_eq!(image_job["payload"]["advanced"]["resolution"], "512x512");
    assert_eq!(
        image_job["payload"]["advanced"]["recipePresetId"],
        "cinematic"
    );

    let (status, null_path_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "city with selected lora",
            "model": "base-model",
            "count": 1,
            "width": 512,
            "height": 512,
            "loras": [{
                "id": "style-lora",
                "name": null,
                "triggerWords": null,
                "compatibility": null,
                "installedPath": null,
                "sourcePath": null
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(null_path_job["payload"]["loras"][0]["id"], "style-lora");
    assert_eq!(null_path_job["payload"]["loras"][0]["name"], "Style LoRA");
    assert_eq!(
        null_path_job["payload"]["loras"][0]["triggerWords"][0],
        "style"
    );
    assert_eq!(
        null_path_job["payload"]["loras"][0]["compatibility"]["families"][0],
        "z-image"
    );
    assert!(null_path_job["payload"]["loras"][0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.ends_with("data/loras/style.safetensors")
            || value.ends_with("data\\loras\\style.safetensors")
            || value.ends_with("loras/style.safetensors")
            || value.ends_with("loras\\style.safetensors")));

    let (status, preset_model_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "city at dawn",
            "count": 1,
            "width": 512,
            "height": 512,
            "recipePresetId": "cinematic"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(preset_model_job["payload"]["model"], "base-model");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/models/base-model/download",
        json!({ "requestedGpu": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "model_download");
    assert_eq!(job["requestedGpu"], "auto");
    assert_eq!(job["payload"]["modelName"], "User Model");
    assert_eq!(job["payload"]["repo"], "owner/model");
    assert_eq!(job["payload"]["files"][0], "*.safetensors");
    assert_eq!(job["payload"]["targetDir"], models[0]["installedPath"]);

    let (status, job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({ "repo": "owner/lora", "name": "Imported LoRA", "files": ["adapter.safetensors"] }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_import");
    assert_eq!(job["payload"]["repo"], "owner/lora");
    assert_eq!(job["payload"]["loraId"], "imported_lora");
    assert_eq!(job["payload"]["scope"], "global");
    assert!(job["payload"]["targetDir"]
        .as_str()
        .is_some_and(|value| value.ends_with("data/loras/imported_lora")
            || value.ends_with("data\\loras\\imported_lora")));
    assert_eq!(job["payload"]["manifestEntry"]["scope"], "global");
    assert!(job["payload"].get("sourcePath").is_none());

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, url_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourceUrl": "https://example.com/loras/detail.safetensors",
            "name": "Detail LoRA",
            "family": "z-image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(url_job["type"], "lora_import");
    assert_eq!(
        url_job["payload"]["sourceUrl"],
        "https://example.com/loras/detail.safetensors"
    );
    assert_eq!(url_job["payload"]["loraId"], "detail_lora");
    assert_eq!(
        url_job["payload"]["manifestEntry"]["source"]["provider"],
        "url"
    );
    assert_eq!(
        url_job["payload"]["manifestEntry"]["source"]["url"],
        "https://example.com/loras/detail.safetensors"
    );
    assert_eq!(url_job["payload"]["manifestEntry"]["family"], "z-image");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let upload_bytes = test_safetensors_bytes();
    let (status, upload_job) = request_multipart_lora_upload(
        app,
        &[
            ("name", "Uploaded Detail"),
            ("scope", "global"),
            ("family", "z-image"),
        ],
        "detail.safetensors",
        &upload_bytes,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(upload_job["type"], "lora_import");
    assert_eq!(upload_job["payload"]["loraId"], "uploaded_detail");
    assert_eq!(upload_job["payload"]["uploadedSourcePath"], true);
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["source"]["provider"],
        "local"
    );
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["files"][0],
        "detail.safetensors"
    );
    let source_path = std::path::PathBuf::from(
        upload_job["payload"]["sourcePath"]
            .as_str()
            .expect("source path"),
    );
    assert_eq!(
        std::fs::read(&source_path).expect("staged upload reads"),
        upload_bytes
    );
    assert_eq!(
        source_path.file_name().and_then(|value| value.to_str()),
        Some("detail.safetensors")
    );

    TEST_MAX_LORA_UPLOAD_BYTES.with(|cap| cap.set(4));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_status, bad_error) = request_multipart_lora_upload(
        app,
        &[("name", "Too Large"), ("scope", "global")],
        "too-large.safetensors",
        b"12345",
    )
    .await;
    TEST_MAX_LORA_UPLOAD_BYTES.with(|cap| cap.set(0));
    assert_eq!(bad_status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        bad_error["detail"],
        "Uploaded LoRA file exceeds the 2GB limit"
    );

    let lora_source_dir = temp_dir.path().join("data").join("loras");
    std::fs::create_dir_all(&lora_source_dir).expect("lora source dir creates");
    let lora_source = lora_source_dir.join("safe-local.safetensors");
    write_test_safetensors(&lora_source);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, source_path_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": lora_source.display().to_string(),
            "name": "Safe Local Source"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        source_path_job["payload"]["manifestEntry"]["source"]["provider"],
        "local"
    );

    let outside_source = temp_dir.path().join("outside.safetensors");
    write_test_safetensors(&outside_source);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_status, bad_error) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": outside_source.display().to_string(),
            "name": "Unsafe Local Source"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
            bad_error["detail"],
            "LoRA sourcePath must be inside app-managed data/loras, project/loras, or staged upload folders"
        );

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/loras/import",
        json!({ "sourceUrl": "file:///tmp/detail.safetensors" }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(bad_error["detail"], "LoRA sourceUrl must use http or https");

    let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/loras/import",
            json!({ "sourceUrl": "https://example.com/loras/detail.safetensors", "family": "unknown-family" }),
        )
        .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Unsupported LoRA family: unknown-family"
    );

    let (status, normalized_family) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourceUrl": "https://example.com/loras/z-detail.safetensors",
            "family": "Z_Image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        normalized_family["payload"]["manifestEntry"]["family"],
        "z-image"
    );

    // sc-1378: architecture detection at import time. The detection
    // policy lets the user supply any family the catalog declares, so
    // expand the catalog now to include the families we exercise below.
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "base-model",
                  "name": "Base Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [
                    { "provider": "huggingface", "repo": "owner/alternate-model", "files": ["*.bin"], "estimatedSizeBytes": 536870912 },
                    { "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"], "default": true, "estimatedSizeBytes": 12884901888 }
                  ],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": { "label": "Base" }
                },
                {
                  "id": "qwen-image-base",
                  "name": "Qwen Image",
                  "family": "qwen-image",
                  "type": "image",
                  "adapter": "qwen_image",
                  "capabilities": ["text_to_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                },
                {
                  "id": "wan-video-base",
                  "name": "Wan Video",
                  "family": "wan-video",
                  "type": "video",
                  "adapter": "wan_video",
                  "capabilities": ["text_to_video"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models rewrites for detection tests");

    let detect_dir = temp_dir.path().join("data").join("loras");
    std::fs::create_dir_all(&detect_dir).expect("detect dir creates");

    // Qwen-Image-shaped file with a mismatched user-supplied family is
    // rejected with both values surfaced in the error message.
    let mismatch_path = detect_dir.join("qwen-as-wan.safetensors");
    write_test_safetensors_with_keys(&mismatch_path, &qwen_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, mismatch_error) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": mismatch_path.display().to_string(),
            "family": "wan-video",
            "name": "Mismatched"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let detail = mismatch_error["detail"].as_str().expect("detail string");
    assert!(
        detail.contains("qwen-image") && detail.contains("wan-video"),
        "mismatch error must surface both detected and supplied families, got: {detail}"
    );

    // Low-block MMDiT tensors are inconclusive rather than treated as
    // Z-Image; sparse Qwen LoRAs can target only early blocks.
    let auto_path = detect_dir.join("low-mmdit-no-autofill.safetensors");
    write_test_safetensors_with_keys(&auto_path, &z_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, auto_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": auto_path.display().to_string(),
            "name": "Auto Family"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(auto_job["payload"]["manifestEntry"].get("family").is_none());

    // Supplied family + inconclusive MMDiT detection succeeds, and the
    // user-supplied family is kept.
    let match_path = detect_dir.join("z-match.safetensors");
    write_test_safetensors_with_keys(&match_path, &z_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, match_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": match_path.display().to_string(),
            "family": "z-image",
            "name": "Matched"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(match_job["payload"]["manifestEntry"]["family"], "z-image");

    // Wan-shaped tensors are detected and accepted when the user agrees.
    let wan_match_path = detect_dir.join("wan-match.safetensors");
    write_test_safetensors_with_keys(&wan_match_path, &wan_video_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, wan_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": wan_match_path.display().to_string(),
            "family": "wan-video",
            "name": "Wan Matched"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(wan_job["payload"]["manifestEntry"]["family"], "wan-video");

    // Inconclusive header (only `__metadata__`) + supplied family is
    // accepted unchanged — the user-supplied label survives.
    let inconclusive_path = detect_dir.join("inconclusive.safetensors");
    write_test_safetensors(&inconclusive_path);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, inconclusive_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": inconclusive_path.display().to_string(),
            "family": "z-image",
            "name": "Inconclusive"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        inconclusive_job["payload"]["manifestEntry"]["family"],
        "z-image"
    );

    // Confident Qwen-Image detection (block count > 40) auto-fills.
    let qwen_path = detect_dir.join("qwen-autofill.safetensors");
    write_test_safetensors_with_keys(&qwen_path, &qwen_image_tensor_keys());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, qwen_job) = request(
        app,
        "POST",
        "/api/v1/loras/import",
        json!({
            "sourcePath": qwen_path.display().to_string(),
            "name": "Qwen Auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(qwen_job["payload"]["manifestEntry"]["family"], "qwen-image");
}

async fn request_multipart_lora_pair_upload(
    app: axum::Router,
    fields: &[(&str, &str)],
    primary: (&str, &[u8]),
    secondary: (&str, &[u8]),
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_LORA_PAIR_BOUNDARY";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    for (part_name, (filename, bytes)) in [("file", primary), ("secondaryFile", secondary)] {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{part_name}\"; filename=\"{filename}\"\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app,
        "POST",
        "/api/v1/loras/import",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

#[tokio::test]
async fn paired_moe_lora_upload_writes_convention_files_and_records_base_model() {
    // sc-1991: a bring-your-own Wan A14B MoE pair uploads as two file parts
    // (`file` = high-noise, `secondaryFile` = low-noise) under one record. The
    // import normalizes both halves to the dot-delimited high/low_noise convention
    // (off-convention upload names included) and persists the chosen A14B baseModel.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let wan_bytes = test_safetensors_bytes_with_keys(&wan_video_tensor_keys());
    let (status, job) = request_multipart_lora_pair_upload(
        app,
        &[
            ("name", "Wan MoE"),
            ("scope", "global"),
            ("family", "wan-video"),
            ("baseModel", "wan_2_2_t2v_14b"),
        ],
        // Community names that do NOT match the convention — must be normalized.
        ("high_noise_model.safetensors", &wan_bytes),
        ("low_noise_model.safetensors", &wan_bytes),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "lora_import");
    let lora_id = job["payload"]["loraId"].as_str().expect("loraId");
    assert_eq!(job["payload"]["manifestEntry"]["family"], "wan-video");
    assert_eq!(
        job["payload"]["manifestEntry"]["baseModel"],
        "wan_2_2_t2v_14b"
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["files"][0],
        format!("{lora_id}.high_noise.safetensors")
    );
    assert_eq!(
        job["payload"]["manifestEntry"]["files"][1],
        format!("{lora_id}.low_noise.safetensors")
    );

    // Both halves staged on disk; the worker renames them on import.
    let primary = std::path::PathBuf::from(
        job["payload"]["sourcePath"]
            .as_str()
            .expect("primary source path"),
    );
    let secondary = std::path::PathBuf::from(
        job["payload"]["secondarySourcePath"]
            .as_str()
            .expect("secondary source path"),
    );
    assert_eq!(std::fs::read(&primary).expect("primary reads"), wan_bytes);
    assert_eq!(
        std::fs::read(&secondary).expect("secondary reads"),
        wan_bytes
    );
    assert_ne!(primary, secondary);
}

async fn request_multipart_model_upload(
    app: axum::Router,
    fields: &[(&str, &str)],
    filename: &str,
    bytes: &[u8],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_MODEL_BOUNDARY";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app,
        "POST",
        "/api/v1/models/import",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

#[tokio::test]
async fn model_import_routes_handle_url_upload_and_family_detection() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                { "id": "z_image_turbo", "name": "Z-Image Turbo", "family": "z-image", "type": "image", "loraCompatibility": { "families": ["z-image"] } }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");

    // URL import without supplied family is accepted, payload echoes
    // the request fields, and target dir lives under data/models/imports.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, url_job) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({
            "sourceUrl": "https://example.com/models/custom.safetensors",
            "name": "Custom Model",
            "type": "image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(url_job["type"], "model_import");
    assert_eq!(url_job["requestedGpu"], "auto");
    assert_eq!(url_job["payload"]["modelId"], "custom_model");
    assert_eq!(url_job["payload"]["modelName"], "Custom Model");
    assert_eq!(
        url_job["payload"]["manifestEntry"]["source"]["provider"],
        "url"
    );
    assert_eq!(url_job["payload"]["manifestEntry"]["type"], "image");
    assert!(url_job["payload"]["targetDir"]
        .as_str()
        .is_some_and(|value| value.contains("models/imports/custom_model")
            || value.contains("models\\imports\\custom_model")));
    assert!(url_job["payload"]["manifestEntry"].get("family").is_none());

    // Repo imports must be owner/name, not arbitrary path-like strings.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (bad_repo_status, bad_repo_error) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({
            "repo": "owner/nested/model",
            "name": "Bad Repo",
            "type": "image"
        }),
    )
    .await;
    assert_eq!(bad_repo_status, StatusCode::BAD_REQUEST);
    assert!(bad_repo_error["detail"]
        .as_str()
        .unwrap_or("")
        .contains("owner/name"));

    // Duplicate id is rejected with an actionable error.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (dup_status, dup_error) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({
            "modelId": "z_image_turbo",
            "sourceUrl": "https://example.com/models/clone.safetensors",
            "name": "Clone"
        }),
    )
    .await;
    assert_eq!(dup_status, StatusCode::BAD_REQUEST);
    assert!(dup_error["detail"]
        .as_str()
        .unwrap_or("")
        .contains("already exists"));

    // Local upload stages bytes, queues an import job, and family
    // detection from a diffusers-style header auto-fills the manifest.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let upload_bytes = test_safetensors_bytes_with_keys(&qwen_image_tensor_keys());
    let (upload_status, upload_job) = request_multipart_model_upload(
        app,
        &[("name", "Auto Family"), ("type", "image")],
        "auto-family.safetensors",
        &upload_bytes,
    )
    .await;
    assert_eq!(upload_status, StatusCode::CREATED);
    assert_eq!(upload_job["type"], "model_import");
    assert_eq!(upload_job["payload"]["modelId"], "auto_family");
    assert_eq!(upload_job["payload"]["uploadedSourcePath"], true);
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["family"],
        "qwen-image"
    );
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["adapter"],
        "qwen_image"
    );
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["capabilities"],
        json!(["text_to_image", "style_variations"])
    );
    assert_eq!(
        upload_job["payload"]["manifestEntry"]["loraCompatibility"]["families"],
        json!(["qwen-image"])
    );
    let source_path = std::path::PathBuf::from(
        upload_job["payload"]["sourcePath"]
            .as_str()
            .expect("source path"),
    );
    assert_eq!(
        std::fs::read(&source_path).expect("staged upload reads"),
        upload_bytes
    );

    // Declared family must match detected family when detection is
    // confident — mismatch is rejected at the API boundary.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (mismatch_status, mismatch_error) = request_multipart_model_upload(
        app,
        &[
            ("name", "Mismatch"),
            ("type", "image"),
            ("family", "z-image"),
        ],
        "mismatch.safetensors",
        &test_safetensors_bytes_with_keys(&qwen_image_tensor_keys()),
    )
    .await;
    assert_eq!(mismatch_status, StatusCode::BAD_REQUEST);
    assert!(mismatch_error["detail"]
        .as_str()
        .unwrap_or("")
        .contains("qwen-image"));

    // Missing source produces an actionable error.
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (empty_status, empty_error) = request(
        app,
        "POST",
        "/api/v1/models/import",
        json!({ "name": "Nothing" }),
    )
    .await;
    assert_eq!(empty_status, StatusCode::BAD_REQUEST);
    assert!(empty_error["detail"]
        .as_str()
        .unwrap_or("")
        .contains("Hugging Face repo"));
}

#[tokio::test]
async fn imported_model_catalog_uses_paths_model_install_marker() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let model_dir = temp_dir.path().join("data/models/imports/custom_model");
    std::fs::create_dir_all(&model_dir).expect("model dir creates");
    std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("marker writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "custom_model",
                    "name": "Custom Model",
                    "type": "image",
                    "family": "z-image",
                    "paths": {{ "model": "{}" }}
                  }}]
                }}"#,
            model_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "custom_model");
    assert_eq!(models[0]["downloadable"], false);
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["installedPath"], model_dir.display().to_string());
}

#[tokio::test]
async fn downloadable_model_catalog_uses_huggingface_cache_install_state() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "base_model",
                "name": "Base Model",
                "type": "image",
                "family": "z-image",
                "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--model/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "StableDiffusionPipeline",
          "scheduler": ["diffusers", "DDPMScheduler"],
          "unet": ["diffusers", "UNet2DConditionModel"],
          "vae": ["diffusers", "AutoencoderKL"],
          "tokenizer": ["transformers", "CLIPTokenizer"]
        }"#,
    )
    .expect("model index writes");
    for (dir, file) in [
        ("scheduler", "scheduler_config.json"),
        ("unet", "config.json"),
        ("vae", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    std::fs::write(
        cache_dir.join("unet/diffusion_pytorch_model.safetensors"),
        "weights",
    )
    .expect("unet weights write");
    std::fs::write(
        cache_dir.join("vae/diffusion_pytorch_model.safetensors"),
        "weights",
    )
    .expect("vae weights write");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "base_model");
    assert_eq!(models[0]["downloadable"], true);
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert!(models[0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.contains("models--owner--model")));
}

#[tokio::test]
async fn downloadable_model_catalog_flags_incomplete_huggingface_snapshots() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "sdxl",
                "name": "SDXL",
                "type": "image",
                "family": "sdxl",
                "downloads": [{ "provider": "huggingface", "repo": "owner/sdxl" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--sdxl/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "StableDiffusionXLPipeline",
          "scheduler": ["diffusers", "EulerDiscreteScheduler"],
          "unet": ["diffusers", "UNet2DConditionModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    for (dir, file) in [
        ("scheduler", "scheduler_config.json"),
        ("unet", "config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    std::fs::write(
        cache_dir.join("unet/diffusion_pytorch_model.safetensors"),
        "weights",
    )
    .expect("unet weights write");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "sdxl");
    assert_eq!(models[0]["installState"], "missing");
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert_eq!(
        models[0]["missingRequiredFiles"],
        json!(["vae/<weights>", "vae/config.json"])
    );
    assert!(models[0]["installedPath"]
        .as_str()
        .is_some_and(|value| value.contains("models--owner--sdxl")));
}

#[tokio::test]
async fn downloadable_model_catalog_ignores_absent_optional_diffusers_components() {
    // Chroma's model_index.json declares `feature_extractor` and `image_encoder`
    // as `[null, null]` — diffusers' sentinel for optional components the pipeline
    // doesn't use, which have no files on disk by design. The health check must
    // not report them as missing, otherwise a fully-installed model is flagged
    // incomplete on every platform.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "chroma1_base",
                "name": "Chroma1-Base",
                "type": "image",
                "family": "chroma",
                "downloads": [{ "provider": "huggingface", "repo": "lodestones/Chroma1-Base" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--lodestones--Chroma1-Base/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "ChromaPipeline",
          "feature_extractor": [null, null],
          "image_encoder": [null, null],
          "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
          "text_encoder": ["transformers", "T5EncoderModel"],
          "tokenizer": ["transformers", "T5Tokenizer"],
          "transformer": ["diffusers", "ChromaTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    for (dir, file) in [
        ("scheduler", "scheduler_config.json"),
        ("text_encoder", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("transformer", "config.json"),
        ("vae", "config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    for dir in ["text_encoder", "transformer", "vae"] {
        std::fs::write(
            cache_dir
                .join(dir)
                .join("diffusion_pytorch_model.safetensors"),
            "weights",
        )
        .expect("component weights write");
    }

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "chroma1_base");
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));
}

#[tokio::test]
async fn downloadable_model_catalog_treats_processor_components_as_weightless() {
    // Qwen-Image-Edit-2511's model_index.json declares a `processor`
    // (Qwen2VLProcessor) — a transformers preprocessing wrapper that carries no
    // model weights and ships `preprocessor_config.json` instead of `config.json`.
    // The health check must not demand `processor/config.json` + `processor/<weights>`
    // (which the repo never contains), otherwise a fully-installed model is flagged
    // incomplete forever and the "Fix" download can never satisfy it.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "qwen_image_edit_2511",
                "name": "Qwen-Image-Edit-2511",
                "type": "image",
                "family": "qwen-image",
                "downloads": [{ "provider": "huggingface", "repo": "Qwen/Qwen-Image-Edit-2511" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "QwenImageEditPlusPipeline",
          "processor": ["transformers", "Qwen2VLProcessor"],
          "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
          "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
          "tokenizer": ["transformers", "Qwen2Tokenizer"],
          "transformer": ["diffusers", "QwenImageTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKLQwenImage"]
        }"#,
    )
    .expect("model index writes");
    // processor ships preprocessor_config.json + tokenizer files, but no
    // config.json and no weights — exactly the real Qwen2VLProcessor layout.
    for (dir, file) in [
        ("processor", "preprocessor_config.json"),
        ("scheduler", "scheduler_config.json"),
        ("text_encoder", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("transformer", "config.json"),
        ("vae", "config.json"),
    ] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join(file), "{}").expect("component config writes");
    }
    for dir in ["text_encoder", "transformer", "vae"] {
        std::fs::write(
            cache_dir
                .join(dir)
                .join("diffusion_pytorch_model.safetensors"),
            "weights",
        )
        .expect("component weights write");
    }

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["id"], "qwen_image_edit_2511");
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));
}

// Helper: write a minimal but COMPLETE set of weight-bearing components
// (text_encoder / transformer / vae, each with config.json + a weight file) into
// a diffusers snapshot dir, so a test can focus on a single weightless component.
fn write_complete_weight_bearing_components(cache_dir: &std::path::Path) {
    for dir in ["text_encoder", "transformer", "vae"] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join("config.json"), "{}").expect("config writes");
        std::fs::write(
            component_dir.join("diffusion_pytorch_model.safetensors"),
            "weights",
        )
        .expect("weights write");
    }
}

fn single_model_manifest(config_dir: &std::path::Path, id: &str, repo: &str) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        format!(
            r#"{{ "schemaVersion": 1, "models": [{{
                "id": "{id}", "name": "{id}", "type": "image", "family": "test",
                "downloads": [{{ "provider": "huggingface", "repo": "{repo}" }}]
            }}] }}"#
        ),
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
}

#[tokio::test]
async fn downloadable_model_catalog_reports_incomplete_cache_for_installed_managed_model() {
    // A model can have SceneWorks' managed completion marker while its Hugging
    // Face cache snapshot is partial. Keep those states independent so the UI
    // can offer "Fix" instead of disabling the primary action as "Ready".
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo = "owner/mixed";
    single_model_manifest(
        &temp_dir.path().join("config/manifests"),
        "mixed_model",
        repo,
    );
    let managed_dir = temp_dir
        .path()
        .join("data/models")
        .join(safe_download_dir(repo));
    std::fs::create_dir_all(&managed_dir).expect("managed dir creates");
    std::fs::write(managed_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("managed marker writes");
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--mixed/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("partial hf cache creates");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["installState"], "installed");
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert_eq!(
        models[0]["missingRequiredFiles"],
        json!(["model_index.json"])
    );
}

#[test]
fn huggingface_cache_health_accepts_readable_snapshot_symlinked_model_index() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let repo_root = temp_dir.path().join("models--owner--symlinked");
    let snapshot = repo_root.join("snapshots/abc123");
    let blobs = repo_root.join("blobs");
    std::fs::create_dir_all(&snapshot).expect("snapshot creates");
    std::fs::create_dir_all(&blobs).expect("blobs creates");
    std::fs::create_dir_all(repo_root.join("refs")).expect("refs creates");
    std::fs::write(repo_root.join("refs/main"), "abc123").expect("ref writes");
    let blob_name = "model-index-blob";
    std::fs::write(
        blobs.join(blob_name),
        r#"{ "_class_name": "EmptyPipeline" }"#,
    )
    .expect("blob writes");
    let link = snapshot.join("model_index.json");
    let relative_target = std::path::Path::new("..")
        .join("..")
        .join("blobs")
        .join(blob_name);
    if create_test_symlink_file(&relative_target, &link).is_err() {
        std::fs::write(&link, r#"{ "_class_name": "EmptyPipeline" }"#)
            .expect("fallback index writes");
    }

    let health = super::models::huggingface_cache_health(&repo_root, &[]);

    assert!(health.installed);
    assert!(!health.incomplete);
    assert!(health.missing_files.is_empty());
}

#[cfg(windows)]
fn create_test_symlink_file(
    target: &std::path::Path,
    link: &std::path::Path,
) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(unix)]
fn create_test_symlink_file(
    target: &std::path::Path,
    link: &std::path::Path,
) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[tokio::test]
async fn downloadable_model_catalog_accepts_weightless_component_with_nonstandard_config_name() {
    // Hardening: completeness for weightless auxiliary components is keyed on
    // "the directory exists and holds a file", not a hard-coded config filename.
    // A processor that ships an unexpected config name must still read complete,
    // so future class variants can't re-trigger a permanent false "incomplete".
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    single_model_manifest(
        &temp_dir.path().join("config/manifests"),
        "weightless_model",
        "owner/weightless",
    );
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--weightless/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "SomePipeline",
          "processor": ["transformers", "SomeFutureProcessor"],
          "text_encoder": ["transformers", "T5EncoderModel"],
          "transformer": ["diffusers", "SomeTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    write_complete_weight_bearing_components(&cache_dir);
    // processor dir exists with a config whose name we do NOT special-case.
    let processor_dir = cache_dir.join("processor");
    std::fs::create_dir_all(&processor_dir).expect("processor dir creates");
    std::fs::write(processor_dir.join("processor_config.json"), "{}").expect("processor config");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["cacheState"], "complete");
    assert_eq!(models[0]["repairAvailable"], false);
    assert_eq!(models[0]["missingRequiredFiles"], json!([]));
}

#[tokio::test]
async fn downloadable_model_catalog_flags_empty_weightless_component_dir() {
    // The hardening must not silently pass everything: a genuinely absent/empty
    // weightless component directory still reports incomplete (partial download).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    single_model_manifest(
        &temp_dir.path().join("config/manifests"),
        "partial_model",
        "owner/partial",
    );
    let cache_dir = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--partial/snapshots/abc123");
    std::fs::create_dir_all(&cache_dir).expect("hf cache creates");
    std::fs::write(
        cache_dir.join("model_index.json"),
        r#"{
          "_class_name": "SomePipeline",
          "tokenizer": ["transformers", "T5Tokenizer"],
          "text_encoder": ["transformers", "T5EncoderModel"],
          "transformer": ["diffusers", "SomeTransformer2DModel"],
          "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    )
    .expect("model index writes");
    write_complete_weight_bearing_components(&cache_dir);
    // tokenizer dir is created but left EMPTY (partial download).
    std::fs::create_dir_all(cache_dir.join("tokenizer")).expect("empty tokenizer dir");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(models[0]["cacheState"], "incomplete");
    assert_eq!(models[0]["repairAvailable"], true);
    assert_eq!(
        models[0]["missingRequiredFiles"],
        json!(["tokenizer/<config>"])
    );
}

#[tokio::test]
async fn model_download_job_forwards_catalog_family_for_worker_reconciliation() {
    // sc-1663: the download job must carry the catalog-declared family so the
    // worker can re-verify the downloaded weights match it (parity with import).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "base_model",
                "name": "Base Model",
                "type": "image",
                "family": "z-image",
                "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
              }]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, job) = request(
        app,
        "POST",
        "/api/v1/models/base_model/download",
        json!({ "requestedGpu": "auto" }),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "model_download");
    assert_eq!(job["payload"]["modelId"], "base_model");
    assert_eq!(job["payload"]["family"], "z-image");
}

#[tokio::test]
async fn lora_catalog_uses_huggingface_cache_install_state() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [{
                "id": "ltx_ic_union",
                "name": "LTX IC Union",
                "family": "ltx-video",
                "icLora": true,
                "conditioningRole": "ic_lora",
                "compatibility": { "families": ["ltx-video"] },
                "source": {
                  "provider": "huggingface",
                  "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                  "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
                }
              }]
            }
            "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");
    let stale_cache_file = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/snapshots/aaa111/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors");
    std::fs::create_dir_all(
        stale_cache_file
            .parent()
            .expect("stale cache file has parent"),
    )
    .expect("stale hf cache creates");
    std::fs::write(&stale_cache_file, b"stale-lora").expect("stale lora cache writes");
    let cache_file = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/snapshots/zzz999/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors");
    std::fs::create_dir_all(cache_file.parent().expect("cache file has parent"))
        .expect("hf cache creates");
    std::fs::write(&cache_file, b"lora").expect("lora cache writes");
    let refs_main = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/refs/main");
    std::fs::create_dir_all(refs_main.parent().expect("refs main has parent"))
        .expect("refs dir creates");
    std::fs::write(&refs_main, b"zzz999").expect("refs main writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(loras[0]["id"], "ltx_ic_union");
    assert_eq!(loras[0]["icLora"], true);
    assert_eq!(loras[0]["conditioningRole"], "ic_lora");
    assert_eq!(loras[0]["installState"], "installed");
    assert_eq!(
        std::path::PathBuf::from(loras[0]["installedPath"].as_str().expect("installed path")),
        cache_file
    );
}

#[test]
fn lora_artifact_paths_exclude_shared_huggingface_cache_files() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let cache_file = temp_dir
        .path()
        .join("data/cache/huggingface/hub/models--owner--repo/snapshots/abc123/lora.safetensors");
    let lora = json!({
        "id": "hf_lora",
        "installedPath": cache_file.display().to_string(),
        "source": {
            "provider": "huggingface",
            "repo": "owner/repo",
            "file": "lora.safetensors"
        }
    });

    assert!(lora_artifact_paths(&lora, temp_dir.path()).is_empty());
}

#[tokio::test]
async fn catalog_delete_routes_remove_manifest_entries_and_owned_artifacts() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let model_dir = temp_dir.path().join("data/models/imports/delete_me");
    let lora_dir = temp_dir.path().join("data/loras/delete_style");
    std::fs::create_dir_all(&model_dir).expect("model dir creates");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("marker writes");
    std::fs::write(lora_dir.join("adapter.safetensors"), b"lora").expect("lora writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "delete_me",
                    "name": "Delete Me",
                    "type": "image",
                    "family": "z-image",
                    "paths": {{ "model": "{}" }}
                  }}]
                }}"#,
            model_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        format!(
            r#"{{
                  "schemaVersion": 1,
                  "loras": [{{
                    "id": "delete_style",
                    "name": "Delete Style",
                    "family": "z-image",
                    "source": {{ "provider": "local", "path": "{}" }}
                  }}]
                }}"#,
            lora_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "moody",
                  "name": "Moody",
                  "workflow": "text_to_image",
                  "model": "delete_me",
                  "loras": [{ "id": "delete_style" }]
                }
              ]
            }
            "#,
    )
    .expect("user presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (model_status, model_delete) = request(
        app.clone(),
        "DELETE",
        "/api/v1/models/delete_me",
        Value::Null,
    )
    .await;
    assert_eq!(model_status, StatusCode::OK);
    assert_eq!(model_delete["removedManifestEntry"], true);
    assert_eq!(model_delete["removedLocalArtifacts"], true);
    assert!(model_delete["warnings"][0]
        .as_str()
        .is_some_and(|warning| warning.contains("Moody")));
    assert!(!model_dir.exists());
    let models_manifest =
        std::fs::read_to_string(config_dir.join("user.models.jsonc")).expect("models reads");
    assert!(!models_manifest.contains("delete_me"));

    let (lora_status, lora_delete) = request(
        app.clone(),
        "DELETE",
        "/api/v1/loras/delete_style?scope=global",
        Value::Null,
    )
    .await;
    assert_eq!(lora_status, StatusCode::OK);
    assert_eq!(lora_delete["removedManifestEntry"], true);
    assert_eq!(lora_delete["removedLocalArtifacts"], true);
    assert!(lora_delete["warnings"][0]
        .as_str()
        .is_some_and(|warning| warning.contains("Moody")));
    assert!(!lora_dir.exists());
    let loras_manifest =
        std::fs::read_to_string(config_dir.join("user.loras.jsonc")).expect("loras reads");
    assert!(!loras_manifest.contains("delete_style"));

    let (models_status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(models_status, StatusCode::OK);
    assert_eq!(models.as_array().expect("models array").len(), 0);
    let (loras_status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(loras_status, StatusCode::OK);
    assert_eq!(loras.as_array().expect("loras array").len(), 0);
}

#[tokio::test]
async fn recipe_preset_crud_routes_persist_global_and_project_presets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "builtin_readonly",
                  "name": "Built-in Readonly",
                  "scope": "builtin",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo"
                }
              ]
            }
            "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"
            // user preset notes survive API writes
            {
              "schemaVersion": 1,
              /* preserve unknown root fields too */
              "futureRoot": true,
              "presets": []
            }
            "#,
    )
    .expect("user recipe presets writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style_lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                },
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                },
                {
                  "id": "deleted_style",
                  "name": "Deleted Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/deleted.safetensors" }
                },
                {
                  "id": "empty_dir_style",
                  "name": "Empty Dir Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/empty-dir" }
                },
                {
                  "id": "unknown_family",
                  "name": "Unknown Family",
                  "triggerWords": [],
                  "compatibility": {},
                  "source": { "provider": "local", "path": "loras/unknown.safetensors" }
                },
                {
                  "id": "no_path_style",
                  "name": "No Path Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    std::fs::create_dir_all(lora_dir.join("empty-dir")).expect("empty lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));
    write_test_safetensors(&lora_dir.join("qwen.safetensors"));
    write_test_safetensors(&lora_dir.join("unknown.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, loras) = request(app.clone(), "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let empty_dir_style = loras
        .as_array()
        .expect("loras array")
        .iter()
        .find(|lora| lora["id"] == "empty_dir_style")
        .expect("empty dir lora listed");
    assert_eq!(empty_dir_style["installState"], "missing");

    // This also pins the positive compatibility path: style_lora is installed and compatible with z_image_turbo.
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Soft Glow",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "order": 30,
            "defaults": { "resolution": "1024x1024" },
            "prompt": { "suffix": "soft glow" },
            "loras": [{ "id": "style_lora", "weight": 0.5 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["id"], "soft_glow");
    assert_eq!(created["scope"], "global");
    assert_eq!(created["builtInLoras"][0]["id"], "style_lora");

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/soft_glow",
        json!({
            "defaults": { "negativePrompt": "noise" },
            "loras": [{ "id": "style_lora", "weight": 0.75 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["defaults"]["negativePrompt"], "noise");
    assert_eq!(updated["loras"][0]["weight"], 0.75);

    let (status, duplicate) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets/soft_glow/duplicate",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(duplicate["id"], "soft_glow_copy");
    assert_eq!(duplicate["name"], "Soft Glow Copy");
    assert_eq!(duplicate["loras"][0]["id"], "style_lora");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
    let (status, project_preset) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "project_soft_glow",
            "name": "Project Soft Glow",
            "scope": "project",
            "projectId": project_id,
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "order": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(project_preset["scope"], "project");
    assert!(project_path.join("recipes/presets.jsonc").is_file());

    for (id, name) in [("beta_order", "Beta Order"), ("alpha_order", "Alpha Order")] {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "id": id,
                "name": name,
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "order": 10
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
    }

    let (status, ordered) = request(
            app.clone(),
            "GET",
            &format!(
                "/api/v1/recipe-presets?projectId={project_id}&workflow=text_to_image&model=z_image_turbo"
            ),
            Value::Null,
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let ordered_ids = ordered
        .as_array()
        .unwrap()
        .iter()
        .map(|preset| preset["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        ordered_ids,
        vec![
            "builtin_readonly",
            "alpha_order",
            "beta_order",
            "soft_glow",
            "soft_glow_copy",
            "project_soft_glow"
        ]
    );

    let (status, scoped) = request(
        app.clone(),
        "GET",
        "/api/v1/recipe-presets?scope=global&workflow=text_to_image&model=z_image_turbo",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(scoped
        .as_array()
        .unwrap()
        .iter()
        .all(|preset| preset["scope"] == "global"));

    let (status, readonly_error) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/builtin_readonly",
        json!({ "name": "Nope" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        readonly_error["detail"],
        "Built-in recipe presets are read-only"
    );

    let (status, project_updated) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/recipe-presets/project_soft_glow?projectId={project_id}"),
        json!({ "prompt": { "suffix": "project update" } }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(project_updated["prompt"]["suffix"], "project update");

    let (status, _, bytes) = request_raw(
        app.clone(),
        "DELETE",
        "/api/v1/recipe-presets/soft_glow",
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let archived: Value = serde_json::from_slice(&bytes).expect("archive response parses");
    assert_eq!(archived["archived"], true);

    let (status, _, bytes) = request_raw(
        app.clone(),
        "DELETE",
        &format!("/api/v1/recipe-presets/project_soft_glow?projectId={project_id}"),
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let archived: Value = serde_json::from_slice(&bytes).expect("project archive response parses");
    assert_eq!(archived["archived"], true);

    let (status, visible) =
        request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!visible
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "soft_glow"));

    let (status, archived_visible) = request(
        app.clone(),
        "GET",
        "/api/v1/recipe-presets?includeArchived=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(archived_visible
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "soft_glow" && preset["archived"] == true));

    let (status, unarchived) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/soft_glow",
        json!({ "archived": false }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unarchived["archived"], false);

    let saved_manifest_text = std::fs::read_to_string(config_dir.join("user.recipe-presets.jsonc"))
        .expect("user recipe preset manifest reads");
    assert!(saved_manifest_text.starts_with(API_MANAGED_MANIFEST_HEADER));
    assert!(!saved_manifest_text.contains("// user preset notes survive API writes"));
    assert!(!saved_manifest_text.contains("/* preserve unknown root fields too */"));
    let saved_manifest: Value = serde_json::from_str(&strip_jsonc_comments(&saved_manifest_text))
        .expect("saved manifest parses");
    assert_eq!(saved_manifest["futureRoot"], true);

    let (status, second_update) = request(
        app.clone(),
        "PATCH",
        "/api/v1/recipe-presets/soft_glow",
        json!({ "order": 31 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second_update["order"], 31);
    let second_manifest_text =
        std::fs::read_to_string(config_dir.join("user.recipe-presets.jsonc"))
            .expect("user recipe preset manifest reads after second write");
    assert!(second_manifest_text.starts_with(API_MANAGED_MANIFEST_HEADER));
    assert_eq!(
        second_manifest_text
            .matches(API_MANAGED_MANIFEST_HEADER)
            .count(),
        1
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "Bad Id",
            "name": "Bad Id",
            "model": "z_image_turbo",
            "workflow": "text_to_image"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset id must use lowercase letters, numbers, dashes, or underscores"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Bad Order",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "order": "high"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset order must be an integer"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Bad Workflow",
            "model": "z_image_turbo",
            "workflow": "text_to_video"
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Model z_image_turbo does not support workflow text_to_video"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Too Many LoRAs",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [
                { "id": "style_one" },
                { "id": "style_two" },
                { "id": "style_three" },
                { "id": "style_four" }
            ]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe presets can include at most 3 LoRAs"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Overweighted LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "style_one", "weight": 2.5 }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA weight must be between -2 and 2"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Missing LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "missing_lora" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA not found: missing_lora"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Deleted LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "deleted_style" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA is not installed: deleted_style"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "No Path LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "no_path_style" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "Recipe preset LoRA is not installed: no_path_style"
    );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Unknown Family LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "unknown_family" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
            bad_error["detail"],
            "LoRA unknown_family has no declared family; cannot verify compatibility with model z_image_turbo"
        );

    let (bad_status, bad_error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Wrong Family LoRA",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bad_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    let create_one = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "concurrent_one",
            "name": "Concurrent One",
            "model": "z_image_turbo",
            "workflow": "text_to_image"
        }),
    );
    let create_two = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "id": "concurrent_two",
            "name": "Concurrent Two",
            "model": "z_image_turbo",
            "workflow": "text_to_image"
        }),
    );
    let ((status_one, _), (status_two, _)) = tokio::join!(create_one, create_two);
    assert_eq!(status_one, StatusCode::CREATED);
    assert_eq!(status_two, StatusCode::CREATED);
    let (status, concurrent_presets) = request(
        app.clone(),
        "GET",
        "/api/v1/recipe-presets?scope=global&includeArchived=true",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(concurrent_presets
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "concurrent_one"));
    assert!(concurrent_presets
        .as_array()
        .unwrap()
        .iter()
        .any(|preset| preset["id"] == "concurrent_two"));

    let (bad_status, bad_error) = request(
        app,
        "GET",
        "/api/v1/recipe-presets?workflow=bogus",
        Value::Null,
    )
    .await;
    assert_eq!(bad_status, StatusCode::BAD_REQUEST);
    assert_eq!(bad_error["detail"], "Unsupported recipe preset workflow");
}

#[tokio::test]
async fn recipe_preset_accepts_full_studio_snapshot_and_rejects_bad_defaults() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style_lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // A full studio snapshot: literal prompt + cfg/steps/sampler + a weighted LoRA.
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Atrium Look",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "modes": ["text_to_image", "character_image", "style_variations"],
            "defaults": {
                "prompt": "a portrait in the atrium",
                "negativePrompt": "blurry",
                "resolution": "1024x1024",
                "count": 4,
                "mode": "character_image",
                "guidanceScale": 5.0,
                "steps": 28,
                "sampler": "default",
                "ipAdapterScale": 0.8
            },
            "loras": [{ "id": "style_lora", "weight": 0.65 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["id"], "atrium_look");
    // The flatten-extra defaults round-trip intact through persistence.
    assert_eq!(created["defaults"]["prompt"], "a portrait in the atrium");
    assert_eq!(created["defaults"]["guidanceScale"], 5.0);
    assert_eq!(created["defaults"]["steps"], 28);
    assert_eq!(created["defaults"]["sampler"], "default");
    assert_eq!(created["defaults"]["mode"], "character_image");
    assert_eq!(created["builtInLoras"][0]["id"], "style_lora");
    assert_eq!(created["builtInLoras"][0]["weight"], 0.65);

    // Re-reading the catalog returns the persisted snapshot.
    let (status, listed) = request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let saved = listed
        .as_array()
        .expect("presets array")
        .iter()
        .find(|preset| preset["id"] == "atrium_look")
        .expect("saved preset listed");
    assert_eq!(saved["defaults"]["prompt"], "a portrait in the atrium");
    assert_eq!(saved["defaults"]["steps"], 28);

    // An out-of-range guidance scale is rejected.
    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Too Hot",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "defaults": { "guidanceScale": 999.0 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("guidanceScale"),
        "unexpected error: {error}"
    );

    // A non-integer steps value is rejected.
    let (status, _error) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        json!({
            "name": "Bad Steps",
            "model": "z_image_turbo",
            "workflow": "text_to_image",
            "defaults": { "steps": 3.5 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn empty_builtin_preset_and_lora_manifests_ship_empty_catalogs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, presets) =
        request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(presets.as_array().expect("presets array").len(), 0);

    let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(loras.as_array().expect("loras array").len(), 0);
}

#[tokio::test]
async fn legacy_preset_read_defaults_do_not_select_uninstalled_models() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "missing_image_model",
                  "name": "Missing Image Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [{ "provider": "huggingface", "repo": "owner/missing-model", "files": ["*.safetensors"] }],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                { "id": "legacy_text", "name": "Legacy Text", "modes": ["text_to_image"] }
              ]
            }
            "#,
    )
    .expect("user recipe presets writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, presets) = request(app, "GET", "/api/v1/recipe-presets", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let preset = &presets.as_array().expect("presets array")[0];
    assert_eq!(preset["workflow"], "text_to_image");
    assert!(preset.get("model").is_none());
    assert_eq!(
        preset["appliedDefaults"]["notes"][0],
        "workflow inferred from legacy modes as text_to_image"
    );
    assert!(preset["appliedDefaults"]["notes"]
        .as_array()
        .expect("notes array")
        .iter()
        .all(|note| !note
            .as_str()
            .unwrap_or_default()
            .contains("model defaulted")));
}

#[test]
fn model_download_size_helpers_match_contract_shapes() {
    let siblings = json!([
        { "rfilename": "model-00001.safetensors", "size": 100 },
        { "rfilename": "model-00002.safetensors", "size": "200" },
        { "rfilename": "README.md", "size": 50 },
        { "rfilename": "unknown.bin" }
    ]);
    let siblings = siblings.as_array().expect("siblings array");
    assert_eq!(
        super::download_size_from_siblings(siblings, &["*.safetensors".to_owned()]),
        Some(300)
    );
    assert_eq!(
        super::download_size_from_siblings(siblings, &["*.ckpt".to_owned()]),
        None
    );
    assert_eq!(super::json_size_to_u64(&json!("200.5")), None);
    assert_eq!(super::format_bytes(0), "0 B");
    assert_eq!(super::format_bytes(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(
        super::manifest_download_size_bytes(
            &json!({ "downloads": [] }),
            &json!({ "estimatedSizeBytes": "4096" })
        ),
        Some(4096)
    );
    assert_eq!(
        super::manifest_download_size_bytes(&json!({ "sizeBytes": 2048 }), &json!({})),
        Some(2048)
    );
    assert_eq!(
        super::quote_huggingface_repo("owner/model name"),
        "owner/model%20name"
    );
    assert!(super::model_download(&json!({
        "downloads": [{ "repo": "owner/model" }]
    }))
    .is_none());
    assert_eq!(
            super::model_download(&json!({
                "downloads": [
                    { "provider": "huggingface", "repo": "owner/fallback", "estimatedSizeBytes": 1024 },
                    { "provider": "huggingface", "repo": "owner/default", "default": true, "estimatedSizeBytes": 4096 }
                ]
            }))
            .and_then(|download| download.get("repo").and_then(Value::as_str).map(str::to_owned)),
            Some("owner/default".to_owned())
        );
    let mut cache = super::ModelSizeCache::default();
    let key = ("owner/model".to_owned(), vec!["*.safetensors".to_owned()]);
    cache.insert(key.clone(), 300);
    assert_eq!(cache.get(&key), Some(Some(300)));
    // sc-4169: failed estimates are negative-cached — `Some(None)` tells the
    // caller to skip the network — and expire after the TTL (a cache miss).
    let failed = ("owner/offline".to_owned(), vec!["*.safetensors".to_owned()]);
    cache.insert_failure(failed.clone());
    assert_eq!(cache.get(&failed), Some(None));
    cache.insert_failure_expiring_at(failed.clone(), std::time::Instant::now());
    assert_eq!(
        cache.get(&failed),
        None,
        "expired negative entry must be a miss"
    );
    // A later successful estimate replaces a cached failure.
    cache.insert_failure(failed.clone());
    cache.insert(failed.clone(), 700);
    assert_eq!(cache.get(&failed), Some(Some(700)));
    assert!(super::allow_pattern_matches(
        "model-7.safetensors",
        &["model-[0-9].safetensors".to_owned()]
    ));
    if cfg!(windows) {
        assert!(super::allow_pattern_matches(
            "Model.SAFETENSORS",
            &["*.safetensors".to_owned()]
        ));
    }
}

#[test]
fn platform_tagged_downloads_resolve_per_os() {
    // A video model that ships a native MLX-convert checkpoint (macOS) and a diffusers/torch
    // checkpoint (Windows/Linux) for the same model id (sc-3240).
    let model = json!({
        "downloads": [
            { "provider": "huggingface", "repo": "Wan-AI/Wan2.2-TI2V-5B", "platforms": ["macos"] },
            { "provider": "huggingface", "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "platforms": ["windows", "linux"] }
        ]
    });
    let resolved_repo = |model: &Value| {
        super::model_download(model).and_then(|download| {
            download
                .get("repo")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
    };

    // macOS keeps the native MLX-convert source...
    let mut mac = model.clone();
    super::retain_downloads_for_os(&mut mac, "macos");
    assert_eq!(
        resolved_repo(&mac),
        Some("Wan-AI/Wan2.2-TI2V-5B".to_owned())
    );

    // ...Windows/Linux keep the diffusers/torch repo.
    for os in ["windows", "linux"] {
        let mut other = model.clone();
        super::retain_downloads_for_os(&mut other, os);
        assert_eq!(
            resolved_repo(&other),
            Some("Wan-AI/Wan2.2-TI2V-5B-Diffusers".to_owned()),
            "os={os}"
        );
    }

    // Untagged single-repo models are untouched on every OS.
    let mut agnostic = json!({
        "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
    });
    super::retain_downloads_for_os(&mut agnostic, "macos");
    assert_eq!(
        agnostic["downloads"].as_array().map(Vec::len),
        Some(1),
        "agnostic downloads must not be filtered"
    );
}

#[test]
fn lora_family_filter_shapes_match_contract_fallbacks() {
    let shapes = [
        json!({ "families": ["z-image"] }),
        json!({ "compatibleFamilies": ["z-image"] }),
        json!({ "modelFamilies": ["z-image"] }),
        json!({ "compatibility": { "families": ["z-image"] } }),
        json!({ "family": "z-image" }),
    ];
    for lora in shapes {
        assert_eq!(super::lora_families(&lora), vec!["z-image".to_owned()]);
    }
}

#[tokio::test]
async fn malformed_manifest_returns_stable_server_error() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{ "models": [ /*"#,
    )
    .expect("manifest writes");
    std::fs::write(config_dir.join("user.models.jsonc"), r#"{ "models": [] }"#)
        .expect("manifest writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(app, "GET", "/api/v1/models", Value::Null).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.starts_with("Failed to parse manifest")));
}

#[tokio::test]
async fn generation_routes_reject_invalid_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({ "projectId": "project-1", "prompt": "x".repeat(4001) }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "image_to_video",
            "prompt": "missing source image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn timeline_routes_reject_invalid_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created_project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Invalid Timeline Project" }),
    )
    .await;
    let project_id = created_project["id"]
        .as_str()
        .expect("project id")
        .to_owned();

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({ "name": "Main timeline", "aspectRatio": "4:3", "fps": 30 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (_, mut timeline) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/timelines"),
        json!({ "name": "Main timeline" }),
    )
    .await;
    let timeline_id = timeline["id"].as_str().expect("timeline id").to_owned();
    timeline["tracks"][0]["items"] = json!([
        {
            "id": "item-1",
            "trackId": "track_main",
            "assetId": "asset-1",
            "type": "video",
            "displayName": "Clip",
            "sourceIn": 4,
            "sourceOut": 2,
            "timelineStart": 0,
            "timelineEnd": 4
        }
    ]);
    let (status, _) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline.clone() }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    timeline["tracks"][0]["items"][0]["sourceOut"] = json!(6);
    timeline["tracks"][0]["kind"] = json!("audio_v2");
    let (status, _) = request(
        app,
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[test]
fn frame_extract_rejects_non_finite_playhead() {
    let result = super::validate_frame_extract(&super::FrameExtractRequest {
        playhead_seconds: f64::NAN,
        intended_use: "reuse".to_owned(),
        requested_gpu: "auto".to_owned(),
    });

    assert!(result.is_err());
}

#[test]
fn image_dimension_cap_covers_sensenova_buckets() {
    // Raised so SenseNova-U1's true trained buckets (largest 3456) pass.
    assert_eq!(super::MAX_IMAGE_DIMENSION, 4096);
    assert!(super::validate_dimension(2720, "width", super::MAX_IMAGE_DIMENSION).is_ok());
    assert!(super::validate_dimension(3456, "height", super::MAX_IMAGE_DIMENSION).is_ok());
    assert!(super::validate_dimension(4096, "width", super::MAX_IMAGE_DIMENSION).is_ok());
    assert!(super::validate_dimension(4097, "width", super::MAX_IMAGE_DIMENSION).is_err());
    assert!(super::validate_dimension(255, "width", super::MAX_IMAGE_DIMENSION).is_err());
}

#[test]
fn vqa_job_validation_requires_question_and_asset() {
    let base = super::VqaJobRequest {
        project_id: "project-1".to_owned(),
        project_name: None,
        source_asset_id: "asset_1".to_owned(),
        question: "What is in this image?".to_owned(),
        model: "sensenova_u1_8b".to_owned(),
        max_new_tokens: 256,
        requested_gpu: "auto".to_owned(),
        advanced: serde_json::Map::new(),
    };
    assert!(super::validate_vqa_job(&base).is_ok());

    // The UI's length presets are all valid.
    for tokens in [256u32, 512, 1024] {
        let mut request = base.clone();
        request.max_new_tokens = tokens;
        assert!(super::validate_vqa_job(&request).is_ok());
    }

    let mut blank_question = base.clone();
    blank_question.question = "   ".to_owned();
    assert!(super::validate_vqa_job(&blank_question).is_err());

    let mut missing_asset = base.clone();
    missing_asset.source_asset_id = String::new();
    assert!(super::validate_vqa_job(&missing_asset).is_err());

    let mut missing_project = base.clone();
    missing_project.project_id = String::new();
    assert!(super::validate_vqa_job(&missing_project).is_err());

    let mut too_many_tokens = base.clone();
    too_many_tokens.max_new_tokens = 4096;
    assert!(super::validate_vqa_job(&too_many_tokens).is_err());
}

#[test]
fn interleave_job_validation_bounds_prompt_images_and_assets() {
    let base = super::InterleaveJobRequest {
        project_id: "project-1".to_owned(),
        project_name: None,
        prompt: "A short illustrated guide to brewing tea".to_owned(),
        source_asset_ids: Vec::new(),
        model: "sensenova_u1_8b".to_owned(),
        max_images: 6,
        width: 1024,
        height: 1024,
        seed: None,
        requested_gpu: "auto".to_owned(),
        advanced: serde_json::Map::new(),
    };
    assert!(super::validate_interleave_job(&base).is_ok());

    // Optional input images (it2i) are allowed.
    let mut with_sources = base.clone();
    with_sources.source_asset_ids = vec!["asset_1".to_owned(), "asset_2".to_owned()];
    assert!(super::validate_interleave_job(&with_sources).is_ok());

    let mut blank_prompt = base.clone();
    blank_prompt.prompt = "   ".to_owned();
    assert!(super::validate_interleave_job(&blank_prompt).is_err());

    let mut missing_project = base.clone();
    missing_project.project_id = String::new();
    assert!(super::validate_interleave_job(&missing_project).is_err());

    let mut zero_images = base.clone();
    zero_images.max_images = 0;
    assert!(super::validate_interleave_job(&zero_images).is_err());

    let mut too_many_images = base.clone();
    too_many_images.max_images = 11;
    assert!(super::validate_interleave_job(&too_many_images).is_err());

    let mut blank_asset = base.clone();
    blank_asset.source_asset_ids = vec!["  ".to_owned()];
    assert!(super::validate_interleave_job(&blank_asset).is_err());

    let mut tiny = base.clone();
    tiny.width = 64;
    assert!(super::validate_interleave_job(&tiny).is_err());
}

#[tokio::test]
async fn project_file_route_serves_files_and_rejects_traversal() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Files" }),
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());
    let media_path = project_path.join("assets/images/image.png");
    std::fs::write(&media_path, b"image-bytes").expect("media writes");
    let outside_path = temp_dir.path().join("data").join("outside.txt");
    std::fs::write(outside_path, b"nope").expect("outside writes");

    let (status, headers, bytes) = request_raw(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/files/assets/images/image.png"),
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bytes, b"image-bytes");
    assert_eq!(
        headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("image/png")
    );

    let (status, _, bytes) = request_raw(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/files/%2E%2E%2F%2E%2E%2Foutside.txt"),
        Body::empty(),
        &[],
    )
    .await;
    let error: Value = serde_json::from_slice(&bytes).expect("json error parses");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Invalid project file path");

    let (status, _, bytes) = request_raw(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/files/%2E%2E%5C%2E%2E%5Coutside.txt"),
        Body::empty(),
        &[],
    )
    .await;
    let error: Value = serde_json::from_slice(&bytes).expect("json error parses");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "Invalid project file path");
}

#[tokio::test]
async fn project_file_route_serves_byte_ranges() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Ranges" }),
    )
    .await;
    let project_id = created["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());
    let media_path = project_path.join("assets/videos/clip.mp4");
    std::fs::write(&media_path, b"0123456789").expect("media writes");
    let uri = format!("/api/v1/projects/{project_id}/files/assets/videos/clip.mp4");

    // A full request advertises range support so WebKit knows it can seek.
    let (status, headers, bytes) = request_raw(app.clone(), "GET", &uri, Body::empty(), &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bytes, b"0123456789");
    assert_eq!(
        headers.get("accept-ranges").and_then(|v| v.to_str().ok()),
        Some("bytes")
    );

    // A bounded range yields 206 with the exact slice and Content-Range.
    let (status, headers, bytes) = request_raw(
        app.clone(),
        "GET",
        &uri,
        Body::empty(),
        &[("range", "bytes=2-5")],
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(bytes, b"2345");
    assert_eq!(
        headers.get("content-range").and_then(|v| v.to_str().ok()),
        Some("bytes 2-5/10")
    );
    assert_eq!(
        headers.get("accept-ranges").and_then(|v| v.to_str().ok()),
        Some("bytes")
    );

    // An open-ended range serves to EOF (this is how WebKit fetches the
    // trailing moov atom on a non-faststart MP4).
    let (status, _, bytes) = request_raw(
        app.clone(),
        "GET",
        &uri,
        Body::empty(),
        &[("range", "bytes=7-")],
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(bytes, b"789");

    // An unsatisfiable range is rejected with 416.
    let (status, _, _) =
        request_raw(app, "GET", &uri, Body::empty(), &[("range", "bytes=99-")]).await;
    assert_eq!(status, StatusCode::RANGE_NOT_SATISFIABLE);
}

#[tokio::test]
async fn character_studio_routes_manage_references_loras_and_test_jobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let config_dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                }
              ]
            }
            "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Characters" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "reference.png",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (status, character) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters"),
        json!({ "name": "Mira", "type": "person" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(character["name"], "Mira");
    let character_id = character["id"].as_str().expect("character id").to_owned();

    let (status, with_reference) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/references"),
        json!({ "assetId": asset_id, "approved": false }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        with_reference["references"][0]["asset"]["displayName"],
        "reference.png"
    );

    let (status, updated) = request(
        app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/references/{asset_id}"),
        json!({ "approved": true, "role": "hero" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["approvedReferences"][0]["assetId"], asset_id);

    let sidecar_path = project_path.join(
        asset["sidecarPath"]
            .as_str()
            .expect("asset sidecar path")
            .replace('/', std::path::MAIN_SEPARATOR_STR),
    );
    let asset_sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(sidecar_path).expect("asset sidecar reads"))
            .expect("asset sidecar parses");
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"][0]["characterId"],
        character_id
    );
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"][0]["approved"],
        true
    );

    let (status, with_look) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/looks"),
            json!({ "name": "Rain coat", "approvedReferenceIds": [asset_id], "recipeSettings": { "style": "noir" } }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(with_look["looks"][0]["recipeSettings"]["style"], "noir");
    let look_id = with_look["looks"][0]["id"]
        .as_str()
        .expect("look id")
        .to_owned();

    let lora_dir = data_dir.join("loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    let lora_source = lora_dir.join("mira.safetensors");
    write_test_safetensors(&lora_source);
    let (status, with_lora) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
        json!({
            "name": "Mira LoRA",
            "sourcePath": lora_source.display().to_string(),
            "compatibility": { "families": ["z-image"] },
            "triggerWords": ["mira"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(with_lora["loras"][0]["copiedIntoProject"], true);
    let project_lora_path = project_path.join(
        with_lora["loras"][0]["projectPath"]
            .as_str()
            .expect("project lora path")
            .replace('/', std::path::MAIN_SEPARATOR_STR),
    );
    assert_eq!(
        std::fs::read(project_lora_path).expect("lora copied"),
        test_safetensors_bytes()
    );

    let (status, test_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
        json!({ "prompt": "portrait", "lookId": look_id, "count": 2 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(test_job["type"], "image_generate");
    assert_eq!(test_job["payload"]["mode"], "character_image");
    assert_eq!(test_job["payload"]["characterId"], character_id);
    // Regression (sc-2074): the worker's image_request_from_job requires payload.projectId;
    // the test-job handler must inject it (it isn't carried by the column alone).
    assert_eq!(test_job["payload"]["projectId"], project_id);
    assert_eq!(
        test_job["payload"]["advanced"]["approvedReferenceIds"][0],
        asset_id
    );

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/archive"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, visible) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/characters"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(visible.as_array().unwrap().len(), 0);
    let (status, archived) = request(
        app,
        "GET",
        &format!("/api/v1/projects/{project_id}/characters?includeArchived=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(archived.as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn worker_heartbeat_interrupts_previous_active_job_through_http() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_generate"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_generate", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    let job_id = created["id"].as_str().expect("job id is string");
    // The owning worker reports at least one heartbeat for the job, so a
    // later idle heartbeat is a genuine restart (not a claim race) and must
    // reclaim the now-orphaned active job.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "busy", "currentJobId": job_id, "loadedModels": [] }),
    )
    .await;

    let (status, worker) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "idle", "currentJobId": null, "loadedModels": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(worker["currentJobId"], Value::Null);

    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
    assert_eq!(job["workerId"], Value::Null);
}

#[tokio::test]
async fn access_token_is_enforced_on_protected_routes() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, access) = request(app.clone(), "GET", "/api/v1/access", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(access["authRequired"], true);

    let (status, error) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "SceneWorks access token required");

    let (status, jobs) = request_with_headers(
        app,
        "GET",
        "/api/v1/jobs",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(jobs, json!([]));
}

#[tokio::test]
async fn public_health_withholds_host_paths_when_a_token_is_configured() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");

    // No token: single-user/local, directories stay for diagnostics.
    let open_app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, health) = request(open_app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["authRequired"], false);
    assert!(health.get("directories").is_some());

    // Token configured but /health is public: don't leak absolute host paths.
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let guarded_app = create_app(settings).expect("app creates");
    let (status, health) = request(guarded_app, "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["authRequired"], true);
    assert!(health.get("directories").is_none());
}

#[test]
fn requires_token_only_gates_api_paths() {
    // Non-API paths (embedded UI / SPA fallback) must never require a token,
    // or the browser cannot load the bundle to prompt for one.
    assert!(!requires_token("/"));
    assert!(!requires_token("/assets/index-abc.js"));
    assert!(!requires_token("/projects/some-id"));
    // Public API paths stay open; other API paths stay gated.
    assert!(!requires_token("/api/v1/health"));
    assert!(requires_token("/api/v1/jobs"));
    assert!(requires_token("/api/v1/projects"));
}

#[tokio::test]
async fn embedded_ui_root_is_reachable_with_access_token_set() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    // With a token configured and no header, the embedded UI root and assets
    // must not be blocked by auth (404 here under default features since the
    // bundle isn't embedded; the point is it is NOT 401).
    let (status, _) = request(app.clone(), "GET", "/", Value::Null).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = request(app.clone(), "GET", "/assets/app.js", Value::Null).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED);
    // API routes stay protected.
    let (status, _) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[cfg(feature = "embed-web")]
#[test]
fn embedded_ui_csp_locks_down_scripts_but_allows_app_resources() {
    let csp = super::web_assets::CONTENT_SECURITY_POLICY;
    // The whole point: scripts only from this origin, no inline/eval escape hatch.
    assert!(csp.contains("script-src 'self'"));
    assert!(!csp.contains("script-src 'self' 'unsafe-inline'"));
    assert!(!csp.contains("unsafe-eval"));
    // Resources the app genuinely needs.
    assert!(csp.contains("default-src 'self'"));
    assert!(csp.contains("font-src 'self' https://fonts.gstatic.com data:"));
    assert!(csp.contains("https://fonts.googleapis.com"));
    assert!(csp.contains("img-src 'self' data: blob:"));
    // Tauri IPC for the navigated desktop webview.
    assert!(csp.contains("ipc:"));
    // Hardening directives.
    assert!(csp.contains("object-src 'none'"));
    assert!(csp.contains("frame-ancestors 'none'"));
}

#[test]
fn inprocess_worker_defaults_to_cpu_utility() {
    // Default (and blank override) → cpu so utility capabilities are
    // advertised regardless of the ambient SCENEWORKS_GPU_ID.
    assert_eq!(inprocess_worker_gpu_id(None), "cpu");
    assert_eq!(inprocess_worker_gpu_id(Some("   ".to_owned())), "cpu");
    // Explicit override is honored.
    assert_eq!(inprocess_worker_gpu_id(Some("auto".to_owned())), "auto");
    assert_eq!(inprocess_worker_gpu_id(Some("0".to_owned())), "0");
}

#[tokio::test]
async fn bearer_token_is_accepted_for_access_verification() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, verified) = request_with_headers(
        app,
        "POST",
        "/api/v1/auth/verify",
        Value::Null,
        &[("authorization", "Bearer secret-token")],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(verified["ok"], true);
}

#[tokio::test]
async fn event_tickets_are_protected_and_match_contract_shape() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "SceneWorks access token required");

    let (status, ticket) = request_with_headers(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        Value::Null,
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ticket["ticket"]
        .as_str()
        .is_some_and(|value| value.len() == 32 && value.chars().all(|c| c.is_ascii_hexdigit())));
    assert_eq!(ticket["expiresInSeconds"], 30);

    let (status, error) = request(
        app,
        "GET",
        "/api/v1/jobs/events?ticket=missing",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(error["detail"], "Invalid or expired event stream ticket");
}

#[tokio::test]
async fn lagged_event_subscribers_are_disconnected() {
    let hub = EventHub::default();
    let mut stream = hub.subscribe();

    for index in 0..EVENT_BUFFER_SIZE {
        hub.publish(EventMessage {
            event: "job.updated".to_owned(),
            data: json!({ "index": index }).to_string(),
        });
    }
    hub.publish(EventMessage {
        event: "job.updated".to_owned(),
        data: json!({ "index": EVENT_BUFFER_SIZE }).to_string(),
    });

    for _ in 0..EVENT_BUFFER_SIZE {
        assert!(stream.next().await.is_some());
    }
    assert!(stream.next().await.is_none());
}

#[test]
fn heartbeat_event_matches_contract_wire_shape() {
    assert_eq!(HEARTBEAT_SSE_DATA, "{}");
    assert_eq!(HEARTBEAT_SSE_WIRE, "event: heartbeat\ndata: {}\n\n");
}

#[tokio::test]
async fn cors_preflight_allows_frontend_origin_and_token_header() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let request = Request::builder()
        .method("OPTIONS")
        .uri("/api/v1/jobs")
        .header("origin", "http://localhost:5173")
        .header("access-control-request-method", "POST")
        .header("access-control-request-headers", "X-SceneWorks-Token")
        .body(Body::empty())
        .expect("request builds");

    let response = app.oneshot(request).await.expect("response returns");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .and_then(|value| value.to_str().ok()),
        Some("http://localhost:5173")
    );
    assert!(response
        .headers()
        .get("access-control-allow-headers")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("x-sceneworks-token")));
}

/// The watchdog must stay pending while the parent lives and resolve once it
/// exits — the desktop-sidecar orphan fix. Mirrors the Python worker's
/// parent-death test: spawn a dummy parent, confirm it isn't flagged alive
/// falsely, kill it, and assert the future then resolves promptly.
#[cfg(unix)]
#[tokio::test]
async fn parent_death_resolves_when_watched_parent_exits() {
    let mut parent = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .expect("spawn dummy parent");
    let pid = parent.id() as i32;

    assert!(
        super::pid_alive(pid),
        "freshly spawned parent reads as dead"
    );
    // Still alive -> the watchdog must not resolve within a poll cycle.
    assert!(
        tokio::time::timeout(Duration::from_millis(200), super::parent_death(Some(pid)),)
            .await
            .is_err(),
        "watchdog resolved while the parent was still alive"
    );

    parent.kill().expect("kill dummy parent");
    parent.wait().expect("reap dummy parent");
    assert!(!super::pid_alive(pid), "reaped parent still reads as alive");

    // Now gone -> the watchdog resolves on its next check.
    tokio::time::timeout(
        super::PARENT_POLL_INTERVAL + Duration::from_secs(2),
        super::parent_death(Some(pid)),
    )
    .await
    .expect("watchdog did not resolve after the parent exited");
}

/// No configured parent (server/Docker) -> the watchdog future never resolves.
#[cfg(unix)]
#[tokio::test]
async fn parent_death_never_fires_without_a_parent_pid() {
    assert!(
        tokio::time::timeout(Duration::from_millis(200), super::parent_death(None))
            .await
            .is_err(),
        "watchdog fired with no parent PID configured"
    );
}

/// PIDs of 0 or 1 (and unset/garbage) yield no parent to watch.
#[cfg(unix)]
#[test]
fn parent_pid_to_watch_rejects_init_and_invalid_values() {
    use std::env;
    // Serialize: the helper reads a process-global env var.
    for (value, expected) in [
        (Some("0"), None),
        (Some("1"), None),
        (Some("-5"), None),
        (Some(" not-a-pid "), None),
        (Some(""), None),
        (Some(" 4242 "), Some(4242_i32)),
        (None, None),
    ] {
        match value {
            Some(v) => env::set_var("SCENEWORKS_PARENT_PID", v),
            None => env::remove_var("SCENEWORKS_PARENT_PID"),
        }
        assert_eq!(super::parent_pid_to_watch(), expected, "value={value:?}");
    }
    env::remove_var("SCENEWORKS_PARENT_PID");
}

#[tokio::test]
async fn credentials_routes_store_redact_and_delete() {
    let temp = tempfile::tempdir().expect("tempdir");
    let settings = test_settings(&temp);

    // Save a credential; PUT returns the updated, redacted listing.
    let (status, body) = request(
        create_app(settings.clone()).expect("app creates"),
        "PUT",
        "/api/v1/credentials",
        json!({ "host": "https://Civitai.com", "label": "Civit.ai", "scheme": "query", "token": "secret-key" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let list = body.as_array().expect("array body");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["host"], "civitai.com"); // normalized
    assert_eq!(list[0]["label"], "Civit.ai");
    assert_eq!(list[0]["scheme"], "query");
    assert_eq!(list[0]["present"], true);
    assert!(
        list[0].get("token").is_none(),
        "listing must not include the token"
    );
    assert!(
        !body.to_string().contains("secret-key"),
        "token leaked in the response"
    );

    // A separate GET is likewise redacted.
    let (status, body) = request(
        create_app(settings.clone()).expect("app creates"),
        "GET",
        "/api/v1/credentials",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.to_string().contains("secret-key"));

    // An empty token is rejected.
    let (status, _) = request(
        create_app(settings.clone()).expect("app creates"),
        "PUT",
        "/api/v1/credentials",
        json!({ "host": "huggingface.co", "token": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Delete returns the now-empty listing.
    let (status, body) = request(
        create_app(settings).expect("app creates"),
        "DELETE",
        "/api/v1/credentials/civitai.com",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.as_array().expect("array body").is_empty());
}

#[tokio::test]
async fn credentials_routes_require_the_access_token() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut settings = test_settings(&temp);
    settings.access_token = "s3cret".to_owned();

    let (status, _) = request(
        create_app(settings.clone()).expect("app creates"),
        "GET",
        "/api/v1/credentials",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let (status, _) = request_with_headers(
        create_app(settings).expect("app creates"),
        "GET",
        "/api/v1/credentials",
        Value::Null,
        &[("x-sceneworks-token", "s3cret")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Snapshot tests for recipe presets JSON round-trip parity.
/// These tests capture the endpoint responses before and after the Value→typed-contract conversion.
/// The conversion must preserve JSON structure, field order, null vs absent, and number formats.
///
/// To update snapshots after validating the conversion preserves parity:
/// 1. Run tests with `SNAPSHOT_UPDATE=true` to capture new baselines
/// 2. Compare old snapshots against new to verify no structural changes
#[cfg(test)]
mod recipe_presets_parity {
    use super::*;
    use serde_json::json;

    fn setup_recipe_preset_fixtures(temp_dir: &tempfile::TempDir) {
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("config dir creates");

        // Builtin presets: full schema representation
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "default_t2i",
                  "name": "Default T2I",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "modes": ["text_to_image"],
                  "order": 1,
                  "defaults": {
                    "count": 1,
                    "resolution": "1024x1024",
                    "negativePrompt": ""
                  },
                  "prompt": {
                    "prefix": "",
                    "suffix": ""
                  },
                  "loras": [],
                  "ui": {
                    "description": "Default text-to-image generation"
                  }
                },
                {
                  "id": "cinematic",
                  "name": "Cinematic",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "modes": ["text_to_image"],
                  "order": 10,
                  "defaults": {
                    "count": 4,
                    "resolution": "1280x720",
                    "negativePrompt": "flat lighting, low contrast"
                  },
                  "prompt": {
                    "prefix": "cinematic",
                    "suffix": "cinematic lighting, volumetric"
                  },
                  "loras": [
                    {
                      "id": "style-lora",
                      "loraId": "style-lora",
                      "sourceUrl": "https://example.com/loras/cinematic.safetensors",
                      "name": "Cinematic Style",
                      "displayName": "Cinematic Style",
                      "compatibility": { "families": ["z-image"] },
                      "weight": 0.75,
                      "trigger": "cinematic style"
                    }
                  ],
                  "ui": {
                    "description": "Cinematic lighting and composition"
                  }
                }
              ]
            }
            "#,
        )
        .expect("builtin presets write");

        // User presets: minimal schema (tests merging + defaults)
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "cinematic",
                  "name": "My Cinematic",
                  "model": "z_image_turbo",
                  "workflow": "text_to_image",
                  "defaults": {
                    "count": 2,
                    "resolution": "1280x720"
                  },
                  "prompt": {
                    "suffix": "my custom lighting"
                  }
                },
                {
                  "id": "legacy_edit",
                  "name": "Legacy Edit",
                  "model": "z_image_turbo",
                  "modes": ["edit_image"],
                  "builtInLoras": [
                    {
                      "id": "style-lora",
                      "weight": 0.25
                    }
                  ]
                }
              ]
            }
            "#,
        )
        .expect("user presets write");

        // Builtin models for workflow validation
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [],
                  "paths": { "model": "data/models/z_image_turbo" },
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models write");

        // Builtin loras for compatibility validation
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style-lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": ["cinematic style"],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
        )
        .expect("builtin loras write");

        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras write");

        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "entries": [] }"#,
        )
        .expect("user models write");

        // Install marker for z_image_turbo
        let model_dir = temp_dir.path().join("data/models/z_image_turbo");
        std::fs::create_dir_all(&model_dir).expect("model dir creates");
        std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("marker writes");

        // LoRA artifact
        let lora_dir = temp_dir.path().join("data/loras");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        write_test_safetensors(&lora_dir.join("style.safetensors"));
    }

    #[tokio::test]
    async fn recipe_presets_list_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(app, "GET", "/api/v1/recipe-presets", Value::Null).await;
        assert_eq!(status, StatusCode::OK);

        let presets = response.as_array().expect("response is array");
        // Builtin: default_t2i, cinematic
        // User: cinematic (merge), legacy_edit (new)
        // Result: default_t2i, cinematic (merged), legacy_edit
        assert_eq!(presets.len(), 3, "builtin+user merged presets");

        // Find and verify the cinematic preset (builtin + user merge)
        let cinematic = presets
            .iter()
            .find(|p| p["id"] == "cinematic")
            .expect("cinematic preset exists");

        // Verify merge: user values override builtin
        assert_eq!(
            cinematic["name"], "My Cinematic",
            "user name overrides builtin"
        );
        assert_eq!(cinematic["scope"], "global");
        assert_eq!(cinematic["workflow"], "text_to_image", "from builtin");
        assert_eq!(cinematic["model"], "z_image_turbo", "from builtin");
        assert_eq!(cinematic["defaults"]["count"], 2, "from user override");
        assert_eq!(cinematic["defaults"]["resolution"], "1280x720");
        assert!(
            cinematic["defaults"]["negativePrompt"].is_null(),
            "user didn't specify, builtin is empty"
        );
        assert_eq!(
            cinematic["prompt"]["suffix"], "my custom lighting",
            "user override"
        );
        // prefix is omitted when empty (skip_serializing_if = "is_empty" behavior)
        assert!(
            cinematic["prompt"]["prefix"].is_null()
                || cinematic["prompt"]["prefix"].as_str().is_some()
        );
        assert!(cinematic["builtInLoras"].is_array(), "computed loras field");
        assert!(cinematic["manifestPath"].is_string(), "computed field");

        // Verify default_t2i (builtin only)
        let default_t2i = presets
            .iter()
            .find(|p| p["id"] == "default_t2i")
            .expect("default_t2i from builtin");
        assert_eq!(default_t2i["name"], "Default T2I");
        assert_eq!(default_t2i["order"], 1);

        // Verify legacy_edit (user only)
        let legacy_edit = presets
            .iter()
            .find(|p| p["id"] == "legacy_edit")
            .expect("legacy_edit from user");
        assert_eq!(legacy_edit["name"], "Legacy Edit");
        assert_eq!(legacy_edit["workflow"], "edit_image", "inferred from modes");
    }

    #[tokio::test]
    async fn recipe_presets_get_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) =
            request(app, "GET", "/api/v1/recipe-presets/cinematic", Value::Null).await;
        assert_eq!(status, StatusCode::OK);

        let preset = response;
        assert_eq!(preset["id"], "cinematic");
        assert_eq!(preset["name"], "My Cinematic");
        assert_eq!(preset["scope"], "global");
        // Verify both loras and builtInLoras are present
        assert!(preset["loras"].is_array());
        assert!(preset["builtInLoras"].is_array());
    }

    #[tokio::test]
    async fn recipe_presets_create_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(
            app,
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Custom Preset",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "defaults": {
                    "count": 2,
                    "resolution": "1024x1024"
                },
                "prompt": {
                    "suffix": "custom suffix"
                },
                "loras": []
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let preset = response;
        assert_eq!(preset["name"], "Custom Preset");
        assert_eq!(preset["workflow"], "text_to_image");
        assert!(preset["id"].is_string(), "id auto-generated from name");
        assert_eq!(preset["scope"], "global");
        assert!(preset["createdAt"].is_string());
        assert!(preset["updatedAt"].is_string());
    }

    #[tokio::test]
    async fn recipe_presets_update_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(
            app,
            "PATCH",
            "/api/v1/recipe-presets/cinematic",
            json!({
                "name": "Updated Cinematic",
                "defaults": {
                    "count": 6,
                    "negativePrompt": "blurry"
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let preset = response;
        assert_eq!(preset["id"], "cinematic");
        assert_eq!(preset["name"], "Updated Cinematic");
        assert_eq!(preset["defaults"]["count"], 6);
        assert_eq!(preset["defaults"]["negativePrompt"], "blurry");
    }

    #[tokio::test]
    async fn recipe_presets_duplicate_snapshot() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        setup_recipe_preset_fixtures(&temp_dir);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, response) = request(
            app,
            "POST",
            "/api/v1/recipe-presets/cinematic/duplicate",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let preset = response;
        assert!(preset["id"]
            .as_str()
            .is_some_and(|id| id.contains("cinematic")));
        assert!(preset["name"]
            .as_str()
            .is_some_and(|name| name.contains("Cinematic")));
        assert!(preset["id"] != "cinematic", "new id is different");
        assert!(preset["name"] != "My Cinematic", "new name is different");
    }
}

#[test]
fn resolve_base_model_path_descends_into_hf_snapshot() {
    // Trainers read their weight tree (z-image/sdxl: tokenizer/ text_encoder/ unet|transformer/
    // vae/; ltx: transformer.safetensors / vae_*.safetensors / connector.safetensors) from inside
    // the HF snapshot dir, not the repo cache root. Resolving to the repo root made every
    // HF-cache base model fail at trainer load — z-image "tokenizer: No such file or directory",
    // sdxl "read vocab.json: No such file or directory", ltx "Path must point to a local file".
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = super::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "z_image_turbo")
        .expect("z_image_turbo target");
    let repo = target.base_model_repo.clone().expect("repo set");

    // Materialize an HF hub cache: refs/main -> a snapshot holding the tokenizer.
    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let revision = "abc123";
    let snapshot = repo_root.join("snapshots").join(revision);
    std::fs::create_dir_all(snapshot.join("tokenizer")).expect("create snapshot tree");
    std::fs::write(snapshot.join("tokenizer").join("tokenizer.json"), "{}")
        .expect("write tokenizer.json");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), revision).expect("write refs/main");

    let resolved = resolve_base_model_path(&target, &data_dir);

    assert_eq!(
        resolved,
        snapshot.display().to_string(),
        "resolver must descend into the snapshot dir, not stop at the repo root"
    );
    assert!(
        std::path::Path::new(&resolved)
            .join("tokenizer")
            .join("tokenizer.json")
            .is_file(),
        "the component tree must be reachable from the resolved path"
    );
}

#[test]
fn resolve_base_model_path_prefers_converted_mlx_dir_for_conversion_models() {
    // `requiresConversion` models (Wan) keep usable weights in <data>/models/mlx/<id>, while the
    // HF cache holds only the native *source* checkpoint the converter consumes. Resolving Wan
    // training to the HF source made the trainer fail ("wan umt5 tokenizer: No such file"); it
    // must read the converted dir, mirroring inference's local_mlx_dir.
    let temp = tempfile::tempdir().expect("tempdir");
    let data_dir = temp.path().join("data");

    let target = super::builtin_training_targets()
        .targets
        .into_iter()
        .find(|t| t.base_model == "wan_2_2")
        .expect("wan_2_2 (TI2V-5B) target");
    let repo = target.base_model_repo.clone().expect("repo set");

    // Materialize BOTH: the HF source snapshot (native checkpoint) and the converted MLX dir.
    // The converted dir must win.
    let repo_root = huggingface_repo_cache_path(&data_dir, &repo).expect("repo cache path");
    let snapshot = repo_root.join("snapshots").join("rev0");
    std::fs::create_dir_all(&snapshot).expect("create snapshot");
    std::fs::create_dir_all(repo_root.join("refs")).expect("create refs");
    std::fs::write(repo_root.join("refs").join("main"), "rev0").expect("write refs/main");

    let converted = data_dir.join("models").join("mlx").join("wan_2_2");
    std::fs::create_dir_all(&converted).expect("create converted dir");
    std::fs::write(converted.join("config.json"), "{}").expect("write config.json");

    let resolved = resolve_base_model_path(&target, &data_dir);

    assert_eq!(
        resolved,
        converted.display().to_string(),
        "conversion models must resolve to the converted MLX dir, not the HF source snapshot"
    );

    // Without the converted dir (config.json gates it), it falls back to the HF snapshot.
    std::fs::remove_file(converted.join("config.json")).expect("remove config.json");
    assert_eq!(
        resolve_base_model_path(&target, &data_dir),
        snapshot.display().to_string(),
        "with no converted dir, fall back to the HF snapshot"
    );
}

#[test]
fn builtin_manifest_registers_the_prompt_refine_model() {
    // sc-5605: the native prompt_refine worker (prompt_refine_jobs.rs) resolves an
    // already-cached HF snapshot via huggingface_snapshot_dir and does NOT auto-download
    // (unlike the retired Python PromptRefiner's from_pretrained). The refine LLM must
    // therefore be a provisionable catalog artifact so Model Manager can download it into
    // the HF cache the worker reads from. This guards the real manifest entry against
    // accidental removal and against the repo string drifting from the worker's
    // DEFAULT_REFINE_MODEL.
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "prompt_refine_llama_3_2_3b")
        .expect("prompt_refine_llama_3_2_3b is registered in the catalog");
    // A non-generation utility entry (mirrors the upscalers): absent from the image/video
    // studio pickers, present + downloadable in Model Manager.
    assert_eq!(model["type"], "utility");
    let download = model["downloads"]
        .as_array()
        .and_then(|downloads| downloads.first())
        .expect("a download entry");
    assert_eq!(download["provider"], "huggingface");
    // Must match the worker's DEFAULT_REFINE_MODEL (prompt_refine_jobs.rs) — the string the
    // worker passes to huggingface_snapshot_dir.
    assert_eq!(
        download["repo"], "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "manifest repo must match the worker's DEFAULT_REFINE_MODEL"
    );
    // The catalog install-state path must resolve the same HF repo the worker does.
    assert_eq!(
        model["paths"]["model"],
        "${HF_CACHE}/huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    );
}

#[test]
fn builtin_manifest_registers_the_joycaption_model() {
    // sc-5620: the native captioner (caption_jobs.rs, the training_caption job) resolves an
    // already-cached HF snapshot via the same resolve_app_managed_model_dir seam and does NOT
    // auto-download. JoyCaption must be a provisionable catalog artifact (same gap as sc-5605's
    // prompt_refine). Guards the entry + repo == caption_jobs::JOY_CAPTION_MODEL + the cache path.
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/manifests/builtin.models.jsonc");
    let raw = std::fs::read_to_string(&manifest_path).expect("read builtin.models.jsonc");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&raw)).expect("parse builtin.models.jsonc");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "joycaption_beta_one")
        .expect("joycaption_beta_one is registered in the catalog");
    assert_eq!(model["type"], "utility");
    let download = model["downloads"]
        .as_array()
        .and_then(|downloads| downloads.first())
        .expect("a download entry");
    assert_eq!(download["provider"], "huggingface");
    // Must match the worker's JOY_CAPTION_MODEL (caption_jobs.rs) — the string the worker passes
    // to huggingface_snapshot_dir.
    assert_eq!(
        download["repo"], "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "manifest repo must match the worker's JOY_CAPTION_MODEL"
    );
    assert_eq!(
        model["paths"]["model"],
        "${HF_CACHE}/fancyfeast/llama-joycaption-beta-one-hf-llava"
    );
}
