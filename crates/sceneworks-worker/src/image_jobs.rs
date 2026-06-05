//! Native MLX image generation jobs (epic 3018).
//!
//! On macOS the worker links the `mlx-gen` engine (epic 2337) in-process and runs
//! generation with no Python adapter/sidecar. This scaffold (sc-3019) wires the
//! `image_generate` dispatch + capability so a job round-trips through the Rust GPU
//! worker and proves the engine is linked and its model registry resolves; the real
//! `ImageRequest` DTO + asset writer land in sc-3020 and per-family inference in
//! sc-3022.. . Off macOS the engine is absent, so the job is rejected — non-macOS
//! workers never advertise `image_generate`, making that path unreachable in
//! practice; it only keeps the dispatch arm cross-platform.

use super::*;

// Force the Z-Image provider crate to link so its `inventory::submit!` registration
// survives linker GC: an only-declared-but-never-referenced dependency can have its
// link-section statics dropped, taking the model registration with it. Each
// per-family story (sc-3022..) adds its provider crate + a matching `use … as _;`
// here. See mlx-gen-z-image/tests/registry.rs (which names "the SceneWorks worker").
#[cfg(target_os = "macos")]
use mlx_gen_z_image as _;

/// Dispatch handler for `JobType::ImageGenerate`. The scaffold reports progress and
/// returns a result describing the linked engine; it does not run inference yet.
pub(crate) async fn run_image_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    check_cancel(api, &job.id, "Worker canceled the image job before start.").await?;
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Preparing native MLX image worker.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let result = image_generate_scaffold_result(job)?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Native MLX image worker scaffold ready (no inference yet — sc-3020/sc-3022+).",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// macOS: prove `mlx-gen` is linked and its link-time registry is populated.
/// `descriptor()` reads only the `inventory` statics — it loads no weights and
/// initializes no Metal device — so this is safe to call from the scaffold.
#[cfg(target_os = "macos")]
fn image_generate_scaffold_result(job: &JobSnapshot) -> WorkerResult<JsonObject> {
    let mut registered: Vec<String> = mlx_gen::registry::generators()
        .map(|registration| (registration.descriptor)().id.to_owned())
        .collect();
    registered.sort();
    registered.dedup();

    let requested_model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let mut result = JsonObject::new();
    result.insert("scaffold".to_owned(), Value::Bool(true));
    result.insert("engine".to_owned(), Value::String("mlx-gen".to_owned()));
    result.insert(
        "registeredModels".to_owned(),
        Value::Array(registered.into_iter().map(Value::String).collect()),
    );
    result.insert("requestedModel".to_owned(), Value::String(requested_model));
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    Ok(result)
}

/// Non-macOS: the MLX engine is not linked. Unreachable in practice (only the macOS
/// Apple-Silicon worker advertises `image_generate`); kept so the dispatch arm
/// compiles on every platform.
#[cfg(not(target_os = "macos"))]
fn image_generate_scaffold_result(_job: &JobSnapshot) -> WorkerResult<JsonObject> {
    Err(WorkerError::InvalidPayload(
        "Native MLX image generation is only available on the macOS (Apple Silicon) GPU worker."
            .to_owned(),
    ))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn image_job(model: &str) -> JobSnapshot {
        serde_json::from_value(serde_json::json!({
            "id": "job-img-1",
            "type": "image_generate",
            "status": "running",
            "projectId": null,
            "projectName": null,
            "payload": { "model": model },
            "result": {},
            "requestedGpu": "mlx",
            "assignedGpu": "mlx",
            "workerId": "test-mlx-worker",
            "progress": 0.0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 1,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": false,
            "createdAt": "2026-06-05T00:00:00Z",
            "updatedAt": "2026-06-05T00:00:00Z",
            "startedAt": null,
            "completedAt": null,
            "canceledAt": null,
            "lastHeartbeatAt": null
        }))
        .expect("valid image_generate job snapshot")
    }

    #[test]
    fn scaffold_result_reports_linked_engine_and_registry() {
        let job = image_job("z_image_turbo");
        let result = image_generate_scaffold_result(&job).expect("scaffold result on macOS");
        assert_eq!(
            result.get("engine").and_then(Value::as_str),
            Some("mlx-gen")
        );
        assert_eq!(result.get("scaffold").and_then(Value::as_bool), Some(true));
        assert_eq!(
            result.get("requestedModel").and_then(Value::as_str),
            Some("z_image_turbo")
        );
        // The Z-Image provider linked into the worker self-registered via inventory:
        // proof the cross-crate, link-time mlx-gen registry resolves inside our binary
        // (and that the `use mlx_gen_z_image as _;` defeats linker GC of the static).
        let registered = result
            .get("registeredModels")
            .and_then(Value::as_array)
            .expect("registeredModels array");
        assert!(
            registered
                .iter()
                .any(|m| m.as_str() == Some("z_image_turbo")),
            "expected z_image_turbo in registry, got {registered:?}"
        );
    }
}
