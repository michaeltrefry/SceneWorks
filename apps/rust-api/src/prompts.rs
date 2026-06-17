use super::*;

/// Enqueue a `prompt_refine` job: a lightweight, non-GPU job that asks an
/// OpenAI-compatible LLM to rewrite the user's prompt to follow the selected
/// model's prompt guide. The job runs in the Python worker (which reuses the
/// vendored Lens reasoner's calling approach) and the client reads the refined
/// prompt from the completed job's `result.refinedPrompt`.
pub(crate) async fn create_prompt_refine_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<PromptRefineRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let prompt = payload.prompt.trim();
    if prompt.is_empty() {
        return Err(ApiError::bad_request("Prompt cannot be empty"));
    }

    let mut job_payload = JsonObject::new();
    job_payload.insert("prompt".to_owned(), Value::String(prompt.to_owned()));

    let workflow = payload
        .workflow
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("image")
        .to_owned();
    job_payload.insert("workflow".to_owned(), Value::String(workflow));

    if let Some(model_id) = payload.model_id.as_deref() {
        if !model_id.trim().is_empty() {
            job_payload.insert(
                "modelId".to_owned(),
                Value::String(model_id.trim().to_owned()),
            );
        }
    }
    if let Some(guide) = payload.guide.as_deref() {
        if !guide.trim().is_empty() {
            job_payload.insert("guide".to_owned(), Value::String(guide.to_owned()));
        }
    }
    // Magic-prompt expansion (sc-5997): the worker swaps in Ideogram's caption system
    // prompt and the aspect ratio steers its layout/bbox decisions.
    if let Some(task) = payload.task.as_deref() {
        if !task.trim().is_empty() {
            job_payload.insert("task".to_owned(), Value::String(task.trim().to_owned()));
        }
    }
    if let Some(aspect_ratio) = payload.aspect_ratio.as_deref() {
        if !aspect_ratio.trim().is_empty() {
            job_payload.insert(
                "aspectRatio".to_owned(),
                Value::String(aspect_ratio.trim().to_owned()),
            );
        }
    }

    let job = create_generation_job(
        state,
        JobType::PromptRefine,
        None,
        None,
        job_payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}
