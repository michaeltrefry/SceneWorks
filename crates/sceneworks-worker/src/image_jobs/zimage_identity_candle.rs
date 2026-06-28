// Candle (Windows/CUDA) Z-Image identity-init route for Image Studio "With Character" (sc-8409, epic
// 4406) — the off-Mac sibling of the macOS MLX generic lane's Z-Image identity img2img path
// (`resolve_zimage_identity_init` in zimage.rs, sc-3146). A `character_image` job with a chosen
// `referenceAssetId` (and `advanced.referenceStrength > 0`) seeds the Z-Image-Turbo denoise FROM the
// reference latents — carrying the character's identity into the variation — instead of falling through
// to plain txt2img (which drops the reference entirely, the pre-existing gap this story closes).
//
// **Why a bespoke lane.** `z_image_turbo` IS a candle txt2img id, so without this a With-Character job
// would be caught by the generic `generate_candle_stream` (txt2img) branch and silently drop the
// reference. The candle `zimage_edit_candle` lane is `edit_image`-only (`sourceAssetId`), so it does not
// cover the identity-init shape (`character_image` + `referenceAssetId`). This reuses the SAME candle
// `ZImageEdit` engine the edit lane drives (the Turbo DiT + a strength-derived source-latent init) — the
// only difference from the edit lane is the source-keying (the identity `referenceAssetId` instead of the
// edit `sourceAssetId`) and the `character_image` mode gate.
//
// **Parity with macOS.** The engage condition mirrors the macOS `zimage_identity_strength` gate EXACTLY
// (`advanced.referenceStrength > 0` AND a non-empty `referenceAssetId`), so candle runs identity img2img
// precisely when the MLX generic lane does — a With-Character job WITHOUT a positive `referenceStrength`
// stays plain txt2img on both backends.
//
// **Face-likeness scoring (sc-4411 seam).** Once the route exists, each finished image is scored against
// the chosen reference face through the SHARED generator-agnostic seam (`build_face_likeness_scorer` +
// `score_generated_image`, source resolved by `resolve_character_image_likeness_source`), exactly as the
// macOS generic lane and the other identity lanes do — source embedded ONCE, reused across the N images,
// non-fatal, the hot-path pixel clone gated behind `scorer.is_some()`, non-frontal → honest N/A.
//
// `include!`d into the `image_jobs` module (carrying the candle cfg), so it shares that module's imports
// (`ImageRequest`/`Settings`/`WorkerResult`/`advanced`/`load_reference_image`/`fit_engine_image`/
// `resolve_character_image_likeness_source`/`ensure_face_stack_dir`/`ZImageEdit`/`drive_gen_items_scored`/
// `score_generated_image`/`consume_gen_events`/`resolve_seed`/`Image`/… all in scope). It also reuses the
// edit lane's `resolve_zimage_edit_candle_base` / `zimage_edit_candle_steps` helpers (same engine, same
// base snapshot resolution) — both live in the sibling `zimage_edit_candle.rs` include.

/// The adapter/engine id recorded on candle Z-Image identity-init assets + telemetry (distinct from the
/// txt2img `candle_z_image`, the `candle_zimage_edit`, and the `candle_zimage_control` lanes).
const ZIMAGE_IDENTITY_CANDLE_ENGINE: &str = "candle_zimage_identity";

/// Model ids the candle Z-Image identity-init route accepts: only `z_image_turbo` (the With-Character
/// target — the candle z-image engine is the distilled Turbo). The dedicated `z_image_edit` id is an
/// edit-mode id (handled by `zimage_edit_candle`), not a character target.
fn is_zimage_identity_candle_model(model: &str) -> bool {
    model == "z_image_turbo"
}

/// The clamped identity img2img-init strength for a candle Z-Image With-Character job, or `None` when the
/// identity init does NOT engage. `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty
/// `referenceAssetId` is present — the EXACT engage gate of the macOS `zimage_identity_strength`
/// (zimage.rs, sc-3146), so candle runs identity img2img precisely when the MLX generic lane does.
///
/// `strength` is the user value clamped to `[0.05, 1.0]` and carries the mflux `image_strength`
/// convention **verbatim** (no numeric inversion): higher strength → later denoise start
/// (`init_time_step`) → output stays closer to the reference. Pure (request only) so the parity-sensitive
/// gate + clamp are unit-testable without asset I/O. (Deliberately duplicates the macOS helper, which
/// lives in the macOS-only `zimage.rs` include — the same per-lane-helper pattern the candle siblings use,
/// kept in lockstep by the shared parity comment.)
fn zimage_identity_candle_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// True when this is a candle-eligible Z-Image identity-init job: a `z_image_turbo` Image Studio
/// "With Character" generation (`mode == "character_image"`) with a chosen `referenceAssetId` and a
/// positive `referenceStrength`, that is NOT an angle set / pose-library set (those are already routed to
/// their own lanes — the candle InstantID angle/pose paths and the Z-Image strict-control lane — and
/// scored there), and whose Turbo base resolves locally. Mirrors `jobs_store::zimage_identity_candle_\
/// eligible` (minus the local weight-resolve check) so the worker and router agree.
fn zimage_identity_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_zimage_identity_candle_model(&request.model)
        && request.mode == "character_image"
        && zimage_identity_candle_strength(request).is_some()
        // Angle / pose sets are `character_image` too, but route to (and score on) their own lanes —
        // exclude both so this plain With-Character lane never steals them (it sits BEFORE the Z-Image
        // strict-control lane in the dispatch). Mirrors `resolve_character_image_likeness_source`.
        && pose_entries(request).is_empty()
        && !advanced::flag(&request.advanced, "angleSet")
        && matches!(
            resolve_zimage_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Flat telemetry recorded on candle Z-Image identity-init assets. No guidance — Z-Image-Turbo is
/// guidance-distilled. Mirrors `zimage_edit_candle_raw_settings`, keyed to the `character_image` mode +
/// the identity engine id (so the sidecar attributes the route distinctly from the edit lane).
fn zimage_identity_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    strength: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Verbatim mflux `image_strength` (higher ⇒ closer to the reference) — the same key the recipe
    // records for the identity init on the MLX side.
    raw.insert("referenceStrength".to_owned(), json!(strength));
    raw.insert(
        "mode".to_owned(),
        Value::String("character_image".to_owned()),
    );
    raw.insert(
        "identityEngine".to_owned(),
        Value::String(ZIMAGE_IDENTITY_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Z-Image identity-init generation: resolve the reference + base on the async side, fit the
/// reference to the render size (honoring `fit_mode`), then load `ZImageEdit` once + generate each image
/// on the blocking thread, seeding the denoise from the reference latents at `referenceStrength`.
/// `request.count` images, each its own seed, all carrying the same character identity. Z-Image-Turbo is
/// distilled (no CFG / negative prompt), so the request carries no guidance. Each finished image is scored
/// against the reference face through the shared sc-4411 seam (the `!Send` scorer built ONCE on the
/// generator thread, source embedded once, reused across the N images; non-fatal; the pixel clone gated
/// behind `scorer.is_some()`). Reuses [`drive_gen_items_scored`] + [`consume_gen_events`].
async fn generate_candle_zimage_identity_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let base = resolve_zimage_edit_candle_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Z-Image base (Z-Image-Turbo) weights not found".to_owned())
    })?;

    let (width, height) = (request.width, request.height);
    // The identity init strength engages this lane (the dispatch gate guarantees `Some`); resolve it
    // again here as the source of truth for the engine + the recipe.
    let strength = zimage_identity_candle_strength(request).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Z-Image identity init requires a referenceAssetId + referenceStrength > 0".to_owned(),
        )
    })?;
    // Identity img2img source = the chosen character `referenceAssetId` (the reference shown in the Image
    // Studio thumbnail), fit to the render geometry honoring `fit_mode` (the provider also resizes
    // internally, but pre-fitting avoids stretching an off-aspect reference). Guaranteed non-empty by the
    // strength gate above.
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("zimage_identity_candle_strength guarantees a non-empty referenceAssetId")
        .to_owned();
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &reference_id,
        project_path,
    )?;
    let source = fit_engine_image(source, width, height, &request.fit_mode)?;

    let steps = zimage_edit_candle_steps(request);
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(ZIMAGE_EDIT_CANDLE_DEFAULT_REPO)
        .to_owned();
    let raw_settings = zimage_identity_candle_raw_settings(request, &repo, steps, strength);

    // Identity-likeness scoring (sc-4411): resolve the source identity (the CURRENT job's
    // `referenceAssetId`, so changing it changes the scored source) + stage the antelopev2 face stack
    // (the same bundle InstantID uses; a no-op if cached). The dispatch gate guarantees a plain
    // With-Character job, so `resolve_character_image_likeness_source` resolves the same reference the
    // img2img init uses (decoded raw, not fit — better for face detection). All non-fatal: a missing
    // reference / staging failure → no scorer → scores omitted, the generation still renders.
    let likeness_source = resolve_character_image_likeness_source(request, settings, project_path);
    let face_stack_dir = match &likeness_source {
        Some(_) => match ensure_face_stack_dir(api, settings, job).await {
            Ok(dir) => Some(dir),
            Err(error) => {
                tracing::warn!(error = %error, "With-Character (z-image identity) face-stack staging failed; likeness scores omitted");
                None
            }
        },
        None => None,
    };
    // Keep the source only if the face stack staged (otherwise no scorer can be built).
    let likeness_source = face_stack_dir.as_ref().and(likeness_source);

    // Per-image work items: (seed, prompt) — `request.count` identity-init renders of the same reference.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "zimage_identity",
        0,
        move || {
            let model = ZImageEdit::load(&ZImageEditPaths { base }).map_err(|error| {
                WorkerError::Engine(format!("Z-Image identity load failed: {error}"))
            })?;
            Ok((model, source))
        },
        move |(model, source), tx, cancel| {
            // Per-job identity-likeness scorer built ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); source embedded once, reused across every output (sc-4411). `None`
            // ⇒ non-fatal staging/construction failure ⇒ scores omitted.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            drive_gen_items_scored(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = ZImageEditRequest {
                    prompt,
                    width,
                    height,
                    steps: steps as usize,
                    strength,
                    seed: seed as u64,
                    cancel: cancel.clone(),
                };
                let out = match model.generate(&req, &source, &mut *on_progress) {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Z-Image identity generation failed: {error}"
                        )));
                    }
                };
                let (out_w, out_h, pixels) = (out.width, out.height, out.pixels);
                // Score this finished image against the cached source embedding (sc-4411). Image build +
                // pixel clone is paid ONLY when a scorer exists (a With-Character generation); non-frontal
                // → honest detected:false N/A; `None` scorer ⇒ field omitted.
                let face_likeness = scorer.as_ref().and_then(|scorer| {
                    crate::face_likeness::score_generated_image(
                        Some(scorer),
                        &Image {
                            width: out_w,
                            height: out_h,
                            pixels: pixels.clone(),
                        },
                        likeness_source_ref.as_deref(),
                    )
                });
                Ok(Some((seed, out_w, out_h, pixels, face_likeness)))
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        ZIMAGE_IDENTITY_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
