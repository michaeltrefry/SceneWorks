enum GenEvent {
    Step {
        index: usize,
        current: u32,
        total: u32,
    },
    Decoding {
        index: usize,
    },
    Image {
        index: usize,
        seed: i64,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        /// Pre-built `faceLikeness` sidecar block (epic 4406, sc-4409) for this image, or `None`
        /// when the producing path did not score it. `consume_gen_events` inserts a `Some` block
        /// verbatim into the per-image `rawAdapterSettings` under
        /// [`face_likeness::FACE_LIKENESS_FACT_KEY`] — the omit-when-absent persistence seam, so a
        /// path that doesn't score (every non-angle-set path) is untouched.
        face_likeness: Option<JsonObject>,
    },
}

type GeneratedImage = (i64, u32, u32, Vec<u8>);

/// A generated image plus its optional pre-built `faceLikeness` sidecar block (sc-4409). Returned by
/// the per-item closure of [`drive_gen_items_scored`] so the identity-likeness post-pass (used by all
/// four angle-set lanes — InstantID, FLUX.2 edit, Qwen-Edit, SenseNova-U1) can attach a per-image
/// score without disturbing the shared [`GeneratedImage`] tuple every other generator returns.
type ScoredGeneratedImage = (i64, u32, u32, Vec<u8>, Option<JsonObject>);

fn send_gen_progress(tx: &tokio::sync::mpsc::Sender<GenEvent>, index: usize, progress: Progress) {
    let event = match progress {
        Progress::Step { current, total } => GenEvent::Step {
            index,
            current,
            total,
        },
        Progress::Decoding => GenEvent::Decoding { index },
    };
    let _ = tx.blocking_send(event);
}

fn send_generated_image(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
    image: GeneratedImage,
) -> bool {
    let (seed, width, height, pixels) = image;
    tx.blocking_send(GenEvent::Image {
        index,
        seed,
        width,
        height,
        pixels,
        face_likeness: None,
    })
    .is_ok()
}

/// Like [`send_generated_image`] but carries the optional pre-built `faceLikeness` block (sc-4409).
fn send_scored_generated_image(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    index: usize,
    image: ScoredGeneratedImage,
) -> bool {
    let (seed, width, height, pixels, face_likeness) = image;
    tx.blocking_send(GenEvent::Image {
        index,
        seed,
        width,
        height,
        pixels,
        face_likeness,
    })
    .is_ok()
}

fn drive_gen_items<I, Item, F>(
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    items: I,
    mut generate: F,
) -> WorkerResult<()>
where
    I: IntoIterator<Item = Item>,
    F: FnMut(usize, Item, &mut dyn FnMut(Progress)) -> WorkerResult<Option<GeneratedImage>>,
{
    for (index, item) in items.into_iter().enumerate() {
        let mut on_progress = |progress| send_gen_progress(&tx, index, progress);
        let Some(image) = generate(index, item, &mut on_progress)? else {
            break;
        };
        if !send_generated_image(&tx, index, image) {
            break;
        }
        // Return image N's retained Metal buffer cache to the system before image N+1
        // allocates, so a multi-image batch doesn't stack each image's transient working
        // set on top of the already-resident model weights and cross the unified-memory
        // ceiling — an OS memory-pressure SIGKILL (Jetsam) that the dense SenseNova-U1 8B
        // family hits first (sc-5567). Frees only freed/retained buffers; the cached
        // generator's live weight arrays are untouched.
        release_gen_cache_between_items();
    }
    Ok(())
}

/// Like [`drive_gen_items`] but the per-item closure additionally returns an optional pre-built
/// `faceLikeness` sidecar block (sc-4409), carried through to `consume_gen_events` for per-image
/// persistence. Used by all four angle-set lanes — InstantID, FLUX.2 edit, Qwen-Edit, and
/// SenseNova-U1 — each of which scores every finished view against the per-job cached source identity
/// embedding on its generation thread (the `!Send` face stack lives there). Every non-scoring path
/// keeps using [`drive_gen_items`].
//
// The scored producers are all face-backend paths; off-Mac they compile only with the candle backend
// (the angle-set scorer's backend legs are cfg-gated the same way), so allow this dead when neither
// face backend is present.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
fn drive_gen_items_scored<I, Item, F>(
    tx: tokio::sync::mpsc::Sender<GenEvent>,
    items: I,
    mut generate: F,
) -> WorkerResult<()>
where
    I: IntoIterator<Item = Item>,
    F: FnMut(usize, Item, &mut dyn FnMut(Progress)) -> WorkerResult<Option<ScoredGeneratedImage>>,
{
    for (index, item) in items.into_iter().enumerate() {
        let mut on_progress = |progress| send_gen_progress(&tx, index, progress);
        let Some(image) = generate(index, item, &mut on_progress)? else {
            break;
        };
        if !send_scored_generated_image(&tx, index, image) {
            break;
        }
        release_gen_cache_between_items();
    }
    Ok(())
}

/// Release MLX's freed-buffer cache between batch images so peak memory doesn't carry
/// forward across a `drive_gen_items` loop (sc-5567). `clear_cache()` returns only the
/// retained-for-reuse buffers to the OS — live arrays (the cached model weights) are not
/// touched — so the one-time reallocation cost on the next image is negligible against a
/// tens-of-seconds generation, and far cheaper than an OOM kill. No-op off macOS: the
/// Windows/CUDA candle lane shares this loop but has no `mlx_rs` dependency.
#[cfg(target_os = "macos")]
fn release_gen_cache_between_items() {
    mlx_rs::memory::clear_cache();
}

#[cfg(not(target_os = "macos"))]
fn release_gen_cache_between_items() {}

// Shared by the macOS MLX paths and the Windows/CUDA candle InstantID lane (sc-5491): both load a
// `!Send` engine on the blocking thread and stream per-item events back. `G` is the loaded model
// (MLX `Box<dyn Generator>` or candle `InstantId`) — created and consumed inside the one
// `spawn_blocking`, so it never needs to be `Send`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn start_gen_stream<G, L, D>(
    job_id: String,
    engine_id: &'static str,
    adapter_count: usize,
    load: L,
    drive: D,
) -> (
    CancelFlag,
    tokio::sync::mpsc::Receiver<GenEvent>,
    tokio::task::JoinHandle<WorkerResult<()>>,
)
where
    L: FnOnce() -> WorkerResult<G> + Send + 'static,
    D: FnOnce(G, tokio::sync::mpsc::Sender<GenEvent>, CancelFlag) -> WorkerResult<()>
        + Send
        + 'static,
{
    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    let blocking_cancel = cancel.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        emit_load_event(
            "image_pipeline_load_start",
            &job_id,
            engine_id,
            adapter_count,
        );
        let generator = load()?;
        emit_load_event(
            "image_pipeline_load_complete",
            &job_id,
            engine_id,
            adapter_count,
        );
        drive(generator, tx, blocking_cancel)
    });
    (cancel, rx, blocking)
}

fn start_cached_gen_stream<D>(
    job_id: String,
    engine_id: &'static str,
    adapter_count: usize,
    spec: LoadSpec,
    load_error_context: String,
    drive: D,
) -> (
    CancelFlag,
    tokio::sync::mpsc::Receiver<GenEvent>,
    tokio::task::JoinHandle<WorkerResult<()>>,
)
where
    D: FnOnce(&dyn Generator, tokio::sync::mpsc::Sender<GenEvent>, CancelFlag) -> WorkerResult<()>
        + Send
        + 'static,
{
    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(64);
    let blocking_cancel = cancel.clone();
    let blocking = tokio::spawn(async move {
        emit_load_event(
            "image_pipeline_load_start",
            &job_id,
            engine_id,
            adapter_count,
        );
        crate::generator_cache::with_cached_generator(
            engine_id,
            spec,
            load_error_context,
            move |generator| {
                emit_load_event(
                    "image_pipeline_load_complete",
                    &job_id,
                    engine_id,
                    adapter_count,
                );
                drive(generator, tx, blocking_cancel)
            },
        )
        .await
    });
    (cancel, rx, blocking)
}

/// True when this job can run real in-process inference: the model is a linked,
/// engine-backed family and its weights resolve locally.
/// Fail-loud gate for the stub fallback (sc-4176): Some(message) when the
/// requested model id is a known MLX engine model but its weights snapshot
/// can't be resolved (partially deleted HF cache, stale refs, missing
/// modelPath). None when the model isn't engine-backed (the stub is its
/// intended path) or the weights resolve. MLX-only (uses `mlx_model` + the macOS
/// `resolve_weights_dir`); the candle lane has no equivalent stub-gap check.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_weights_gap(request: &ImageRequest, settings: &Settings) -> Option<String> {
    let model = mlx_model(&request.model)?;
    match resolve_weights_dir(request, settings) {
        Ok(Some(_)) => return None,
        Err(error) => return Some(error.to_string()),
        Ok(None) => {}
    }
    Some(format!(
        "{}: MLX weights not found or incomplete (Hugging Face repo {}). \
         Re-download the model in Model Manager, then retry.",
        request.model,
        model_repo(request, &model),
    ))
}
