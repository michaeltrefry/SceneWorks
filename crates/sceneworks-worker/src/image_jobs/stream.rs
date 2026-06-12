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
    },
}

type GeneratedImage = (i64, u32, u32, Vec<u8>);

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
    }
    Ok(())
}

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
/// intended path) or the weights resolve.
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
