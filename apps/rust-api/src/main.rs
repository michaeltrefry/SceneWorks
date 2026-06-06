#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The desktop app launches this same binary a second time as a standalone
    // GPU worker — the Apple-Silicon MLX worker (sc-3289) — by setting
    // SCENEWORKS_WORKER_ONLY=1. The binary already links the mlx-gen engine, so
    // reusing it avoids bundling a second multi-hundred-MB sidecar.
    if std::env::var("SCENEWORKS_WORKER_ONLY").is_ok_and(|value| value.trim() == "1") {
        return sceneworks_rust_api::run_worker().await;
    }
    sceneworks_rust_api::run().await
}
