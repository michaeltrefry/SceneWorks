#[tokio::main]
async fn main() -> Result<(), sceneworks_rust_worker::WorkerError> {
    sceneworks_rust_worker::run().await
}
