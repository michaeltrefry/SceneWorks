#[tokio::main]
async fn main() -> Result<(), sceneworks_worker::WorkerError> {
    sceneworks_worker::run().await
}
