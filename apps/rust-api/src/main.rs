#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    sceneworks_rust_api::run().await
}
