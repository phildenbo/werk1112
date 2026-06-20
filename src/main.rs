#[tokio::main]
async fn main() -> anyhow::Result<()> {
    werk1112::cli::run_from_env().await
}
