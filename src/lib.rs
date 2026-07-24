#[cfg(all(
    any(feature = "release-linux", feature = "release-windows"),
    not(feature = "candle-cuda")
))]
compile_error!("Linux and Windows release artifacts must compile Candle CUDA support.");

#[cfg(all(feature = "release-macos-apple-silicon", not(feature = "metal")))]
compile_error!("macOS Apple Silicon release artifacts must compile Candle Metal support.");

pub mod api;
pub mod api_keys;
pub mod backend;
pub mod banner;
pub mod capabilities;
pub mod cli;
pub mod inference;
pub mod inference_service;
pub mod media_cli;
pub mod media_companion;
pub mod model_store;
pub mod openai;
pub mod runtime_planner;
