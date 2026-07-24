//! HTTP API façade.
//!
//! State, routing, chat, media handling, responses, and tests live in focused
//! submodules while the historical public API remains available here.

mod chat;
mod media;
mod response;
mod router;
mod state;

#[cfg(test)]
mod tests;

pub use router::{router, serve};
pub use state::{ApiState, PromptOptionsResolver};
