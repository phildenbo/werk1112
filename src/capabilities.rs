//! Canonical model and runtime capabilities.
//!
//! The public surface is re-exported from this facade so callers can keep
//! using `crate::capabilities::*` while the implementation stays grouped by
//! responsibility.

macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $($(#[$variant_meta:meta])* $variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(
            Debug,
            Clone,
            Copy,
            serde::Serialize,
            serde::Deserialize,
            PartialEq,
            Eq,
            Hash,
        )]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($(#[$variant_meta])* $variant),+
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(match self {
                    $(Self::$variant => $value),+
                })
            }
        }

        impl std::str::FromStr for $name {
            type Err = String;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
                match normalized.as_str() {
                    $($value => Ok(Self::$variant)),+,
                    _ => Err(format!("unknown {}: {value}", stringify!($name))),
                }
            }
        }
    };
}

mod component;
mod modality;
mod repository;
mod task;

pub use component::{ModelComponent, ModelComponentKind};
pub use modality::{InputModality, OutputModality};
pub use repository::RepositoryLayout;
pub use task::InferenceTask;

#[cfg(test)]
mod tests;
