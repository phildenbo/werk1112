use std::{env, fs, path::Path};

use crate::inference::{HostResources, RuntimeAccelerator};

pub fn detect_host_resources() -> HostResources {
    let host_memory_bytes = fs::read_to_string("/proc/meminfo").ok().and_then(|data| {
        data.lines()
            .find(|line| line.starts_with("MemAvailable:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kib| kib.saturating_mul(1024))
    });
    let accelerator_memory_bytes = env::var("WERK_ACCELERATOR_MEMORY_BYTES")
        .ok()
        .and_then(|value| value.parse().ok());
    HostResources {
        host_memory_bytes,
        accelerator_memory_bytes,
        accelerator: Some(format!("{:?}", detected_accelerator()).to_ascii_lowercase()),
    }
}

pub(super) fn detected_accelerator() -> RuntimeAccelerator {
    if let Ok(value) = env::var("WERK_MEDIA_ACCELERATOR")
        && !value.trim().is_empty()
    {
        return match value.to_ascii_lowercase().as_str() {
            "cuda" => RuntimeAccelerator::Cuda,
            "rocm" | "hip" => RuntimeAccelerator::Rocm,
            "mps" | "metal" => RuntimeAccelerator::Mps,
            "mlx" => RuntimeAccelerator::Mlx,
            "cpu" => RuntimeAccelerator::Cpu,
            _ => RuntimeAccelerator::Other,
        };
    }
    #[cfg(target_os = "macos")]
    {
        return RuntimeAccelerator::Mps;
    }
    #[cfg(not(target_os = "macos"))]
    {
        if Path::new("/dev/nvidiactl").exists()
            || accelerator_env_is_enabled("CUDA_VISIBLE_DEVICES")
        {
            RuntimeAccelerator::Cuda
        } else if Path::new("/dev/kfd").exists()
            || accelerator_env_is_enabled("ROCR_VISIBLE_DEVICES")
        {
            RuntimeAccelerator::Rocm
        } else {
            RuntimeAccelerator::Cpu
        }
    }
}

fn accelerator_env_is_enabled(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty()
            && !matches!(
                value.to_ascii_lowercase().as_str(),
                "-1" | "none" | "disabled" | "void"
            )
    })
}
