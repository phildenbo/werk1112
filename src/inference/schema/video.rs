use super::super::types::{ParameterDescriptor, ParameterType};
use super::builders::{
    add_named_descriptors, bool_descriptor, enum_descriptor, integer_descriptor, number_descriptor,
};
use crate::capabilities::InferenceTask;

pub(super) fn video_descriptors(task: InferenceTask) -> Vec<ParameterDescriptor> {
    let mut values = vec![
        integer_descriptor(
            "video.width",
            "--width",
            "Width",
            "Output width in pixels",
            "geometry",
            832,
            64,
            16384,
        ),
        integer_descriptor(
            "video.height",
            "--height",
            "Height",
            "Output height in pixels",
            "geometry",
            480,
            64,
            16384,
        ),
        integer_descriptor(
            "video.frames",
            "--frames",
            "Frames",
            "Number of generated frames",
            "temporal",
            81,
            1,
            100_000,
        ),
        number_descriptor(
            "video.fps",
            "--fps",
            "FPS",
            "Frames per second",
            "temporal",
            24.0,
            0.1,
            1000.0,
            0.1,
        ),
        integer_descriptor(
            "video.batch_size",
            "--batch-size",
            "Batch size",
            "Videos processed per batch",
            "generation",
            1,
            1,
            64,
        ),
        integer_descriptor(
            "video.num_videos",
            "--num-videos",
            "Video count",
            "Number of videos to create",
            "generation",
            1,
            1,
            256,
        ),
        integer_descriptor(
            "video.steps",
            "--steps",
            "Steps",
            "Denoising steps",
            "sampling",
            30,
            1,
            2000,
        ),
        number_descriptor(
            "video.guidance",
            "--guidance",
            "Guidance",
            "Guidance scale",
            "sampling",
            6.0,
            0.0,
            100.0,
            0.1,
        ),
        integer_descriptor(
            "video.seed",
            "--seed",
            "Seed",
            "Random seed",
            "sampling",
            0,
            0,
            i64::MAX,
        ),
        bool_descriptor(
            "video.temporal_vae_tiling",
            "--temporal-vae-tiling",
            "Temporal VAE tiling",
            "Tile temporal VAE operations",
            "memory",
            false,
        ),
        enum_descriptor(
            "video.output_format",
            "--output-format",
            "Output format",
            "Video container",
            "output",
            "mp4",
            &["mp4", "gif"],
        ),
    ];
    add_named_descriptors(
        &mut values,
        "video",
        "geometry",
        &[
            ("aspect_ratio", ParameterType::String),
            ("duration", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "video",
        "sampling",
        &[
            ("sampler", ParameterType::String),
            ("scheduler", ParameterType::String),
            ("eta", ParameterType::Number),
            ("denoise_strength", ParameterType::Number),
            ("noise_augmentation", ParameterType::Number),
            ("sigma_min", ParameterType::Number),
            ("sigma_max", ParameterType::Number),
            ("rho", ParameterType::Number),
            ("sigmas", ParameterType::List),
            ("shift", ParameterType::Number),
            ("flow_shift", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "video",
        "motion",
        &[
            ("motion_strength", ParameterType::Number),
            ("motion_bucket", ParameterType::Integer),
            ("temporal_guidance", ParameterType::Number),
            ("temporal_consistency", ParameterType::Number),
            ("camera_motion", ParameterType::String),
            ("camera_strength", ParameterType::Number),
            ("camera_keyframes", ParameterType::List),
            ("prompt_keyframes", ParameterType::List),
            ("guidance_schedule", ParameterType::List),
            ("denoise_schedule", ParameterType::List),
            ("adapters", ParameterType::List),
            ("adapter_schedule", ParameterType::List),
        ],
    );
    add_named_descriptors(
        &mut values,
        "video",
        "conditioning",
        &[
            ("final_image", ParameterType::Path),
            ("mask_video", ParameterType::Path),
            ("pose_video", ParameterType::Path),
            ("depth_video", ParameterType::Path),
            ("control_video", ParameterType::Path),
            ("controls", ParameterType::List),
            ("reference_images", ParameterType::List),
            ("reference_audio", ParameterType::Path),
            ("image_strength", ParameterType::Number),
            ("video_strength", ParameterType::Number),
            ("first_frame_strength", ParameterType::Number),
            ("last_frame_strength", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "video",
        "windowing",
        &[
            ("context_frames", ParameterType::Integer),
            ("context_overlap", ParameterType::Integer),
            ("context_stride", ParameterType::Integer),
            ("window_size", ParameterType::Integer),
            ("window_overlap", ParameterType::Integer),
            ("looping", ParameterType::Boolean),
            ("loop_blend_frames", ParameterType::Integer),
            ("extension_frames", ParameterType::Integer),
            ("extension_direction", ParameterType::Enumeration),
            ("tile_width", ParameterType::Integer),
            ("tile_height", ParameterType::Integer),
            ("tile_frames", ParameterType::Integer),
            ("tile_overlap", ParameterType::Integer),
            ("decode_chunk_size", ParameterType::Integer),
        ],
    );
    add_named_descriptors(
        &mut values,
        "video",
        "postprocessing",
        &[
            ("frame_interpolation", ParameterType::Boolean),
            ("interpolation_factor", ParameterType::Number),
            ("upscaling", ParameterType::Boolean),
            ("upscale_scale", ParameterType::Number),
            ("stabilization", ParameterType::Boolean),
            ("codec", ParameterType::String),
            ("pixel_format", ParameterType::String),
            ("bitrate", ParameterType::String),
            ("crf", ParameterType::Integer),
            ("encoding_preset", ParameterType::String),
            ("include_audio", ParameterType::Boolean),
            ("output_path", ParameterType::Path),
        ],
    );
    if matches!(
        task,
        InferenceTask::VideoUpscaling | InferenceTask::FrameInterpolation
    ) {
        values.retain(|descriptor| match descriptor.category.as_str() {
            "geometry" | "temporal" | "generation" | "sampling" | "memory" | "output"
            | "windowing" | "postprocessing" => true,
            "motion" => matches!(
                descriptor.path.as_str(),
                "video.motion_strength"
                    | "video.motion_bucket"
                    | "video.temporal_guidance"
                    | "video.temporal_consistency"
            ),
            _ => false,
        });
    }
    values
}
