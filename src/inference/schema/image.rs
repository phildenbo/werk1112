use super::super::types::{ParameterDescriptor, ParameterType};
use super::builders::{
    add_named_descriptors, bool_descriptor, enum_descriptor, integer_descriptor, number_descriptor,
};
use crate::capabilities::InferenceTask;

pub(super) fn image_descriptors(task: InferenceTask) -> Vec<ParameterDescriptor> {
    let mut values = vec![
        integer_descriptor(
            "image.width",
            "--width",
            "Width",
            "Output width in pixels",
            "geometry",
            1024,
            64,
            32768,
        ),
        integer_descriptor(
            "image.height",
            "--height",
            "Height",
            "Output height in pixels",
            "geometry",
            1024,
            64,
            32768,
        ),
        integer_descriptor(
            "image.batch_size",
            "--batch-size",
            "Batch size",
            "Images processed per batch",
            "generation",
            1,
            1,
            256,
        ),
        integer_descriptor(
            "image.num_images",
            "--num-images",
            "Image count",
            "Number of images to create",
            "generation",
            1,
            1,
            1024,
        ),
        integer_descriptor(
            "image.steps",
            "--steps",
            "Steps",
            "Denoising steps",
            "sampling",
            28,
            1,
            1000,
        ),
        number_descriptor(
            "image.guidance",
            "--guidance",
            "Guidance",
            "Classifier-free guidance scale",
            "sampling",
            7.0,
            0.0,
            100.0,
            0.1,
        ),
        integer_descriptor(
            "image.seed",
            "--seed",
            "Seed",
            "Random seed",
            "sampling",
            0,
            0,
            i64::MAX,
        ),
        bool_descriptor(
            "image.vae_tiling",
            "--image-vae-tiling",
            "VAE tiling",
            "Tile VAE encode and decode",
            "memory",
            false,
        ),
        bool_descriptor(
            "image.vae_slicing",
            "--image-vae-slicing",
            "VAE slicing",
            "Slice VAE batches",
            "memory",
            false,
        ),
        enum_descriptor(
            "image.output_format",
            "--output-format",
            "Output format",
            "Image output format",
            "output",
            "png",
            &["png", "jpeg", "webp"],
        ),
    ];
    add_named_descriptors(
        &mut values,
        "image",
        "geometry",
        &[
            ("aspect_ratio", ParameterType::String),
            ("resize_mode", ParameterType::Enumeration),
            ("crop_mode", ParameterType::Enumeration),
            ("target_width", ParameterType::Integer),
            ("target_height", ParameterType::Integer),
        ],
    );
    add_named_descriptors(
        &mut values,
        "image",
        "sampling",
        &[
            ("subseed", ParameterType::Integer),
            ("variation_strength", ParameterType::Number),
            ("guidance_rescale", ParameterType::Number),
            ("true_cfg", ParameterType::Number),
            ("sampler", ParameterType::String),
            ("scheduler", ParameterType::String),
            ("prediction_type", ParameterType::String),
            ("eta", ParameterType::Number),
            ("denoise_strength", ParameterType::Number),
            ("noise_strength", ParameterType::Number),
            ("sigma_min", ParameterType::Number),
            ("sigma_max", ParameterType::Number),
            ("rho", ParameterType::Number),
            ("sigmas", ParameterType::List),
            ("shift", ParameterType::Number),
            ("dynamic_shift", ParameterType::Boolean),
            ("clip_skip", ParameterType::Integer),
            ("prompt_weighting", ParameterType::String),
            ("prompt_token_limit", ParameterType::Integer),
            ("image_strength", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "image",
        "inpainting",
        &[
            ("mask_blur", ParameterType::Number),
            ("mask_expand", ParameterType::Integer),
            ("mask_invert", ParameterType::Boolean),
            ("mask_fill", ParameterType::Enumeration),
            ("inpaint_area", ParameterType::Enumeration),
            ("padding", ParameterType::Integer),
            ("preserve_unmasked", ParameterType::Boolean),
        ],
    );
    add_named_descriptors(
        &mut values,
        "image",
        "conditioning",
        &[
            ("controls", ParameterType::List),
            ("control_type", ParameterType::String),
            ("control_model", ParameterType::Path),
            ("control_weight", ParameterType::Number),
            ("control_start", ParameterType::Number),
            ("control_end", ParameterType::Number),
            ("control_preprocessor", ParameterType::String),
            ("reference_images", ParameterType::List),
            ("reference_weight", ParameterType::Number),
            ("reference_start", ParameterType::Number),
            ("reference_end", ParameterType::Number),
            ("identity_preservation", ParameterType::Number),
            ("color_preservation", ParameterType::Number),
            ("composition_preservation", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "image",
        "adapters",
        &[
            ("loras", ParameterType::List),
            ("adapters", ParameterType::List),
            ("adapter_weight", ParameterType::Number),
            ("text_encoder_weight", ParameterType::Number),
            ("transformer_weight", ParameterType::Number),
            ("adapter_start", ParameterType::Number),
            ("adapter_end", ParameterType::Number),
        ],
    );
    add_named_descriptors(
        &mut values,
        "image",
        "high_resolution",
        &[
            ("high_resolution_fix", ParameterType::Boolean),
            ("upscale_scale", ParameterType::Number),
            ("second_sampler", ParameterType::String),
            ("second_scheduler", ParameterType::String),
            ("second_denoise", ParameterType::Number),
            ("refiner", ParameterType::Path),
            ("refiner_switch_point", ParameterType::Number),
            ("vae", ParameterType::Path),
            ("vae_precision", ParameterType::String),
            ("tiled_encode", ParameterType::Boolean),
            ("tiled_decode", ParameterType::Boolean),
        ],
    );
    add_named_descriptors(
        &mut values,
        "image",
        "postprocessing",
        &[
            ("safety_checker_policy", ParameterType::Enumeration),
            ("watermark_policy", ParameterType::Enumeration),
            ("face_restoration", ParameterType::Boolean),
            ("color_correction", ParameterType::Boolean),
            ("post_upscaling", ParameterType::String),
            ("output_path", ParameterType::Path),
        ],
    );
    if task == InferenceTask::ImageUpscaling {
        values.retain(|descriptor| {
            matches!(
                descriptor.category.as_str(),
                "geometry"
                    | "generation"
                    | "memory"
                    | "output"
                    | "high_resolution"
                    | "postprocessing"
            )
        });
    }
    values
}
