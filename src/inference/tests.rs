use super::*;
use crate::{
    capabilities::{
        InferenceTask, InputModality, ModelComponent, OutputModality, RepositoryLayout,
    },
    media_cli::{AudioCommands, ImageCommands, VideoCommands},
    model_store::{
        CURRENT_MANIFEST_SCHEMA_VERSION, ModelFile, ModelFormat, ModelManifest, ModelMetadata,
        ModelSource,
    },
};
use clap::CommandFactory;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(clap::Parser)]
struct ImageSchemaCli {
    #[command(subcommand)]
    command: ImageCommands,
}

#[derive(clap::Parser)]
struct VideoSchemaCli {
    #[command(subcommand)]
    command: VideoCommands,
}

#[derive(clap::Parser)]
struct AudioSchemaCli {
    #[command(subcommand)]
    command: AudioCommands,
}

fn assert_schema_flags_exist_on_subcommand<C: CommandFactory>(
    task: InferenceTask,
    subcommand: &str,
) {
    let command = C::command();
    let public_flags = command
        .find_subcommand(subcommand)
        .unwrap_or_else(|| panic!("missing public subcommand {subcommand}"))
        .get_arguments()
        .filter_map(|argument| argument.get_long())
        .map(|flag| format!("--{flag}"))
        .collect::<BTreeSet<_>>();
    let missing = parameter_schema(task)
        .into_iter()
        .filter(|descriptor| !descriptor.path.starts_with("routing."))
        .filter(|descriptor| !public_flags.contains(&descriptor.cli_flag))
        .map(|descriptor| (descriptor.path, descriptor.cli_flag))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "{task} advertises flags absent from '{subcommand}': {missing:?}"
    );
}

fn image_manifest() -> ModelManifest {
    let mut metadata = ModelMetadata {
        schema_version: CURRENT_MANIFEST_SCHEMA_VERSION,
        family: Some("flux".to_string()),
        repository_layout: RepositoryLayout::Diffusers,
        tasks: vec![InferenceTask::ImageGeneration, InferenceTask::ImageEditing],
        input_modalities: vec![InputModality::Text, InputModality::Image],
        output_modalities: vec![OutputModality::Image],
        components: vec![ModelComponent::new(
            crate::capabilities::ModelComponentKind::Transformer,
            "files/transformer",
        )],
        ..Default::default()
    };
    metadata
        .generation_defaults
        .insert("image.steps".to_string(), Value::from(35));
    ModelManifest {
        id: "flux".to_string(),
        source: ModelSource::LocalPath {
            path: "fixture".to_string(),
        },
        format: ModelFormat::SafeTensors,
        architecture: Some("FluxTransformer2DModel".to_string()),
        tokenizer_path: None,
        config_path: None,
        model_path: Some("files/transformer/model.safetensors".to_string()),
        backend: "media-companion".to_string(),
        created_unix: 1,
        files: vec![ModelFile {
            path: "files/transformer/model.safetensors".to_string(),
            size: 1_000_000_000,
            checksum: "crc32:0".to_string(),
        }],
        artifacts: Vec::new(),
        metadata,
    }
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageGeneration);
    request.prompt = Some("an orbital station".to_string());
    request
}

#[test]
fn boolean_override_has_three_states() {
    assert!(OverrideBool::Inherit.resolve(true));
    assert!(!OverrideBool::Inherit.resolve(false));
    assert!(OverrideBool::Enabled.resolve(false));
    assert!(!OverrideBool::Disabled.resolve(true));
    assert_eq!(OverrideBool::Inherit.explicit(), None);
}

#[test]
fn list_override_distinguishes_inherit_replace_add_and_clear() {
    let inherited = vec!["base"];
    assert_eq!(
        ListOverride::<&str>::Inherit.resolve(&inherited),
        vec!["base"]
    );
    assert_eq!(
        ListOverride::Replace(vec!["new"]).resolve(&inherited),
        vec!["new"]
    );
    assert_eq!(
        ListOverride::Add(vec!["extra"]).resolve(&inherited),
        vec!["base", "extra"]
    );
    assert!(ListOverride::<&str>::Clear.resolve(&inherited).is_empty());
}

#[test]
fn list_override_operations_apply_end_to_end_with_provenance() {
    let mut manifest = image_manifest();
    manifest
        .metadata
        .generation_defaults
        .insert("image.loras".to_string(), json!(["base.safetensors"]));
    let mut add = request();
    add.parameters.insert(
        "loras".to_string(),
        ParameterValue::from_json(json!({
            "operation": "add",
            "values": ["style.safetensors"]
        }))
        .unwrap(),
    );
    let effective = resolve_request(&manifest, add, &ResolutionContext::default()).unwrap();
    assert_eq!(
        effective.parameter("image.loras"),
        Some(&ParameterValue::List(vec![
            "base.safetensors".into(),
            "style.safetensors".into()
        ]))
    );
    assert_eq!(
        effective.parameters["image.loras"].source,
        ParameterSource::RequestOverride
    );

    let mut clear = request();
    clear.parameters.insert(
        "image.loras".to_string(),
        ParameterValue::from_json(json!({"operation": "clear"})).unwrap(),
    );
    let effective = resolve_request(&manifest, clear, &ResolutionContext::default()).unwrap();
    assert_eq!(
        effective.parameter("image.loras"),
        Some(&ParameterValue::List(Vec::new()))
    );
}

#[test]
fn defaults_follow_specificity_and_track_provenance() {
    let manifest = image_manifest();
    let mut request = request();
    request
        .parameters
        .insert("steps".to_string(), 42_i64.into());
    let mut context = ResolutionContext::default();
    context
        .runtime_defaults
        .insert("image.guidance".to_string(), 2.0_f64.into());
    context
        .hardware_profile
        .insert("image.width".to_string(), 768_i64.into());
    context
        .user_profile
        .insert("width".to_string(), 896_i64.into());

    let effective = resolve_request(&manifest, request, &context).unwrap();
    assert_eq!(effective.u64_parameter("image.steps"), Some(42));
    assert_eq!(
        effective.parameters["image.steps"].source,
        ParameterSource::RequestOverride
    );
    assert_eq!(effective.u64_parameter("image.width"), Some(896));
    assert_eq!(
        effective.parameters["image.width"].source,
        ParameterSource::UserProfile
    );
    assert_eq!(effective.f64_parameter("image.guidance"), Some(2.0));
    assert_eq!(
        effective.parameters["image.guidance"].source,
        ParameterSource::RuntimeDefault
    );
}

#[test]
fn manifest_constraints_enrich_schema_and_validate_effective_values() {
    let mut manifest = image_manifest();
    manifest.metadata.parameter_constraints.insert(
        "width".to_string(),
        json!({
            "default": 512,
            "minimum": 320,
            "maximum": 768,
            "step": 64,
            "allowed_values": [320, 512, 768]
        }),
    );
    let schema = parameter_schema_for_manifest(InferenceTask::ImageGeneration, &manifest).unwrap();
    let width = schema
        .iter()
        .find(|descriptor| descriptor.path == "image.width")
        .unwrap();
    assert_eq!(width.minimum, Some(320_i64.into()));
    assert_eq!(width.maximum, Some(768_i64.into()));
    assert_eq!(width.step, Some(64_i64.into()));

    let mut invalid = request();
    invalid
        .parameters
        .insert("image.width".to_string(), 640_i64.into());
    assert!(
        resolve_request(&manifest, invalid, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("must be one of")
    );
}

#[test]
fn invalid_model_default_is_reported_instead_of_silently_dropped() {
    let mut manifest = image_manifest();
    manifest
        .metadata
        .generation_defaults
        .insert("image.width".to_string(), json!({"not": "an integer"}));
    assert!(
        resolve_request(&manifest, request(), &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("expects Integer")
    );
}

#[test]
fn family_default_is_overridden_by_model_default() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    assert_eq!(effective.u64_parameter("image.steps"), Some(35));
    assert_eq!(
        effective.parameters["image.steps"].source,
        ParameterSource::ModelDefault
    );
    assert_eq!(effective.f64_parameter("image.guidance"), Some(3.5));
    assert_eq!(
        effective.parameters["image.guidance"].source,
        ParameterSource::ModelFamilyDefault
    );
}

#[test]
fn validation_rejects_ranges_and_unknown_parameters() {
    let manifest = image_manifest();
    let mut invalid = request();
    invalid.parameters.insert("width".to_string(), 1_i64.into());
    assert!(
        resolve_request(&manifest, invalid, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("minimum")
    );

    let mut unknown = request();
    unknown
        .parameters
        .insert("telepathy".to_string(), true.into());
    assert!(
        resolve_request(&manifest, unknown, &ResolutionContext::default())
            .unwrap_err()
            .to_string()
            .contains("unknown parameter")
    );
}

#[test]
fn strict_parameter_support_rejects_ignored_override() {
    let manifest = image_manifest();
    let mut request = request();
    request
        .parameters
        .insert("steps".to_string(), 12_i64.into());
    let mut context = ResolutionContext::default();
    context
        .parameter_support
        .insert("image.steps".to_string(), ParameterSupportStatus::Ignored);
    assert!(
        resolve_request(&manifest, request.clone(), &context)
            .unwrap_err()
            .to_string()
            .contains("ignored")
    );
    request.routing.parameter_policy = ParameterPolicy::Warn;
    let effective = resolve_request(&manifest, request, &context).unwrap();
    assert_eq!(effective.warnings.len(), 1);
}

#[test]
fn schema_is_machine_readable_and_complete_for_core_image_values() {
    let schema = parameter_schema(InferenceTask::ImageGeneration);
    let paths = schema
        .iter()
        .map(|descriptor| descriptor.path.as_str())
        .collect::<BTreeSet<_>>();
    assert!(paths.contains("routing.parameter_policy"));
    assert!(paths.contains("image.width"));
    assert!(paths.contains("image.controls"));
    assert!(paths.contains("image.loras"));
    assert!(paths.contains("image.vae_tiling"));
    assert!(paths.contains("image.output_path"));
    let width = schema
        .iter()
        .find(|descriptor| descriptor.path == "image.width")
        .unwrap();
    assert!(width.affects_memory);
    assert!(width.affects_runtime);
    assert_eq!(width.minimum, Some(64_i64.into()));
}

#[test]
fn schema_uses_real_primary_flags_for_structured_and_repeatable_values() {
    let flags = [
        (
            InferenceTask::ImageGeneration,
            "image.controls",
            "--image-control",
        ),
        (
            InferenceTask::ImageGeneration,
            "image.loras",
            "--image-lora",
        ),
        (
            InferenceTask::VideoGeneration,
            "video.prompt_keyframes",
            "--video-prompt-keyframe",
        ),
        (InferenceTask::MusicGeneration, "audio.genres", "--genre"),
        (
            InferenceTask::MusicGeneration,
            "audio.instruments",
            "--music-instrument",
        ),
        (
            InferenceTask::MusicGeneration,
            "audio.variations",
            "--num-variations",
        ),
        (
            InferenceTask::MusicGeneration,
            "audio.output_format",
            "--format",
        ),
        (InferenceTask::TextToSpeech, "tts.output_format", "--format"),
        (
            InferenceTask::TextToSpeech,
            "tts.loudness",
            "--loudness-lufs",
        ),
        (InferenceTask::SpeechToText, "stt.operation", "--task"),
        (
            InferenceTask::SpeechToText,
            "stt.temperature_fallbacks",
            "--temperature-fallback",
        ),
    ];
    for (task, path, expected) in flags {
        let schema = parameter_schema(task);
        assert_eq!(
            schema
                .iter()
                .find(|descriptor| descriptor.path == path)
                .map(|descriptor| descriptor.cli_flag.as_str()),
            Some(expected),
            "{path}"
        );
    }
}

#[test]
fn task_schemas_only_advertise_flags_on_their_public_subcommands() {
    for task in [
        InferenceTask::ImageGeneration,
        InferenceTask::ImageEditing,
        InferenceTask::ImageVariation,
        InferenceTask::ImageInpainting,
        InferenceTask::ImageOutpainting,
    ] {
        let subcommand = if task == InferenceTask::ImageGeneration {
            "generate"
        } else {
            "edit"
        };
        assert_schema_flags_exist_on_subcommand::<ImageSchemaCli>(task, subcommand);
    }
    assert_schema_flags_exist_on_subcommand::<ImageSchemaCli>(
        InferenceTask::ImageUpscaling,
        "upscale",
    );

    assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(
        InferenceTask::VideoGeneration,
        "generate",
    );
    assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(
        InferenceTask::ImageToVideo,
        "animate",
    );
    for task in [
        InferenceTask::VideoToVideo,
        InferenceTask::VideoInpainting,
        InferenceTask::VideoExtension,
    ] {
        assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(task, "transform");
    }
    for task in [
        InferenceTask::VideoUpscaling,
        InferenceTask::FrameInterpolation,
    ] {
        assert_schema_flags_exist_on_subcommand::<VideoSchemaCli>(task, "upscale");
    }

    for task in [
        InferenceTask::AudioGeneration,
        InferenceTask::MusicGeneration,
        InferenceTask::SongContinuation,
        InferenceTask::SongVariation,
    ] {
        assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(task, "generate");
    }
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(InferenceTask::TextToSpeech, "speak");
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(
        InferenceTask::SpeechToText,
        "transcribe",
    );
    assert_schema_flags_exist_on_subcommand::<AudioSchemaCli>(
        InferenceTask::StemSeparation,
        "separate",
    );
}

#[test]
fn prepared_tasks_without_public_subcommands_do_not_advertise_task_flags() {
    for task in [
        InferenceTask::VoiceConversion,
        InferenceTask::StemGeneration,
        InferenceTask::AudioEnhancement,
    ] {
        assert!(
            parameter_schema(task)
                .iter()
                .all(|descriptor| descriptor.path.starts_with("routing.")),
            "{task}"
        );
    }
}

#[test]
fn task_specific_schema_filtering_keeps_resolution_consistent() {
    let image_upscale = parameter_schema(InferenceTask::ImageUpscaling);
    assert!(
        image_upscale
            .iter()
            .all(|descriptor| descriptor.path != "image.steps")
    );
    assert!(
        image_upscale
            .iter()
            .all(|descriptor| descriptor.path != "image.controls")
    );

    let video_upscale = parameter_schema(InferenceTask::VideoUpscaling);
    assert!(
        video_upscale
            .iter()
            .all(|descriptor| descriptor.path != "video.reference_images")
    );
    assert!(
        video_upscale
            .iter()
            .all(|descriptor| descriptor.path != "video.prompt_keyframes")
    );

    let stem_paths = parameter_schema(InferenceTask::StemSeparation)
        .into_iter()
        .filter(|descriptor| descriptor.path.starts_with("audio."))
        .map(|descriptor| descriptor.path)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        stem_paths,
        [
            "audio.channels",
            "audio.normalization",
            "audio.num_stems",
            "audio.output_format",
            "audio.output_path",
            "audio.sample_rate",
            "audio.segment_duration",
            "audio.segment_overlap",
            "audio.stems",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );

    let mut manifest = image_manifest();
    manifest.metadata.tasks = vec![InferenceTask::ImageUpscaling];
    let mut request = InferenceRequest::new("flux", InferenceTask::ImageUpscaling);
    request.inputs.push(InferenceInput {
        modality: InputModality::Image,
        role: "image".to_string(),
        source: InferenceInputSource::Path {
            path: "input.png".to_string(),
        },
        mime_type: Some("image/png".to_string()),
    });
    let effective = resolve_request(&manifest, request, &ResolutionContext::default()).unwrap();
    assert!(!effective.parameters.contains_key("image.steps"));
    assert!(effective.parameters.contains_key("image.upscale_scale"));
}

#[test]
fn image_estimate_scales_with_resolution_and_tiling() {
    let manifest = image_manifest();
    let low = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let mut high_request = request();
    high_request
        .parameters
        .insert("width".to_string(), 2048_i64.into());
    high_request
        .parameters
        .insert("height".to_string(), 2048_i64.into());
    let high = resolve_request(&manifest, high_request, &ResolutionContext::default()).unwrap();
    let resources = HostResources::default();
    let low_estimate = estimate_workload(&manifest, &low, &resources);
    let high_estimate = estimate_workload(&manifest, &high, &resources);
    assert!(
        high_estimate.accelerator_peak_bytes.unwrap()
            > low_estimate.accelerator_peak_bytes.unwrap()
    );

    let mut tiled_request = request();
    tiled_request
        .parameters
        .insert("vae_tiling".to_string(), true.into());
    let tiled = resolve_request(&manifest, tiled_request, &ResolutionContext::default()).unwrap();
    assert!(
        estimate_workload(&manifest, &tiled, &resources).accelerator_peak_bytes
            < low_estimate.accelerator_peak_bytes
    );
}

#[test]
fn offload_permission_does_not_change_estimate_until_planner_selects_it() {
    let manifest = image_manifest();
    let baseline = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let mut permitted_request = request();
    permitted_request.routing.allow_cpu_offload = OverrideBool::Enabled;
    let permitted =
        resolve_request(&manifest, permitted_request, &ResolutionContext::default()).unwrap();
    let resources = HostResources::default();

    let baseline_estimate = estimate_workload(&manifest, &baseline, &resources);
    let permitted_estimate = estimate_workload(&manifest, &permitted, &resources);
    assert_eq!(
        permitted_estimate.accelerator_peak_bytes,
        baseline_estimate.accelerator_peak_bytes
    );
    assert_eq!(
        permitted_estimate.host_peak_bytes,
        baseline_estimate.host_peak_bytes
    );

    let mut selected = permitted;
    selected
        .parameters
        .get_mut("routing.allow_cpu_offload")
        .unwrap()
        .source = ParameterSource::BackendAdjustment;
    assert!(
        estimate_workload(&manifest, &selected, &resources).host_peak_bytes
            > baseline_estimate.host_peak_bytes
    );
}

#[test]
fn planner_only_selects_gpu_offload_under_degrade_policy() {
    let manifest = image_manifest();
    let mut permitted_request = request();
    permitted_request.routing.allow_cpu_offload = OverrideBool::Enabled;
    permitted_request.routing.allow_sequential_offload = OverrideBool::Disabled;
    permitted_request.routing.allow_component_offload = OverrideBool::Disabled;
    permitted_request.routing.fallback_policy = Some("degrade".to_string());
    let effective =
        resolve_request(&manifest, permitted_request, &ResolutionContext::default()).unwrap();
    let mut estimate = estimate_workload(&manifest, &effective, &HostResources::default());
    estimate.fit = FitAssessment::LikelyOom;
    let candidate = |id: &str, accelerator: RuntimeAccelerator| InferenceRuntimeCandidate {
        id: id.to_string(),
        backend: "media-companion".to_string(),
        accelerator,
        available: true,
        availability_reason: None,
        supported_tasks: vec![InferenceTask::ImageGeneration],
        supported_layouts: vec![RepositoryLayout::Diffusers],
        supported_formats: vec![ModelFormat::SafeTensors],
        supported_families: Vec::new(),
        supported_architectures: Vec::new(),
        parameter_support: BTreeMap::new(),
        supports_offloading: true,
        supports_compile: false,
        supports_batching: true,
        priority: 100,
    };
    let plan = plan_execution(
        &manifest,
        &effective,
        &estimate,
        &[
            candidate("media-companion-cpu", RuntimeAccelerator::Cpu),
            candidate("media-companion-cuda", RuntimeAccelerator::Cuda),
        ],
    );
    let cpu = plan
        .candidates
        .iter()
        .find(|candidate| candidate.runtime_id == "media-companion-cpu")
        .unwrap();
    assert_eq!(cpu.status, PlanCandidateStatus::Rejected);
    assert!(cpu.degradations.is_empty());
    let cuda = plan
        .candidates
        .iter()
        .find(|candidate| candidate.runtime_id == "media-companion-cuda")
        .unwrap();
    assert_eq!(cuda.status, PlanCandidateStatus::Accepted);
    assert_eq!(cuda.degradations, vec![ExecutionDegradation::CpuOffload]);

    let mut no_degrade = effective;
    no_degrade
        .parameters
        .get_mut("routing.fallback_policy")
        .unwrap()
        .value = "backend".into();
    let plan = plan_execution(
        &manifest,
        &no_degrade,
        &estimate,
        &[candidate("media-companion-cuda", RuntimeAccelerator::Cuda)],
    );
    assert_eq!(plan.candidates[0].status, PlanCandidateStatus::Rejected);
    assert!(plan.candidates[0].degradations.is_empty());
}

#[test]
fn planner_scores_fallbacks_and_never_applies_quality_downgrade() {
    let manifest = image_manifest();
    let effective = resolve_request(&manifest, request(), &ResolutionContext::default()).unwrap();
    let estimate = estimate_workload(
        &manifest,
        &effective,
        &HostResources {
            host_memory_bytes: Some(16_000_000_000),
            accelerator_memory_bytes: Some(16_000_000_000),
            accelerator: Some("cuda".to_string()),
        },
    );
    let candidates = vec![
        InferenceRuntimeCandidate {
            id: "unavailable".to_string(),
            backend: "diffusers".to_string(),
            accelerator: RuntimeAccelerator::Cuda,
            available: false,
            availability_reason: Some("missing".to_string()),
            supported_tasks: vec![InferenceTask::ImageGeneration],
            supported_layouts: vec![RepositoryLayout::Diffusers],
            supported_formats: vec![ModelFormat::SafeTensors],
            supported_families: Vec::new(),
            supported_architectures: Vec::new(),
            parameter_support: BTreeMap::new(),
            supports_offloading: true,
            supports_compile: true,
            supports_batching: true,
            priority: 1000,
        },
        InferenceRuntimeCandidate {
            id: "working".to_string(),
            backend: "media-companion".to_string(),
            accelerator: RuntimeAccelerator::Cpu,
            available: true,
            availability_reason: None,
            supported_tasks: vec![InferenceTask::ImageGeneration],
            supported_layouts: vec![RepositoryLayout::Diffusers],
            supported_formats: vec![ModelFormat::SafeTensors],
            supported_families: Vec::new(),
            supported_architectures: Vec::new(),
            parameter_support: BTreeMap::new(),
            supports_offloading: true,
            supports_compile: false,
            supports_batching: true,
            priority: 500,
        },
    ];
    let plan = plan_execution(&manifest, &effective, &estimate, &candidates);
    assert_eq!(plan.selected_runtime.as_deref(), Some("working"));
    assert!(
        plan.candidates
            .iter()
            .any(|candidate| candidate.runtime_id == "unavailable"
                && candidate.status == PlanCandidateStatus::Rejected)
    );
    assert!(plan.model_or_quality_downgrades.is_empty());
}

#[test]
fn conversation_content_serializes_typed_media_and_tools() {
    let content = ConversationContent::ToolResult(ToolResultContent {
        call_id: "call-1".to_string(),
        result: json!({"ok": true}),
        content: vec![ConversationContent::Image(MediaContent {
            url: Some("/v1/outputs/out.png".to_string()),
            path: None,
            mime_type: "image/png".to_string(),
            metadata: BTreeMap::new(),
        })],
    });
    let value = serde_json::to_value(content).unwrap();
    assert_eq!(value["type"], "tool_result");
    assert_eq!(value["content"]["content"][0]["type"], "image");
}
