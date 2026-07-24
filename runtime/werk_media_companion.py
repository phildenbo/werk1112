#!/usr/bin/env python3
"""Offline media companion for Werk.

The process handles exactly one operation. It reads one JSON object from stdin
and writes exactly one JSON object to stdout. Model loading is deliberately
local-only: this module never installs packages and never downloads a model.
"""

import contextlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import mimetypes
import os
import shutil
import sys
import time
import traceback
import uuid
import wave
from pathlib import Path


PROTOCOL_VERSION = 1
COMPANION_VERSION = "1.0.0"

for _name in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "DIFFUSERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
):
    os.environ[_name] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


IMAGE_TASKS = {
    "image_generation",
    "image_editing",
    "image_variation",
    "image_inpainting",
    "image_outpainting",
    "image_upscaling",
}
VIDEO_TASKS = {
    "video_generation",
    "image_to_video",
    "video_to_video",
    "video_inpainting",
    "video_extension",
    "video_upscaling",
    "frame_interpolation",
}
AUDIO_GENERATION_TASKS = {
    "audio_generation",
    "music_generation",
}
TTS_TASKS = {"text_to_speech"}
ASR_TASKS = {"speech_to_text"}
DECLARED_UNSUPPORTED_TASKS = {
    "song_continuation",
    "song_variation",
    "voice_conversion",
    "stem_generation",
    "stem_separation",
    "audio_enhancement",
}

ORCHESTRATOR_PARAMETERS = {
    "routing.backend",
    "routing.fallback_policy",
    "routing.parameter_policy",
    "routing.performance_preference",
    "routing.profile",
    "routing.quality",
    "routing.timeout",
}
TORCH_RUNTIME_PARAMETERS = {
    "routing.accelerator",
    "routing.device",
    "routing.precision",
}
DIFFUSERS_ROUTING_PARAMETERS = {
    "routing.allow_component_offload",
    "routing.allow_cpu_offload",
    "routing.allow_sequential_offload",
}
IMAGE_ADAPTER_PARAMETERS = {
    "image.batch_size",
    "image.guidance",
    "image.height",
    "image.loras",
    "image.num_images",
    "image.output_format",
    "image.seed",
    "image.steps",
    "image.vae_slicing",
    "image.vae_tiling",
    "image.width",
}
VIDEO_ADAPTER_PARAMETERS = {
    "video.batch_size",
    "video.fps",
    "video.frames",
    "video.guidance",
    "video.height",
    "video.num_videos",
    "video.output_format",
    "video.seed",
    "video.steps",
    "video.temporal_vae_tiling",
    "video.width",
}
DIFFUSERS_AUDIO_PARAMETERS = {
    "audio.duration",
    "audio.guidance",
    "audio.lyrics",
    "audio.output_format",
    "audio.seed",
    "audio.steps",
    "audio.variations",
}
TRANSFORMERS_AUDIO_PARAMETERS = {
    "audio.guidance",
    "audio.lyrics",
    "audio.output_format",
    "audio.seed",
}
TTS_ADAPTER_PARAMETERS = {
    "tts.output_format",
}
ASR_ADAPTER_PARAMETERS = {
    "stt.beam_size",
    "stt.initial_prompt",
    "stt.language",
    "stt.operation",
    "stt.output_format",
    "stt.segment_timestamps",
    "stt.temperature",
    "stt.word_timestamps",
}


class CompanionFailure(Exception):
    def __init__(self, code, message, detail=None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.detail = detail


def fail(code, message, detail=None):
    raise CompanionFailure(code, message, detail)


def normalized_name(value):
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalized_parameters(payload):
    value = payload.get("effective_parameters", payload.get("parameters", {}))
    if value is None:
        return {}
    if not isinstance(value, dict):
        fail("invalid_request", "effective_parameters must be a JSON object")
    normalized = {}
    for key, item in value.items():
        if isinstance(item, dict) and "value" in item:
            item = item["value"]
        name = normalized_name(key)
        normalized[name] = item
        # Werk's canonical schema uses dotted paths (for example
        # ``image.width``).  Pipeline adapters consume the task-local leaf
        # while the full path remains available for diagnostics.
        if "." in name:
            normalized.setdefault(name.rsplit(".", 1)[1], item)
    if payload.get("prompt") is not None:
        normalized["prompt"] = payload["prompt"]
    if payload.get("negative_prompt") is not None:
        normalized["negative_prompt"] = payload["negative_prompt"]
    return normalized


def explicit_parameter_paths(payload):
    raw = payload.get("explicit_parameters", [])
    if raw is None:
        return set()
    if not isinstance(raw, list):
        fail("invalid_request", "explicit_parameters must be a JSON array of parameter paths")
    paths = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            fail(
                "invalid_request",
                "explicit_parameters entries must be non-empty strings",
            )
        paths.add(normalized_name(value))
    return paths


def requested_parameter_policy(payload, parameters):
    value = payload.get("parameter_policy")
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if value is None:
        value = parameters.get("routing.parameter_policy")
    if value is None:
        value = parameters.get("parameter_policy")
    policy = normalized_name(value or "strict")
    if policy not in {"strict", "warn", "permissive"}:
        fail(
            "invalid_parameter",
            "parameter_policy must be strict, warn, or permissive",
        )
    return policy


class ExplicitParameterGuard:
    def __init__(self, payload, task, adapter, parameters):
        self.task = task
        self.adapter = adapter
        self.policy = requested_parameter_policy(payload, parameters)
        self.explicit = explicit_parameter_paths(payload)
        if payload.get("negative_prompt") is not None:
            self.explicit.add("negative_prompt")
        if task in ASR_TASKS and payload.get("prompt") is not None:
            self.explicit.add("stt.initial_prompt")
        self.unsupported = {}
        self.warnings = []

    def validate_supported(self, supported):
        unsupported = {
            path: (
                f"the {self.adapter} adapter for task '{self.task}' does not "
                "consume this parameter"
            )
            for path in self.explicit
            if path not in supported
        }
        self._record(unsupported)

    def reject(self, path, reason):
        path = normalized_name(path)
        if path in self.explicit:
            self._record({path: reason})

    def reject_overridden(self, path, winner):
        self.reject(
            path,
            f"it is overridden by explicit parameter '{normalized_name(winner)}'",
        )

    def metadata(self):
        return {
            "policy": self.policy,
            "explicit_parameters": sorted(self.explicit),
            "unsupported_explicit_parameters": sorted(self.unsupported),
            "unsupported_reasons": dict(sorted(self.unsupported.items())),
        }

    def without_unsupported(self, parameters):
        sanitized = dict(parameters)
        for path in self.unsupported:
            sanitized.pop(path, None)
            if "." in path:
                sanitized.pop(path.rsplit(".", 1)[1], None)
            if path == "stt.initial_prompt" and self.task in ASR_TASKS:
                sanitized.pop("prompt", None)
        return sanitized

    def _record(self, unsupported):
        added = {}
        for path, reason in unsupported.items():
            if path not in self.unsupported:
                self.unsupported[path] = reason
                added[path] = reason
        if not added:
            return
        if self.policy == "strict":
            paths = sorted(added)
            fail(
                "unsupported_parameter",
                "explicit parameters are not supported by the selected media adapter",
                {
                    "adapter": self.adapter,
                    "task": self.task,
                    "parameters": paths,
                    "reasons": {path: added[path] for path in paths},
                },
            )
        if self.policy == "warn":
            for path in sorted(added):
                self.warnings.append(
                    f"explicit parameter '{path}' is unsupported by the "
                    f"{self.adapter} adapter and was ignored: {added[path]}"
                )


def execution_adapter(model_path, task):
    if task in IMAGE_TASKS | VIDEO_TASKS:
        return "diffusers"
    if task in AUDIO_GENERATION_TASKS:
        root = model_path if model_path.is_dir() else model_path.parent
        return (
            "diffusers_audio"
            if (root / "model_index.json").is_file()
            else "transformers_audio"
        )
    if task in TTS_TASKS:
        return "transformers_tts"
    if task in ASR_TASKS:
        return "transformers_asr"
    return None


def supported_explicit_parameters(task, adapter):
    supported = set(ORCHESTRATOR_PARAMETERS)
    supported.update(TORCH_RUNTIME_PARAMETERS)
    if adapter in {"diffusers", "diffusers_audio"}:
        supported.update(DIFFUSERS_ROUTING_PARAMETERS)
        supported.add("negative_prompt")
    if task in IMAGE_TASKS:
        supported.update(IMAGE_ADAPTER_PARAMETERS)
    elif task in VIDEO_TASKS:
        supported.update(VIDEO_ADAPTER_PARAMETERS)
    elif adapter == "diffusers_audio":
        supported.update(DIFFUSERS_AUDIO_PARAMETERS)
    elif adapter == "transformers_audio":
        supported.update(TRANSFORMERS_AUDIO_PARAMETERS)
    elif adapter == "transformers_tts":
        supported.update(TTS_ADAPTER_PARAMETERS)
    elif adapter == "transformers_asr":
        supported.update(ASR_ADAPTER_PARAMETERS)
    return supported


def input_values(payload, parameters):
    values = payload.get("inputs", {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        fail("invalid_request", "inputs must be a JSON object")
    merged = {
        normalized_name(key): value
        for key, value in values.items()
        if value is not None
    }
    for key in (
        "source",
        "input",
        "image",
        "input_image",
        "initial_image",
        "final_image",
        "mask",
        "mask_image",
        "mask_video",
        "source_video",
        "input_video",
        "video",
        "input_audio",
        "source_audio",
        "audio",
        "reference_audio",
    ):
        if payload.get(key) is not None:
            merged[key] = payload[key]
        elif parameters.get(key) is not None and key not in merged:
            merged[key] = parameters[key]
    if "video" in merged:
        merged.setdefault("source_video", merged["video"])
    if "audio" in merged:
        merged.setdefault("source_audio", merged["audio"])
    return merged


def validate_adapter_inputs(task, adapter, inputs):
    allowed = set()
    if task in IMAGE_TASKS:
        allowed.update(
            {
                "image",
                "input_image",
                "initial_image",
                "final_image",
                "mask",
                "mask_image",
            }
        )
    elif task in VIDEO_TASKS:
        allowed.update(
            {
                "image",
                "input_image",
                "initial_image",
                "final_image",
                "video",
                "source_video",
                "input_video",
                "mask",
                "mask_video",
            }
        )
    elif adapter == "transformers_asr":
        allowed.update(
            {
                "audio",
                "source_audio",
                "input_audio",
                "source",
                "input",
            }
        )
    unsupported = sorted(set(inputs) - allowed)
    if unsupported:
        fail(
            "unsupported_parameter",
            f"task '{task}' does not support the supplied media input roles",
            {"input_roles": unsupported},
        )


def required_string(value, name):
    if not isinstance(value, str) or not value.strip():
        fail("invalid_request", f"{name} must be a non-empty string")
    return value.strip()


def prompt_with_lyrics(prompt, lyrics):
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    lyrics = lyrics.strip() if isinstance(lyrics, str) else ""
    if not lyrics:
        return prompt or None
    if not prompt:
        return lyrics
    lyrics_block = f"Lyrics:\n{lyrics}"
    if prompt == lyrics or prompt == lyrics_block or prompt.endswith(f"\n\n{lyrics_block}"):
        return prompt
    return f"{prompt}\n\n{lyrics_block}"


def local_model_path(payload):
    raw = required_string(payload.get("model_path"), "model_path")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        fail("model_not_found", f"local model path does not exist: {path}")
    return path


def output_directory(payload):
    raw = required_string(payload.get("output_dir"), "output_dir")
    path = Path(raw).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        fail("output_error", f"cannot create output directory: {path}", str(error))
    if not path.is_dir():
        fail("output_error", f"output path is not a directory: {path}")
    return path


def module_status(module_name, distribution=None):
    available = importlib.util.find_spec(module_name) is not None
    version = None
    detail = None
    if available:
        try:
            version = importlib.metadata.version(distribution or module_name)
        except Exception:
            detail = "module found; version unavailable"
    else:
        detail = "not installed (optional)"
    return {
        "available": available,
        "version": version,
        "detail": detail,
    }


def dependency_snapshot():
    dependencies = {
        "torch": module_status("torch"),
        "diffusers": module_status("diffusers"),
        "transformers": module_status("transformers"),
        "PIL": module_status("PIL", "Pillow"),
        "numpy": module_status("numpy"),
        "soundfile": module_status("soundfile"),
        "scipy": module_status("scipy"),
        "librosa": module_status("librosa"),
        "torchaudio": module_status("torchaudio"),
        "imageio": module_status("imageio"),
        "imageio_ffmpeg": module_status("imageio_ffmpeg", "imageio-ffmpeg"),
        "av": module_status("av"),
    }
    ffmpeg = shutil.which("ffmpeg")
    dependencies["ffmpeg"] = {
        "available": ffmpeg is not None,
        "version": None,
        "detail": ffmpeg or "ffmpeg executable not found (optional)",
    }
    return dependencies


def require_module(module_name, distribution=None, purpose=None):
    try:
        return importlib.import_module(module_name)
    except Exception as error:
        package = distribution or module_name
        suffix = f" for {purpose}" if purpose else ""
        fail(
            "missing_dependency",
            f"optional dependency '{package}' is required{suffix}",
            str(error),
        )


def command_health(_payload):
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "companion_version": COMPANION_VERSION,
        "python_version": sys.version.split()[0],
        "offline": True,
        "dependencies": dependency_snapshot(),
    }


def task_capability(task, available, runtime, reason=None):
    return {
        "task": task,
        "available": bool(available),
        "runtime": runtime,
        "model_dependent": True,
        "reason": reason,
    }


def command_capabilities(_payload):
    deps = dependency_snapshot()
    image_ready = (
        deps["torch"]["available"]
        and deps["diffusers"]["available"]
        and deps["PIL"]["available"]
    )
    video_ready = image_ready and (
        deps["imageio"]["available"]
        or deps["imageio_ffmpeg"]["available"]
        or deps["ffmpeg"]["available"]
    )
    transformers_audio_ready = (
        deps["torch"]["available"]
        and deps["transformers"]["available"]
        and deps["numpy"]["available"]
    )
    generative_audio_ready = (
        deps["torch"]["available"]
        and deps["numpy"]["available"]
        and (
            deps["transformers"]["available"]
            or deps["diffusers"]["available"]
        )
    )
    model_dependent_reason = (
        "adapter dependencies are available; exact model, pipeline, and "
        "parameter compatibility is confirmed when the concrete pipeline is loaded for execution"
    )
    capabilities = []
    for task in sorted(IMAGE_TASKS):
        capabilities.append(
            task_capability(
                task,
                image_ready,
                "diffusers",
                model_dependent_reason
                if image_ready
                else "requires torch, diffusers and Pillow",
            )
        )
    for task in sorted(VIDEO_TASKS):
        capabilities.append(
            task_capability(
                task,
                video_ready,
                "diffusers",
                model_dependent_reason
                if video_ready
                else "requires torch, diffusers, Pillow and a video encoder",
            )
        )
    for task in sorted(AUDIO_GENERATION_TASKS):
        capabilities.append(
            task_capability(
                task,
                generative_audio_ready,
                "diffusers-or-transformers",
                model_dependent_reason
                if generative_audio_ready
                else "requires torch, numpy, and either diffusers or transformers",
            )
        )
    for task in sorted(TTS_TASKS | ASR_TASKS):
        capabilities.append(
            task_capability(
                task,
                transformers_audio_ready,
                "transformers",
                model_dependent_reason
                if transformers_audio_ready
                else "requires torch, transformers and numpy",
            )
        )
    for task in sorted(DECLARED_UNSUPPORTED_TASKS):
        capabilities.append(
            task_capability(
                task,
                False,
                None,
                "the generic companion has no reliable local adapter for this task",
            )
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "offline": True,
        "capabilities": capabilities,
        "parameter_policy": {
            "default": "strict",
            "supported": ["strict", "warn", "permissive"],
            "scope": "explicit_parameters are checked against the selected task and adapter",
        },
        "input_modalities": ["text", "image", "video", "audio"],
        "output_modalities": ["image", "video", "audio", "text"],
    }


def read_json_file(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def file_inventory(root):
    if root.is_file():
        return [root]
    files = []
    for directory, names, file_names in os.walk(root, followlinks=False):
        names[:] = [
            name
            for name in names
            if name not in {".git", ".cache", "__pycache__", "outputs"}
        ]
        base = Path(directory)
        for name in file_names:
            files.append(base / name)
    return files


def model_probe(path):
    root = path if path.is_dir() else path.parent
    model_index = read_json_file(root / "model_index.json")
    config = read_json_file(root / "config.json")
    generation_config = read_json_file(root / "generation_config.json")
    files = file_inventory(path)
    names = {item.name.lower() for item in files}
    directories = {
        item.name
        for item in root.iterdir()
        if item.is_dir()
    } if root.is_dir() else set()

    if model_index:
        layout = "diffusers"
    elif config or "config.json" in names:
        layout = "transformers"
    elif path.is_file():
        layout = "single_file"
    else:
        layout = "custom"

    class_name = str(model_index.get("_class_name") or "")
    model_type = str(config.get("model_type") or "")
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [str(architectures)]
    pipeline_tag = str(
        config.get("pipeline_tag")
        or model_index.get("pipeline_tag")
        or generation_config.get("pipeline_tag")
        or ""
    )
    searchable = " ".join(
        [path.name, class_name, model_type, pipeline_tag]
        + [str(item) for item in architectures]
    ).lower()

    tasks = []
    if model_index:
        if any(word in searchable for word in ("video", "animatediff", "cogvideo", "wan")):
            tasks.extend(["video_generation", "image_to_video"])
        elif any(word in searchable for word in ("audio", "music", "audioldm")):
            tasks.extend(["audio_generation", "music_generation"])
        else:
            tasks.append("image_generation")
            if "inpaint" in searchable:
                tasks.append("image_inpainting")
            if any(word in searchable for word in ("img2img", "image2image")):
                tasks.append("image_editing")
    if any(word in searchable for word in ("whisper", "speech_to_text", "automatic-speech")):
        tasks.append("speech_to_text")
    if any(
        word in searchable
        for word in ("speecht5", "bark", "vits", "fastspeech", "text-to-speech")
    ):
        tasks.append("text_to_speech")
    if any(
        word in searchable
        for word in ("musicgen", "audiogen", "text-to-audio", "text_to_audio")
    ):
        tasks.extend(["audio_generation", "music_generation"])
    if not tasks and config:
        # A generic Transformers repository remains cataloguable. Execution is
        # intentionally model-dependent instead of guessing a text model is media.
        tasks = []

    components = sorted(
        name
        for name in directories
        if normalized_name(name)
        in {
            "transformer",
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "scheduler",
            "encoder",
            "decoder",
            "vocoder",
            "feature_extractor",
            "controlnet",
            "adapter",
        }
    )
    weight_extensions = {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".onnx",
        ".gguf",
        ".ckpt",
        ".npz",
    }
    weight_files = [
        item for item in files if item.suffix.lower() in weight_extensions
    ]
    return {
        "layout": layout,
        "class_name": class_name or None,
        "model_type": model_type or None,
        "architectures": architectures,
        "pipeline_tag": pipeline_tag or None,
        "tasks": sorted(set(tasks)),
        "components": components,
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "weight_payload_bytes": sum(safe_size(item) for item in weight_files),
        "config": config,
    }


def task_dependency_ready(task, deps):
    if task in IMAGE_TASKS:
        return (
            deps["torch"]["available"]
            and deps["diffusers"]["available"]
            and deps["PIL"]["available"]
        ), "requires torch, diffusers and Pillow"
    if task in VIDEO_TASKS:
        base = (
            deps["torch"]["available"]
            and deps["diffusers"]["available"]
            and deps["PIL"]["available"]
        )
        encoder = (
            deps["imageio"]["available"]
            or deps["imageio_ffmpeg"]["available"]
            or deps["ffmpeg"]["available"]
        )
        return base and encoder, "requires torch, diffusers, Pillow and a video encoder"
    if task in AUDIO_GENERATION_TASKS:
        return (
            deps["torch"]["available"]
            and deps["numpy"]["available"]
            and (
                deps["diffusers"]["available"]
                or deps["transformers"]["available"]
            )
        ), "requires torch, numpy, and either diffusers or transformers"
    if task in TTS_TASKS | ASR_TASKS:
        return (
            deps["torch"]["available"]
            and deps["transformers"]["available"]
            and deps["numpy"]["available"]
        ), "requires torch, transformers and numpy"
    return False, "task has no generic companion adapter"


def command_probe_model(payload):
    path = local_model_path(payload)
    probe = model_probe(path)
    requested_task = normalized_name(payload.get("task"))
    dependencies = dependency_snapshot()
    supported = None
    reasons = []
    if requested_task:
        ready, dependency_reason = task_dependency_ready(requested_task, dependencies)
        recognized = (
            requested_task in probe["tasks"]
            or (
                probe["layout"] == "diffusers"
                and requested_task in IMAGE_TASKS | VIDEO_TASKS
            )
        )
        supported = ready and recognized
        if not ready:
            reasons.append(dependency_reason)
        if not recognized:
            reasons.append(
                "model metadata does not advertise this task; support remains model-dependent"
            )
    if reasons:
        detail = "; ".join(reasons)
    elif requested_task and supported:
        detail = (
            "local dependencies and model metadata match the requested task; "
            "exact pipeline and parameter compatibility remains model-dependent"
        )
    elif requested_task:
        detail = "requested task support could not be established"
    else:
        detail = "model metadata probe completed; no task was requested"
    return {
        "model_path": str(path),
        "supported": supported,
        "reasons": reasons,
        "detail": detail,
        "probe": {
            key: value
            for key, value in probe.items()
            if key != "config"
        },
        "offline": True,
    }


def safe_size(path):
    try:
        return path.stat().st_size
    except Exception:
        return 0


def positive_int(parameters, key, default, minimum=1):
    value = parameters.get(key, default)
    if isinstance(value, bool):
        fail("invalid_parameter", f"{key} must be an integer")
    try:
        value = int(value)
    except Exception:
        fail("invalid_parameter", f"{key} must be an integer")
    if value < minimum:
        fail("invalid_parameter", f"{key} must be at least {minimum}")
    return value


def positive_float(parameters, key, default, minimum=0.0):
    value = parameters.get(key, default)
    if isinstance(value, bool):
        fail("invalid_parameter", f"{key} must be numeric")
    try:
        value = float(value)
    except Exception:
        fail("invalid_parameter", f"{key} must be numeric")
    if not math.isfinite(value) or value < minimum:
        fail("invalid_parameter", f"{key} must be at least {minimum}")
    return value


def command_estimate(payload):
    path = local_model_path(payload)
    task = normalized_name(payload.get("task"))
    if not task:
        fail("invalid_request", "task is required for workload estimation")
    parameters = normalized_parameters(payload)
    adapter = execution_adapter(path, task)
    if adapter is None:
        fail("unsupported_task", f"no estimate adapter for task '{task}'")
    parameter_guard = ExplicitParameterGuard(
        payload,
        task,
        adapter,
        parameters,
    )
    parameter_guard.validate_supported(
        supported_explicit_parameters(task, adapter)
    )
    parameters = parameter_guard.without_unsupported(parameters)
    probe = model_probe(path)
    files = file_inventory(path)
    download_size = sum(safe_size(item) for item in files)
    weight_payload = probe["weight_payload_bytes"]
    if weight_payload == 0:
        weight_payload = download_size

    batch = positive_int(parameters, "batch_size", 1)
    count = positive_int(
        parameters,
        "num_images"
        if task in IMAGE_TASKS
        else "num_videos"
        if task in VIDEO_TASKS
        else "num_variations",
        1,
    )
    assumptions = [
        "weights are loaded from the local repository without conversion",
        "activation memory is a conservative architecture-independent heuristic",
    ]
    warnings = []
    recommendations = []
    # The public allow_* values are planner permissions, not execution
    # commands.  Only Werk's internal flags represent a degradation selected
    # for the concrete runtime.
    offload = bool(
        parameters.get("_werk_enable_cpu_offload")
        or parameters.get("_werk_enable_sequential_offload")
        or parameters.get("_werk_enable_component_offload")
    )
    precision = normalized_name(parameters.get("precision") or "auto")
    precision_scale = {
        "fp16": 0.6,
        "float16": 0.6,
        "half": 0.6,
        "bf16": 0.6,
        "bfloat16": 0.6,
        "fp8": 0.4,
        "float8": 0.4,
        "int8": 0.4,
        "int4": 0.3,
        "nf4": 0.3,
    }.get(precision, 1.0)
    if precision != "auto":
        assumptions.append(
            f"activation estimate is scaled for requested precision '{precision}'"
        )
    attention = normalized_name(parameters.get("attention_backend") or "auto")
    attention_scale = {
        "flash": 0.78,
        "flash_attention": 0.78,
        "xformers": 0.78,
        "sliced": 0.65,
        "sdpa": 0.88,
    }.get(attention, 1.0)
    if attention != "auto":
        assumptions.append(
            f"activation estimate is scaled for attention backend '{attention}'"
        )
    component_overhead = len(probe["components"]) * 32 * 1024**2
    assumptions.append(
        f"{len(probe['components'])} detected model component(s) contribute runtime overhead"
    )

    if task in IMAGE_TASKS:
        width = positive_int(parameters, "width", 1024)
        height = positive_int(parameters, "height", 1024)
        steps = positive_int(parameters, "steps", 28)
        pixels = width * height * batch
        activations = int(pixels * 4 * 32 * precision_scale * attention_scale)
        accelerator_peak = int(
            weight_payload * (0.72 if offload else 1.12)
            + activations
            + component_overhead
        )
        host_peak = int(
            weight_payload * (1.15 if offload else 0.30)
            + 768 * 1024**2
            + component_overhead
        )
        output_size = int(width * height * 1.6 * count)
        assumptions.append(f"{width}x{height}, batch {batch}, {steps} denoising steps")
        if width * height > 1024 * 1024:
            recommendations.append("enable VAE tiling if the selected pipeline supports it")
    elif task in VIDEO_TASKS:
        width = positive_int(parameters, "width", 832)
        height = positive_int(parameters, "height", 480)
        frames = positive_int(parameters, "frames", parameters.get("num_frames", 49))
        fps = positive_float(parameters, "fps", 24.0, 0.1)
        pixels = width * height * frames * batch
        window = positive_int(parameters, "window_size", frames)
        active_frames = min(frames, window)
        activations = int(
            width
            * height
            * active_frames
            * batch
            * 4
            * 42
            * precision_scale
            * attention_scale
        )
        accelerator_peak = int(
            weight_payload * (0.68 if offload else 1.10)
            + activations
            + component_overhead
        )
        host_peak = int(
            weight_payload * (1.20 if offload else 0.35)
            + 1536 * 1024**2
            + component_overhead
        )
        duration = frames / fps
        bitrate = positive_int(parameters, "bitrate", 8_000_000)
        output_size = int(duration * bitrate / 8 * count)
        assumptions.append(
            f"{width}x{height}, {frames} frames at {fps:g} fps, active window {active_frames}"
        )
        recommendations.append("use temporal windowing or tiling when full-frame fit is marginal")
    elif task in AUDIO_GENERATION_TASKS | TTS_TASKS | ASR_TASKS:
        duration = positive_float(parameters, "duration", 30.0, 0.01)
        sample_rate = positive_int(parameters, "sample_rate", 44_100)
        channels = positive_int(parameters, "channels", 2)
        stems = max(1, len(parameters.get("stems", []) or []))
        working_audio = int(duration * sample_rate * channels * 4 * stems)
        accelerator_peak = int(
            weight_payload * (0.80 if offload else 1.08)
            + working_audio * 8 * precision_scale
            + component_overhead
        )
        host_peak = int(
            weight_payload * (1.15 if offload else 0.35)
            + working_audio * 3
            + component_overhead
        )
        bit_depth = positive_int(parameters, "bit_depth", 16)
        output_size = int(duration * sample_rate * channels * bit_depth / 8 * stems)
        assumptions.append(
            f"{duration:g}s, {sample_rate} Hz, {channels} channel(s), {stems} output stem(s)"
        )
        if task in ASR_TASKS:
            output_size = max(4096, int(duration * 80))
            assumptions.append("speech-to-text output is structured text rather than PCM audio")
    else:
        fail("unsupported_task", f"no estimate adapter for task '{task}'")

    if weight_payload == download_size:
        warnings.append("weight files were not identified precisely; all repository bytes were counted")
    warnings.extend(parameter_guard.warnings)
    confidence = "heuristic"
    return {
        "task": task,
        "model_path": str(path),
        "download_size_bytes": download_size,
        "weight_payload_bytes": weight_payload,
        "accelerator_peak_bytes": max(0, accelerator_peak),
        "host_peak_bytes": max(0, host_peak),
        "output_size_bytes": max(0, output_size),
        "confidence": confidence,
        "fit_assessment": "unknown",
        "assumptions": assumptions,
        "warnings": warnings,
        "parameter_support": parameter_guard.metadata(),
        "recommendations": recommendations,
        "backend": "werk-media-companion",
        "offline": True,
    }


def torch_runtime(parameters):
    torch = require_module("torch", purpose="media inference")
    requested = normalized_name(
        parameters.get("device")
        or parameters.get("accelerator")
        or "auto"
    )
    if requested in {"auto", ""}:
        if bool(torch.cuda.is_available()):
            device = "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and bool(torch.backends.mps.is_available())
        ):
            device = "mps"
        else:
            device = "cpu"
    elif requested in {"cuda", "rocm", "hip"}:
        if not bool(torch.cuda.is_available()):
            fail("accelerator_unavailable", f"{requested} was requested but torch reports no GPU")
        device = "cuda"
    elif requested in {"mps", "metal"}:
        if (
            getattr(torch.backends, "mps", None) is None
            or not bool(torch.backends.mps.is_available())
        ):
            fail("accelerator_unavailable", "MPS/Metal was requested but is unavailable")
        device = "mps"
    elif requested == "cpu":
        device = "cpu"
    else:
        fail("unsupported_parameter", f"unsupported device/accelerator '{requested}'")

    precision = normalized_name(parameters.get("precision") or "auto")
    if precision in {"float32", "fp32", "f32"}:
        dtype = torch.float32
    elif precision in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif precision in {"float16", "fp16", "f16", "half"}:
        dtype = torch.float16
    elif precision in {"auto", ""}:
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    else:
        fail("unsupported_parameter", f"unsupported precision '{precision}'")
    return torch, device, dtype


def seeded_generator(torch, device, parameters):
    seed = parameters.get("seed")
    if seed is None:
        return None, None
    try:
        seed = int(seed)
    except Exception:
        fail("invalid_parameter", "seed must be an integer")
    generator_device = device if device in {"cuda", "cpu"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed & ((1 << 63) - 1))
    return generator, seed


def local_input_path(value, name):
    if isinstance(value, list):
        if not value:
            fail("invalid_request", f"{name} list is empty")
        value = value[0]
    path = Path(required_string(value, name)).expanduser().resolve()
    if not path.is_file():
        fail("input_not_found", f"{name} does not exist or is not a file: {path}")
    return path


def load_image(value, name):
    image_module = require_module("PIL.Image", "Pillow", "image input")
    path = local_input_path(value, name)
    try:
        with image_module.open(path) as image:
            return image.convert("RGB")
    except Exception as error:
        fail("invalid_input", f"failed to load {name}: {path}", str(error))


def load_video_frames(value, name):
    path = local_input_path(value, name)
    imageio = require_module("imageio.v3", "imageio", "video input")
    image_module = require_module("PIL.Image", "Pillow", "video input")
    try:
        return [image_module.fromarray(frame).convert("RGB") for frame in imageio.imiter(path)]
    except Exception as error:
        fail("invalid_input", f"failed to decode {name}: {path}", str(error))


def supports_keyword(callable_value, keyword):
    try:
        signature = inspect.signature(callable_value)
    except Exception:
        return True
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def filtered_kwargs(
    callable_value,
    values,
    required=(),
    parameter_guard=None,
    parameter_paths=None,
):
    result = {}
    unsupported = []
    parameter_paths = parameter_paths or {}
    for key, value in values.items():
        if value is None:
            continue
        if supports_keyword(callable_value, key):
            result[key] = value
        elif key in required:
            unsupported.append(key)
        elif parameter_guard is not None and key in parameter_paths:
            paths = parameter_paths[key]
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                parameter_guard.reject(
                    path,
                    f"the selected pipeline call does not accept keyword '{key}'",
                )
    if unsupported:
        fail(
            "unsupported_parameter",
            "selected pipeline cannot accept required parameters",
            {"parameters": unsupported},
        )
    return result


def load_diffusers_pipeline(model_path, task, has_image, torch, dtype):
    diffusers = require_module("diffusers", purpose=task)
    class_name = None
    if task in {"image_inpainting", "image_outpainting"}:
        class_name = "AutoPipelineForInpainting"
    elif task in {
        "image_editing",
        "image_variation",
        "image_upscaling",
    }:
        class_name = "AutoPipelineForImage2Image"
    elif task in VIDEO_TASKS and has_image:
        class_name = "AutoPipelineForImage2Video"
    elif task == "image_generation":
        class_name = "AutoPipelineForText2Image"

    pipeline_class = getattr(diffusers, class_name, None) if class_name else None
    if pipeline_class is None:
        pipeline_class = getattr(diffusers, "DiffusionPipeline", None)
    if pipeline_class is None:
        fail("missing_dependency", "installed diffusers has no compatible pipeline class")

    load_kwargs = {"local_files_only": True, "torch_dtype": dtype}
    try:
        if model_path.is_file() and hasattr(pipeline_class, "from_single_file"):
            return pipeline_class.from_single_file(str(model_path), **load_kwargs)
        return pipeline_class.from_pretrained(str(model_path), **load_kwargs)
    except TypeError:
        # Older Diffusers releases may use the deprecated dtype spelling.
        load_kwargs.pop("torch_dtype", None)
        load_kwargs["dtype"] = dtype
        try:
            if model_path.is_file() and hasattr(pipeline_class, "from_single_file"):
                return pipeline_class.from_single_file(str(model_path), **load_kwargs)
            return pipeline_class.from_pretrained(str(model_path), **load_kwargs)
        except Exception as error:
            fail("model_load_failed", f"failed to load local Diffusers model: {model_path}", str(error))
    except Exception as error:
        fail("model_load_failed", f"failed to load local Diffusers model: {model_path}", str(error))


def configure_diffusers_pipeline(
    pipeline,
    device,
    task,
    parameters,
    warnings,
    parameter_guard,
):
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    sequential = bool(parameters.get("_werk_enable_sequential_offload"))
    cpu_offload = bool(
        parameters.get("_werk_enable_cpu_offload")
        or parameters.get("_werk_enable_component_offload")
    )
    # Diffusers/Accelerate CPU-offload hooks target a CUDA-style accelerator.
    # Never invoke them for a CPU (or MPS) execution attempt.
    if device != "cuda":
        sequential = False
        cpu_offload = False
    try:
        configured = False
        if sequential and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
            configured = True
        elif sequential:
            parameter_guard.reject(
                "routing.allow_sequential_offload",
                "the selected pipeline has no sequential CPU offload hook",
            )
        if not configured and cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            configured = True
        elif not configured and cpu_offload:
            for path in (
                "routing.allow_cpu_offload",
                "routing.allow_component_offload",
            ):
                parameter_guard.reject(
                    path,
                    "the selected pipeline has no model CPU offload hook",
                )
        if not configured and hasattr(pipeline, "to"):
            pipeline.to(device)
    except CompanionFailure:
        raise
    except Exception as error:
        fail("backend_configuration_failed", "failed to configure pipeline device/offload", str(error))

    tiling = bool(
        parameters.get("vae_tiling")
        or parameters.get("temporal_vae_tiling")
    )
    tiling_path = (
        "video.temporal_vae_tiling"
        if task in VIDEO_TASKS
        else "image.vae_tiling"
    )
    if tiling:
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        elif getattr(pipeline, "vae", None) is not None and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        else:
            parameter_guard.reject(
                tiling_path,
                "VAE tiling is unavailable on the selected pipeline",
            )
            if tiling_path not in parameter_guard.explicit:
                warnings.append(
                    "VAE tiling requested by resolved defaults but unavailable on "
                    "the selected pipeline"
                )
    if bool(parameters.get("vae_slicing")):
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        elif getattr(pipeline, "vae", None) is not None and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        else:
            parameter_guard.reject(
                "image.vae_slicing",
                "VAE slicing is unavailable on the selected pipeline",
            )
            if "image.vae_slicing" not in parameter_guard.explicit:
                warnings.append(
                    "VAE slicing requested by resolved defaults but unavailable on "
                    "the selected pipeline"
                )
    if bool(parameters.get("attention_slicing")) and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()


def apply_loras(pipeline, parameters, warnings, parameter_guard):
    loras = parameters.get("loras") or parameters.get("lora") or []
    if isinstance(loras, dict):
        loras = [loras]
    if not loras:
        return
    if not isinstance(loras, list):
        fail("invalid_parameter", "LoRA adapters must be a list")
    if not hasattr(pipeline, "load_lora_weights"):
        parameter_guard.reject(
            "image.loras",
            "the selected pipeline has no LoRA loading hook",
        )
        if "image.loras" in parameter_guard.explicit:
            return
        fail("unsupported_parameter", "LoRA adapters are not supported by the selected pipeline")
    names = []
    weights = []
    for index, item in enumerate(loras):
        if isinstance(item, str):
            path = local_input_path(item, f"loras[{index}]")
            weight = 1.0
        elif isinstance(item, dict):
            path = local_input_path(
                item.get("model") or item.get("path"),
                f"loras[{index}].model",
            )
            weight = float(item.get("weight", 1.0))
        else:
            fail("invalid_parameter", "each LoRA entry must be a path or object")
        name = f"werk_lora_{index}"
        try:
            pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=name,
                local_files_only=True,
            )
        except TypeError:
            pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=name,
            )
        except Exception as error:
            fail("adapter_load_failed", f"failed to load LoRA: {path}", str(error))
        names.append(name)
        weights.append(weight)
    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(names, adapter_weights=weights)
    elif any(weight != 1.0 for weight in weights):
        parameter_guard.reject(
            "image.loras",
            "the pipeline loaded LoRA files but cannot apply their explicit weights",
        )
        if "image.loras" not in parameter_guard.explicit:
            warnings.append(
                "pipeline loaded LoRA adapters but cannot apply adapter weights "
                "from resolved defaults"
            )


def diffusers_call_values(
    task,
    parameters,
    inputs,
    torch,
    device,
    parameter_guard,
):
    prompt = parameters.get("prompt") or parameters.get("description") or ""
    if task not in {"frame_interpolation", "video_upscaling", "image_upscaling"}:
        prompt = required_string(prompt, "effective_parameters.prompt")
    generator, seed = seeded_generator(torch, device, parameters)
    namespace = "video" if task in VIDEO_TASKS else "image"
    if namespace == "video":
        count_key = "num_videos_per_prompt"
        count_path = (
            "video.num_videos"
            if parameters.get("num_videos") is not None
            else "video.batch_size"
        )
        count_value = (
            parameters.get("num_videos")
            if parameters.get("num_videos") is not None
            else parameters.get("batch_size")
        )
        if (
            parameters.get("num_videos") is not None
            and parameters.get("batch_size") is not None
        ):
            parameter_guard.reject_overridden(
                "video.batch_size",
                "video.num_videos",
            )
    else:
        count_key = "num_images_per_prompt"
        count_path = (
            "image.num_images"
            if parameters.get("num_images") is not None
            else "image.batch_size"
        )
        count_value = (
            parameters.get("num_images")
            if parameters.get("num_images") is not None
            else parameters.get("batch_size")
        )
        if (
            parameters.get("num_images") is not None
            and parameters.get("batch_size") is not None
        ):
            parameter_guard.reject_overridden(
                "image.batch_size",
                "image.num_images",
            )
    values = {
        "prompt": prompt or None,
        "negative_prompt": parameters.get("negative_prompt"),
        "width": parameters.get("width"),
        "height": parameters.get("height"),
        "num_inference_steps": parameters.get("steps"),
        "guidance_scale": parameters.get("guidance_scale", parameters.get("guidance")),
        "guidance_rescale": parameters.get("guidance_rescale"),
        "strength": parameters.get(
            "image_strength",
            parameters.get("video_strength", parameters.get("strength")),
        ),
        "eta": parameters.get("eta"),
        "num_frames": parameters.get("frames", parameters.get("num_frames")),
        "fps": parameters.get("fps"),
        "decode_chunk_size": parameters.get("decode_chunk_size"),
        "motion_bucket_id": parameters.get("motion_bucket"),
        "noise_aug_strength": parameters.get("noise_augmentation"),
        "generator": generator,
    }
    values[count_key] = count_value
    parameter_paths = {
        "negative_prompt": "negative_prompt",
        "width": f"{namespace}.width",
        "height": f"{namespace}.height",
        "num_inference_steps": f"{namespace}.steps",
        "guidance_scale": f"{namespace}.guidance",
        count_key: count_path,
        "num_frames": "video.frames",
        "generator": f"{namespace}.seed",
    }
    image_value = (
        inputs.get("input_image")
        or inputs.get("initial_image")
        or inputs.get("image")
        or (
            inputs.get("input")
            if task in IMAGE_TASKS | {"image_to_video"}
            else None
        )
        or (
            inputs.get("source")
            if task in IMAGE_TASKS | {"image_to_video"}
            else None
        )
    )
    if image_value is not None:
        values["image"] = load_image(image_value, "input image")
    final_image = inputs.get("final_image")
    if final_image is not None:
        values["last_image"] = load_image(final_image, "final image")
    mask_value = inputs.get("mask_image") or inputs.get("mask")
    if mask_value is not None:
        values["mask_image"] = load_image(mask_value, "mask image")
    mask_video_value = inputs.get("mask_video")
    if mask_video_value is not None:
        values["mask_video"] = load_video_frames(mask_video_value, "mask video")
    video_value = (
        inputs.get("source_video")
        or inputs.get("input_video")
        or inputs.get("video")
        or inputs.get("source")
        or inputs.get("input")
    )
    if video_value is not None:
        values["video"] = load_video_frames(video_value, "source video")
    required = ["prompt"]
    if task in {
        "image_editing",
        "image_variation",
        "image_inpainting",
        "image_outpainting",
        "image_upscaling",
        "image_to_video",
    }:
        required.append("image")
        if "image" not in values:
            fail("invalid_request", f"task '{task}' requires a local input image")
    if task == "image_inpainting":
        required.append("mask_image")
        if "mask_image" not in values:
            fail("invalid_request", f"task '{task}' requires a local mask input")
    if task == "video_inpainting":
        required.append("mask_video")
        if "mask_video" not in values:
            fail("invalid_request", f"task '{task}' requires a local mask video")
    if task in {
        "video_to_video",
        "video_inpainting",
        "video_extension",
        "video_upscaling",
        "frame_interpolation",
    }:
        required.append("video")
        if "video" not in values:
            fail("invalid_request", f"task '{task}' requires a local source video")
    return values, required, seed, parameter_paths


def image_format(parameters):
    value = normalized_name(parameters.get("output_format") or parameters.get("format") or "png")
    aliases = {"jpg": "jpeg"}
    value = aliases.get(value, value)
    if value not in {"png", "jpeg", "webp"}:
        fail("unsupported_parameter", f"unsupported image output format '{value}'")
    return value


def image_mime(format_name):
    return {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }[format_name]


def ensure_pil_image(value):
    image_module = require_module("PIL.Image", "Pillow", "image output")
    if hasattr(value, "save"):
        return value
    numpy = require_module("numpy", purpose="image output")
    array = numpy.asarray(value)
    if array.dtype.kind == "f":
        array = numpy.clip(array * 255.0, 0, 255).astype("uint8")
    return image_module.fromarray(array)


def save_images(images, output_dir, task, parameters, identifier):
    if not isinstance(images, (list, tuple)):
        images = [images]
    format_name = image_format(parameters)
    suffix = "jpg" if format_name == "jpeg" else format_name
    outputs = []
    for index, raw in enumerate(images):
        image = ensure_pil_image(raw)
        path = output_dir / f"{task}-{identifier}-{index + 1}.{suffix}"
        save_kwargs = {}
        if format_name == "jpeg":
            # ``quality`` is the leaf alias of canonical
            # ``routing.quality`` (quality/balanced/latency policy), not a
            # numeric JPEG encoder setting.
            save_kwargs["quality"] = 95
        image.save(path, format=format_name.upper(), **save_kwargs)
        outputs.append(
            output_record(
                path,
                image_mime(format_name),
                width=int(image.width),
                height=int(image.height),
            )
        )
    return outputs


def frame_batches(value):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return [[value]]
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return [list(batch) for batch in value]
    return [list(value)]


def export_video(frames, path, fps, format_name):
    frames = [ensure_pil_image(frame) for frame in frames]
    if not frames:
        fail("backend_error", "video pipeline returned no frames")
    if format_name == "gif":
        try:
            duration_ms = max(1, round(1000.0 / float(fps)))
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            return
        except Exception as error:
            fail("encoding_failed", f"failed to encode animated GIF: {path}", str(error))
    try:
        utils = require_module("diffusers.utils", "diffusers", "video export")
        exporter = getattr(utils, "export_to_video", None)
        if exporter is not None:
            exporter(frames, str(path), fps=float(fps))
            return
    except CompanionFailure:
        raise
    except Exception:
        pass
    imageio = require_module("imageio.v3", "imageio", "video export")
    numpy = require_module("numpy", purpose="video export")
    try:
        imageio.imwrite(
            path,
            numpy.stack([numpy.asarray(frame) for frame in frames]),
            fps=float(fps),
        )
    except Exception as error:
        fail("encoding_failed", f"failed to encode video: {path}", str(error))


def execute_diffusers(
    payload,
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
):
    torch, device, dtype = torch_runtime(parameters)
    has_image = any(
        key in inputs for key in ("image", "input_image", "initial_image")
    )
    pipeline = load_diffusers_pipeline(model_path, task, has_image, torch, dtype)
    warnings = []
    configure_diffusers_pipeline(
        pipeline,
        device,
        task,
        parameters,
        warnings,
        parameter_guard,
    )
    apply_loras(pipeline, parameters, warnings, parameter_guard)
    values, required, seed, parameter_paths = diffusers_call_values(
        task,
        parameters,
        inputs,
        torch,
        device,
        parameter_guard,
    )
    if (
        "num_videos_per_prompt" in values
        and not supports_keyword(pipeline.__call__, "num_videos_per_prompt")
        and supports_keyword(pipeline.__call__, "num_images_per_prompt")
    ):
        values["num_images_per_prompt"] = values.pop("num_videos_per_prompt")
        parameter_paths["num_images_per_prompt"] = parameter_paths.pop(
            "num_videos_per_prompt"
        )
    if (
        "mask_video" in values
        and not supports_keyword(pipeline.__call__, "mask_video")
    ):
        alternate_mask_keyword = next(
            (
                keyword
                for keyword in ("mask", "mask_image")
                if supports_keyword(pipeline.__call__, keyword)
            ),
            None,
        )
        if alternate_mask_keyword is not None:
            values[alternate_mask_keyword] = values.pop("mask_video")
            required = [
                alternate_mask_keyword if key == "mask_video" else key
                for key in required
            ]
    kwargs = filtered_kwargs(
        pipeline.__call__,
        values,
        required=required,
        parameter_guard=parameter_guard,
        parameter_paths=parameter_paths,
    )
    try:
        with torch.inference_mode():
            result = pipeline(**kwargs)
    except Exception as error:
        fail("execution_failed", f"Diffusers pipeline failed for task '{task}'", str(error))

    if task in IMAGE_TASKS:
        images = getattr(result, "images", None)
        if images is None and isinstance(result, dict):
            images = result.get("images")
        if not images:
            fail("backend_error", "Diffusers pipeline returned no images")
        outputs = save_images(images, output_dir, task, parameters, identifier)
    else:
        frames = getattr(result, "frames", None)
        if frames is None and isinstance(result, dict):
            frames = result.get("frames")
        batches = frame_batches(frames)
        if not batches:
            fail("backend_error", "Diffusers pipeline returned no video frames")
        fps = positive_float(parameters, "fps", 24.0, 0.1)
        format_name = normalized_name(
            parameters.get("output_format") or parameters.get("format") or "mp4"
        )
        if format_name not in {"mp4", "gif"}:
            fail(
                "unsupported_parameter",
                f"unsupported direct video output format '{format_name}'",
            )
        outputs = []
        for index, batch in enumerate(batches):
            path = output_dir / f"{task}-{identifier}-{index + 1}.{format_name}"
            export_video(batch, path, fps, format_name)
            first = ensure_pil_image(batch[0])
            outputs.append(
                output_record(
                    path,
                    "video/mp4" if format_name == "mp4" else "image/gif",
                    width=int(first.width),
                    height=int(first.height),
                    duration=len(batch) / fps,
                    metadata={"frames": len(batch), "fps": fps},
                )
            )
    return outputs, warnings, {
        "runtime": "diffusers",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "translated_parameters": sorted(kwargs),
    }


def execute_diffusers_audio(
    model_path,
    task,
    parameters,
    output_dir,
    identifier,
    parameter_guard,
):
    torch, device, dtype = torch_runtime(parameters)
    pipeline = load_diffusers_pipeline(
        model_path,
        task,
        False,
        torch,
        dtype,
    )
    warnings = []
    configure_diffusers_pipeline(
        pipeline,
        device,
        task,
        parameters,
        warnings,
        parameter_guard,
    )
    prompt = prompt_with_lyrics(
        parameters.get("prompt") or parameters.get("description"),
        parameters.get("lyrics"),
    )
    prompt = required_string(prompt, "effective_parameters.prompt/description")
    generator, seed = seeded_generator(torch, device, parameters)
    values = {
        "prompt": prompt,
        "negative_prompt": parameters.get("negative_prompt"),
        "num_inference_steps": parameters.get("steps"),
        "guidance_scale": parameters.get("guidance_scale", parameters.get("guidance")),
        "audio_length_in_s": parameters.get("duration"),
        "num_waveforms_per_prompt": parameters.get(
            "num_variations",
            parameters.get("batch_size"),
        ),
        "generator": generator,
    }
    kwargs = filtered_kwargs(
        pipeline.__call__,
        values,
        required=("prompt",),
        parameter_guard=parameter_guard,
        parameter_paths={
            "negative_prompt": "negative_prompt",
            "num_inference_steps": "audio.steps",
            "guidance_scale": "audio.guidance",
            "audio_length_in_s": "audio.duration",
            "num_waveforms_per_prompt": "audio.variations",
            "generator": "audio.seed",
        },
    )
    try:
        with torch.inference_mode():
            result = pipeline(**kwargs)
    except Exception as error:
        fail("execution_failed", f"Diffusers audio pipeline failed for '{task}'", str(error))
    audios = getattr(result, "audios", None)
    if audios is None and isinstance(result, dict):
        audios = result.get("audios")
        if audios is None:
            audios = result.get("audio")
    if audios is None:
        fail("backend_error", "Diffusers pipeline returned no audio")

    numpy = require_module("numpy", purpose="Diffusers audio output")
    array = numpy.asarray(audios)
    if array.ndim <= 1:
        waveforms = [array]
    else:
        waveforms = [array[index] for index in range(array.shape[0])]
    sample_rate = 0
    if sample_rate <= 0:
        for component_name in ("vocoder", "vae"):
            component = getattr(pipeline, component_name, None)
            config = getattr(component, "config", None)
            value = getattr(config, "sampling_rate", None)
            if value:
                sample_rate = int(value)
                break
    if sample_rate <= 0:
        sample_rate = 16_000
        warnings.append(
            "pipeline did not expose a sampling rate; output metadata assumes 16000 Hz"
        )
    format_name = normalized_name(
        parameters.get("output_format") or parameters.get("format") or "wav"
    )
    if format_name not in {"wav", "flac", "ogg"}:
        fail(
            "unsupported_parameter",
            f"unsupported direct audio output format '{format_name}'",
        )
    outputs = []
    for index, waveform in enumerate(waveforms):
        path = output_dir / f"{task}-{identifier}-{index + 1}.{format_name}"
        channels, duration = write_audio(path, waveform, sample_rate, format_name)
        outputs.append(
            output_record(
                path,
                mimetypes.guess_type(path.name)[0] or f"audio/{format_name}",
                duration=duration,
                metadata={"sample_rate": sample_rate, "channels": channels},
            )
        )
    return outputs, warnings, {
        "runtime": "diffusers",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "translated_parameters": sorted(kwargs),
    }


def transformers_device(device):
    if device == "cuda":
        return 0
    if device == "mps":
        return "mps"
    return -1


def load_transformers_pipeline(model_path, pipeline_task, parameters):
    torch, device, dtype = torch_runtime(parameters)
    transformers = require_module("transformers", purpose=pipeline_task)
    factory = getattr(transformers, "pipeline", None)
    if factory is None:
        fail("missing_dependency", "installed transformers package has no pipeline API")
    kwargs = {
        "task": pipeline_task,
        "model": str(model_path),
        "device": transformers_device(device),
        "trust_remote_code": False,
        "model_kwargs": {"local_files_only": True},
    }
    if device != "cpu":
        try:
            factory_parameters = inspect.signature(factory).parameters
        except Exception:
            factory_parameters = {}
        if "dtype" in factory_parameters:
            kwargs["dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    try:
        pipeline = factory(**kwargs)
    except Exception as error:
        fail(
            "model_load_failed",
            f"failed to load local Transformers pipeline '{pipeline_task}' from {model_path}",
            str(error),
        )
    return pipeline, torch, device, dtype


def audio_array_and_rate(result, _parameters):
    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict):
        fail("backend_error", "audio pipeline returned an unsupported result shape")
    audio = result.get("audio")
    rate = (
        result.get("sampling_rate")
        or result.get("sample_rate")
        or 44_100
    )
    if audio is None:
        fail("backend_error", "audio pipeline result has no 'audio' value")
    try:
        rate = int(rate)
    except Exception:
        fail("backend_error", "audio pipeline returned an invalid sampling rate")
    return audio, rate


def write_audio(path, audio, sample_rate, format_name):
    numpy = require_module("numpy", purpose="audio output")
    array = numpy.asarray(audio)
    array = numpy.squeeze(array)
    if array.ndim == 1:
        channels = 1
        interleaved = array
    elif array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
            array = array.T
        channels = int(array.shape[1])
        interleaved = array.reshape(-1)
    else:
        fail("backend_error", f"unsupported audio tensor shape: {array.shape}")
    if format_name == "wav":
        if interleaved.dtype.kind == "f":
            pcm = (
                numpy.clip(interleaved, -1.0, 1.0) * 32767.0
            ).astype("<i2")
        elif interleaved.dtype.itemsize <= 2:
            pcm = interleaved.astype("<i2")
        else:
            pcm = numpy.clip(interleaved, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
    else:
        soundfile = require_module("soundfile", purpose=f"{format_name} audio output")
        try:
            soundfile.write(str(path), array, sample_rate, format=format_name.upper())
        except Exception as error:
            fail("encoding_failed", f"failed to encode {format_name} audio", str(error))
    frames = int(array.shape[0] if array.ndim > 1 else array.size)
    return channels, frames / float(sample_rate)


def execute_audio_generation(
    model_path,
    task,
    parameters,
    output_dir,
    identifier,
    parameter_guard,
):
    pipeline, torch, device, dtype = load_transformers_pipeline(
        model_path,
        "text-to-audio",
        parameters,
    )
    prompt = prompt_with_lyrics(
        parameters.get("prompt") or parameters.get("description"),
        parameters.get("lyrics"),
    )
    prompt = required_string(prompt, "effective_parameters.prompt/description")
    generator, seed = seeded_generator(torch, device, parameters)
    call_values = {
        "forward_params": {
            key: value
            for key, value in {
                "guidance_scale": parameters.get("guidance_scale", parameters.get("guidance")),
                "do_sample": parameters.get("temperature", 1.0) != 0,
                "temperature": parameters.get("temperature"),
                "top_k": parameters.get("top_k"),
                "top_p": parameters.get("top_p"),
                "max_new_tokens": parameters.get("max_new_tokens"),
                "generator": generator,
            }.items()
            if value is not None
        }
    }
    if not call_values["forward_params"]:
        call_values = {}
    try:
        result = pipeline(prompt, **call_values)
    except TypeError:
        if call_values:
            for path in ("audio.seed", "audio.guidance"):
                parameter_guard.reject(
                    path,
                    "the selected Transformers audio pipeline rejected generation kwargs",
                )
        try:
            result = pipeline(prompt)
        except Exception as error:
            fail("execution_failed", f"Transformers audio pipeline failed for '{task}'", str(error))
    except Exception as error:
        fail("execution_failed", f"Transformers audio pipeline failed for '{task}'", str(error))
    audio, sample_rate = audio_array_and_rate(result, parameters)
    format_name = normalized_name(parameters.get("output_format") or parameters.get("format") or "wav")
    if format_name not in {"wav", "flac", "ogg"}:
        fail("unsupported_parameter", f"unsupported direct audio output format '{format_name}'")
    path = output_dir / f"{task}-{identifier}.{format_name}"
    channels, duration = write_audio(path, audio, sample_rate, format_name)
    return [output_record(
        path,
        mimetypes.guess_type(path.name)[0] or f"audio/{format_name}",
        duration=duration,
        metadata={"sample_rate": sample_rate, "channels": channels},
    )], [], {
        "runtime": "transformers",
        "pipeline_task": "text-to-audio",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
    }


def execute_tts(model_path, task, parameters, output_dir, identifier):
    pipeline, _torch, device, dtype = load_transformers_pipeline(
        model_path,
        "text-to-speech",
        parameters,
    )
    text = required_string(parameters.get("text") or parameters.get("prompt"), "effective_parameters.text")
    kwargs = {}
    for key in ("speaker_embeddings", "vocoder", "generate_kwargs"):
        if key in parameters:
            kwargs[key] = parameters[key]
    try:
        result = pipeline(text, **kwargs)
    except Exception as error:
        fail("execution_failed", "Transformers text-to-speech pipeline failed", str(error))
    audio, sample_rate = audio_array_and_rate(result, parameters)
    format_name = normalized_name(parameters.get("output_format") or parameters.get("format") or "wav")
    if format_name not in {"wav", "flac", "ogg"}:
        fail("unsupported_parameter", f"unsupported direct audio output format '{format_name}'")
    path = output_dir / f"{task}-{identifier}.{format_name}"
    channels, duration = write_audio(path, audio, sample_rate, format_name)
    return [output_record(
        path,
        mimetypes.guess_type(path.name)[0] or f"audio/{format_name}",
        duration=duration,
        metadata={"sample_rate": sample_rate, "channels": channels},
    )], [], {
        "runtime": "transformers",
        "pipeline_task": "text-to-speech",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
    }


def transcript_segments(result):
    segments = []
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if not isinstance(chunks, list):
        return segments
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
            continue
        try:
            start = max(0.0, float(timestamp[0] or 0.0))
            end = max(start, float(timestamp[1] or start))
        except (TypeError, ValueError):
            continue
        segments.append((start, end, str(chunk.get("text") or "").strip()))
    return segments


def transcript_timestamp(seconds, decimal_marker="."):
    milliseconds = max(0, round(float(seconds) * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{millis:03d}"
    )


def save_transcription_outputs(result, output_dir, task, identifier, parameters):
    format_name = normalized_name(parameters.get("output_format") or "json")
    format_name = {"txt": "text"}.get(format_name, format_name)
    if format_name not in {"json", "text", "srt", "vtt", "tsv"}:
        fail(
            "unsupported_parameter",
            f"unsupported transcription output format '{format_name}'",
        )
    text = str(result.get("text") or "")
    if format_name == "json":
        path = output_dir / f"{task}-{identifier}.json"
        atomic_json_write(path, result)
        return [output_record(path, "application/json")]
    if format_name == "text":
        path = output_dir / f"{task}-{identifier}.txt"
        path.write_text(text, encoding="utf-8")
        return [output_record(path, "text/plain")]

    segments = transcript_segments(result)
    if not segments:
        fail(
            "backend_error",
            f"transcription format '{format_name}' requires timestamped pipeline chunks",
        )
    if format_name == "srt":
        body = "\n\n".join(
            f"{index}\n"
            f"{transcript_timestamp(start, ',')} --> "
            f"{transcript_timestamp(end, ',')}\n{text}"
            for index, (start, end, text) in enumerate(segments, start=1)
        )
        mime_type = "application/x-subrip"
    elif format_name == "vtt":
        entries = "\n\n".join(
            f"{transcript_timestamp(start)} --> {transcript_timestamp(end)}\n{text}"
            for start, end, text in segments
        )
        body = f"WEBVTT\n\n{entries}"
        mime_type = "text/vtt"
    else:
        body = "start\tend\ttext\n" + "\n".join(
            f"{round(start * 1000)}\t{round(end * 1000)}\t"
            f"{text.replace(chr(9), ' ').replace(chr(10), ' ')}"
            for start, end, text in segments
        )
        mime_type = "text/tab-separated-values"
    path = output_dir / f"{task}-{identifier}.{format_name}"
    path.write_text(f"{body.rstrip()}\n", encoding="utf-8")
    return [output_record(path, mime_type)]


def execute_asr(
    model_path,
    task,
    parameters,
    inputs,
    output_dir,
    identifier,
    parameter_guard,
):
    pipeline, _torch, device, dtype = load_transformers_pipeline(
        model_path,
        "automatic-speech-recognition",
        parameters,
    )
    source = (
        inputs.get("input_audio")
        or inputs.get("source_audio")
        or inputs.get("audio")
        or inputs.get("source")
        or inputs.get("input")
        or parameters.get("input_audio")
    )
    source_path = local_input_path(source, "input audio")
    kwargs = {}
    return_timestamps = parameters.get("word_timestamps")
    timestamp_paths = []
    if return_timestamps:
        kwargs["return_timestamps"] = "word"
        timestamp_paths.append("stt.word_timestamps")
        if parameters.get("segment_timestamps"):
            parameter_guard.reject_overridden(
                "stt.segment_timestamps",
                "stt.word_timestamps",
            )
    elif parameters.get("segment_timestamps"):
        kwargs["return_timestamps"] = True
        timestamp_paths.append("stt.segment_timestamps")
    generate_kwargs = {
        key: value
        for key, value in {
            "language": parameters.get("language"),
            "task": (
                parameters.get("operation")
                or parameters.get("mode")
                or parameters.get("transcription_task")
            ),
            "temperature": parameters.get("temperature"),
            "num_beams": parameters.get("beam_size"),
        }.items()
        if value is not None
    }
    generate_paths = []
    if parameters.get("language") is not None:
        generate_paths.append("stt.language")
    if parameters.get("operation") is not None:
        generate_paths.append("stt.operation")
    if parameters.get("temperature") is not None:
        generate_paths.append("stt.temperature")
    if parameters.get("beam_size") is not None:
        generate_paths.append("stt.beam_size")
    initial_prompt = parameters.get("initial_prompt") or parameters.get("prompt")
    if initial_prompt:
        prompt_ids = getattr(
            getattr(pipeline, "tokenizer", None),
            "get_prompt_ids",
            None,
        )
        if prompt_ids is None:
            parameter_guard.reject(
                "stt.initial_prompt",
                "the selected ASR pipeline tokenizer cannot encode prompt IDs",
            )
        else:
            try:
                generate_kwargs["prompt_ids"] = prompt_ids(
                    required_string(initial_prompt, "initial_prompt"),
                    return_tensors="pt",
                )
                generate_paths.append("stt.initial_prompt")
            except Exception as error:
                parameter_guard.reject(
                    "stt.initial_prompt",
                    f"the ASR tokenizer rejected the initial prompt: {error}",
                )
    if generate_kwargs:
        kwargs["generate_kwargs"] = generate_kwargs
    kwargs = filtered_kwargs(
        pipeline.__call__,
        kwargs,
        parameter_guard=parameter_guard,
        parameter_paths={
            "return_timestamps": timestamp_paths,
            "generate_kwargs": generate_paths,
        },
    )
    try:
        result = pipeline(str(source_path), **kwargs)
    except Exception as error:
        fail("execution_failed", "Transformers speech-to-text pipeline failed", str(error))
    if not isinstance(result, dict):
        result = {"text": str(result)}
    text = str(result.get("text") or "")
    outputs = save_transcription_outputs(
        result,
        output_dir,
        task,
        identifier,
        parameters,
    )
    return outputs, [], {
        "runtime": "transformers",
        "pipeline_task": "automatic-speech-recognition",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "text": text,
    }


def output_record(
    path,
    mime_type,
    width=None,
    height=None,
    duration=None,
    metadata=None,
):
    return {
        "path": str(path),
        "mime_type": mime_type,
        "size": safe_size(path),
        "width": width,
        "height": height,
        "duration": duration,
        "metadata": metadata or {},
    }


def atomic_json_write(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except Exception:
            pass


def command_execute(payload):
    model_path = local_model_path(payload)
    task = normalized_name(payload.get("task"))
    if not task:
        fail("invalid_request", "task is required")
    parameters = normalized_parameters(payload)
    adapter = execution_adapter(model_path, task)
    if adapter is None:
        fail("unsupported_task", f"no executable companion adapter for task '{task}'")
    parameter_guard = ExplicitParameterGuard(
        payload,
        task,
        adapter,
        parameters,
    )
    parameter_guard.validate_supported(
        supported_explicit_parameters(task, adapter)
    )
    parameters = parameter_guard.without_unsupported(parameters)
    inputs = input_values(payload, parameters)
    validate_adapter_inputs(task, adapter, inputs)
    output_dir = output_directory(payload)
    identifier = uuid.uuid4().hex
    started = time.time()

    if adapter == "diffusers":
        outputs, warnings, backend_metadata = execute_diffusers(
            payload,
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
        )
    elif adapter == "diffusers_audio":
        outputs, warnings, backend_metadata = execute_diffusers_audio(
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
            parameter_guard,
        )
    elif adapter == "transformers_audio":
        outputs, warnings, backend_metadata = execute_audio_generation(
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
            parameter_guard,
        )
    elif adapter == "transformers_tts":
        outputs, warnings, backend_metadata = execute_tts(
            model_path,
            task,
            parameters,
            output_dir,
            identifier,
        )
    elif adapter == "transformers_asr":
        outputs, warnings, backend_metadata = execute_asr(
            model_path,
            task,
            parameters,
            inputs,
            output_dir,
            identifier,
            parameter_guard,
        )
    else:
        fail("internal_error", f"unknown execution adapter '{adapter}'")

    warnings.extend(parameter_guard.warnings)
    parameters = parameter_guard.without_unsupported(parameters)
    backend_metadata["parameter_support"] = parameter_guard.metadata()

    metadata = {
        "id": identifier,
        "task": task,
        "model_path": str(model_path),
        "runtime": "werk-media-companion",
        "backend": backend_metadata,
        "effective_parameters": parameters,
        "outputs": outputs,
        "warnings": warnings,
        "created_unix": int(started),
        "elapsed_seconds": max(0.0, time.time() - started),
        "offline": True,
    }
    metadata_path = output_dir / f"{task}-{identifier}.metadata.json"
    atomic_json_write(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)
    return {
        "task": task,
        "outputs": outputs,
        "metadata": metadata,
        "warnings": warnings,
    }


def dispatch(operation, payload):
    commands = {
        "health": command_health,
        "capabilities": command_capabilities,
        "probe-model": command_probe_model,
        "estimate": command_estimate,
        "execute": command_execute,
    }
    handler = commands.get(operation)
    if handler is None:
        fail("unknown_command", f"unknown companion command '{operation}'")
    return handler(payload)


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def response_error(error):
    if isinstance(error, CompanionFailure):
        code = error.code
        message = error.message
        detail = error.detail
    else:
        code = "internal_error"
        message = str(error) or error.__class__.__name__
        detail = None
        if os.environ.get("WERK_MEDIA_DEBUG") in {"1", "true", "yes", "on"}:
            detail = traceback.format_exc()
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "detail": json_safe(detail),
        },
    }


def main():
    original_stdout = sys.stdout
    try:
        if len(sys.argv) != 2:
            fail("invalid_command", "expected exactly one command argument")
        operation = normalized_name(sys.argv[1]).replace("_", "-")
        try:
            payload = json.load(sys.stdin)
        except Exception as error:
            fail("invalid_json", "stdin must contain one JSON object", str(error))
        if not isinstance(payload, dict):
            fail("invalid_request", "stdin JSON value must be an object")
        # Third-party imports and pipelines occasionally print progress or
        # warnings to stdout. Redirect all such output to stderr so stdout
        # remains a single protocol object.
        with contextlib.redirect_stdout(sys.stderr):
            body = dispatch(operation, payload)
        response = {"ok": True}
        if isinstance(body, dict):
            response.update(body)
        else:
            response["result"] = body
    except BaseException as error:
        response = response_error(error)

    encoded = json.dumps(json_safe(response), ensure_ascii=False, separators=(",", ":"))
    original_stdout.write(encoded)
    original_stdout.write("\n")
    original_stdout.flush()


if __name__ == "__main__":
    main()
