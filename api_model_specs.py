# api_model_specs.py

from typing import Final

from api_types import (
    GenerateVideoModelsSpecsResponse,
    LTXAspectRatio,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
)
from runtime_config.model_download_specs import (
    ALL_LTX_LOCAL_MODEL_IDS,
    get_ltx_model_spec,
)

ALL_DURATIONS_1_TO_30: Final = tuple(range(1, 31))


def _resolution_spec(*, durations_by_fps: dict[int, list[int]]) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            int(fps): [int(duration) for duration in durations]
            for fps, durations in durations_by_fps.items()
        }
    )


ltx_api_model_specs: Final[tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...]] = (
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (Fast)",
            supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "720p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "1080p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "1440p": _resolution_spec(durations_by_fps={24: [5, 10], 30: [5, 10]}),
                "2160p": _resolution_spec(durations_by_fps={24: [5], 30: [5]}),
            },
            a2v_supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: [5, 10, 20]}),
                "720p": _resolution_spec(durations_by_fps={24: [5, 10, 20]}),
                "1080p": _resolution_spec(durations_by_fps={24: [5, 10]}),
            },
        ),
    ),
    (
        "fast_hq",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (Fast HQ)",
            supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "720p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "1080p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "1440p": _resolution_spec(durations_by_fps={24: [5, 10], 30: [5, 10]}),
                "2160p": _resolution_spec(durations_by_fps={24: [5], 30: [5]}),
            },
            a2v_supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: [5, 10, 20]}),
                "720p": _resolution_spec(durations_by_fps={24: [5, 10, 20]}),
                "1080p": _resolution_spec(durations_by_fps={24: [5, 10]}),
            },
        ),
    ),
    (
        "pro",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (PRO)",
            supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10], 60: [5], 120: [5]}),
                "720p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "1080p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 25: [5, 10], 30: [5, 10]}),
                "1440p": _resolution_spec(durations_by_fps={24: [5, 10], 25: [5, 10], 30: [5]}),
                "2160p": _resolution_spec(durations_by_fps={24: [5]}),
            },
            a2v_supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: [5, 10, 20], 30: [5, 10]}),
                "720p": _resolution_spec(durations_by_fps={24: [5, 10]}),
                "1080p": _resolution_spec(durations_by_fps={24: [5], 30: [5]}),
            },
        ),
    ),
)


def _pairs_to_items(
    pairs: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...],
) -> list[LTXVideoGenerationModelSpecItem]:
    return [
        LTXVideoGenerationModelSpecItem(pipeline=pipeline, spec=spec)
        for pipeline, spec in pairs
    ]


def get_local_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    return [
        LTXVideoGenerationModelSpecItem(
            pipeline=model_id,  # type: ignore[arg-type]
            spec=get_ltx_model_spec(model_id),
        )
        for model_id in ALL_LTX_LOCAL_MODEL_IDS
    ]


def get_api_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    return _pairs_to_items(ltx_api_model_specs)


def build_generate_video_model_specs_response() -> GenerateVideoModelsSpecsResponse:
    return GenerateVideoModelsSpecsResponse(
        local_models=get_local_video_generation_model_specs(),
        api_models=get_api_video_generation_model_specs(),
        upscalers=[
            "ltx-2.3-spatial-upscaler-x1.5-1.0",
            "ltx-2.3-spatial-upscaler-x2-1.1",
        ],
    )


def _get_resolution_spec(
    spec: LTXVideoGenerationSpec,
    resolution: LTXVideoGenResolution,
    *,
    is_a2v: bool = False,
) -> LTXVideoGenerationResolutionSpec | None:
    if is_a2v and spec.a2v_supported_resolutions_durations is not None:
        return spec.a2v_supported_resolutions_durations.get(resolution)
    return spec.supported_resolutions_durations.get(resolution)


def get_supported_durations(
    resolution_spec: LTXVideoGenerationResolutionSpec | None,
    fps: LTXVideoGenFps,
) -> list[LTXVideoGenDuration]:
    if resolution_spec is None:
        return []
    return resolution_spec.fps_to_durations.get(fps, [])


def validate_generate_video_request(req: "GenerateVideoRequest") -> None:
    model_map = {item.pipeline: item.spec for item in get_api_video_generation_model_specs()}
    spec = model_map.get(req.model)
    if spec is None:
        raise ValueError(f"Unsupported model: {req.model}")

    resolution_spec = _get_resolution_spec(spec, req.resolution, is_a2v=bool(req.audioPath))
    if resolution_spec is None:
        raise ValueError(f"Unsupported resolution for model {req.model}: {req.resolution}")

    supported_durations = get_supported_durations(resolution_spec, req.fps)
    if req.duration not in supported_durations:
        raise ValueError(
            f"Unsupported duration {req.duration} for model {req.model}, "
            f"resolution {req.resolution}, fps {req.fps}"
        )
