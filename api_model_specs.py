# api_model_specs.py
from typing import Final

from api_types import (
    GenerateVideoModelsSpecsResponse,
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

# همه اعداد ۱ تا ۳۰
ALL_DURATIONS_1_TO_30: Final = tuple(range(1, 31))


def _resolution_spec(*, durations_by_fps: dict[int, list[int]]) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            int(fps): [int(duration) for duration in durations]
            for fps, durations in durations_by_fps.items()
        }
    )


# مشخصات مدل‌های API (Fast, Fast HQ, PRO)
ltx_api_model_specs: Final[tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...]] = (

    # ---------- حالت Fast ----------
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (Fast)",
            supported_resolutions_durations={
                "4k": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                }),
                "2k": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                }),
                "1080p": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                }),
                "720p": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                    120: ALL_DURATIONS_1_TO_30,
                }),
                "480p": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                    120: ALL_DURATIONS_1_TO_30,
                    240: ALL_DURATIONS_1_TO_30,
                }),
            },
            a2v_supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
                "720p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
                "1080p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
            },
        ),
    ),

    # ---------- حالت Fast HQ ----------
    (
        "fast_hq",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (Fast HQ)",
            supported_resolutions_durations={
                "4k": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                }),
                "2k": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                }),
                "1080p": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                }),
                "720p": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                    120: ALL_DURATIONS_1_TO_30,
                }),
                "480p": _resolution_spec(durations_by_fps={
                    15: ALL_DURATIONS_1_TO_30,
                    24: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                    120: ALL_DURATIONS_1_TO_30,
                    240: ALL_DURATIONS_1_TO_30,
                }),
            },
            a2v_supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
                "720p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
                "1080p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
            },
        ),
    ),

    # ---------- حالت PRO ----------
    (
        "pro",
        LTXVideoGenerationSpec(
            display_name="LTX 2.3 (PRO)",
            supported_resolutions_durations={
                "2k": _resolution_spec(durations_by_fps={
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                }),
                "1080p": _resolution_spec(durations_by_fps={
                    24: ALL_DURATIONS_1_TO_30,
                    25: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                }),
                "720p": _resolution_spec(durations_by_fps={
                    24: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                }),
                "480p": _resolution_spec(durations_by_fps={
                    24: ALL_DURATIONS_1_TO_30,
                    30: ALL_DURATIONS_1_TO_30,
                    50: ALL_DURATIONS_1_TO_30,
                    60: ALL_DURATIONS_1_TO_30,
                    120: ALL_DURATIONS_1_TO_30,
                }),
            },
            a2v_supported_resolutions_durations={
                "480p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
                "720p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
                "1080p": _resolution_spec(durations_by_fps={24: ALL_DURATIONS_1_TO_30}),
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
        # ✅ لیست آپ‌اسکیل‌ها
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
