# api_types.py

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


ModelCheckpointID: TypeAlias = Literal[
    "ltx-2.3-22b-dev",
    "ltx-2.3-22b-distilled",
    "ltx-2.3-spatial-upscaler-x1.5-1.0",
    "ltx-2.3-spatial-upscaler-x2-1.1",
]

LTXLocalModelId: TypeAlias = Literal[
    "ltx-2.3-22b-dev",
    "ltx-2.3-22b-distilled",
]

LTXVideoGenResolution: TypeAlias = Literal[
    "480p",
    "720p",
    "1080p",
    "1440p",
    "2160p",
]

LTXVideoGenDuration: TypeAlias = Annotated[int, Field(ge=1, le=30)]

LTXVideoGenFps: TypeAlias = Literal[15, 24, 25, 30, 60, 120, 240]

LTXVideoGenPipeline: TypeAlias = Literal["fast", "fast_hq", "pro"]

LTXAspectRatio: TypeAlias = Literal["16:9", "9:16", "1:1", "4:3", "3:4"]


class LTXVideoGenerationResolutionSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]


class LTXVideoGenerationSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    display_name: str
    supported_resolutions_durations: dict[
        LTXVideoGenResolution,
        LTXVideoGenerationResolutionSpec,
    ]
    a2v_supported_resolutions_durations: dict[
        LTXVideoGenResolution,
        LTXVideoGenerationResolutionSpec,
    ] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]
    upscalers: list[ModelCheckpointID]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str = Field(..., min_length=1)
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    upscaler_id: ModelCheckpointID = "ltx-2.3-spatial-upscaler-x1.5-1.0"
    cameraMotion: str | None = None
    negativePrompt: str = ""
    duration: LTXVideoGenDuration = 5
    fps: LTXVideoGenFps = 24
    audio: str | None = None
    imagePath: str | None = None
    audioPath: str | None = None
    aspectRatio: LTXAspectRatio = "16:9"


class GenerateVideoResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    videoPath: str | None = None
    message: str | None = None
