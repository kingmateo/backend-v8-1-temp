from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


LTXVideoGenPipeline: TypeAlias = Literal["fast", "fast_hq", "pro"]
LTXVideoGenResolution: TypeAlias = Literal["480p", "720p", "1080p", "1440p", "2160p"]
LTXVideoGenFps: TypeAlias = Literal[15, 24, 30, 60]
LTXAspectRatio: TypeAlias = Literal["16:9", "9:16"]

# محدودیت مدت تولید بین 1 تا 30 ثانیه
LTXVideoGenDuration = Annotated[int, Field(ge=1, le=30)]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    prompt: str = Field(..., min_length=1)
    negativePrompt: str = Field(default="")
    model: LTXVideoGenPipeline = Field(default="fast")
    resolution: LTXVideoGenResolution = Field(default="1080p")
    duration: LTXVideoGenDuration = Field(default=5)
    fps: LTXVideoGenFps = Field(default=24)
    upscaler: str | None = Field(default=None)
    aspectRatio: LTXAspectRatio = Field(default="16:9")
    audioPath: str | None = Field(default=None)
    imagePath: str | None = Field(default=None)


class GenerateVideoResponse(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    videoPath: str | None = None
    message: str | None = None


class LTXVideoGenerationResolutionDurations(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    fps_to_durations: dict[LTXVideoGenFps, list[int]]


class LTXVideoGenerationSpec(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    display_name: str
    supported_resolutions_durations: dict[
        LTXVideoGenResolution,
        LTXVideoGenerationResolutionDurations,
    ]


class LTXVideoGenerationModelSpecItem(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    models: list[LTXVideoGenerationModelSpecItem]
