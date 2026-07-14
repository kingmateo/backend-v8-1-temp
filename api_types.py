# api_types.py
from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

ModelCheckpointID = Literal[
    "ltx-2.3-22b-dev",
    "ltx-2.3-22b-distilled",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-spatial-upscaler-x1.5-1.0",
    "ltx-2.3-spatial-upscaler-x2-1.1",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "z-image-turbo",
]

VideoCameraMotion = Literal[
    "none",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "zoom_in",
    "zoom_out",
    "roll_clockwise",
    "roll_counter_clockwise",
    "static",
]


class HTTPErrorResponse(BaseModel):
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")


class GenerateTextRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt: NonEmptyPrompt


class GenerateTextResponse(BaseModel):
    text: str


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt: NonEmptyPrompt
    negativePrompt: str = ""
    imagePath: str | None = None
    aspectRatio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "1:1"


class GenerateImageResponse(BaseModel):
    imagePath: str


LTXVideoGenResolution = Literal["480p", "720p", "1080p", "1440p", "2160p", "2k", "4k"]
LTXVideoGenDuration = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30
]
LTXVideoGenFps = Literal[15, 24, 25, 30, 50, 60, 120, 240]
LTXVideoGenPipeline = Literal["fast", "fast_hq", "pro"]


class LTXVideoGenerationResolutionSpec(BaseModel):
    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]


class LTXVideoGenerationSpec(BaseModel):
    display_name: str
    supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]
    a2v_supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]
    # ✅ لیست آپ‌اسکیل‌های قابل انتخاب
    upscalers: list[ModelCheckpointID] = [
        "ltx-2.3-spatial-upscaler-x1.5-1.0",
        "ltx-2.3-spatial-upscaler-x2-1.1",
    ]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt: NonEmptyPrompt
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    upscaler: ModelCheckpointID = "ltx-2.3-spatial-upscaler-x1.5-1.0"
    cameraMotion: VideoCameraMotion = "none"
    duration: LTXVideoGenDuration = 5
    fps: LTXVideoGenFps = 24
    cfgScale: float = Field(default=7.0, ge=1.0, le=20.0)
    steps: int = Field(default=8, ge=1, le=100)
    seed: int | None = None
    imageConditioning: ImageConditioningInput | None = None
