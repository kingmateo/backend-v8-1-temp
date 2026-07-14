# api_types.py

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HTTPErrorResponse(BaseModel):
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")


class ImageConditioningInput(BaseModel):
    path: str = Field(..., description="Path to the conditioning image")
    frame_idx: int = Field(..., description="Frame index to apply conditioning")
    strength: float = Field(..., description="Conditioning strength")


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate video from")
    negative_prompt: str = Field("", description="Text prompt to negatively influence generation")
    width: int = Field(512, description="Width of the generated video")
    height: int = Field(512, description="Height of the generated video")
    num_frames: int = Field(24, description="Number of frames in the generated video")
    frame_rate: float = Field(8.0, description="Frame rate of the generated video")
    camera_motion: str = Field("static", description="Camera motion type")
    seed: int | None = Field(None, description="Seed for reproducibility")
    model: Literal["fast", "fast_hq", "pro"] = Field(
        "fast", description="Type of video generation model to use"
    )
    ref_image: str | None = Field(None, description="Path to reference image for image-to-video")
    image_conditioning: list[ImageConditioningInput] | None = Field(
        None, description="List of conditioning images"
    )


class GenerateVideoResponse(BaseModel):
    video_path: str | None = Field(None, description="Path to the generated video file")
    audio_path: str | None = Field(None, description="Path to the generated audio file")


class UpscaleVideoRequest(BaseModel):
    video_path: str = Field(..., description="Path to the video to upscale")
    upscaler_model: str = Field(..., description="Name of the upscaler model to use")


class UpscaleVideoResponse(BaseModel):
    video_path: str | None = Field(None, description="Path to the upscaled video file")


class CancelCancellingResponse(BaseModel):
    status: Literal["cancelling"] = Field("cancelling")
    id: str


class CancelNoActiveGenerationResponse(BaseModel):
    status: Literal["no_active_generation"] = Field("no_active_generation")


CancelResponse = CancelCancellingResponse | CancelNoActiveGenerationResponse


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None


VideoModelType = Literal["fast", "fast_hq", "pro"]
LTXVideoGenResolution = Literal["480p", "720p", "1080p", "1440p", "2160p", "2k", "4k"]


class LTXVideoGenerationModelSpecItem(BaseModel):
    model_type: VideoModelType
    resolution: LTXVideoGenResolution
    cp_id: str
    upscale_cp_id: str


class LTXVideoGenerationModelSpec(BaseModel):
    models: list[LTXVideoGenerationModelSpecItem]
