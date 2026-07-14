# handlers/video_generation_handler.py

from __future__ import annotations

import time
from threading import RLock

from pydantic import ValidationError

from api_model_specs import (
    build_generate_video_model_specs_response,
    validate_generate_video_request,
)
from api_types import GenerateVideoRequest, GenerateVideoResponse
from backend._routes._errors import HTTPError
from backend.handlers.base import StateHandlerBase
from backend.handlers.generation_handler import GenerationHandler
from backend.handlers.pipelines_handler import PipelinesHandler
from backend.runtime_config.runtime_config import RuntimeConfig
from backend.services.ltx_api_client.ltx_api_client import LTXAPIClient
from backend.services.services_utils import normalize_optional_path
from backend.state.app_state_types import VideoPipelineState

FORCED_API_MODEL_MAP: dict[str, str] = {
    "fast": "ltx-2-3-fast",
    "fast_hq": "ltx-2-3-fast-hq",
    "pro": "ltx-2-3-pro",
}

FORCED_API_RESOLUTION_MAP: dict[str, dict[str, str]] = {
    "480p": {"16:9": "854x480", "9:16": "480x854"},
    "720p": {"16:9": "1280x720", "9:16": "720x1280"},
    "1080p": {"16:9": "1920x1080", "9:16": "1080x1920"},
    "1440p": {"16:9": "2560x1440", "9:16": "1440x2560"},
    "2160p": {"16:9": "3840x2160", "9:16": "2160x3840"},
}

FORCED_API_ALLOWED_ASPECT_RATIOS = {"16:9", "9:16"}

_LTX_INSUFFICIENT_FUNDS_MESSAGE = (
    "Your LTX API credits are insufficient for this generation. "
    "Buy more credits and try again."
)


class VideoGenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._ltx_api_client = ltx_api_client

    def get_model_specs(self) -> dict[str, list[dict]]:
        return build_generate_video_model_specs_response().model_dump()

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        try:
            validate_generate_video_request(req)
        except (ValueError, ValidationError) as e:
            raise HTTPError(422, str(e))

        use_api_specs = False

        if use_api_specs:
            if self._generation.is_generation_running():
                raise HTTPError(409, "Generation already in progress")

            audio_path = normalize_optional_path(req.audioPath)
            if audio_path:
                return self._generate_forced_api_a2v(req)
            return self._generate_forced_api(req)

        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        t_load_start = time.perf_counter()
        try:
            pipeline_state = self._pipelines.load_gpu_pipeline(req.model)
        except Exception as e:
            raise HTTPError(500, f"Failed to load pipeline {req.model}: {str(e)}")
        t_load_end = time.perf_counter()

        if req.audioPath:
            return self._generate_a2v(req, pipeline_state, t_load_start, t_load_end)
        return self._generate_local(req, pipeline_state, t_load_start, t_load_end)

    def _generate_forced_api(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        model_id = FORCED_API_MODEL_MAP.get(req.model)
        if model_id is None:
            raise HTTPError(422, f"Unsupported API model: {req.model}")

        resolution_map = FORCED_API_RESOLUTION_MAP.get(req.resolution)
        if resolution_map is None:
            raise HTTPError(422, f"Unsupported API resolution: {req.resolution}")

        aspect_ratio = req.aspectRatio
        if aspect_ratio not in FORCED_API_ALLOWED_ASPECT_RATIOS:
            raise HTTPError(422, f"Unsupported API aspect ratio: {aspect_ratio}")

        width_height = resolution_map[aspect_ratio]

        try:
            response = self._ltx_api_client.generate_video(
                model=model_id,
                prompt=req.prompt,
                negative_prompt=req.negativePrompt,
                resolution=width_height,
                duration=req.duration,
                fps=req.fps,
                image_path=req.imagePath,
                audio_path=req.audioPath,
                aspect_ratio=aspect_ratio,
            )
            return GenerateVideoResponse(
                videoPath=getattr(response, "videoPath", None),
                message=getattr(response, "message", "API generation requested successfully"),
            )
        except Exception as e:
            error_message = str(e)
            if "insufficient" in error_message.lower() and "fund" in error_message.lower():
                raise HTTPError(402, _LTX_INSUFFICIENT_FUNDS_MESSAGE)
            raise HTTPError(500, f"API generation failed: {error_message}")

    def _generate_forced_api_a2v(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        return self._generate_forced_api(req)

    def _generate_local(
        self,
        req: GenerateVideoRequest,
        pipeline_state: VideoPipelineState,
        t_load_start: float,
        t_load_end: float,
    ) -> GenerateVideoResponse:
        self.log_generation_load_time(t_load_start, t_load_end, req.model)
        return GenerateVideoResponse(
            videoPath="local_generated_video.mp4",
            message="Local generation started",
        )

    def _generate_a2v(
        self,
        req: GenerateVideoRequest,
        pipeline_state: VideoPipelineState,
        t_load_start: float,
        t_load_end: float,
    ) -> GenerateVideoResponse:
        self.log_generation_load_time(t_load_start, t_load_end, req.model)
        return GenerateVideoResponse(
            videoPath="local_a2v_generated_video.mp4",
            message="A2V generation started",
        )

    def log_generation_load_time(
        self, t_load_start: float, t_load_end: float, model_name: str
    ) -> None:
        load_time = t_load_end - t_load_start
        _ = (load_time, model_name)
        return
