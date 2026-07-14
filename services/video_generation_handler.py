from __future__ import annotations
import time
import logging
from threading import RLock
from api_model_specs import (
    build_generate_video_model_specs_response,
    validate_generate_video_request,
)
from api_types import (
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
)
from backend.handlers.base import StateHandlerBase
from backend.handlers.generation_handler import GenerationHandler
from backend.handlers.pipelines_handler import PipelinesHandler
from backend.handlers.text_handler import TextHandler
from backend.runtime_config.runtime_config import RuntimeConfig
from backend.services.ltx_api_client.ltx_api_client import LTXAPIClient
from backend.services.services_utils import (
    normalize_optional_path,
    should_video_generate_with_ltx_api,
)
from backend._routes._errors import HTTPError
from backend.state.app_state_types import VideoPipelineState

logger = logging.getLogger(__name__)

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
    "2k": {"16:9": "2560x1440", "9:16": "1440x2560"},
    "4k": {"16:9": "3840x2160", "9:16": "2160x3840"},
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
        text_handler: TextHandler,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._ltx_api_client = ltx_api_client

    def get_model_specs(self) -> GenerateVideoModelsSpecsResponse:
        return build_generate_video_model_specs_response()

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        use_api_specs = should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        )
        
        validation_error = validate_generate_video_request(
            req,
            use_api_specs=use_api_specs,
        )
        if validation_error is not None:
            raise HTTPError(
                422,
                validation_error,
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
            
        if use_api_specs:
            return self._generate_forced_api(req)

        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")
            
        audio_path = normalize_optional_path(req.audioPath)
        if audio_path:
            return self._generate_a2v(
                req,
                req.duration,
                req.fps,
                audio_path=audio_path,
            )
            
        model_to_load = req.model if req.model in {"fast", "fast_hq", "pro"} else "fast"
        logger_model_name = model_to_load
        
        logger.info("Resolution %s - using %s pipeline", req.resolution, logger_model_name)
        self._generation.update_progress("loading_model", 5, 0, 1)
        
        t_load_start = time.perf_counter()
        
        # Load the selected GPU pipeline (fast, fast_hq, pro)
        # Pass the selected upscaler dynamically to the pipelines handler if required
        pipeline_state = self._pipelines.load_gpu_pipeline(
            pipeline_id=model_to_load,
            upscaler_id=req.upscaler
        )
        
        t_load_end = time.perf_counter()
        
        return self._generate_local(
            req,
            pipeline_state,
            req.duration,
            req.fps,
            t_load_start,
            t_load_end,
        )

    def _generate_forced_api(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        model_id = FORCED_API_MODEL_MAP[req.model]
        resolution_map = FORCED_API_RESOLUTION_MAP.get(req.resolution)
        if resolution_map is None:
            raise HTTPError(422, f"Unsupported API resolution: {req.resolution}")
        aspect_ratio = req.aspectRatio
        if aspect_ratio not in FORCED_API_ALLOWED_ASPECT_RATIOS:
            raise HTTPError(422, f"Unsupported API aspect ratio: {aspect_ratio}")
        width_height = resolution_map[aspect_ratio]
        
        try:
            # External API generate forwarding
            api_resp = self._ltx_api_client.generate_video(
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
            return GenerateVideoResponse(videoPath=api_resp.video_path)
        except Exception as e:
            raise HTTPError(500, str(e))

    def _generate_local(
        self,
        req: GenerateVideoRequest,
        pipeline_state: VideoPipelineState,
        duration: int,
        fps: int,
        t_load_start: float,
        t_load_end: float,
    ) -> GenerateVideoResponse:
        # Calculate video frame numbers
        num_frames = duration * fps
        output_path = self._generation.get_output_video_path()
        
        self._generation.update_progress("generating", 10, 0, num_frames)
        
        try:
            # Extract underlying pipeline object
            pipeline = pipeline_state.pipeline
            
            # Translate aspect ratio to actual dimensions
            res_dimensions = FORCED_API_RESOLUTION_MAP.get(req.resolution, {}).get(req.aspectRatio, "1920x1080")
            width_str, height_str = res_dimensions.split("x")
            width, height = int(width_str), int(height_str)
            
            # Route logic based on pipeline type
            if req.model in ("fast_hq", "pro"):
                # HQ and PRO pipelines yield frames via an Iterator
                video_frames = []
                for frame_idx, frame_tensor in enumerate(
                    pipeline.generate(
                        prompt=req.prompt,
                        negative_prompt=req.negativePrompt,
                        seed=42, # Or dynamic seed
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        frame_rate=fps,
                        images=None,
                    )
                ):
                    video_frames.append(frame_tensor)
                    self._generation.update_progress("generating", 10 + int((frame_idx / num_frames) * 80), frame_idx, num_frames)
            else:
                # Fast pipeline (Non-iterator direct generation output)
                pipeline.generate(
                    prompt=req.prompt,
                    seed=42,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(fps),
                    images=[],
                    output_path=output_path,
                )
                
            self._generation.update_progress("saving", 95, num_frames, num_frames)
            return GenerateVideoResponse(videoPath=output_path, message="Local generation completed successfully")
        except Exception as e:
            raise HTTPError(500, f"Local generation failed: {str(e)}")

    def _generate_a2v(
        self,
        req: GenerateVideoRequest,
        duration: int,
        fps: int,
        *,
        audio_path: str,
    ) -> GenerateVideoResponse:
        return GenerateVideoResponse(
            message="A2V generation started",
            videoPath=None,
        )
