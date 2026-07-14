# app_handler.py
from __future__ import annotations

import threading
from collections.abc import Iterator
import torch
from api_types import GenerateVideoRequest, GenerateVideoResponse
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from handlers.video_generation_handler import VideoGenerationHandler
from runtime_config.runtime_config import RuntimeConfig
from state.app_state import AppState
from api_model_specs import build_generate_video_model_specs_response

class AppHandler:
    def __init__(
        self,
        state: AppState,
        lock: threading.RLock,
        config: RuntimeConfig,
        pipelines_handler: PipelinesHandler,
        video_generation_handler: VideoGenerationHandler,
        text_handler: TextHandler | None = None,
        generation_handler: GenerationHandler | None = None,
    ) -> None:
        self._state = state
        self._lock = lock
        self._config = config
        self._pipelines_handler = pipelines_handler
        self._video_generation_handler = video_generation_handler
        self._text_handler = text_handler
        self._generation_handler = generation_handler

    def generate_video(
        self,
        req: GenerateVideoRequest,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
    ) -> GenerateVideoResponse:
        with self._lock:
            return self._video_generation_handler.generate_video(req)

    def get_model_specs(self) -> dict:
        response = build_generate_video_model_specs_response()
        return response.model_dump()
