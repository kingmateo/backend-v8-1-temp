# app_handler.py

from __future__ import annotations

import threading
from collections.abc import Iterator

import torch

from api_types import GenerateVideoRequest
from backend.handlers.generation_handler import GenerationHandler
from backend.handlers.pipelines_handler import PipelinesHandler
from backend.handlers.text_handler import TextHandler
from backend.handlers.video_generation_handler import VideoGenerationHandler
from backend.runtime_config.runtime_config import RuntimeConfig
from backend.state.app_state import AppState


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
    ) -> Iterator[torch.Tensor]:
        with self._lock:
            return self._video_generation_handler.generate(
                req, seed, height, width, num_frames, frame_rate
            )

    def get_model_specs(self) -> dict[str, list[dict]]:
        return self._video_generation_handler.get_model_specs()
