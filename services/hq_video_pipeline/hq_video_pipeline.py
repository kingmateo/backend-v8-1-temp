# services/hq_video_pipeline/hq_video_pipeline.py
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from api_types import AudioOrNone, GenerateVideoRequest, ImageConditioningInput


class HQVideoPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        hq_steps: int = 16,
        hq_cfg_scale: float = 7.0,
    ) -> HQVideoPipeline:
        ...

    def generate(
        self,
        request: GenerateVideoRequest,
        audio: AudioOrNone = None,
        image_conditioning: ImageConditioningInput | None = None,
        progress_callback=None,
    ) -> Iterator[bytes]:
        ...
