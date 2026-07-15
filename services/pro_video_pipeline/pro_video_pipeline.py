# services/pro_video_pipeline/pro_video_pipeline.py
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from api_types import AudioOrNone, GenerateVideoRequest, ImageConditioningInput


class ProVideoPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        pro_steps: int = 32,
        pro_cfg_scale: float = 9.0,
    ) -> ProVideoPipeline:
        ...

    def generate(
        self,
        request: GenerateVideoRequest,
        audio: AudioOrNone = None,
        image_conditioning: ImageConditioningInput | None = None,
        progress_callback=None,
    ) -> Iterator[bytes]:
        ...
