# services/fast_video_pipeline/fast_video_pipeline.py
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from api_types import AudioOrNone, GenerateVideoRequest, ImageConditioningInput


class FastVideoPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
    ) -> FastVideoPipeline:
        ...

    def generate(
        self,
        request: GenerateVideoRequest,
        audio: AudioOrNone = None,
        image_conditioning: ImageConditioningInput | None = None,
        progress_callback=None,
    ) -> Iterator[bytes]:
        ...
