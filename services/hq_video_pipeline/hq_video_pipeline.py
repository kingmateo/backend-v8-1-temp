from __future__ import annotations

from typing import ClassVar, Iterator, Literal, Protocol

import torch
from PIL import Image

from services.interfaces import AudioOrNone, ImageConditioningInput


class HQVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["fast_hq"]]

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
    ) -> "HQVideoPipeline":
        ...

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        image: Image.Image | None,
        images: list[ImageConditioningInput] | None,
        audio: AudioOrNone,
        camera_motion: str,
        negative_prompt: str,
    ) -> Iterator[torch.Tensor]:
        ...
