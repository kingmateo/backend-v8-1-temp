# services/interfaces.py

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import torch
from PIL import Image

from api_types import GenerateVideoResponse, ImageConditioningInput, LTXVideoGenResolution
from services.services_utils import AudioOrNone, TilingConfigType


@runtime_checkable
class VideoPipeline(Protocol):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput] | None,
        audio: AudioOrNone,
        camera_motion: str,
        negative_prompt: str,
        tiling_config: TilingConfigType | None,
        resolution: LTXVideoGenResolution | None,
        upscaler: str | None,
    ) -> Iterator[torch.Tensor] | GenerateVideoResponse:
        ...

    @abstractmethod
    def warmup(self, output_path: str) -> None:
        ...

    @abstractmethod
    def compile_transformer(self) -> None:
        ...


__all__ = [
    "Any",
    "AudioOrNone",
    "Image",
    "ImageConditioningInput",
    "Iterator",
    "LTXVideoGenResolution",
    "Protocol",
    "GenerateVideoResponse",
    "TilingConfigType",
    "VideoPipeline",
    "abstractmethod",
    "runtime_checkable",
]
