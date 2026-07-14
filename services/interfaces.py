# services/interfaces.py
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from api_types import (
    GenerateVideoResponse,
    ImageConditioningInput,
    LTXVideoGenResolution,
)
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
        """تولید ویدئو"""
        ...

    @abstractmethod
    def warmup(self, output_path: str) -> None:
        """گرم کردن pipeline"""
        ...

    @abstractmethod
    def compile_transformer(self) -> None:
        """کامپایل Transformer"""
        ...


__all__ = [
    "AudioOrNone",
    "ImageConditioningInput",
    "Iterator",
    "LTXVideoGenResolution",
    "GenerateVideoResponse",
    "TilingConfigType",
    "VideoPipeline",
]
