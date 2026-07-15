# services/interfaces.py
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from api_types import GenerateVideoRequest, ImageConditioningInput
from services.services_utils import AudioOrNone


@runtime_checkable
class VideoPipeline(Protocol):
    @abstractmethod
    def generate(
        self,
        request: GenerateVideoRequest,
        audio: AudioOrNone = None,
        image_conditioning: ImageConditioningInput | None = None,
        progress_callback=None,
    ) -> Iterator[bytes]:
        """تولید ویدئو و بازگشت به‌صورت جریانی"""
        ...

    @abstractmethod
    def warmup(self, output_path: str) -> None:
        """گرم کردن pipeline"""
        ...

    @abstractmethod
    def compile_transformer(self) -> None:
        """کامپایل Transformer"""
        ...
