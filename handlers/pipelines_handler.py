# handlers/pipelines_handler.py
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

import torch

from api_types import GenerateVideoRequest, ImageConditioningInput
from handlers.base import StateHandlerBase
from services.fast_video_pipeline import FastVideoPipeline
from services.hq_video_pipeline import HQVideoPipeline
from services.pro_video_pipeline import ProVideoPipeline
from services.interfaces import AudioOrNone, VideoPipeline
from services.services_utils import device_supports_fp8
from runtime_config.runtime_config import RuntimeConfig

if TYPE_CHECKING:
    from state.app_state import AppState

logger = logging.getLogger(__name__)

PipelineKind: TypeAlias = Literal["fast", "fast_hq", "pro"]


class PipelinesHandler(StateHandlerBase):
    pipeline_kind_to_class: ClassVar[dict[PipelineKind, type[VideoPipeline]]] = {
        "fast": FastVideoPipeline,
        "fast_hq": HQVideoPipeline,
        "pro": ProVideoPipeline,
    }

    def __init__(self, state: "AppState", lock: threading.RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)
        self._pipelines: dict[PipelineKind, VideoPipeline | None] = {
            "fast": None,
            "fast_hq": None,
            "pro": None,
        }
        self._pipeline_loading_locks: dict[PipelineKind, threading.Lock] = {
            kind: threading.Lock() for kind in self._pipelines
        }

    def load_gpu_pipeline(self, pipeline_kind: PipelineKind) -> VideoPipeline:
        if pipeline_kind not in self._pipelines:
            raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

        with self._pipeline_loading_locks[pipeline_kind]:
            pipeline = self._pipelines[pipeline_kind]
            if pipeline is None:
                logger.info("Loading pipeline: %s", pipeline_kind)

                pipeline_class = self.pipeline_kind_to_class[pipeline_kind]

                # انتخاب مسیر checkpoint بر اساس نوع pipeline
                if pipeline_kind in ("fast", "fast_hq"):
                    checkpoint_path = self.config.pipeline_checkpoint_path_fast
                elif pipeline_kind == "pro":
                    checkpoint_path = self.config.pipeline_checkpoint_path_pro
                else:
                    raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

                upsampler_path = self.config.pipeline_upsampler_path
                gemma_root = self.config.gemma_root
                device = self.config.device
                streaming_prefetch_count = self.config.streaming_prefetch_count

                if pipeline_kind == "fast":
                    pipeline = pipeline_class.create(
                        checkpoint_path=checkpoint_path,
                        gemma_root=gemma_root,
                        upsampler_path=upsampler_path,
                        device=torch.device(device),
                        streaming_prefetch_count=streaming_prefetch_count,
                    )
                elif pipeline_kind == "fast_hq":
                    pipeline = pipeline_class.create(
                        checkpoint_path=checkpoint_path,
                        gemma_root=gemma_root,
                        upsampler_path=upsampler_path,
                        device=torch.device(device),
                        streaming_prefetch_count=streaming_prefetch_count,
                    )
                elif pipeline_kind == "pro":
                    pipeline = pipeline_class.create(
                        checkpoint_path=checkpoint_path,
                        gemma_root=gemma_root,
                        upsampler_path=upsampler_path,
                        device=torch.device(device),
                        streaming_prefetch_count=streaming_prefetch_count,
                    )
                else:
                    raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

                self._pipelines[pipeline_kind] = pipeline

            if pipeline is None:
                raise RuntimeError(f"Failed to load pipeline: {pipeline_kind}")

            return pipeline

    def unload_pipeline(self, pipeline_kind: PipelineKind) -> None:
        if pipeline_kind not in self._pipelines:
            return

        with self._pipeline_loading_locks[pipeline_kind]:
            pipeline = self._pipelines[pipeline_kind]
            if pipeline is not None:
                logger.info("Unloading pipeline: %s", pipeline_kind)
                if hasattr(pipeline, "unload"):
                    pipeline.unload()
                self._pipelines[pipeline_kind] = None
                torch.cuda.empty_cache()

    def generate_video(
        self,
        pipeline_kind: PipelineKind,
        request: GenerateVideoRequest,
        audio: AudioOrNone = None,
        image_conditioning: ImageConditioningInput | None = None,
        progress_callback=None,
    ) -> Iterator[bytes]:
        pipeline = self.load_gpu_pipeline(pipeline_kind)
        yield from pipeline.generate(
            request=request,
            audio=audio,
            image_conditioning=image_conditioning,
            progress_callback=progress_callback,
        )
