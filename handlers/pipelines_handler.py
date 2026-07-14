# handlers/pipelines_handler.py

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

import torch

from api_types import GenerateVideoRequest, ImageConditioningInput
from backend.handlers.base import StateHandlerBase
from backend.handlers.pipelines.fast_video_pipeline import LTXFastVideoPipeline
from backend.handlers.pipelines.hq_video_pipeline import LTXHQVideoPipeline
from backend.handlers.pipelines.pro_video_pipeline import LTXProVideoPipeline
from backend.runtime_config.runtime_config import RuntimeConfig
from backend.services.interfaces import AudioOrNone, VideoPipeline
from backend.services.services_utils import device_supports_fp8

if TYPE_CHECKING:
    from backend.state.app_state import AppState

logger = logging.getLogger(__name__)

PipelineKind: TypeAlias = Literal["fast", "fast_hq", "pro"]


class PipelinesHandler(StateHandlerBase):
    pipeline_kind_to_class: ClassVar[dict[PipelineKind, type[VideoPipeline]]] = {
        "fast": LTXFastVideoPipeline,
        "fast_hq": LTXHQVideoPipeline,
        "pro": LTXProVideoPipeline,
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

                checkpoint_path = self.config.pipeline_checkpoint_path
                upsampler_path = self.config.pipeline_upsampler_path
                gemma_root = self.config.gemma_root
                device = self.config.device
                streaming_prefetch_count = self.config.streaming_prefetch_count

                if pipeline_kind == "fast":
                    pipeline = pipeline_class.create(
                        checkpoint_path=checkpoint_path,
                        gemma_root=gemma_root,
                        upsampler_path=upsampler_path,
                        device=device,
                        streaming_prefetch_count=streaming_prefetch_count,
                    )
                elif pipeline_kind == "fast_hq":
                    pipeline = pipeline_class.create(
                        checkpoint_path=checkpoint_path,
                        gemma_root=gemma_root,
                        upsampler_path=upsampler_path,
                        device=device,
                        streaming_prefetch_count=streaming_prefetch_count,
                        hq_steps=self.config.pipeline_hq_steps,
                        hq_cfg_scale=self.config.pipeline_hq_cfg_scale,
                    )
                elif pipeline_kind == "pro":
                    pipeline = pipeline_class.create(
                        checkpoint_path=checkpoint_path,
                        gemma_root=gemma_root,
                        upsampler_path=upsampler_path,
                        device=device,
                        streaming_prefetch_count=streaming_prefetch_count,
                        pro_steps=self.config.pipeline_pro_steps,
                        pro_cfg_scale=self.config.pipeline_pro_cfg_scale,
                    )
                else:
                    raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

                self._pipelines[pipeline_kind] = pipeline

            if pipeline is None:
                raise RuntimeError(f"Failed to load pipeline: {pipeline_kind}")
            return pipeline

    def unload_pipeline(self, pipeline_kind: PipelineKind) -> None:
        if pipeline_kind not in self._pipelines:
            raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

        with self._pipeline_loading_locks[pipeline_kind]:
            if self._pipelines[pipeline_kind] is not None:
                logger.info("Unloading pipeline: %s", pipeline_kind)
                self._pipelines[pipeline_kind] = None

    def unload_all_pipelines(self) -> None:
        for kind in self._pipelines:
            self.unload_pipeline(kind)

    def get_pipeline(self, pipeline_kind: PipelineKind) -> VideoPipeline:
        if pipeline_kind not in self._pipelines:
            raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

        pipeline = self._pipelines[pipeline_kind]
        if pipeline is None:
            raise RuntimeError(f"Pipeline {pipeline_kind} not loaded")
        return pipeline

    def generate_video(
        self,
        pipeline_kind: PipelineKind,
        req: GenerateVideoRequest,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        audio: AudioOrNone,
    ) -> Iterator[torch.Tensor]:
        pipeline = self.get_pipeline(pipeline_kind)

        if pipeline_kind == "fast":
            video, _ = pipeline.generate(  # type: ignore[arg-type]
                prompt=req.prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                audio=audio,
                camera_motion=req.cameraMotion,
                negative_prompt=req.negativePrompt,
                tiling_config=None,
                resolution=None,
                upscaler=None,
            )
            if isinstance(video, Iterator):
                yield from video
            else:
                yield video  # type: ignore[misc]
            return

        yield from pipeline.generate(  # type: ignore[call-arg, misc]
            prompt=req.prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio=audio,
            camera_motion=req.cameraMotion,
            negative_prompt=req.negativePrompt,
            tiling_config=None,
            resolution=None,
            upscaler=None,
        )
