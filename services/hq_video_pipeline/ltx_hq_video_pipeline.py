from __future__ import annotations

import logging
from typing import ClassVar, Final, Iterator, Literal

import torch
from PIL import Image

from services.hq_video_pipeline.hq_video_pipeline import HQVideoPipeline
from services.interfaces import AudioOrNone, ImageConditioningInput
from services.ltx_pipeline_common import LTXPipelineCommon
from services.services_utils import (
    TilingConfigType,
    default_tiling_config,
    encode_video_output,
    video_chunks_number,
)

logger = logging.getLogger(__name__)

PIPELINE_KIND: Final = "fast_hq"


class LTXHQVideoPipeline(HQVideoPipeline):
    pipeline_kind: ClassVar[Literal["fast_hq"]] = PIPELINE_KIND

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        hq_steps: int = 16,
        hq_cfg_scale: float = 7.0,
    ):
        self.hq_steps = hq_steps
        self.hq_cfg_scale = hq_cfg_scale
        self.common_pipeline = LTXPipelineCommon(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
        )

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        hq_steps: int = 16,
        hq_cfg_scale: float = 7.0,
    ) -> "LTXHQVideoPipeline":
        return LTXHQVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            hq_steps=hq_steps,
            hq_cfg_scale=hq_cfg_scale,
        )

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        image: Image.Image | None = None,
        images: list[ImageConditioningInput] | None = None,
        audio: AudioOrNone = None,
        camera_motion: str = "static",
        negative_prompt: str = "",
    ) -> Iterator[torch.Tensor]:
        tiling_config: TilingConfigType = default_tiling_config(
            height=height,
            width=width,
            tiling_factor=self.common_pipeline.tiling_factor,
        )
        n_chunks = video_chunks_number(num_frames=num_frames)

        for frame_index, latents in enumerate(
            self.common_pipeline.inference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                num_frames=num_frames,
                height=height,
                width=width,
                steps=self.hq_steps,
                cfg_scale=self.hq_cfg_scale,
                image=image,
                images=images,
                audio=audio,
                camera_motion=camera_motion,
                tiling_config=tiling_config,
                n_chunks=n_chunks,
                frame_rate=frame_rate,
            )
        ):
            yield encode_video_output(
                latents=latents,
                upsampler=self.common_pipeline.upsampler,
                device=self.common_pipeline.device,
                tiling_config=tiling_config,
                frame_idx=frame_index,
                video_chunk_idx=frame_index // n_chunks,
                num_video_chunks=n_chunks,
                num_frames=num_frames,
                output_format="tensor",
            )
