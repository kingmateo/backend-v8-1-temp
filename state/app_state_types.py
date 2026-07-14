from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.a2v_pipeline.a2v_pipeline import A2VPipeline
from services.depth_processor_pipeline.depth_processor_pipeline import (
    DepthProcessorPipeline,
)
from services.fast_video_pipeline.fast_video_pipeline import FastVideoPipeline
from services.hq_video_pipeline.hq_video_pipeline import HQVideoPipeline
from services.ic_lora_pipeline.ic_lora_pipeline import IcLoraPipeline
from services.image_generation_pipeline.image_generation_pipeline import (
    ImageGenerationPipeline,
)
from services.pose_processor_pipeline.pose_processor_pipeline import (
    PoseProcessorPipeline,
)
from services.pro_video_pipeline.pro_video_pipeline import ProVideoPipeline
from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.text_encoder.text_encoder import TextEncoder
from state.app_settings import AppSettings
from state.conditioning_cache import ConditioningCache

VideoPipelineInstance = FastVideoPipeline | HQVideoPipeline | ProVideoPipeline

@dataclass
class DownloadingSession:
    download_id: str
    model_id: str
    started_at: float

@dataclass
class ActiveGeneration:
    generation_id: str
    started_at: float
    output_path: Path | None = None
    meta: dict[str, Any] | None = None

@dataclass
class TextEncoderState:
    text_encoder: TextEncoder
    gemma_root: Path
    is_loaded: bool = True

@dataclass
class VideoPipelineState:
    pipeline: VideoPipelineInstance
    is_compiled: bool
    model_type: str
    upscaler_id: str | None = None

@dataclass
class ICLoraState:
    pipeline: IcLoraPipeline

@dataclass
class A2VPipelineState:
    pipeline: A2VPipeline

@dataclass
class RetakePipelineState:
    pipeline: RetakePipeline

@dataclass
class DepthProcessorState:
    pipeline: DepthProcessorPipeline

@dataclass
class PoseProcessorState:
    pipeline: PoseProcessorPipeline

@dataclass
class ImageGenerationPipelineState:
    pipeline: ImageGenerationPipeline

@dataclass
class CpuSlot:
    active_pipeline: DepthProcessorState | PoseProcessorState | TextEncoderState | None = None

@dataclass
class GpuSlot:
    active_pipeline: (
        VideoPipelineState
        | ICLoraState
        | A2VPipelineState
        | RetakePipelineState
        | ImageGenerationPipelineState
        | None
    ) = None

@dataclass
class AppState:
    downloading_session: DownloadingSession | None = None
    gpu_slot: GpuSlot | None = None
    active_generation: ActiveGeneration | None = None
    cpu_slot: CpuSlot | None = None
    text_encoder: TextEncoderState | None = None
    app_settings: AppSettings = field(default_factory=AppSettings)
    conditioning_cache: ConditioningCache = field(default_factory=ConditioningCache)
