from __future__ import annotations

from typing import Literal

from services.a2v_pipeline.a2v_pipeline import A2VPipeline
from services.depth_processor_pipeline.depth_processor_pipeline import (
    DepthProcessorPipeline,
)
from services.fast_video_pipeline.fast_video_pipeline import FastVideoPipeline
from services.gpu_cleaner.gpu_cleaner import GpuCleaner
from services.gpu_info.gpu_info import GpuInfo
from services.hq_video_pipeline.hq_video_pipeline import HQVideoPipeline
from services.http_client.http_client import HTTPClient
from services.ic_lora_pipeline.ic_lora_pipeline import IcLoraPipeline
from services.image_generation_pipeline.image_generation_pipeline import (
    ImageGenerationPipeline,
)
from services.ltx_api_client.ltx_api_client import LTXAPIClient
from services.model_downloader.model_downloader import ModelDownloader
from services.pose_processor_pipeline.pose_processor_pipeline import (
    PoseProcessorPipeline,
)
from services.pro_video_pipeline.pro_video_pipeline import ProVideoPipeline
from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.task_runner.task_runner import TaskRunner
from services.text_encoder.text_encoder import TextEncoder
from services.video_processor.video_processor import VideoProcessor
from services.zit_api_client.zit_api_client import ZitAPIClient

VideoPipelineModelType = Literal["fast", "fast_hq", "pro"]

__all__ = [
    "A2VPipeline",
    "DepthProcessorPipeline",
    "FastVideoPipeline",
    "GpuCleaner",
    "GpuInfo",
    "HQVideoPipeline",
    "HTTPClient",
    "IcLoraPipeline",
    "ImageGenerationPipeline",
    "LTXAPIClient",
    "ModelDownloader",
    "PoseProcessorPipeline",
    "ProVideoPipeline",
    "RetakePipeline",
    "TaskRunner",
    "TextEncoder",
    "VideoPipelineModelType",
    "VideoProcessor",
    "ZitAPIClient",
]
