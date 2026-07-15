# runtime_config/runtime_config.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda", "mps"]
LocalGenerationsMode = Literal["performance", "balanced", "none"]


@dataclass(slots=True)
class RuntimeConfig:
    # General config
    app_dir: Path
    local_generations_mode: LocalGenerationsMode = "performance"
    device: DeviceType = "cpu"

    # Model config
    models_dir: Path = field(init=False)

    # نام فایل‌های مدل (بر اساس اطلاعات Hugging Face)
    FAST_MODEL_FILENAME = "ltx-2.3-22b-distilled.safetensors"
    PRO_MODEL_FILENAME = "ltx-2.3-22b-dev.safetensors"
    UPSCALER_MODEL_FILENAME = "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"

    # این فیلدها بعداً در __post_init__ مقداردهی می‌شوند
    pipeline_checkpoint_path_fast: str = ""
    pipeline_checkpoint_path_pro: str = ""
    pipeline_upsampler_path: str = ""

    gemma_root: str | None = None

    # Pipeline specific config
    pipeline_hq_steps: int = 16
    pipeline_hq_cfg_scale: float = 7.0
    pipeline_pro_steps: int = 32
    pipeline_pro_cfg_scale: float = 9.0

    # API config
    ltx_api_endpoint: str = "https://api.ltx.monster"
    ltx_api_key: str = ""

    # Service config
    pipeline_download_workers: int = 8
    task_runner_threads: int = 2

    # Other config
    streaming_prefetch_count: int | None = None

    def __post_init__(self) -> None:
        # 1. تنظیم پوشه مدل‌ها
        self.models_dir = self.app_dir / "models"

        # 2. ساخت مسیر کامل برای هر فایل مدل
        fast_model_path = self.models_dir / self.FAST_MODEL_FILENAME
        pro_model_path = self.models_dir / self.PRO_MODEL_FILENAME
        upscaler_model_path = self.models_dir / self.UPSCALER_MODEL_FILENAME

        # 3. بررسی وجود فایل‌ها و مقداردهی فیلدها
        if not fast_model_path.exists():
            raise FileNotFoundError(
                f"مدل Fast (Distilled) در مسیر زیر پیدا نشد:\n{fast_model_path}\n"
                f"لطفاً فایل '{self.FAST_MODEL_FILENAME}' را از آدرس زیر دانلود و در این پوشه قرار دهید:\n"
                "https://huggingface.co/Lightricks/LTX-2.3/tree/main"
            )
        self.pipeline_checkpoint_path_fast = str(fast_model_path)

        if not pro_model_path.exists():
            raise FileNotFoundError(
                f"مدل PRO (Dev) در مسیر زیر پیدا نشد:\n{pro_model_path}\n"
                f"لطفاً فایل '{self.PRO_MODEL_FILENAME}' را از آدرس زیر دانلود و در این پوشه قرار دهید:\n"
                "https://huggingface.co/Lightricks/LTX-2.3/tree/main"
            )
        self.pipeline_checkpoint_path_pro = str(pro_model_path)

        if not upscaler_model_path.exists():
            raise FileNotFoundError(
                f"مدل Upscaler در مسیر زیر پیدا نشد:\n{upscaler_model_path}\n"
                f"لطفاً فایل '{self.UPSCALER_MODEL_FILENAME}' را از آدرس زیر دانلود و در این پوشه قرار دهید:\n"
                "https://huggingface.co/Lightricks/LTX-2.3/tree/main"
            )
        self.pipeline_upsampler_path = str(upscaler_model_path)

        # 4. تنظیمات مربوط به دستگاه
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            self.device = "cpu"

        logger.info("Running on %s", self.device)

        # 5. تنظیمات مربوط به Prefetch
        self.streaming_prefetch_count = {
            "performance": 2,
            "balanced": 1,
            "none": 0,
        }[self.local_generations_mode]


def create_runtime_config(app_dir: Path) -> RuntimeConfig:
    """یک نمونه از RuntimeConfig را با بررسی خودکار مدل‌ها ایجاد می‌کند."""
    config = RuntimeConfig(app_dir=app_dir)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    return config
