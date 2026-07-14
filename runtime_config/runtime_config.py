# runtime_config/runtime_config.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Literal

import torch
import yaml

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

    # ✅ مسیرهای جداگانه برای مدل‌های مختلف
    pipeline_checkpoint_path_fast: str = ""   # مسیر مدل distilled (برای Fast و Fast HQ)
    pipeline_checkpoint_path_pro: str = ""    # مسیر مدل dev (برای PRO)
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
        self.models_dir = self.app_dir / "models"

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            self.device = "cpu"

        logger.info("Running on %s", self.device)

        self.streaming_prefetch_count = {
            "performance": 2,
            "balanced": 1,
            "none": 0,
        }[self.local_generations_mode]


def load_runtime_config(app_dir: Path, lock: RLock) -> RuntimeConfig:
    config_path = app_dir / "runtime_config.yaml"
    base_config = RuntimeConfig(app_dir=app_dir)

    if not config_path.exists():
        logger.info("runtime_config.yaml not found, using default config")
        base_config.models_dir.mkdir(parents=True, exist_ok=True)
        return base_config

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("runtime_config.yaml must contain a mapping")

        with lock:
            config = RuntimeConfig(app_dir=app_dir, **config_dict)  # type: ignore[arg-type]

        logger.info("Loaded runtime config from %s", config_path)

    except Exception as e:
        logger.exception("Failed to load runtime config: %s", e)
        config = base_config

    config.models_dir.mkdir(parents=True, exist_ok=True)
    return config
