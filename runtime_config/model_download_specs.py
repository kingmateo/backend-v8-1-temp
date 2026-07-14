# runtime_config/model_download_specs.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

FAST_VIDEO_MODEL_CP_ID = "ltx-2.3-22b-distilled"
FAST_VIDEO_UPSCALE_CP_ID = "ltx-2.3-spatial-upscaler-x2-1.0"
HQ_VIDEO_MODEL_CP_ID = "ltx-2.3-22b-dev"
HQ_VIDEO_UPSCALE_CP_ID = "ltx-2.3-spatial-upscaler-x2-1.1"
PRO_VIDEO_MODEL_CP_ID = "ltx-2.3-22b-dev"
PRO_VIDEO_UPSCALE_CP_ID = "ltx-2.3-spatial-upscaler-x2-1.1"

ALL_LTX_LOCAL_MODEL_IDS = [
    FAST_VIDEO_MODEL_CP_ID,
    FAST_VIDEO_UPSCALE_CP_ID,
    HQ_VIDEO_MODEL_CP_ID,
    HQ_VIDEO_UPSCALE_CP_ID,
]

class LTXLocalModelRelevant(BaseModel):
    supported_pipelines: list[tuple[str, object]]

class CheckpointSpec(BaseModel):
    model_id: str
    subfolder: str | None = None

class LTXModelSpec(BaseModel):
    model_cp: CheckpointSpec
    upscale_cp: CheckpointSpec
    relevance: object = None
    supported_pipelines: list[tuple[str, object]] = []

def get_ltx_model_spec(model_id: str) -> LTXModelSpec:
    from api_model_specs import ltx_api_model_specs
    supported = [item for item in ltx_api_model_specs if item[0] in ("fast", "fast_hq")]
    
    if model_id == FAST_VIDEO_MODEL_CP_ID:
        return LTXModelSpec(
            model_cp=CheckpointSpec(model_id=model_id, subfolder="unet"),
            upscale_cp=CheckpointSpec(model_id=FAST_VIDEO_UPSCALE_CP_ID, subfolder="unet"),
            relevance=LTXLocalModelRelevant(supported_pipelines=supported),
            supported_pipelines=supported
        )
    return LTXModelSpec(
        model_cp=CheckpointSpec(model_id=model_id, subfolder="unet"),
        upscale_cp=CheckpointSpec(model_id=HQ_VIDEO_UPSCALE_CP_ID, subfolder="unet"),
        relevance=LTXLocalModelRelevant(supported_pipelines=supported),
        supported_pipelines=supported
    )

def get_model_cp_path(models_dir: Path, spec: CheckpointSpec) -> Path:
    return models_dir / spec.model_id / (spec.subfolder or "")

def get_existing_cp_path(models_dir: Path, spec: CheckpointSpec) -> Path:
    path = get_model_cp_path(models_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    return path

def get_downloaded_ltx_model_id(models_dir: Path) -> str | None:
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        if (models_dir / model_id).exists():
            return model_id
    return None
