# runtime_config/model_download_specs.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, assert_never

from api_types import (
    LTXLocalModelId,
    LTXVideoGenPipeline,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    ModelCheckpointID,
)

LTXLocalModelRelevance: TypeAlias = "LTXLocalModelDeprecated | LTXLocalModelRelevant"


@dataclass(frozen=True, slots=True)
class ModelCheckpointSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool = False
    repo_id: str | None = None
    description: str = ""

    @property
    def name(self) -> str:
        return str(self.relative_path)


@dataclass(frozen=True, slots=True)
class LTXLocalModelDeprecated:
    pass


@dataclass(frozen=True, slots=True)
class LTXLocalModelRelevant:
    upgrade_messages: dict[LTXLocalModelId, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LtxIcLorasSpec:
    depth_cp: ModelCheckpointID
    canny_cp: ModelCheckpointID
    pose_cp: ModelCheckpointID


@dataclass(frozen=True, slots=True)
class LTXLocalModelSpec:
    model_cp: ModelCheckpointID
    upscale_cp: ModelCheckpointID
    text_encoder_cp: ModelCheckpointID | None = None
    ic_loras_spec: LtxIcLorasSpec | None = None
    relevance: LTXLocalModelDeprecated | LTXLocalModelRelevant | None = None
    supported_pipelines: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...] = ()


def _resolution_spec(*, fps_to_durations: dict[int, tuple[int, ...]]) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={fps: list(durations) for fps, durations in fps_to_durations.items()}
    )


def get_model_cp_spec(cp_id: ModelCheckpointID) -> ModelCheckpointSpec:
    match cp_id:
        case "ltx-2.3-22b-dev":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-dev.safetensors"),
                expected_size_bytes=46_100_000_000,
                repo_id="Lightricks/LTX-2.3",
                description="Full/dev transformer model",
            )
        case "ltx-2.3-22b-distilled":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
                expected_size_bytes=43_000_000_000,
                repo_id="Lightricks/LTX-2.3",
                description="Main transformer model",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.0":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                expected_size_bytes=1_900_000_000,
                repo_id="Lightricks/LTX-2.3-Upscalers",
                description="2x upscaler",
            )
        case "ltx-2.3-spatial-upscaler-x1.5-1.0":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
                expected_size_bytes=1_064_576_000,
                repo_id="Lightricks/LTX-2.3-Upscalers",
                description="1.5x upscaler",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.1":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
                expected_size_bytes=1_900_000_000,
                repo_id="Lightricks/LTX-2.3-Upscalers",
                description="2x upscaler v1.1",
            )
        case "ltx-2.3-22b-ic-lora-union-control-ref0.5":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
                expected_size_bytes=900_000_000,
                repo_id="Lightricks/LTX-2.3",
                description="IC LoRA union control reference",
            )
        case "dpt-hybrid-midas":
            return ModelCheckpointSpec(
                relative_path=Path("dpt_hybrid-midas-501f0c75.pt"),
                expected_size_bytes=470_000_000,
                repo_id="Intel/dpt-hybrid-midas",
                description="MiDaS depth model",
            )
        case "yolox-l-torchscript":
            return ModelCheckpointSpec(
                relative_path=Path("yolox_l.torchscript"),
                expected_size_bytes=340_000_000,
                repo_id="Megvii-BaseDetection/YOLOX",
                description="YOLOX-L TorchScript model",
            )
        case "dw-ll-ucoco-384-bs5":
            return ModelCheckpointSpec(
                relative_path=Path("dw-ll_ucoco_384_bs5.pth"),
                expected_size_bytes=200_000_000,
                repo_id="Caffe/dwpose",
                description="DWpose model",
            )
        case "gemma-3-12b-it-qat-q4_0-unquantized":
            return ModelCheckpointSpec(
                relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
                expected_size_bytes=8_000_000_000,
                repo_id="google/gemma-3-12b-it",
                description="Gemma 3 12B text encoder",
            )
        case "z-image-turbo":
            return ModelCheckpointSpec(
                relative_path=Path("z-image-turbo"),
                expected_size_bytes=6_000_000_000,
                repo_id="z-image/turbo",
                description="Z-Image-Turbo upscaler",
            )
        case _:
            assert_never(cp_id)


def get_ltx_model_spec(model_id: LTXLocalModelId) -> LTXLocalModelSpec:
    fast_spec = LTXVideoGenerationSpec(
        display_name="LTX-2.3 Fast",
        supported_resolutions_durations={
            "480p": _resolution_spec(fps_to_durations={24: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30)}),
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30),
                    30: (1, 2, 3, 4, 5, 6, 8, 10, 12),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (1, 2, 3, 4, 5, 6, 8, 10, 12),
                    30: (1, 2, 3, 4, 5, 6, 8, 10),
                }
            ),
            "1440p": _resolution_spec(fps_to_durations={24: (1, 2, 3, 4, 5), 30: (1, 2, 3, 4)}),
            "2160p": _resolution_spec(fps_to_durations={24: (1, 2, 3), 30: (1, 2)}),
        },
    )

    fast_hq_spec = LTXVideoGenerationSpec(
        display_name="LTX-2.3 Fast HQ",
        supported_resolutions_durations={
            "480p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "720p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "1080p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "1440p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10), 30: (5, 6, 8)}),
            "2160p": _resolution_spec(fps_to_durations={24: (5, 6, 8), 30: (5, 6, 8)}),
        },
        a2v_supported_resolutions_durations={
            "480p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "720p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "1080p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10), 30: (5, 6, 8, 10)}),
        },
    )

    pro_spec = LTXVideoGenerationSpec(
        display_name="LTX-2.3 Pro",
        supported_resolutions_durations={
            "480p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10), 60: (5,), 120: (5,)}),
            "720p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "1080p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 25: (5, 6, 8, 10), 30: (5, 6, 8, 10)}),
            "1440p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10), 25: (5, 6, 8, 10), 30: (5, 6, 8)}),
            "2160p": _resolution_spec(fps_to_durations={24: (5,)}),
        },
        a2v_supported_resolutions_durations={
            "480p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10, 12), 30: (5, 6, 8, 10)}),
            "720p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10), 30: (5, 6, 8, 10)}),
            "1080p": _resolution_spec(fps_to_durations={24: (5, 6, 8, 10), 30: (5, 6, 8, 10)}),
        },
    )

    common_ic_loras = LtxIcLorasSpec(
        depth_cp="dpt-hybrid-midas",
        canny_cp="yolox-l-torchscript",
        pose_cp="dw-ll-ucoco-384-bs5",
    )

    match model_id:
        case "ltx-2.3-22b-distilled":
            return LTXLocalModelSpec(
                model_cp="ltx-2.3-22b-distilled",
                upscale_cp="ltx-2.3-spatial-upscaler-x1.5-1.0",
                text_encoder_cp="gemma-3-12b-it-qat-q4_0-unquantized",
                ic_loras_spec=common_ic_loras,
                relevance=LTXLocalModelRelevant(
                    upgrade_messages={
                        "ltx-2.3-22b-dev": "Upgrade to Pro for higher quality outputs.",
                    }
                ),
                supported_pipelines=(
                    ("fast", fast_spec),
                    ("fast_hq", fast_hq_spec),
                ),
            )
        case "ltx-2.3-22b-dev":
            return LTXLocalModelSpec(
                model_cp="ltx-2.3-22b-dev",
                upscale_cp="ltx-2.3-spatial-upscaler-x2-1.1",
                text_encoder_cp="gemma-3-12b-it-qat-q4_0-unquantized",
                ic_loras_spec=common_ic_loras,
                relevance=LTXLocalModelRelevant(
                    upgrade_messages={
                        "ltx-2.3-22b-distilled": "Switch to Fast for lower latency.",
                    }
                ),
                supported_pipelines=(("pro", pro_spec),),
            )
        case _:
            assert_never(model_id)


ALL_LTX_LOCAL_MODEL_IDS: tuple[LTXLocalModelId, ...] = (
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-dev",
)
