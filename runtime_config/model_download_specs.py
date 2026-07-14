from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from api_types import (
    LTXLocalModelId,
    LTXVideoGenPipeline,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    ModelCheckpointID,
)


@dataclass(frozen=True, slots=True)
class ModelCheckpointSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name


@dataclass(frozen=True, slots=True)
class LTXLocalModelDeprecated:
    pass


@dataclass(frozen=True, slots=True)
class LTXLocalModelRelevant:
    upgrade_messages: dict[LTXLocalModelId, str]


LTXLocalModelRelevance = LTXLocalModelDeprecated | LTXLocalModelRelevant


@dataclass(frozen=True, slots=True)
class LtxIcLorasSpec:
    depth_cp: ModelCheckpointID
    canny_cp: ModelCheckpointID
    pose_cp: ModelCheckpointID


@dataclass(frozen=True, slots=True)
class LTXLocalModelSpec:
    model_cp: ModelCheckpointID
    upscale_cp: ModelCheckpointID
    text_encoder_cp: ModelCheckpointID
    ic_loras_spec: LtxIcLorasSpec
    relevance: LTXLocalModelRelevance
    supported_pipelines: tuple[
        tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...
    ]


def _resolution_spec(
    *,
    fps_to_durations: dict[int, tuple[int, ...]],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
    )


def get_model_cp_spec(cp_id: ModelCheckpointID) -> ModelCheckpointSpec:
    match cp_id:
        case "ltx-2.3-22b-dev":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-dev.safetensors"),
                expected_size_bytes=46_100_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Full/dev transformer model",
            )
        case "ltx-2.3-22b-distilled":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
                expected_size_bytes=43_000_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="Main transformer model",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.0":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                expected_size_bytes=1_900_000_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="2x upscaler",
            )
        case "ltx-2.3-spatial-upscaler-x1.5-1.0":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
                expected_size_bytes=1_064_576_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="1.5x upscaler",
            )
        case "ltx-2.3-spatial-upscaler-x2-1.1":
            return ModelCheckpointSpec(
                relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
                expected_size_bytes=972_406_000,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3",
                description="2x upscaler v1.1",
            )
        case "ltx-2.3-22b-ic-lora-union-control-ref0.5":
            return ModelCheckpointSpec(
                relative_path=Path(
                    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
                ),
                expected_size_bytes=654_465_352,
                is_folder=False,
                repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                description="Union IC-LoRA control model",
            )
        case "dpt-hybrid-midas":
            return ModelCheckpointSpec(
                relative_path=Path("dpt-hybrid-midas"),
                expected_size_bytes=500_000_000,
                is_folder=True,
                repo_id="Intel/dpt-hybrid-midas",
                description="DPT-Hybrid MiDaS depth processor",
            )
        case "yolox-l-torchscript":
            return ModelCheckpointSpec(
                relative_path=Path("yolox_l.torchscript.pt"),
                expected_size_bytes=217_697_649,
                is_folder=False,
                repo_id="hr16/yolox-onnx",
                description="YOLOX person detector for pose preprocessing",
            )
        case "dw-ll-ucoco-384-bs5":
            return ModelCheckpointSpec(
                relative_path=Path("dw-ll_ucoco_384_bs5.torchscript.pt"),
                expected_size_bytes=135_059_124,
                is_folder=False,
                repo_id="hr16/DWPose-TorchScript-BatchSize5",
                description="DW Pose TorchScript processor",
            )
        case "gemma-3-12b-it-qat-q4_0-unquantized":
            return ModelCheckpointSpec(
                relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
                expected_size_bytes=25_000_000_000,
                is_folder=True,
                repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
                description="Gemma text encoder (bfloat16)",
            )
        case "z-image-turbo":
            return ModelCheckpointSpec(
                relative_path=Path("Z-Image-Turbo"),
                expected_size_bytes=31_000_000_000,
                is_folder=True,
                repo_id="Tongyi-MAI/Z-Image-Turbo",
                description="Z-Image-Turbo model for text-to-image generation",
            )
        case _:
            assert_never(cp_id)


def get_ltx_model_spec(model_id: LTXLocalModelId) -> LTXLocalModelSpec:
    fast_spec = LTXVideoGenerationSpec(
        display_name="LTX-2.3 Fast",
        supported_resolutions_durations={
            "540p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14, 16, 18, 20),
                }
            ),
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14, 16, 18, 20),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14, 16, 18, 20),
                }
            ),
        },
        a2v_supported_resolutions_durations={
            "540p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14, 16, 18, 20),
                }
            ),
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14, 16, 18, 20),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14, 16, 18, 20),
                }
            ),
        },
    )

    fast_hq_spec = LTXVideoGenerationSpec(
        display_name="LTX-2.3 Fast HQ",
        supported_resolutions_durations={
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14),
                    30: (5, 6, 8, 10, 12),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12),
                    30: (5, 6, 8, 10),
                }
            ),
            "1440p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10),
                    30: (5, 6, 8, 10),
                }
            ),
            "2160p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8),
                    30: (5, 6, 8),
                }
            ),
        },
        a2v_supported_resolutions_durations={
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12, 14),
                    30: (5, 6, 8, 10, 12),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12),
                    30: (5, 6, 8, 10),
                }
            ),
        },
    )

    pro_spec = LTXVideoGenerationSpec(
        display_name="LTX-2.3 Pro",
        supported_resolutions_durations={
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12),
                    30: (5, 6, 8, 10),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10),
                    30: (5, 6, 8, 10),
                }
            ),
            "1440p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10),
                    30: (5, 6, 8),
                }
            ),
            "2160p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8),
                    30: (5, 6, 8),
                }
            ),
        },
        a2v_supported_resolutions_durations={
            "720p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10, 12),
                    30: (5, 6, 8, 10),
                }
            ),
            "1080p": _resolution_spec(
                fps_to_durations={
                    24: (5, 6, 8, 10),
                    30: (5, 6, 8, 10),
                }
            ),
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
                supported_pipelines=(
                    ("pro", pro_spec),
                ),
            )
        case _:
            assert_never(model_id)


ALL_LTX_LOCAL_MODEL_IDS: tuple[LTXLocalModelId, ...] = (
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-dev",
)
