from __future__ import annotations

import logging
from threading import RLock

from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
)
from runtime_config.runtime_config import RuntimeConfig
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    GpuCleaner,
    HQVideoPipeline,
    IcLoraPipeline,
    ImageGenerationPipeline,
    PoseProcessorPipeline,
    ProVideoPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    GpuSlot,
    ICLoraState,
    ImageGenerationPipelineState,
    RetakePipelineState,
    VideoPipelineState,
)

logger = logging.getLogger(__name__)


def streaming_prefetch_count_for_mode(mode: str) -> int:
    if mode == "performance":
        return 2
    if mode == "balanced":
        return 1
    return 0


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        hq_video_pipeline_class: type[HQVideoPipeline],
        pro_video_pipeline_class: type[ProVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner

        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._hq_video_pipeline_class = hq_video_pipeline_class
        self._pro_video_pipeline_class = pro_video_pipeline_class

        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class

    @property
    def models_dir(self):
        return self.config.models_dir

    @property
    def _runtime_device(self) -> str:
        return self.config.device

    def _assert_invariants(self) -> None:
        return

    def _require_downloaded_ltx_model_id(self):
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            raise RuntimeError("No downloaded LTX local model found.")
        return model_id

    def _pipeline_matches_model_type(self, model_type: VideoPipelineModelType) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline)):
                return getattr(pipeline, "pipeline_kind", None) == model_type
            case _:
                return False

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state

        if state.is_compiled:
            return state

        if self._runtime_device == "mps":
            logger.info(
                "Skipping torch.compile() for %s - not supported on MPS",
                state.pipeline.pipeline_kind,
            )
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)

        return state

    def _select_video_pipeline_class(
        self,
        model_type: VideoPipelineModelType,
    ) -> (
        type[FastVideoPipeline]
        | type[HQVideoPipeline]
        | type[ProVideoPipeline]
    ):
        if model_type == "fast":
            return self._fast_video_pipeline_class
        if model_type == "fast_hq":
            return self._hq_video_pipeline_class
        if model_type == "pro":
            return self._pro_video_pipeline_class
        raise ValueError(f"Unsupported video pipeline model type: {model_type}")

    def _create_video_pipeline(self, model_type: VideoPipelineModelType) -> VideoPipelineState:
        gemma_root = self._text_handler.resolve_gemma_root()
        model_id = self._require_downloaded_ltx_model_id()
        spec = get_ltx_model_spec(model_id)

        checkpoint_path = str(get_existing_cp_path(self.models_dir, spec.model_cp))
        upsampler_path = str(get_existing_cp_path(self.models_dir, spec.upscale_cp))

        pipeline_class = self._select_video_pipeline_class(model_type)

        pipeline = pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
        )
        return self._compile_if_enabled(state)

    def _evict_gpu_pipeline_for_swap(self) -> None:
        with self._lock:
            self.state.gpu_slot = None
        self._gpu_cleaner.clean()

    def load_gpu_pipeline(self, model_type: VideoPipelineModelType) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        state: VideoPipelineState | None = None

        with self._lock:
            if self._pipeline_matches_model_type(model_type):
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as existing_state):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(model_type)
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state)
                self._assert_invariants()

        return state

    def _install_text_patches_if_needed(self) -> None:
        return

    def load_ic_lora_pipeline(self) -> ICLoraState:
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=ICLoraState() as existing_state):
                    return existing_state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        pipeline = self._ic_lora_pipeline_class.create(
            self.config.device,
        )
        state = ICLoraState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()

        return state

    def load_a2v_pipeline(self) -> A2VPipelineState:
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as existing_state):
                    return existing_state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        pipeline = self._a2v_pipeline_class.create(self.config.device)
        state = A2VPipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()

        return state

    def load_retake_pipeline(self) -> RetakePipelineState:
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=RetakePipelineState() as existing_state):
                    return existing_state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        pipeline = self._retake_pipeline_class.create(self.config.device)
        state = RetakePipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()

        return state

    def load_image_generation_pipeline(self) -> ImageGenerationPipelineState:
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=ImageGenerationPipelineState() as existing_state):
                    return existing_state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        checkpoint_path = str(get_existing_cp_path(self.models_dir, IMG_GEN_MODEL_CP_ID))
        pipeline = self._image_generation_pipeline_class.create(
            checkpoint_path,
            self.config.device,
        )
        state = ImageGenerationPipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()

        return state
