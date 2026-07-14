# app_factory.py

from __future__ import annotations

import logging
import threading
from pathlib import Path

from backend.handlers.generation_handler import GenerationHandler
from backend.handlers.pipelines_handler import PipelinesHandler
from backend.handlers.text_handler import TextHandler
from backend.handlers.video_generation_handler import VideoGenerationHandler
from backend.runtime_config.runtime_config import RuntimeConfig, load_runtime_config
from backend.services.generation_service import GenerationService
from backend.services.ltx_api_client.ltx_api_client import LTXAPIClient
from backend.services.model_downloader.model_downloader import ModelDownloader
from backend.services.pipelines.fast_video_pipeline import LTXFastVideoPipeline
from backend.services.pipelines.hq_video_pipeline import LTXHQVideoPipeline
from backend.services.pipelines.pro_video_pipeline import LTXProVideoPipeline
from backend.services.text_encoder.text_encoder import TextEncoder
from backend.state.app_state import AppState
from backend.state.app_state_types import AppSettings
from backend.web.app import create_app

logger = logging.getLogger(__name__)


def create_application(
    app_dir: Path, lock: threading.RLock
) -> tuple[AppState, threading.Thread]:
    config = load_runtime_config(app_dir, lock)

    # Services
    ltx_api_client = LTXAPIClient(config.ltx_api_endpoint, config.ltx_api_key)
    model_downloader = ModelDownloader(config.models_dir, config.pipeline_download_workers)
    text_encoder = TextEncoder(config.gemma_root)
    generation_service = GenerationService(lock)

    # State
    app_state = AppState(
        app_settings=AppSettings(
            use_torch_compile=True,
            force_api_generations=False,
            local_generations_mode=config.local_generations_mode,
        ),
        gpu_slot=None,
        generation_progress=None,
        ltx_client_state=ltx_api_client.get_state(),
        model_downloader_state=model_downloader.get_state(),
        text_encoder_state=text_encoder.get_state(),
        generation_service_state=generation_service.get_state(),
    )

    # Handlers
    text_handler = TextHandler(app_state, lock, text_encoder)
    pipelines_handler = PipelinesHandler(
        app_state,
        lock,
        text_handler=text_handler,
        fast_video_pipeline_class=LTXFastVideoPipeline,
        hq_video_pipeline_class=LTXHQVideoPipeline,
        pro_video_pipeline_class=LTXProVideoPipeline,
        config=config,
    )
    generation_handler = GenerationHandler(app_state, lock, generation_service)
    video_generation_handler = VideoGenerationHandler(
        app_state,
        lock,
        generation_handler=generation_handler,
        pipelines_handler=pipelines_handler,
        ltx_api_client=ltx_api_client,
        config=config,
    )

    app = create_app(
        app_state=app_state,
        video_generation_handler=video_generation_handler,
    )

    def run_background_tasks() -> None:
        model_downloader.run_background_task()

    background_thread = threading.Thread(target=run_background_tasks, daemon=True)
    background_thread.start()

    return app, background_thread
