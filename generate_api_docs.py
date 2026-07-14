# generate_api_docs.py
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, cast

from app_factory import create_app
from app_handler import AppHandler
from runtime_config.runtime_config import RuntimeConfig
from state.app_state import AppState
from handlers.pipelines_handler import PipelinesHandler
from handlers.video_generation_handler import VideoGenerationHandler
import torch

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "api_docs.json"

def _build_schema() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        app_data = tmp_root / "app_data"
        default_models_dir = app_data / "models"
        outputs_dir = tmp_root / "outputs"
        for directory in (app_data, default_models_dir, outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        config = RuntimeConfig(
            device=torch.device("cpu"),
            app_data_dir=app_data,
            default_models_dir=default_models_dir,
            outputs_dir=outputs_dir,
            settings_file=app_data / "settings.json",
            ltx_api_base_url="https://api.ltx.video",
            local_generations_mode="full_models_loading",
            use_sage_attention=False,
            camera_motion_prompts={},
            default_negative_prompt="test",
            dev_mode=False,
            backend_port=8000,
        )

        lock = torch.multiprocessing.RLock()
        state = AppState()
        
        pipelines = cast(Any, None)
        video_handler = VideoGenerationHandler(state=state, lock=lock, config=config)

        handler = AppHandler(
            state=state,
            lock=lock,
            config=config,
            pipelines_handler=pipelines,
            video_generation_handler=video_handler
        )
        
        app = create_app(handler=handler)
        return app.openapi()

def main() -> None:
    parser = argparse.ArgumentParser(description="Export API documentation to JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output API docs JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    schema = _build_schema()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"Wrote API documentation to {output}")

if __name__ == "__main__":
    main()
