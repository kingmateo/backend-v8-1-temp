# generate_api_docs.py

from __future__ import annotations

import json
from pathlib import Path

from runtime_config.runtime_config import RuntimeConfig
from runtime_config.runtime_policy import LocalGenerationMode
from state.app_settings import AppSettings
from app_factory import create_app
from app_handler import build_default_service_bundle, build_initial_state


def main() -> None:
    dummy_config = RuntimeConfig(
        device="cpu",
        app_data_dir=Path("dummy_app_data"),
        default_models_dir=Path("dummy_models"),
        outputs_dir=Path("dummy_outputs"),
        settings_file=Path("dummy_settings.yaml"),
        ltx_api_base_url="http://dummy.api",
        local_generations_mode=LocalGenerationMode.performance,
        use_sage_attention=False,
        camera_motion_prompts={},
        default_negative_prompt="",
        dev_mode=False,
        backend_port=8000,
    )

    dummy_settings = AppSettings(
        use_torch_compile=True,
        force_api_generations=False,
        local_generations_mode=LocalGenerationMode.performance,
    )

    services = build_default_service_bundle()
    handler = build_initial_state(
        config=dummy_config,
        default_settings=dummy_settings,
        services=services,
    )

    app = create_app(handler=handler, title="LTX Video Generation Server")

    openapi_schema = app.openapi()
    output_path = Path("api_docs.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    print(f"API docs generated successfully at {output_path.resolve()}")


if __name__ == "__main__":
    main()
