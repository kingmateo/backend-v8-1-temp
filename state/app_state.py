# state/app_state.py
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from runtime_config.runtime_config import RuntimeConfig


class AppState:
    """وضعیت کلی برنامه"""

    def __init__(self, app_dir: Path) -> None:
        self.app_dir = app_dir
        self.lock = threading.RLock()
        self.config = RuntimeConfig(app_dir=app_dir)
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self._data[key] = value
