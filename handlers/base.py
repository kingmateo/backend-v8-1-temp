# handlers/base.py
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from runtime_config.runtime_config import RuntimeConfig

if TYPE_CHECKING:
    from state.app_state import AppState


class StateHandlerBase:
    """کلاس پایه برای همه Handlerها"""

    def __init__(self, state: "AppState", lock: threading.RLock, config: RuntimeConfig) -> None:
        self.state = state
        self.lock = lock
        self.config = config
