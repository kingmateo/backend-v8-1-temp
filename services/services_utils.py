# services/services_utils.py
from __future__ import annotations

import logging
from typing import Union

import torch

# نوع داده صوتی (اختیاری)
AudioOrNone = Union[bytes, None]

# نوع داده برای JSON
JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, dict[str, "JSONValue"], list["JSONValue"]]

# نوع داده برای Tiling Config
TilingConfigType = dict[str, int | float | bool | None] | None


def device_supports_fp8(device: torch.device) -> bool:
    """بررسی اینکه دستگاه از FP8 پشتیبانی می‌کند یا نه"""
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device) >= (8, 9)
    return False


def get_device_type(device: str | torch.device | None) -> str:
    """دریافت نوع دستگاه به صورت رشته"""
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return device.type
    return device


def sync_device(device: str | torch.device) -> None:
    """همگام‌سازی دستگاه (برای CUDA)"""
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_device_cache(device: str | torch.device) -> None:
    """پاک کردن کش دستگاه"""
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.empty_cache()
