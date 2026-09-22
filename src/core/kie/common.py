"""Shared helpers for the KIE engines."""

import torch


def resolve_device(requested: str) -> str:
    """Resolve the requested device to a concrete torch device.

    Args:
        requested: "auto", "cuda" or "cpu".

    Returns:
        The concrete device, mapping "auto" to CUDA when available.
    """
    if requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    ):
        return "cuda"
    return "cpu"
