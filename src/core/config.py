"""Typed access to the JSON service configurations."""

import json
from functools import cache
from pathlib import Path

from src.schemas.config import CordConfig, EvaluationConfig
from src.schemas.service import LayoutLmv3ServiceConfig


@cache
def load_config(path: str | Path) -> dict:
    """Load and decode a JSON configuration file.

    Args:
        path: Path to the configuration file. Relative paths are resolved
            against the current working directory.

    Returns:
        The decoded configuration mapping.
    """
    config_path = Path(path).resolve()
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_cord_config(path: str | Path) -> CordConfig:
    """Load and validate the CORD dataset configuration.

    Args:
        path: Path to the CORD configuration file.

    Returns:
        The validated CORD configuration.
    """
    return CordConfig.model_validate(load_config(path))


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load and validate the evaluation configuration.

    Args:
        path: Path to the evaluation configuration file.

    Returns:
        The validated evaluation configuration.
    """
    return EvaluationConfig.model_validate(load_config(path))


def load_layoutlmv3_service_config(path: str | Path) -> LayoutLmv3ServiceConfig:
    """Load and validate the LayoutLMv3 evaluation service configuration.

    Args:
        path: Path to the service configuration file.

    Returns:
        The validated LayoutLMv3 service configuration.
    """
    return LayoutLmv3ServiceConfig.model_validate(load_config(path))