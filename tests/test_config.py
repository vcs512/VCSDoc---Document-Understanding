"""Tests for the configuration provider."""

from src.core.config import (
    load_config,
    load_cord_config,
    load_donut_service_config,
    load_evaluation_config,
)
from src.schemas.config import CordConfig, EvaluationConfig
from src.schemas.service import DonutServiceConfig


def test_load_config_resolves_an_absolute_path() -> None:
    """A relative config path should resolve to an absolute existing file."""
    config = load_config("configs/cord.json")
    assert config["dataset_id"] == "naver-clova-ix/cord-v2"


def test_load_cord_config() -> None:
    """The CORD config should parse with the expected values."""
    config = load_cord_config("configs/cord.json")
    assert isinstance(config, CordConfig)
    assert config.dataset_id == "naver-clova-ix/cord-v2"
    assert "train" in config.splits
    assert config.cache_dir == "data/cache"


def test_load_evaluation_config() -> None:
    """The evaluation config should parse with the expected values."""
    config = load_evaluation_config("configs/evaluation.json")
    assert isinstance(config, EvaluationConfig)
    assert config.detection.iou_threshold == 0.5
    assert "menu.etc" in config.ignore_categories


def test_load_donut_service_config() -> None:
    """The Donut service config should parse with the expected values."""
    config = load_donut_service_config("configs/evaluate_donut.json")
    assert isinstance(config, DonutServiceConfig)
    assert config.split == "test"
    assert config.donut.model_dir == "checkpoints/donut-base-finetuned-cord-v2"
    assert config.donut.task_prompt == "<s_cord-v2>"
    assert config.donut.device == "auto"
