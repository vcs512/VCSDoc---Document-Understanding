"""Semantic entity recognition (SER) metric over grouped entities."""

from collections import Counter

from src.core.evaluation.common import (
    compute_f1,
    normalized_key,
)
from src.schemas.cord import CordEntity
from src.schemas.evaluation import MetricResult
from src.schemas.kie import KieEntity


def entity_level_f1(
    gold: list[CordEntity],
    predicted: list[KieEntity],
    name: str = "ser",
) -> MetricResult:
    """Compare ground truth and predicted semantic entities.

    Gold entities come from grouping CORD lines by group id and category,
    predicted entities from a token-based or image-to-text KIE model.

    Args:
        gold: Ground truth entities.
        predicted: Predicted entities.
        name: Metric identifier.

    Returns:
        Entity-level micro F1 plus per-category scores.
    """
    gold_keys = Counter(
        normalized_key(entity.category, entity.text) for entity in gold
    )
    predicted_keys = Counter(
        normalized_key(entity.category, entity.text) for entity in predicted
    )
    return compute_f1(gold_keys, predicted_keys, name)
