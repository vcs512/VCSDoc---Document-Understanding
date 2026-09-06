"""Token-level F1 metric over labeled spans."""

from collections import Counter

from src.core.evaluation.common import (
    compute_f1,
    normalized_key,
)
from src.schemas.evaluation import MetricResult
from src.schemas.kie import LabeledSpan


def token_level_f1(
    gold: list[LabeledSpan],
    predicted: list[LabeledSpan],
    name: str = "token_f1",
) -> MetricResult:
    """Compare a set of ground truth and predicted labeled spans.

    Args:
        gold: Ground truth spans (word-level for OCR-based models or
            flattened JSON values for image-to-text models).
        predicted: Predicted spans in the same representation.
        name: Metric identifier.

    Returns:
        Micro-averaged F1 over all spans plus per-category scores.
    """
    gold_keys = Counter(
        normalized_key(span.category, span.text) for span in gold
    )
    predicted_keys = Counter(
        normalized_key(span.category, span.text) for span in predicted
    )
    return compute_f1(gold_keys, predicted_keys, name)
