"""OCR recognition metrics: character and word error rates."""

from src.core.evaluation.levenshtein import (
    edit_distance,
)
from src.schemas.evaluation import RecognitionMetrics


def recognition_metrics(
    gold_texts: list[str],
    predicted_texts: list[str],
    name: str = "recognition",
) -> RecognitionMetrics:
    """Compute CER and WER over aligned ground truth and prediction pairs.

    Sequences must be aligned (e.g. pairs matched by detection boxes).

    Args:
        gold_texts: Ground truth text per matched detection.
        predicted_texts: Recognized text per matched detection.
        name: Metric identifier.

    Returns:
        Character and word error rates plus instance counts.
    """
    character_distance = 0
    word_distance = 0
    gold_characters = 0
    gold_words = 0
    for gold, predicted in zip(gold_texts, predicted_texts):
        character_distance += edit_distance(gold, predicted)
        word_distance += edit_distance(gold.split(), predicted.split())
        gold_characters += len(gold)
        gold_words += len(gold.split())
    return RecognitionMetrics(
        name=name,
        cer=_ratio(character_distance, gold_characters),
        wer=_ratio(word_distance, gold_words),
        num_gt=len(gold_texts),
        num_pred=len(predicted_texts),
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Compute a guarded ratio.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        The ratio, or 0.0 when the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator