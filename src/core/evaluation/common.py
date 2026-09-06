"""Shared helpers for the F1-style metrics."""

from collections import Counter

from src.schemas.evaluation import MetricResult, Score

_BIO_PREFIXES = ("B-", "I-", "E-", "S-")


def normalize_category(category: str) -> str:
    """Strip a BIO/IOB tag prefix and surrounding whitespace.

    Args:
        category: Raw semantic category, possibly token-tagged.

    Returns:
        The normalized category name.
    """
    clean = " ".join(category.split())
    for prefix in _BIO_PREFIXES:
        if clean.startswith(prefix):
            return clean[len(prefix) :]
    return clean


def normalize_text(text: str) -> str:
    """Collapse surrounding and inner whitespace of a text value.

    Args:
        text: Raw text value.

    Returns:
        The whitespace-normalized text.
    """
    return " ".join(text.split())


def normalized_key(category: str, text: str) -> tuple[str, str]:
    """Normalize a category and text into a single comparable key.

    Args:
        category: Raw semantic category.
        text: Raw text value.

    Returns:
        A (category, text) tuple with both parts normalized.
    """
    return normalize_category(category), normalize_text(text)


def compute_f1(gold: Counter, predicted: Counter, name: str) -> MetricResult:
    """Compute micro, per-class and macro F1 between two multisets.

    Args:
        gold: Multiset of normalized (category, text) ground truth keys.
        predicted: Multiset of normalized (category, text) prediction keys.
        name: Metric identifier.

    Returns:
        The metric result with micro overall and macro averaged scores.
    """
    true_positive = sum((gold & predicted).values())
    overall = _score(true_positive, sum(predicted.values()), sum(gold.values()))
    categories = {key[0] for key in gold} | {key[0] for key in predicted}
    per_class: dict[str, Score] = {}
    for category in sorted(categories):
        gold_count = sum(
            count for key, count in gold.items() if key[0] == category
        )
        predicted_count = sum(
            count for key, count in predicted.items() if key[0] == category
        )
        class_tp = sum(
            count
            for key, count in (gold & predicted).items()
            if key[0] == category
        )
        per_class[category] = _score(class_tp, predicted_count, gold_count)
    macro = None
    if per_class:
        macro = Score(
            precision=sum(score.precision for score in per_class.values())
            / len(per_class),
            recall=sum(score.recall for score in per_class.values())
            / len(per_class),
            f1=sum(score.f1 for score in per_class.values()) / len(per_class),
        )
    return MetricResult(
        name=name, overall=overall, per_class=per_class, macro=macro
    )


def _score(true_positive: int, predicted: int, gold: int) -> Score:
    """Build a score triplet from raw counts.

    Args:
        true_positive: Number of correctly predicted instances.
        predicted: Number of predicted instances.
        gold: Number of ground truth instances.

    Returns:
        The precision, recall and F1 score.
    """
    precision = _ratio(true_positive, predicted)
    recall = _ratio(true_positive, gold)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Score(precision=precision, recall=recall, f1=f1)


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
