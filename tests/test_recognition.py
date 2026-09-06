"""Tests for the OCR recognition metrics."""

import pytest

from src.core.evaluation.levenshtein import (
    edit_distance,
)
from src.core.evaluation.recognition import (
    recognition_metrics,
)


def test_edit_distance_identical() -> None:
    """An identical pair should have zero distance."""
    assert edit_distance("hello", "hello") == 0


def test_edit_distance_insertion() -> None:
    """A single extra character should cost one edit."""
    assert edit_distance("hello", "hell") == 1


def test_edit_distance_substitution() -> None:
    """A single substitution should cost one edit."""
    assert edit_distance("hello", "hallo") == 1


def test_edit_distance_empty() -> None:
    """The distance from empty to a string is its length."""
    assert edit_distance("", "abc") == 3


def test_recognition_metrics_cer() -> None:
    """CER should be edits divided by the gold character count."""
    metrics = recognition_metrics(["hello"], ["hallo"])
    assert metrics.cer == pytest.approx(1 / 5)


def test_recognition_metrics_wer() -> None:
    """WER should count word-level edits over the gold word count."""
    metrics = recognition_metrics(["nasi campur"], ["nasi goreng"])
    assert metrics.wer == pytest.approx(1 / 2)


def test_recognition_metrics_perfect() -> None:
    """Identical texts should produce zero error rates."""
    metrics = recognition_metrics(["nasi", "campur"], ["nasi", "campur"])
    assert metrics.cer == pytest.approx(0.0)
    assert metrics.wer == pytest.approx(0.0)
    assert metrics.num_gt == 2
    assert metrics.num_pred == 2


def test_recognition_metrics_empty() -> None:
    """Empty inputs should produce zero error rates without crashing."""
    metrics = recognition_metrics([], [])
    assert metrics.cer == pytest.approx(0.0)
    assert metrics.wer == pytest.approx(0.0)
