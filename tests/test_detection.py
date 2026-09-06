"""Tests for the OCR detection metrics."""

import pytest

from src.core.evaluation.detection import (
    DetectionTracker,
    average_precision,
    match_detections,
)
from src.schemas.bbox import BBox


def box(x1: float, y1: float, x2: float, y2: float) -> BBox:
    """Build a convenience box.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        A BBox with the given coordinates.
    """
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def test_greedy_match_single() -> None:
    """An overlapping prediction should match its ground truth box."""
    matches = match_detections(
        [box(0, 0, 10, 10)], [box(0, 0, 10, 10)], [0.9], 0.5
    )
    assert matches == [(0, 0, pytest.approx(1.0))]


def test_greedy_match_below_threshold() -> None:
    """A prediction below the IoU threshold should remain unmatched."""
    matches = match_detections(
        [box(0, 0, 10, 10)], [box(100, 100, 110, 110)], [0.9], 0.5
    )
    assert matches[0][1] is None
    assert matches[0][2] is None


def test_greedy_match_prefers_higher_confidence() -> None:
    """Two predictions competing for a box should resolve by confidence."""
    gold = [box(0, 0, 10, 10)]
    predicted = [box(8, 8, 18, 18), box(0, 0, 10, 10)]
    scores = [0.4, 0.9]
    matches = match_detections(gold, predicted, scores, 0.5)
    by_index = {index: gt_index for index, gt_index, _iou in matches}
    assert by_index[1] == 0
    assert by_index[0] is None


def test_tracker_metrics_perfect() -> None:
    """Perfect detections should yield average precision and recall of 1.0."""
    tracker = DetectionTracker(iou_threshold=0.5)
    tracker.update([box(0, 0, 10, 10)], [box(0, 0, 10, 10)], [0.9])
    metrics = tracker.metrics()
    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.mean_iou == pytest.approx(1.0)
    assert metrics.num_gt == 1
    assert metrics.num_pred == 1


def test_tracker_metrics_with_noise() -> None:
    """Mixed correct and incorrect detections should lower the scores."""
    tracker = DetectionTracker(iou_threshold=0.5)
    tracker.update(
        [box(0, 0, 10, 10), box(20, 20, 30, 30)],
        [box(0, 0, 10, 10), box(20, 20, 30, 30), box(50, 50, 60, 60)],
        [0.9, 0.8, 0.7],
    )
    metrics = tracker.metrics()
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)


def test_average_precision_ranking() -> None:
    """All relevant hits ranked first should yield an AP of 1.0."""
    ranked = [(0.9, True), (0.8, False), (0.7, False)]
    assert average_precision(ranked, 1) == pytest.approx(1.0)


def test_average_precision_penalizes_confident_false_positive() -> None:
    """A confident false positive before the true positive should lower the AP."""
    ranked = [(0.9, False), (0.8, True)]
    assert average_precision(ranked, 1) == pytest.approx(0.5)
