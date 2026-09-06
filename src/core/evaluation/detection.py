"""OCR detection metrics: IoU matching, average precision, mean IoU."""

from src.schemas.bbox import BBox
from src.schemas.evaluation import DetectionMetrics


class DetectionTracker:
    """Accumulate detection matches across receipts.

    Predictions are matched to ground truth boxes greedily by confidence.

    Args:
        iou_threshold: Minimum IoU required to accept a match.

    Returns:
        A tracker accumulating ground truth and prediction counts.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._num_gt = 0
        self._num_pred = 0
        self._matches: list[tuple[float, bool, float | None]] = []
        self._matched_ious: list[float] = []

    def update(
        self,
        gold_boxes: list[BBox],
        predicted_boxes: list[BBox],
        predicted_scores: list[float],
    ) -> None:
        """Register one receipt with gold and predicted boxes.

        Args:
            gold_boxes: Ground truth detection boxes.
            predicted_boxes: Predicted detection boxes.
            predicted_scores: Confidence scores aligned with the boxes.
        """
        self._num_gt += len(gold_boxes)
        self._num_pred += len(predicted_boxes)
        for prediction_index, gt_index, iou in match_detections(
            gold_boxes,
            predicted_boxes,
            predicted_scores,
            self.iou_threshold,
        ):
            self._matches.append(
                (predicted_scores[prediction_index], gt_index is not None, iou)
            )
            if iou is not None:
                self._matched_ious.append(iou)

    def metrics(self, name: str = "detection") -> DetectionMetrics:
        """Summarize the accumulated detections.

        Args:
            name: Metric identifier.

        Returns:
            Average precision, mean IoU and micro precision/recall.
        """
        true_positive = sum(
            1 for _score, is_true, _iou in self._matches if is_true
        )
        precision = _ratio(true_positive, self._num_pred)
        recall = _ratio(true_positive, self._num_gt)
        mean_iou = _mean(self._matched_ious) if self._matched_ious else 0.0
        ranked = [(score, is_true) for score, is_true, _iou in self._matches]
        return DetectionMetrics(
            name=name,
            average_precision=average_precision(ranked, self._num_gt),
            mean_iou=mean_iou,
            precision=precision,
            recall=recall,
            num_gt=self._num_gt,
            num_pred=self._num_pred,
        )


def match_detections(
    gold_boxes: list[BBox],
    predicted_boxes: list[BBox],
    predicted_scores: list[float],
    iou_threshold: float,
) -> list[tuple[int, int | None, float | None]]:
    """Match ground truth and predicted boxes greedily by confidence.

    Args:
        gold_boxes: Ground truth detection boxes.
        predicted_boxes: Predicted detection boxes.
        predicted_scores: Confidence scores aligned with the predictions.
        iou_threshold: Minimum IoU required to accept a match.

    Returns:
        One (prediction_index, gt_index, iou) entry per prediction, where
        gt_index and iou are None for unmatched (false positive) predictions.
    """
    used = [False] * len(gold_boxes)
    ordered = sorted(
        range(len(predicted_boxes)),
        key=lambda index: predicted_scores[index],
        reverse=True,
    )
    matches: list[tuple[int, int | None, float | None]] = []
    for prediction_index in ordered:
        best_gt_index: int | None = None
        best_iou = iou_threshold
        for gt_index, gold_box in enumerate(gold_boxes):
            if used[gt_index]:
                continue
            iou = gold_box.iou(predicted_boxes[prediction_index])
            if iou >= best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None:
            used[best_gt_index] = True
            matches.append((prediction_index, best_gt_index, best_iou))
        else:
            matches.append((prediction_index, None, None))
    return matches


def average_precision(
    ranked_scores: list[tuple[float, bool]], num_relevant: int
) -> float:
    """Compute the area under the precision-recall curve.

    Args:
        ranked_scores: (confidence, is_true_positive) pairs in any order.
        num_relevant: Total number of ground truth instances.

    Returns:
        Average precision in the range [0, 1].
    """
    ordered = sorted(ranked_scores, key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    average = 0.0
    previous_recall = 0.0
    for _score, is_true in ordered:
        if is_true:
            true_positive += 1
        else:
            false_positive += 1
        total = true_positive + false_positive
        precision = _ratio(true_positive, total)
        recall = _ratio(true_positive, num_relevant)
        average += precision * (recall - previous_recall)
        previous_recall = recall
    return average


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


def _mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list.

    Args:
        values: Non-empty list of numbers.

    Returns:
        The mean of the values.
    """
    return sum(values) / len(values)
