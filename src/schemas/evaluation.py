"""Evaluation results data transfer objects."""

from pydantic import BaseModel


class Score(BaseModel):
    """Precision, recall and F1 triplet.

    Args:
        precision: Precision in the range [0, 1].
        recall: Recall in the range [0, 1].
        f1: Harmonic mean of precision and recall.

    Returns:
        A validated score triplet.
    """

    precision: float
    recall: float
    f1: float


class MetricResult(BaseModel):
    """Summary of a token-level or entity-level F1 evaluation.

    Args:
        name: Metric identifier.
        overall: Micro-averaged score over all instances.
        per_class: Scores per semantic category.
        macro: Mean of the per-category scores, optional.

    Returns:
        A validated F1 metric result.
    """

    name: str
    overall: Score
    per_class: dict[str, Score]
    macro: Score | None = None


class DetectionMetrics(BaseModel):
    """Summary of an OCR detection evaluation.

    Args:
        name: Metric identifier.
        average_precision: Average precision at the configured IoU threshold.
        mean_iou: Mean IoU over boxes matched to a ground truth box.
        precision: Matched predictions divided by all predictions.
        recall: Matched predictions divided by all ground truth boxes.
        num_gt: Number of ground truth boxes.
        num_pred: Number of predicted boxes.

    Returns:
        A validated detection metrics result.
    """

    name: str
    average_precision: float
    mean_iou: float
    precision: float
    recall: float
    num_gt: int
    num_pred: int


class RecognitionMetrics(BaseModel):
    """Summary of an OCR recognition evaluation.

    Args:
        name: Metric identifier.
        cer: Character error rate in the range [0, ...).
        wer: Word error rate in the range [0, ...).
        num_gt: Number of ground truth text units.
        num_pred: Number of predicted text units.

    Returns:
        A validated recognition metrics result.
    """

    name: str
    cer: float
    wer: float
    num_gt: int
    num_pred: int


class EvaluationReport(BaseModel):
    """Aggregated result of evaluating one model on one dataset split.

    Args:
        model: Identifier of the evaluated model.
        split: Dataset split that was evaluated.
        metrics: Evaluated metrics keyed by metric name.

    Returns:
        A validated evaluation report.
    """

    model: str
    split: str
    metrics: dict[str, object]
