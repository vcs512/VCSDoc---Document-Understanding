"""Service configuration data transfer objects."""

from pydantic import BaseModel


class CordConfig(BaseModel):
    """Configuration for the CORD dataset source.

    Args:
        dataset_id: Hugging Face dataset identifier.
        cache_dir: Local directory used as the dataset cache.
        splits: List of available dataset splits.

    Returns:
        A validated CORD configuration.
    """

    dataset_id: str
    cache_dir: str
    splits: list[str]


class DetectionConfig(BaseModel):
    """Configuration for OCR detection evaluation.

    Args:
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        A validated detection configuration.
    """

    iou_threshold: float


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation metrics.

    Args:
        detection: Detection evaluation settings.
        ignore_categories: Semantic categories excluded from KIE metrics.
        batch_size: Number of receipts evaluated per processing chunk.
        ocr_workers: Parallel OCR worker threads, at most the batch size.

    Returns:
        A validated evaluation configuration.
    """

    detection: DetectionConfig
    ignore_categories: list[str]
    batch_size: int = 4
    ocr_workers: int = 4