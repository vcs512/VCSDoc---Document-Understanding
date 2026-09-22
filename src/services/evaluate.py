"""Shared evaluation harness for OCR + KIE model pipelines."""

from collections.abc import Iterable
from itertools import islice

from tqdm import tqdm

from src.core.evaluation.detection import DetectionTracker, match_detections
from src.core.evaluation.recognition import recognition_metrics
from src.core.evaluation.ser import entity_level_f1
from src.core.evaluation.token_f1 import token_level_f1
from src.core.kie.base import KieEngine
from src.core.ocr.base import OcrEngine
from src.core.serialization import JsonTreeFlattener
from src.schemas.config import EvaluationConfig
from src.schemas.cord import CordEntity, CordReceipt
from src.schemas.evaluation import EvaluationReport
from src.schemas.kie import KieEntity, LabeledSpan
from src.schemas.ocr import OcrResult

_MODEL_ID = "paddleocr+layoutlmv3"


class Evaluator:
    """Run the full metric suite over a set of receipts.

    Args:
        ocr: OCR engine producing words and boxes.
        kie: KIE engine consuming the OCR words.
        config: Evaluation settings for matching and category filtering.

    Returns:
        A configured evaluator ready to score a dataset split.
    """

    def __init__(
        self,
        ocr: OcrEngine | None,
        kie: KieEngine,
        config: EvaluationConfig,
    ) -> None:
        self._ocr = ocr
        self._kie = kie
        self._iou_threshold = config.detection.iou_threshold
        self._ignored = set(config.ignore_categories)
        self._batch_size = config.batch_size
        self._ocr_workers = config.ocr_workers

    def evaluate(
        self,
        receipts: Iterable[CordReceipt],
        split: str,
        limit: int | None = None,
        model: str = _MODEL_ID,
        progress: bool = True,
        include_ocr: bool = True,
        tree_based: bool = False,
    ) -> EvaluationReport:
        """Score the engines over the receipts and aggregate the metrics.

        The receipts are processed in chunks whose configured size drives the
        OCR and KIE batch sizes, with a tqdm bar reporting the progress.

        Args:
            receipts: Iterable of receipts to evaluate.
            split: Dataset split being evaluated.
            limit: Optional maximum number of receipts to evaluate.
            model: Model identifier recorded in the report.
            progress: Whether to display the tqdm progress bar.
            include_ocr: Whether to run the OCR engine and report the detec-
                tion and recognition metrics.
            tree_based: Whether the KIE gold derives from the flattened
                gt_parse tree instead of the CORD line annotations.

        Returns:
            The aggregated evaluation report with detection, recognition,
            token-level F1 and SER metrics.
        """
        if include_ocr and self._ocr is None:
            raise ValueError("An OCR engine is required when include_ocr is set.")
        tracker = DetectionTracker(self._iou_threshold)
        recognition_pairs: list[tuple[str, str]] = []
        gold_spans: list[LabeledSpan] = []
        predicted_spans: list[LabeledSpan] = []
        gold_entities: list[CordEntity] = []
        predicted_entities: list[KieEntity] = []
        bar = tqdm(
            total=_progress_total(receipts, limit),
            desc=f"evaluating {split}",
            unit="receipts",
            disable=not progress,
        )
        try:
            for batch in _batch(islice(receipts, limit), self._batch_size):
                ocr_results = (
                    self._ocr.recognize_batch(batch, workers=self._ocr_workers)
                    if include_ocr
                    else [
                        OcrResult(image_id=receipt.image_id, words=[])
                        for receipt in batch
                    ]
                )
                predictions = self._kie.predict_batch(batch, ocr_results)
                for receipt, ocr_result, prediction in zip(
                    batch, ocr_results, predictions
                ):
                    if include_ocr:
                        self._track_ocr(
                            tracker, recognition_pairs, receipt, ocr_result
                        )
                    if tree_based:
                        tree_spans = JsonTreeFlattener.flatten(receipt.gt_parse)
                        gold_spans.extend(tree_spans)
                        gold_entities.extend(
                            CordEntity(
                                category=span.category,
                                group_id=index,
                                text=span.text,
                            )
                            for index, span in enumerate(tree_spans)
                        )
                    else:
                        gold_spans.extend(
                            LabeledSpan(category=token.category, text=token.text)
                            for token in receipt.tokens(self._ignored)
                        )
                        gold_entities.extend(receipt.entities(self._ignored))
                    predicted_spans.extend(prediction.spans)
                    predicted_entities.extend(prediction.entities)
                bar.update(len(batch))
        finally:
            bar.close()
        metrics: dict[str, object] = {}
        if include_ocr:
            metrics["detection"] = tracker.metrics()
            metrics["recognition"] = recognition_metrics(
                [gold for gold, _pred in recognition_pairs],
                [pred for _gold, pred in recognition_pairs],
            )
        metrics["token_f1"] = token_level_f1(gold_spans, predicted_spans)
        metrics["ser"] = entity_level_f1(gold_entities, predicted_entities)
        return EvaluationReport(model=model, split=split, metrics=metrics)

    def _track_ocr(
        self,
        tracker: DetectionTracker,
        recognition_pairs: list[tuple[str, str]],
        receipt: CordReceipt,
        ocr_result: OcrResult,
    ) -> None:
        """Accumulate the OCR detection and recognition contributions.

        Args:
            tracker: Detection accumulator for the full run.
            recognition_pairs: Two-tuples of aligned gold and predicted texts.
            receipt: The receipt under evaluation.
            ocr_result: The OCR result of the receipt.
        """
        gold_words = receipt.ocr_words()
        gold_boxes = [word.bbox for word in gold_words]
        predicted_boxes = [word.bbox for word in ocr_result.words]
        predicted_scores = [word.confidence for word in ocr_result.words]
        tracker.update(gold_boxes, predicted_boxes, predicted_scores)
        matches = match_detections(
            gold_boxes,
            predicted_boxes,
            predicted_scores,
            self._iou_threshold,
        )
        for prediction_index, gold_index, _iou in matches:
            if gold_index is None:
                continue
            recognition_pairs.append(
                (
                    gold_words[gold_index].text,
                    ocr_result.words[prediction_index].text,
                )
            )


def _batch(
    receipts: Iterable[CordReceipt], size: int
) -> Iterable[list[CordReceipt]]:
    """Chunk an iterable of receipts into contiguous batches.

    Args:
        receipts: Receipts to chunk.
        size: Maximum size of each batch.

    Yields:
        One list of receipts per batch, the last one possibly smaller.
    """
    batch: list[CordReceipt] = []
    for receipt in receipts:
        batch.append(receipt)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _progress_total(
    receipts: Iterable[CordReceipt], limit: int | None
) -> int | None:
    """Estimate the total number of receipts for the progress bar.

    Args:
        receipts: Iterable of receipts to evaluate.
        limit: Optional maximum number of receipts to evaluate.

    Returns:
        The expected receipt count when known, None otherwise.
    """
    try:
        total = len(receipts)
    except TypeError:
        return None
    if limit is not None:
        return min(limit, total)
    return total