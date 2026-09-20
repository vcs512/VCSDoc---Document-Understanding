"""Tests for the shared evaluation harness."""

import pytest

from src.core.kie.base import KieEngine
from src.core.ocr.base import OcrEngine
from src.schemas.config import DetectionConfig, EvaluationConfig
from src.schemas.cord import CordReceipt
from src.schemas.evaluation import EvaluationReport
from src.schemas.kie import KieEntity, KiePrediction, LabeledSpan
from src.schemas.ocr import OcrResult
from src.services.evaluate import Evaluator


class PerfectOcrEngine(OcrEngine):
    """OCR engine returning the gold words as its predictions."""

    def recognize(self, receipt: CordReceipt) -> OcrResult:
        """Return the receipt OCR words.

        Args:
            receipt: The receipt under test.

        Returns:
            The exact ground truth OCR result.
        """
        return OcrResult(
            image_id=receipt.image_id, words=receipt.ocr_words()
        )


class GoldKieEngine(KieEngine):
    """KIE engine predicting the exact gold entities and spans."""

    def predict(
        self, receipt: CordReceipt, ocr: OcrResult | None = None
    ) -> KiePrediction:
        """Return the gold entities from the receipt lines.

        Args:
            receipt: The receipt under test.
            ocr: Unused OCR result.

        Returns:
            Spans and entities matching the gold labels.
        """
        prediction = KiePrediction(image_id=receipt.image_id)
        for entity in receipt.entities():
            prediction.entities.append(
                KieEntity(
                    category=entity.category,
                    text=entity.text,
                    group_id=entity.group_id,
                )
            )
        for token in receipt.tokens():
            prediction.spans.append(
                LabeledSpan(category=token.category, text=token.text)
            )
        return prediction


@pytest.fixture
def evaluation_config() -> EvaluationConfig:
    """Build an evaluation configuration for the tests.

    Returns:
        A default evaluation configuration.
    """
    return EvaluationConfig(
        detection=DetectionConfig(iou_threshold=0.5),
        ignore_categories=["menu.etc", "office.dontcare"],
    )


def test_evaluator_returns_report(receipt: CordReceipt, evaluation_config: EvaluationConfig) -> None:
    """The evaluator should aggregate all four metric families."""
    evaluator = Evaluator(PerfectOcrEngine(), GoldKieEngine(), evaluation_config)
    report = evaluator.evaluate([receipt], split="test")
    assert isinstance(report, EvaluationReport)
    assert set(report.metrics) == {
        "detection",
        "recognition",
        "token_f1",
        "ser",
    }


def test_evaluator_perfect_scores(receipt: CordReceipt, evaluation_config: EvaluationConfig) -> None:
    """Perfect OCR and KIE engines should reach perfect metric scores."""
    evaluator = Evaluator(PerfectOcrEngine(), GoldKieEngine(), evaluation_config)
    report = evaluator.evaluate([receipt], split="test")
    assert report.metrics["detection"].average_precision == pytest.approx(1.0)
    assert report.metrics["detection"].recall == pytest.approx(1.0)
    assert report.metrics["detection"].precision == pytest.approx(1.0)
    assert report.metrics["recognition"].cer == pytest.approx(0.0)
    assert report.metrics["recognition"].wer == pytest.approx(0.0)
    assert report.metrics["token_f1"].overall.f1 == pytest.approx(1.0)
    assert report.metrics["ser"].overall.f1 == pytest.approx(1.0)


def test_evaluator_limit(receipt: CordReceipt, evaluation_config: EvaluationConfig) -> None:
    """The limit should truncate the receipt iteration."""
    evaluator = Evaluator(PerfectOcrEngine(), GoldKieEngine(), evaluation_config)
    report = evaluator.evaluate([receipt], split="test", limit=1)
    assert report.metrics["detection"].num_gt == 5


class EmptyKieEngine(KieEngine):
    """KIE engine returning an empty prediction."""

    def predict(
        self, receipt: CordReceipt, ocr: OcrResult | None = None
    ) -> KiePrediction:
        """Return an empty KIE prediction.

        Args:
            receipt: The receipt under test.
            ocr: Unused OCR result.

        Returns:
            A KIE prediction with no spans or entities.
        """
        return KiePrediction(image_id=receipt.image_id, spans=[], entities=[])


def test_evaluator_missing_predictions(receipt: CordReceipt, evaluation_config: EvaluationConfig) -> None:
    """A missing KIE output should still report OCR metrics."""
    evaluator = Evaluator(PerfectOcrEngine(), EmptyKieEngine(), evaluation_config)
    report = evaluator.evaluate([receipt], split="test")
    assert report.metrics["token_f1"].overall.f1 == pytest.approx(0.0)
    assert report.metrics["ser"].overall.f1 == pytest.approx(0.0)
    assert report.metrics["detection"].recall == pytest.approx(1.0)


def test_evaluator_batch_matches_sequential(
    receipt: CordReceipt, evaluation_config: EvaluationConfig
) -> None:
    """The batch size should not change the aggregated metrics."""
    sequential = evaluation_config.model_copy(update={"batch_size": 1})
    batched = evaluation_config.model_copy(update={"batch_size": 3})
    receipts = [receipt, receipt, receipt]
    sequential_report = Evaluator(
        PerfectOcrEngine(), GoldKieEngine(), sequential
    ).evaluate(receipts, split="test", progress=False)
    batched_report = Evaluator(
        PerfectOcrEngine(), GoldKieEngine(), batched
    ).evaluate(receipts, split="test", progress=False)
    assert sequential_report.model_dump() == batched_report.model_dump()


def test_evaluator_progress_bar(
    receipt: CordReceipt, evaluation_config: EvaluationConfig, monkeypatch
) -> None:
    """progress=False should hide the bar while progress=True shows it."""
    bars: list[object] = []
    updates: list[int] = []

    class FakeBar:
        """Record the tqdm construction and update calls."""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            bars.append(self)

        def update(self, count: int) -> None:
            updates.append(count)

        def close(self) -> None:
            pass

    monkeypatch.setattr("src.services.evaluate.tqdm", FakeBar)
    evaluator = Evaluator(PerfectOcrEngine(), GoldKieEngine(), evaluation_config)
    evaluator.evaluate([receipt], split="test", progress=False)
    evaluator.evaluate([receipt], split="test", progress=True)
    assert bars[0].kwargs["disable"] is True
    assert bars[1].kwargs["disable"] is False
    assert bars[1].kwargs["total"] == 1
    assert updates == [1, 1]