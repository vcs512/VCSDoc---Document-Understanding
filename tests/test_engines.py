"""Tests for the OCR and KIE engine interfaces with dummy implementations."""

from abc import ABC

from src.core.kie.base import KieEngine
from src.core.ocr.base import OcrEngine
from src.schemas.cord import CordReceipt
from src.schemas.kie import KiePrediction
from src.schemas.ocr import OcrResult


class DummyOcrEngine(OcrEngine):
    """Dummy OCR engine returning the ground truth words."""

    def recognize(self, receipt: CordReceipt) -> OcrResult:
        """Return the receipt OCR words as an OcrResult.

        Args:
            receipt: The receipt under test.

        Returns:
            The OCR result for the receipt.
        """
        return OcrResult(image_id=receipt.image_id, words=receipt.ocr_words())


class DummyKieEngine(KieEngine):
    """Dummy KIE engine returning an empty prediction."""

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


def test_ocr_engine_is_abstract() -> None:
    """OcrEngine should expose the abstract recognize method marker."""
    assert OcrEngine.recognize.__isabstractmethod__


def test_kie_engine_is_abstract() -> None:
    """KieEngine should expose the abstract predict method marker."""
    assert KieEngine.predict.__isabstractmethod__


def test_ocr_and_kie_are_abc() -> None:
    """Both engines should subclass ABC."""
    assert issubclass(OcrEngine, ABC)
    assert issubclass(KieEngine, ABC)


def test_dummy_ocr_runs(receipt: CordReceipt) -> None:
    """The dummy OCR engine should run without error."""
    result = DummyOcrEngine().recognize(receipt)
    assert len(result.words) == 5


def test_dummy_kie_runs(receipt: CordReceipt) -> None:
    """The dummy KIE engine should run without error."""
    result = DummyKieEngine().predict(receipt)
    assert result.image_id == receipt.image_id
