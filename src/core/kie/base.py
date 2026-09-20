"""Abstract KIE engine interface for evaluation services."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from src.schemas.cord import CordReceipt
from src.schemas.kie import KiePrediction
from src.schemas.ocr import OcrResult


class KieEngine(ABC):
    """Extract semantic entities from a receipt image."""

    @abstractmethod
    def predict(
        self, receipt: CordReceipt, ocr: OcrResult | None = None
    ) -> KiePrediction:
        """Predict semantic entities for one receipt.

        Args:
            receipt: The receipt image and its metadata.
            ocr: Optional OCR words used by token-based models.

        Returns:
            The structured KIE prediction for the receipt.
        """

    def predict_batch(
        self,
        receipts: Sequence[CordReceipt],
        ocr_results: Iterable[OcrResult],
    ) -> list[KiePrediction]:
        """Predict semantic entities for a batch of receipts.

        Args:
            receipts: Receipts to predict, one prediction per receipt.
            ocr_results: Per-receipt OCR words in the same order.

        Returns:
            The KIE predictions in the same order as the input receipts.
        """
        return [
            self.predict(receipt, ocr)
            for receipt, ocr in zip(receipts, ocr_results)
        ]
