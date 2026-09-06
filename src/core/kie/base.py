"""Abstract KIE engine interface for evaluation services."""

from abc import ABC, abstractmethod

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
