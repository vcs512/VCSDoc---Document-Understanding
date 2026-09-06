"""Abstract OCR engine interface for evaluation services."""

from abc import ABC, abstractmethod

from src.schemas.cord import CordReceipt
from src.schemas.ocr import OcrResult


class OcrEngine(ABC):
    """Detect and recognize text on a receipt image."""

    @abstractmethod
    def recognize(self, receipt: CordReceipt) -> OcrResult:
        """Detect words and recognize their text on one receipt.

        Args:
            receipt: The receipt image and its metadata.

        Returns:
            The recognized words ordered by reading order.
        """
