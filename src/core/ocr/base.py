"""Abstract OCR engine interface for evaluation services."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

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

    def recognize_batch(
        self, receipts: Iterable[CordReceipt], workers: int | None = None
    ) -> list[OcrResult]:
        """Recognize a batch of receipts, one result per input receipt.

        Args:
            receipts: Receipts to recognize.
            workers: Optional parallel worker threads, ignored by the base.

        Returns:
            The OCR results in the same order as the input receipts.
        """
        return [self.recognize(receipt) for receipt in receipts]
