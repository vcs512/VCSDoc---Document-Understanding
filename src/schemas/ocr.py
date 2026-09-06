"""OCR-specific data transfer objects."""

from pydantic import BaseModel, Field

from src.schemas.bbox import BBox


class OcrWord(BaseModel):
    """A single word detected and recognized by an OCR engine.

    Args:
        text: Recognized text.
        bbox: Detected bounding box.
        confidence: Recognition confidence in the range [0, 1].
        line_index: Optional index of the source line on the receipt.

    Returns:
        A validated OCR word.
    """

    text: str
    bbox: BBox
    confidence: float = Field(default=0.0, ge=0, le=1)
    line_index: int | None = None


class OcrResult(BaseModel):
    """The raw OCR output for one receipt image.

    Args:
        image_id: Identifier of the source receipt.
        words: Recognized words ordered by layout reading order.

    Returns:
        A validated OCR result.
    """

    image_id: int
    words: list[OcrWord]
