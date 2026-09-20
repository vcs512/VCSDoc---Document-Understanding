"""Service-level configuration data transfer objects."""

from typing import Literal

from pydantic import BaseModel


class OcrServiceConfig(BaseModel):
    """Configuration for the PaddleOCR engine.

    Args:
        lang: OCR recognition language.
        api: OCR API family, "auto", "paddlex3" or "classic2".
        det_model_dir: Optional explicit detection model directory, useful to
            decouple detection from the language default model.
        return_word_box: Whether to produce word-level boxes and texts.
        use_doc_orientation_classify: Whether to normalize the document
            orientation before detection.
        use_doc_unwarping: Whether to unwarp curled documents before detection.
        use_textline_orientation: Whether to correct cropped line orientation.
        device: Inference device, "cpu" or "gpu".

    Returns:
        A validated PaddleOCR configuration.
    """

    lang: str = "korean"
    api: Literal["auto", "paddlex3", "classic2"] = "auto"
    det_model_dir: str | None = None
    return_word_box: bool = True
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = True
    device: str = "cpu"


class KieServiceConfig(BaseModel):
    """Configuration for the LayoutLMv3 KIE engine.

    Args:
        model_dir: Directory holding the fine-tuned token classification model.
        device: Inference device, "auto", "cuda" or "cpu".
        max_length: Maximum token sequence length per receipt.

    Returns:
        A validated LayoutLMv3 configuration.
    """

    model_dir: str
    device: str = "auto"
    max_length: int = 512


class LayoutLmv3ServiceConfig(BaseModel):
    """Configuration of the PaddleOCR + LayoutLMv3 evaluation service.

    Args:
        split: Dataset split evaluated by the service.
        output: Report file written by the service.
        ocr: PaddleOCR engine settings.
        kie: LayoutLMv3 engine settings.

    Returns:
        A validated service configuration.
    """

    split: str = "test"
    output: str = "reports/evaluate-layoutlmv3.json"
    ocr: OcrServiceConfig
    kie: KieServiceConfig