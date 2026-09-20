"""Tests for the PaddleOCR result parsing."""

from pathlib import Path
from typing import ClassVar

from PIL import Image

import src.core.ocr.paddle as paddle_module
from src.core.ocr.paddle import PaddleOcrEngine
from src.schemas.bbox import BBox
from src.schemas.cord import CordReceipt
from src.schemas.ocr import OcrResult, OcrWord
from src.schemas.service import OcrServiceConfig


class FakeClassicOcr:
    """Classic 2.x engine stand-in recording constructor kwargs."""

    instances: ClassVar[list[dict]] = []
    count: ClassVar[int] = 0

    def __init__(self, *args, **kwargs) -> None:
        FakeClassicOcr.instances.append(kwargs)
        FakeClassicOcr.count += 1

    def ocr(self, image, cls=False):
        """Return a classic payload holding a single line."""
        return [
            [
                [[[0, 0], [10, 0], [10, 10], [0, 10]], ["tok", 0.9]],
            ]
        ]


def receipt_with_image(image_id: int) -> CordReceipt:
    """Build a small receipt carrying a real image.

    Args:
        image_id: Identifier of the receipt.

    Returns:
        A receipt whose image triggers the classic OCR path.
    """
    return CordReceipt(
        image_id=image_id,
        split="test",
        image_size=(10, 10),
        lines=[],
        gt_parse={},
        image=Image.new("RGB", (10, 10), "white"),
    )


def test_recognize_batch_parallel(monkeypatch) -> None:
    """Parallel recognition should spawn per-thread engines in order."""
    monkeypatch.setattr(paddle_module, "PaddleOCR", FakeClassicOcr)
    FakeClassicOcr.instances = []
    FakeClassicOcr.count = 0
    engine = PaddleOcrEngine(OcrServiceConfig(api="classic2", device="cpu"))
    results = engine.recognize_batch(
        [receipt_with_image(1), receipt_with_image(2), receipt_with_image(3)],
        workers=3,
    )
    assert [result.image_id for result in results] == [1, 2, 3]
    assert FakeClassicOcr.count >= 2


def test_recognize_batch_sequential(monkeypatch) -> None:
    """A single worker should reuse one engine instance."""
    monkeypatch.setattr(paddle_module, "PaddleOCR", FakeClassicOcr)
    FakeClassicOcr.instances = []
    FakeClassicOcr.count = 0
    engine = PaddleOcrEngine(OcrServiceConfig(api="classic2", device="cpu"))
    results = engine.recognize_batch(
        [receipt_with_image(1), receipt_with_image(2)], workers=1
    )
    assert [result.image_id for result in results] == [1, 2]
    assert FakeClassicOcr.count == 1


def test_det_model_dir_passthrough(monkeypatch) -> None:
    """The configured detector should be forwarded to the classic engine."""
    monkeypatch.setattr(paddle_module, "PaddleOCR", FakeClassicOcr)
    FakeClassicOcr.instances = []
    FakeClassicOcr.count = 0
    engine = PaddleOcrEngine(
        OcrServiceConfig(
            api="classic2",
            device="cpu",
            det_model_dir="~/models/multilingual_det",
        )
    )
    data = engine.recognize_batch([receipt_with_image(1)], workers=1)
    assert data[0].image_id == 1
    assert FakeClassicOcr.instances[0]["det_model_dir"] == str(
        Path("~/models/multilingual_det").expanduser()
    )


def word_payload() -> dict:
    """Build a PaddleOCR payload with word-level boxes.

    Returns:
        A payload with two lines of recognized words.
    """
    return {
        "rec_scores": [0.9, 0.8],
        "rec_texts": ["nasi campur", "1"],
        "text_word": [["nasi", "campur"], ["1"]],
        "text_word_boxes": [
            [[0, 0, 10, 10], [12, 0, 22, 10]],
            [[0, 20, 10, 30]],
        ],
    }


def test_word_level_parsing() -> None:
    """Word-level boxes should produce one OcrWord per word."""
    result = PaddleOcrEngine.to_result(7, word_payload())
    assert isinstance(result, OcrResult)
    assert result.image_id == 7
    assert len(result.words) == 3


def test_word_level_fields() -> None:
    """OcrWords should carry text, box, confidence and line index."""
    result = PaddleOcrEngine.to_result(7, word_payload())
    first, second, third = result.words
    assert first == OcrWord(
        text="nasi",
        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        confidence=0.9,
        line_index=0,
    )
    assert second.line_index == 0
    assert third == OcrWord(
        text="1",
        bbox=BBox(x1=0, y1=20, x2=10, y2=30),
        confidence=0.8,
        line_index=1,
    )


def test_line_level_fallback() -> None:
    """Payloads without word boxes should fall back to line boxes."""
    payload = {
        "rec_scores": [0.95],
        "rec_texts": ["nasi campur"],
        "rec_boxes": [[0, 0, 30, 10]],
    }
    result = PaddleOcrEngine.to_result(7, payload)
    assert result.words == [
        OcrWord(
            text="nasi campur",
            bbox=BBox(x1=0, y1=0, x2=30, y2=10),
            confidence=0.95,
            line_index=0,
        )
    ]


def test_numpy_boxes_are_accepted() -> None:
    """Numpy box arrays should be converted to flat coordinate lists."""
    import numpy as np

    payload = {
        "rec_scores": [0.7],
        "rec_texts": ["x"],
        "rec_boxes": [np.array([1, 2, 3, 4])],
    }
    result = PaddleOcrEngine.to_result(7, payload)
    assert result.words[0].bbox == BBox(x1=1, y1=2, x2=3, y2=4)


def test_empty_payload() -> None:
    """A payload with no detections should produce no words."""
    result = PaddleOcrEngine.to_result(7, {"rec_scores": []})
    assert result.words == []


def test_single_line_does_not_trigger_word_path() -> None:
    """An empty first word line should fall back to line-level boxes."""
    payload = {
        "rec_scores": [0.9],
        "rec_texts": ["abc"],
        "text_word": [[]],
        "text_word_boxes": [[]],
        "rec_boxes": [[0, 0, 10, 10]],
    }
    result = PaddleOcrEngine.to_result(7, payload)
    assert len(result.words) == 1
    assert result.words[0].text == "abc"


def test_flatten_paired_box() -> None:
    """Two-corner polygon boxes should expand to four coordinates."""
    box = [[5, 6], [15, 20]]
    assert PaddleOcrEngine._flatten_box(box) == [5.0, 6.0, 15.0, 20.0]


def test_flatten_flat_box() -> None:
    """Flat box lists should be returned with float coordinates."""
    assert PaddleOcrEngine._flatten_box([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]


def classic_payload() -> list:
    """Build a classic 2.x OCR output payload.

    Returns:
        A single page holding two lines, each a polygon box plus a
        (text, score) pair.
    """
    return [
        [
            [[[0, 0], [30, 0], [30, 10], [0, 10]], ["nasi campur", 0.95]],
            [[[0, 20], [10, 20], [10, 30], [0, 30]], ["1", 0.8]],
        ]
    ]


def test_classic_parsing() -> None:
    """Classic line entries should produce one OcrWord per line."""
    result = PaddleOcrEngine.to_result_classic(7, classic_payload())
    assert isinstance(result, OcrResult)
    assert result.words == [
        OcrWord(
            text="nasi campur",
            bbox=BBox(x1=0, y1=0, x2=30, y2=10),
            confidence=0.95,
            line_index=0,
        ),
        OcrWord(
            text="1",
            bbox=BBox(x1=0, y1=20, x2=10, y2=30),
            confidence=0.8,
            line_index=1,
        ),
    ]


def test_classic_none_payload() -> None:
    """A None classic payload should produce no words."""
    result = PaddleOcrEngine.to_result_classic(7, None)
    assert result.words == []


def test_api_family_explicit() -> None:
    """The api config field should override autodetection."""
    assert PaddleOcrEngine(OcrServiceConfig(api="classic2"))._uses_classic_api()
    assert not PaddleOcrEngine(
        OcrServiceConfig(api="paddlex3")
    )._uses_classic_api()


def test_api_family_auto(monkeypatch) -> None:
    """Auto should pick classic2 when predict is unavailable."""
    classic = type("ClassicOCR", (), {"ocr": lambda self, *a: None})
    monkeypatch.setattr(paddle_module, "PaddleOCR", classic)
    assert PaddleOcrEngine(OcrServiceConfig(api="auto"))._uses_classic_api()
    modern = type(
        "ModernOCR", (), {"predict": lambda self, *a: [], "ocr": lambda self, *a: None}
    )
    monkeypatch.setattr(paddle_module, "PaddleOCR", modern)
    assert not PaddleOcrEngine(OcrServiceConfig(api="auto"))._uses_classic_api()