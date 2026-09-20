"""Tests for the LayoutLMv3 KIE label processing helpers."""

import torch
from PIL import Image

from src.core.kie.layoutlmv3 import (
    LayoutLmv3KieEngine,
    WordData,
    build_prediction,
    first_word_labels,
    normalize_boxes,
    prediction_for_sample,
)
from src.schemas.bbox import BBox
from src.schemas.cord import CordReceipt
from src.schemas.kie import KieEntity, KiePrediction, LabeledSpan
from src.schemas.ocr import OcrResult, OcrWord
from src.schemas.service import KieServiceConfig


class FakeEncoded(dict):
    """BatchEncoding stand-in with a per-sample word id mapping."""

    def __init__(self, word_ids: list[int | None]) -> None:
        super().__init__()
        self._word_ids = word_ids

    def word_ids(self, sample_index: int) -> list[int | None]:
        return self._word_ids

    def to(self, device: str) -> "FakeEncoded":
        return self


def label_logits(
    label_ids: list[int], sequence_length: int, batch_size: int = 1
) -> torch.Tensor:
    """Build logits whose argmax matches the requested label ids.

    Args:
        label_ids: Argmax label id per token position.
        sequence_length: Number of tokens in the sequence.
        batch_size: Number of samples sharing the same peaks.

    Returns:
        A logits tensor with one-hot peaks at the requested ids.
    """
    logits = torch.zeros(batch_size, sequence_length, 10)
    for index, label_id in enumerate(label_ids):
        logits[:, index, label_id] = 1.0
    return logits


FakeModel = type(
    "FakeModel",
    (),
    {
        "config": type(
            "FakeConfig",
            (),
            {
                "id2label": {
                    0: "O",
                    3: "B-MENU.NM",
                    4: "I-MENU.NM",
                    5: "I-MENU.NM",
                    7: "B-TOTAL.TOTAL_PRICE",
                }
            },
        )()
    },
)


def test_normalize_boxes_scales_to_thousand() -> None:
    """Boxes should be scaled from pixels to the 0-1000 coordinate range."""
    boxes = normalize_boxes(
        [BBox(x1=0, y1=0, x2=100, y2=200)], (200, 400)
    )
    assert boxes == [[0, 0, 500, 500]]


def test_normalize_boxes_clamps_to_range() -> None:
    """Out-of-image boxes should be clipped into the 0-1000 range."""
    boxes = normalize_boxes(
        [BBox(x1=0, y1=0, x2=500, y2=500)], (100, 100)
    )
    assert boxes == [[0, 0, 1000, 1000]]


def test_first_word_labels_keeps_first_subtoken() -> None:
    """Only the first sub-token of each word should define its label."""
    word_ids = [None, 0, 0, 0, 1, None]
    label_ids = [0, 3, 4, 5, 7, 0]
    result = first_word_labels(word_ids, label_ids)
    assert result == {0: 3, 1: 7}


def test_build_prediction_spans_and_entities() -> None:
    """Words should produce spans and grouped entities with BIO handling."""
    words = [
        WordData(text="Nasi", line_index=0, label="B-MENU.NM"),
        WordData(text="Campur", line_index=0, label="I-MENU.NM"),
        WordData(text="1", line_index=1, label="B-MENU.CNT"),
        WordData(text="75,000", line_index=2, label="B-TOTAL.TOTAL_PRICE"),
        WordData(text="2", line_index=1, label="B-MENU.CNT"),
    ]
    prediction = build_prediction(7, words)
    assert isinstance(prediction, KiePrediction)
    assert prediction.image_id == 7
    assert prediction.spans == [
        LabeledSpan(category="menu.nm", text="Nasi"),
        LabeledSpan(category="menu.nm", text="Campur"),
        LabeledSpan(category="menu.cnt", text="1"),
        LabeledSpan(category="total.total_price", text="75,000"),
        LabeledSpan(category="menu.cnt", text="2"),
    ]
    assert prediction.entities == [
        KieEntity(category="menu.nm", text="Nasi Campur", group_id=0),
        KieEntity(category="menu.cnt", text="1", group_id=1),
        KieEntity(category="total.total_price", text="75,000", group_id=2),
        KieEntity(category="menu.cnt", text="2", group_id=1),
    ]


def test_build_prediction_skips_o_labels() -> None:
    """"O" labeled words should be excluded from spans and break entity runs."""
    words = [
        WordData(text="Nasi", line_index=0, label="B-MENU.NM"),
        WordData(text="Campur", line_index=0, label="O"),
        WordData(text="Goreng", line_index=0, label="I-MENU.NM"),
    ]
    prediction = build_prediction(7, words)
    assert [span.text for span in prediction.spans] == ["Nasi", "Goreng"]
    assert prediction.entities == [
        KieEntity(category="menu.nm", text="Nasi", group_id=0),
        KieEntity(category="menu.nm", text="Goreng", group_id=0),
    ]


def test_build_prediction_i_without_b_starts_entity() -> None:
    """An I- label with no preceding B- should start a new entity."""
    words = [
        WordData(text="a", line_index=0, label="I-MENU.NM"),
        WordData(text="b", line_index=0, label="I-MENU.NM"),
    ]
    prediction = build_prediction(7, words)
    assert prediction.entities == [
        KieEntity(category="menu.nm", text="a b", group_id=0)
    ]


def test_build_prediction_separates_lines() -> None:
    """Entities should break when the source line index changes."""
    words = [
        WordData(text="a", line_index=0, label="B-MENU.NM"),
        WordData(text="b", line_index=1, label="I-MENU.NM"),
    ]
    prediction = build_prediction(7, words)
    assert [entity.text for entity in prediction.entities] == ["a", "b"]


def test_build_prediction_raw_labels_preserved() -> None:
    """The raw per-word labels should be stored on the prediction."""
    words = [WordData(text="x", line_index=0, label="B-MENU.NM")]
    prediction = build_prediction(7, words)
    assert prediction.raw == ["B-MENU.NM"]


class FakeProcessor:
    """LayoutLMv3Processor stand-in returning a fake encoding.

    Args:
        encodings: Per-sample word id lists assigned in order.
    """

    def __init__(self, encodings: list[list[int | None]]) -> None:
        self._encodings = encodings

    def __call__(self, *args, **kwargs) -> FakeEncoded:
        """Return the first unused fake encoding."""
        return FakeEncoded(self._encodings.pop(0))


class FakeKieModel:
    """Model stand-in returning the given logits tensor."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.config = FakeModel.config

    def __call__(self, **kwargs) -> object:
        return type("Out", (), {"logits": self._logits})()


def receipt_with_image(image_id: int) -> CordReceipt:
    """Build a minimal receipt carrying an image.

    Args:
        image_id: Identifier of the receipt.

    Returns:
        A receipt accepted by the batched KIE path.
    """
    return CordReceipt(
        image_id=image_id,
        split="test",
        image_size=(10, 10),
        lines=[],
        gt_parse={},
        image=Image.new("RGB", (10, 10), "white"),
    )


def test_prediction_for_sample() -> None:
    """Sample logits should decode into spans and entities."""
    words = [
        OcrWord(text="Nasi", bbox=BBox(x1=0, y1=0, x2=1, y2=1), line_index=0),
        OcrWord(text="75,000", bbox=BBox(x1=0, y1=0, x2=1, y2=1), line_index=1),
    ]
    encoded = FakeEncoded([None, 0, 0, 0, 1, None])
    logits = label_logits([0, 3, 4, 5, 7, 0], sequence_length=6)
    prediction = prediction_for_sample(
        7, words, encoded, logits, sample_index=0, model=FakeModel()
    )
    assert prediction.spans == [
        LabeledSpan(category="menu.nm", text="Nasi"),
        LabeledSpan(category="total.total_price", text="75,000"),
    ]
    assert prediction.entities == [
        KieEntity(category="menu.nm", text="Nasi", group_id=0),
        KieEntity(category="total.total_price", text="75,000", group_id=1),
    ]


def test_predict_batch_one_forward_pass(monkeypatch) -> None:
    """The batch should be decoded per receipt with empties preserved."""
    receipts = [receipt_with_image(1), receipt_with_image(2), receipt_with_image(3)]
    ocr_results = [
        OcrResult(
            image_id=1,
            words=[OcrWord(text="x", bbox=BBox(x1=0, y1=0, x2=1, y2=1))],
        ),
        OcrResult(
            image_id=2,
            words=[OcrWord(text="y", bbox=BBox(x1=0, y1=0, x2=1, y2=1))],
        ),
        OcrResult(image_id=3, words=[]),
    ]
    processor = FakeProcessor(
        [[None, 0], [None, 0]]
    )
    logits = label_logits([0, 7], sequence_length=2, batch_size=2)
    model = FakeKieModel(logits)
    engine = LayoutLmv3KieEngine(
        KieServiceConfig(model_dir="checkpoints/layoutlmv3-finetuned-cord")
    )
    monkeypatch.setattr(
        engine,
        "_load",
        lambda: (processor, model, "cpu"),
    )
    predictions = engine.predict_batch(receipts, ocr_results)
    assert [p.image_id for p in predictions] == [1, 2, 3]
    assert predictions[0].spans == [
        LabeledSpan(category="total.total_price", text="x")
    ]
    assert predictions[2].spans == []
    assert predictions[2].entities == []