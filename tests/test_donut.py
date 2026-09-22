"""Tests for the Donut image-to-text KIE engine."""

import json

import pytest
import torch
from PIL import Image

from src.core.kie.donut import (
    DonutKieEngine,
    extract_tree,
    tree_to_prediction,
)
from src.schemas.cord import CordReceipt
from src.schemas.kie import KieEntity, KiePrediction, LabeledSpan
from src.schemas.ocr import OcrResult
from src.schemas.service import DonutEngineConfig

_EOS = "</s>"
_PAD = "<pad>"


class FakeInputIds:
    """Tokenizer output stand-in holding a repeatable id tensor."""

    def __init__(self) -> None:
        self.input_ids = torch.tensor([[7]])


class FakeTokenizer:
    """DonutProcessor tokenizer stand-in decoding preset sequences."""

    def __init__(self, sequences: list[str]) -> None:
        self._sequences = sequences
        self.pad_token = _PAD
        self.eos_token = _EOS
        self.unk_token_id = 3
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, *args, **kwargs) -> FakeInputIds:
        return FakeInputIds()

    def batch_decode(self, sequences: object) -> list[str]:
        return self._sequences


class FakeEncoded:
    """Processor output stand-in with pixel values.

    Args:
        pixel_values: Tensor returned on attribute access.
    """

    def __init__(self, pixel_values: torch.Tensor) -> None:
        self.pixel_values = pixel_values


class FakeProcessor:
    """DonutProcessor stand-in returning a fixed encoding."""

    def __init__(self, sequences: list[str], batch_size: int) -> None:
        self.tokenizer = FakeTokenizer(sequences)
        self._pixel_values = torch.zeros(batch_size, 3, 2, 2)

    def __call__(self, **kwargs) -> FakeEncoded:
        return FakeEncoded(self._pixel_values)

    def batch_decode(self, sequences: object) -> list[str]:
        return self.tokenizer.batch_decode(sequences)


class FakeGenerateModel:
    """VisionEncoderDecoderModel stand-in returning fixed sequences."""

    def __init__(self, sequences: object) -> None:
        self._sequences = sequences
        self.config = type(
            "FakeConfig",
            (),
            {
                "decoder": type(
                    "FakeDecoder",
                    (),
                    {"max_position_embeddings": 20},
                )()
            },
        )()

    def generate(self, *args, **kwargs) -> object:
        return type("FakeOut", (), {"sequences": self._sequences})()


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


def test_extract_tree_parses_json() -> None:
    """The task tag and eos token should be stripped before parsing."""
    sequence = '<s_cord-v2>{"menu": [{"nm": "Nasi"}]}</s>'
    tree = extract_tree(sequence, _EOS, _PAD)
    assert tree == {"menu": [{"nm": "Nasi"}]}


def test_extract_tree_removes_padding() -> None:
    """Padding tokens inside the sequence should be dropped."""
    sequence = '<pad><s_cord-v2>{"nm": "A"}</s><pad>'
    tree = extract_tree(sequence, _EOS, _PAD)
    assert tree == {"nm": "A"}


def test_extract_tree_without_task_tag() -> None:
    """A sequence without the leading task tag should still parse."""
    sequence = '{"total_price": "5,000"}</s>'
    tree = extract_tree(sequence, _EOS, _PAD)
    assert tree == {"total_price": "5,000"}


def test_extract_tree_malformed_raises() -> None:
    """A non-JSON sequence should raise a ValueError."""
    with pytest.raises(ValueError):
        extract_tree("<s_cord-v2>oops{</s>", _EOS, _PAD)


def test_tree_to_prediction() -> None:
    """A parsed tree should produce one span and entity per flattened leaf."""
    tree = {
        "menu": [{"nm": "Nasi", "cnt": "1"}],
        "total": {"total_price": "75,000"},
    }
    prediction = tree_to_prediction(7, tree)
    assert prediction.image_id == 7
    assert prediction.raw == tree
    assert prediction.spans == [
        LabeledSpan(category="menu.0.nm", text="Nasi"),
        LabeledSpan(category="menu.0.cnt", text="1"),
        LabeledSpan(category="total.total_price", text="75,000"),
    ]
    assert prediction.entities == [
        KieEntity(category="menu.0.nm", text="Nasi", group_id=0),
        KieEntity(category="menu.0.cnt", text="1", group_id=1),
        KieEntity(category="total.total_price", text="75,000", group_id=2),
    ]


def receipt_without_image(image_id: int) -> CordReceipt:
    """Build a minimal receipt without an image.

    Args:
        image_id: Identifier of the receipt.

    Returns:
        A receipt excluded from the batched KIE path.
    """
    return CordReceipt(
        image_id=image_id,
        split="test",
        image_size=(10, 10),
        lines=[],
        gt_parse={},
        image=None,
    )


def test_predict_batch_decodes_sequences(monkeypatch) -> None:
    """Generated sequences should decode into predictions per receipt."""
    receipts = [
        receipt_with_image(1),
        receipt_with_image(2),
        receipt_without_image(3),
    ]
    payload = json.dumps({"menu": [{"nm": "Nasi Campur"}]}, ensure_ascii=False)
    sequences = [f"<s_cord-v2>{payload}</s>", f"<s_cord-v2>{payload}</s>"]
    processor = FakeProcessor(sequences, batch_size=2)
    model = FakeGenerateModel([[0, 1, 2], [0, 1, 2]])
    engine = DonutKieEngine(
        DonutEngineConfig(model_dir="checkpoints/donut-base-finetuned-cord-v2")
    )
    engine._max_length = 20
    monkeypatch.setattr(engine, "_load", lambda: (processor, model, "cpu"))
    predictions = engine.predict_batch(
        receipts,
        [
            OcrResult(image_id=1, words=[]),
            OcrResult(image_id=2, words=[]),
            OcrResult(image_id=3, words=[]),
        ],
    )
    assert [prediction.image_id for prediction in predictions] == [1, 2, 3]
    assert predictions[0].spans == [
        LabeledSpan(category="menu.0.nm", text="Nasi Campur")
    ]
    assert predictions[1].spans == predictions[0].spans
    assert predictions[2].spans == []
    assert predictions[2].entities == []


def test_predict_batch_malformed_sequence_empty(monkeypatch) -> None:
    """A degenerate sequence should yield an empty prediction."""
    receipts = [receipt_with_image(1)]
    sequences = ["<s_cord-v2>not json</s>"]
    processor = FakeProcessor(sequences, batch_size=1)
    model = FakeGenerateModel([[0]])
    engine = DonutKieEngine(
        DonutEngineConfig(model_dir="checkpoints/donut-base-finetuned-cord-v2")
    )
    engine._max_length = 20
    monkeypatch.setattr(engine, "_load", lambda: (processor, model, "cpu"))
    prediction = engine.predict_batch(
        receipts, [OcrResult(image_id=1, words=[])]
    )[0]
    assert prediction.spans == []
    assert prediction.entities == []


def test_predict_returns_empty_without_image() -> None:
    """A receipt without an image should yield an empty prediction."""
    engine = DonutKieEngine(
        DonutEngineConfig(model_dir="checkpoints/donut-base-finetuned-cord-v2")
    )
    receipt = CordReceipt(
        image_id=4,
        split="test",
        image_size=(10, 10),
        lines=[],
        gt_parse={},
        image=None,
    )
    prediction = engine.predict(receipt)
    assert isinstance(prediction, KiePrediction)
    assert prediction.image_id == 4
    assert prediction.spans == []
