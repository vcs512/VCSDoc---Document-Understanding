"""Donut image-to-text engine for the evaluation services."""

import json
import re
from collections.abc import Sequence
from typing import cast

import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)

from src.core.kie.base import KieEngine
from src.core.kie.common import resolve_device
from src.core.serialization import JsonTreeFlattener
from src.schemas.cord import CordReceipt
from src.schemas.kie import KieEntity, KiePrediction
from src.schemas.ocr import OcrResult
from src.schemas.service import DonutEngineConfig

_TASK_TAG = re.compile(r"^<.*?>")

_CORNER_TAG = re.compile(r"<s_([A-Za-z0-9_]+)>|</s_([A-Za-z0-9_]+)>|([^<]+)")


def extract_tree(sequence: str, eos: str, pad: str) -> dict:
    """Parse the generated sequence into the raw JSON tree.

    The model emits the task start tag, then either a JSON document or a
    corner-marker sequence (``<s_key>value</s_key>`` nesting) ending with the
    end-of-sequence token. Both markers are stripped before parsing, and the
    tree is rebuilt from whichever representation is present.

    Args:
        sequence: Decoded generated sequence.
        eos: End-of-sequence token.
        pad: Padding token.

    Returns:
        The parsed JSON tree.

    Raises:
        ValueError: When the sequence holds neither valid JSON nor a valid
            corner structure.
    """
    clean = sequence.replace(pad, "").replace(eos, "").strip()
    clean = _TASK_TAG.sub("", clean, count=1).strip()
    try:
        return json.loads(clean)
    except (ValueError, TypeError):
        pass
    return _corner_to_tree(clean)


def _corner_to_tree(sequence: str) -> dict:
    """Rebuild a JSON tree from a Donut corner-marker sequence.

    Indentation-free markup such as ``<s_menu><s_nm>ice</s_nm></s_menu>`` is
    turned into nested dicts, with repeated sibling markers folded into lists
    and empty markers into ``{}``. Leaf values are stripped of surrounding
    whitespace so they match the unspaced ground-truth values.

    Args:
        sequence: A corner-marker sequence without task, pad or eos tokens.

    Returns:
        The reconstructed tree.

    Raises:
        ValueError: When the sequence holds no corner structure.
    """
    events = _corner_events(sequence)
    if not events:
        raise ValueError(f"no JSON or corner structure in sequence: {sequence!r}")
    root, _ = _parse_siblings(events, len(events))
    if not root:
        raise ValueError(f"no JSON or corner structure in sequence: {sequence!r}")
    return root


def _corner_events(sequence: str) -> list[tuple[str, str]]:
    """Turn a corner sequence into an ordered list of parse events.

    Each ``(*open*)``/``close`` marker pair becomes an event with the marker
    key, while the text between two markers becomes a ``text`` event.

    Args:
        sequence: A corner-marker sequence.

    Returns:
        A list of ``("open"|"close"|"text", value)`` events.
    """
    events: list[tuple[str, str]] = []
    for opening, closing, text in _CORNER_TAG.findall(sequence):
        if opening:
            events.append(("open", opening))
        elif closing:
            events.append(("close", closing))
        elif text:
            events.append(("text", text))
    return events


def _parse_siblings(
    events: list[tuple[str, str]], end: int
) -> tuple[dict, int]:
    """Parse a run of sibling corner tokens into a dict.

    Repeated sibling markers are folded into a list; an empty marker pair
    becomes an empty dict. The emitted structure is

    Args:
        events: The ordered corner events.
        end: Exclusive index of this sibling run.

    Returns:
        The built dict and the index just past the last consumed event.
    """
    result: dict = {}
    index = 0
    while index < end:
        kind, key = events[index]
        if kind != "open":
            index += 1
            continue
        node, index = _parse_marker(events, index, end)
        if key in result:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(node)
            else:
                result[key] = [existing, node]
        else:
            result[key] = node
    return result, index


def _parse_marker(
    events: list[tuple[str, str]], index: int, end: int
) -> tuple[object, int]:
    """Parse one ``open`` marker with its nested content into a node.

    Args:
        events: The ordered corner events.
        index: Index of the marker's open event.
        end: Exclusive index of this sibling run.

    Returns:
        The node value and the index just past the marker's close event.
    """
    key = events[index][1]
    index += 1
    inner: list[tuple[str, str]] = []
    text: list[str] = []
    depth = 1
    while index < end and depth:
        kind, value = events[index]
        if kind == "open":
            depth += 1
        elif kind == "close":
            depth -= 1
            if depth == 0:
                index += 1
                break
        else:
            text.append(value)
        inner.append((kind, value))
        index += 1
    if not any(kind == "open" for kind, _ in inner):
        value = "".join(text).strip()
        return (value if value else {}), index
    children, _ = _parse_siblings(inner, len(inner))
    if key in children:
        return children[key], index
    return children, index
    return children, index


def tree_to_prediction(image_id: int, tree: dict) -> KiePrediction:
    """Convert a parsed JSON tree into a KIE prediction.

    Args:
        image_id: Identifier of the source receipt.
        tree: The generated parse tree.

    Returns:
        A prediction with one span and one entity per flattened tree leaf.
    """
    spans = JsonTreeFlattener.flatten(tree)
    entities = [
        KieEntity(category=span.category, text=span.text, group_id=index)
        for index, span in enumerate(spans)
    ]
    return KiePrediction(
        image_id=image_id, raw=tree, spans=spans, entities=entities
    )


class DonutKieEngine(KieEngine):
    """Extract semantic entities with a fine-tuned Donut model.

    Args:
        config: Donut engine settings.

    Returns:
        A configured Donut engine ready for prediction.
    """

    def __init__(self, config: DonutEngineConfig) -> None:
        self._config = config
        self._processor: DonutProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None
        self._device = "cpu"
        self._max_length = 0

    def predict(
        self, receipt: CordReceipt, ocr: OcrResult | None = None
    ) -> KiePrediction:
        """Predict semantic entities for one receipt.

        Args:
            receipt: The receipt image and its metadata.
            ocr: Unused OCR words, Donut is OCR-free.

        Returns:
            The structured KIE prediction for the receipt.
        """
        return self.predict_batch(
            [receipt], [OcrResult(image_id=receipt.image_id, words=[])]
        )[0]

    def predict_batch(
        self,
        receipts: Sequence[CordReceipt],
        ocr_results: Sequence[OcrResult],
    ) -> list[KiePrediction]:
        """Predict semantic entities for receipts in one batched generation.

        Receipts without an image are assigned empty predictions and excluded
        from the batch, so every receipt yields a prediction.

        Args:
            receipts: Receipts to predict.
            ocr_results: Unused OCR words, Donut is OCR-free.

        Returns:
            The KIE predictions in the same order as the input receipts.
        """
        predictions = [
            KiePrediction(image_id=receipt.image_id) for receipt in receipts
        ]
        valid_indexes = [
            index
            for index, receipt in enumerate(receipts)
            if receipt.image is not None
        ]
        if not valid_indexes:
            return predictions
        processor, model, device = self._load()
        pixel_values = processor(
            images=[receipts[index].image for index in valid_indexes],
            return_tensors="pt",
        ).pixel_values
        decoder_input_ids = processor.tokenizer(
            self._config.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.repeat(len(valid_indexes), 1)
        pixel_values = pixel_values.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        with torch.inference_mode():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._max_length,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=self._config.num_beams,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = cast(list[str], processor.batch_decode(outputs.sequences))
        for batch_index, receipt_index in enumerate(valid_indexes):
            try:
                tree = extract_tree(
                    sequences[batch_index],
                    processor.tokenizer.eos_token,
                    processor.tokenizer.pad_token,
                )
            except (ValueError, TypeError):
                tree = {}
            predictions[receipt_index] = tree_to_prediction(
                receipts[receipt_index].image_id, tree
            )
        return predictions

    def _load(self) -> tuple[DonutProcessor, VisionEncoderDecoderModel, str]:
        """Load the processor and model once on first use.

        Returns:
            The processor, the model in eval mode, and the resolved device.
        """
        if self._model is None:
            self._processor = DonutProcessor.from_pretrained(
                self._config.model_dir
            )
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self._config.model_dir,
                torch_dtype=_torch_dtype(self._config.dtype),
            )
            self._device = resolve_device(self._config.device)
            self._max_length = self._config.max_length or (
                self._model.config.decoder.max_position_embeddings
            )
            self._model.to(self._device)
            self._model.eval()
        return self._processor, self._model, self._device


def _torch_dtype(dtype: str) -> torch.dtype:
    """Map a dtype identifier to a torch dtype.

    Args:
        dtype: "float32" or "float16".

    Returns:
        The matching torch dtype.
    """
    if dtype == "float16":
        return torch.float16
    return torch.float32
