"""LayoutLMv3 token classification engine for the evaluation services."""

from collections.abc import Sequence

import torch
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3Tokenizer,
)

from src.core.evaluation.common import normalize_category
from src.core.kie.base import KieEngine
from src.schemas.bbox import BBox
from src.schemas.cord import CordReceipt
from src.schemas.kie import KieEntity, KiePrediction, LabeledSpan
from src.schemas.ocr import OcrResult, OcrWord
from src.schemas.service import KieServiceConfig

_CONTINUATION_PREFIXES = ("I-", "E-")


class WordData:
    """A recognized word paired with its predicted token label.

    Args:
        text: Word text.
        line_index: Index of the source OCR line.
        label: Raw model label such as "B-MENU.NM" or "O".

    Returns:
        A word with its predicted classification label.
    """

    def __init__(self, text: str, line_index: int, label: str) -> None:
        self.text = text
        self.line_index = line_index
        self.label = label


def normalize_boxes(
    bboxes: Sequence[BBox], image_size: tuple[int, int]
) -> list[list[int]]:
    """Scale word boxes into the 0-1000 range expected by the model.

    Args:
        bboxes: Word bounding boxes in pixel coordinates.
        image_size: (width, height) pixel dimensions of the receipt.

    Returns:
        One normalized [x1, y1, x2, y2] box per input box.
    """
    width = max(1, image_size[0])
    height = max(1, image_size[1])
    normalized: list[list[int]] = []
    for box in bboxes:
        x1 = int(min(box.x1, box.x2) / width * 1000)
        y1 = int(min(box.y1, box.y2) / height * 1000)
        x2 = int(max(box.x1, box.x2) / width * 1000)
        y2 = int(max(box.y1, box.y2) / height * 1000)
        normalized.append(
            [min(1000, max(0, value)) for value in (x1, y1, x2, y2)]
        )
    return normalized


def first_word_labels(
    word_ids: Sequence[int | None], token_labels: Sequence[int]
) -> dict[int, int]:
    """Map each word to the label of its first sub-token.

    Args:
        word_ids: Per-token word index, None for special tokens.
        token_labels: Per-token predicted label ids.

    Returns:
        A mapping of word index to its first sub-token label id.
    """
    words: dict[int, int] = {}
    for word_id, label_id in zip(word_ids, token_labels):
        if word_id is None:
            continue
        if word_id not in words:
            words[word_id] = label_id
    return words


def build_prediction(
    image_id: int, words: Sequence[WordData]
) -> KiePrediction:
    """Assemble labeled spans and grouped entities from word labels.

    Args:
        image_id: Identifier of the source receipt.
        words: Recognized words with their predicted model labels.

    Returns:
        A KIE prediction with spans and entities. Entities group consecutive
        words of the same category on the same OCR line.
    """
    spans: list[LabeledSpan] = []
    entities: list[KieEntity] = []
    run_category: str | None = None
    run_group: int | None = None
    run_texts: list[str] = []
    for word in words:
        category = _standard_category(word.label)
        if category is None:
            _flush_entity(entities, run_category, run_group, run_texts)
            run_category, run_group, run_texts = None, None, []
            continue
        spans.append(LabeledSpan(category=category, text=word.text))
        if _continues(category, word, run_category, run_group):
            run_texts.append(word.text)
        else:
            _flush_entity(entities, run_category, run_group, run_texts)
            run_category, run_group, run_texts = category, word.line_index, [word.text]
    _flush_entity(entities, run_category, run_group, run_texts)
    return KiePrediction(
        image_id=image_id,
        raw=[word.label for word in words],
        spans=spans,
        entities=entities,
    )


def _standard_category(label: str) -> str | None:
    """Convert a raw model label to a lowercase semantic category.

    Args:
        label: Raw model label such as "B-MENU.NM" or "O".

    Returns:
        The category without BIO prefix lowercased, or None for "O".
    """
    if not label or label == "O":
        return None
    return normalize_category(label).lower()


def _continues(
    category: str,
    word: WordData,
    run_category: str | None,
    run_group: int | None,
) -> bool:
    """Tell whether a word continues the current entity run.

    Args:
        category: Normalized category of the current word.
        word: The word being evaluated.
        run_category: Category of the open entity run.
        run_group: Line index of the open entity run.

    Returns:
        True for an I-/E- continuation of the same category on the same line.
    """
    return (
        word.label.startswith(_CONTINUATION_PREFIXES)
        and category == run_category
        and word.line_index == run_group
    )


def _flush_entity(
    entities: list[KieEntity],
    category: str | None,
    group_id: int | None,
    texts: list[str],
) -> None:
    """Commit the current entity run into the entity list.

    Args:
        entities: Accumulated entities to append to.
        category: Category of the run, None when no run is open.
        group_id: Line index identifying the run group.
        texts: Text values accumulated for the run.
    """
    if category is None or not texts:
        return
    entities.append(
        KieEntity(category=category, text=" ".join(texts), group_id=group_id)
    )


def prediction_for_sample(
    image_id: int,
    words: Sequence[OcrWord],
    encoded: object,
    logits: object,
    sample_index: int,
    model: object,
) -> KiePrediction:
    """Decode the model logits of one batch sample into a prediction.

    Args:
        image_id: Identifier of the source receipt.
        words: Recognized OCR words of the receipt.
        encoded: The batched processor encoding holding per-sample word ids.
        logits: The full-batch model output logits.
        sample_index: Index of the sample within the batch.
        model: The trained token classification model exposing id2label.

    Returns:
        The structured KIE prediction of the sample.
    """
    label_ids = logits[sample_index].argmax(dim=-1).tolist()
    token_labels = [model.config.id2label[index] for index in label_ids]
    word_labels = first_word_labels(encoded.word_ids(sample_index), token_labels)
    word_data = [
        WordData(
            text=word.text,
            line_index=word.line_index
            if word.line_index is not None
            else index,
            label=word_labels[index],
        )
        for index, word in enumerate(words)
        if index in word_labels
    ]
    return build_prediction(image_id, word_data)


class LayoutLmv3KieEngine(KieEngine):
    """Extract semantic entities with a fine-tuned LayoutLMv3 model.

    Args:
        config: LayoutLMv3 engine settings.

    Returns:
        A configured LayoutLMv3 engine ready for prediction.
    """

    def __init__(self, config: KieServiceConfig) -> None:
        self._config = config
        self._processor: LayoutLMv3Processor | None = None
        self._model: LayoutLMv3ForTokenClassification | None = None
        self._device = "cpu"

    def predict(
        self, receipt: CordReceipt, ocr: OcrResult | None = None
    ) -> KiePrediction:
        """Predict semantic entities for one receipt.

        Args:
            receipt: The receipt image and its metadata.
            ocr: PaddleOCR words consumed as token inputs.

        Returns:
            The structured KIE prediction for the receipt.
        """
        ocr_result = (
            ocr if ocr is not None else OcrResult(image_id=receipt.image_id)
        )
        return self.predict_batch([receipt], [ocr_result])[0]

    def predict_batch(
        self,
        receipts: Sequence[CordReceipt],
        ocr_results: Sequence[OcrResult],
    ) -> list[KiePrediction]:
        """Predict semantic entities for receipts in one batched forward pass.

        Receipts without OCR words or images are assigned empty predictions
        and excluded from the batch, so every receipt yields a prediction.

        Args:
            receipts: Receipts to predict.
            ocr_results: Per-receipt OCR words in the same order.

        Returns:
            The KIE predictions in the same order as the input receipts.
        """
        predictions = [
            KiePrediction(image_id=receipt.image_id) for receipt in receipts
        ]
        valid_indexes = [
            index
            for index, (receipt, ocr) in enumerate(zip(receipts, ocr_results))
            if ocr.words and receipt.image is not None
        ]
        if not valid_indexes:
            return predictions
        processor, model, device = self._load()
        encoded = processor(
            images=[receipts[index].image for index in valid_indexes],
            text=[
                [word.text for word in ocr_results[index].words]
                for index in valid_indexes
            ],
            boxes=[
                normalize_boxes(
                    [word.bbox for word in ocr_results[index].words],
                    receipts[index].image_size,
                )
                for index in valid_indexes
            ],
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        )
        encoded = encoded.to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        for batch_index, receipt_index in enumerate(valid_indexes):
            predictions[receipt_index] = prediction_for_sample(
                receipts[receipt_index].image_id,
                ocr_results[receipt_index].words,
                encoded,
                logits,
                batch_index,
                model,
            )
        return predictions

    def _load(self) -> tuple[LayoutLMv3Processor, LayoutLMv3ForTokenClassification, str]:
        """Load the processor and model once on first use.

        Returns:
            The processor, the model in eval mode, and the resolved device.
        """
        if self._model is None:
            tokenizer = LayoutLMv3Tokenizer.from_pretrained(self._config.model_dir)
            image_processor = LayoutLMv3ImageProcessor.from_pretrained(
                self._config.model_dir, apply_ocr=False
            )
            self._processor = LayoutLMv3Processor(
                image_processor=image_processor, tokenizer=tokenizer
            )
            self._device = _resolve_device(self._config.device)
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                self._config.model_dir
            )
            self._model.to(self._device)
            self._model.eval()
        return self._processor, self._model, self._device


def _resolve_device(requested: str) -> str:
    """Resolve the requested device to a concrete torch device.

    Args:
        requested: "auto", "cuda" or "cpu".

    Returns:
        The concrete device, mapping "auto" to CUDA when available.
    """
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"