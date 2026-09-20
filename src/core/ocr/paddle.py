"""PaddleOCR engine for the evaluation services."""

import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import paddle
from paddleocr import PaddleOCR

from src.core.ocr.base import OcrEngine
from src.schemas.bbox import BBox
from src.schemas.cord import CordReceipt
from src.schemas.ocr import OcrResult, OcrWord
from src.schemas.service import OcrServiceConfig

_ENGINE_CREATION_LOCK = threading.Lock()


def _cap_intraop_threads() -> None:
    """Cap Paddle intra-op threading to avoid oversubscription.

    Each parallel OCR worker runs its own engine, so every engine would
    otherwise spin up one thread per CPU core. The call is thread-scoped and
    only applied when the underlying Paddle version exposes it.
    """
    core = getattr(paddle.framework, "core", None)
    set_num_threads = getattr(core, "set_num_threads", None)
    if set_num_threads is not None:
        set_num_threads(1)


class PaddleOcrEngine(OcrEngine):
    """Detect and recognize receipt text with PaddleOCR.

    The engine disables the document preprocessor and unwarping so that the
    returned boxes keep the original image coordinates used by the gold
    annotations.

    Args:
        config: PaddleOCR engine settings.

    Returns:
        A configured PaddleOCR engine ready for recognition.
    """

    def __init__(self, config: OcrServiceConfig) -> None:
        self._config = config
        self._ocr: object | None = None
        self._local = threading.local()

    def recognize(self, receipt: CordReceipt) -> OcrResult:
        """Detect words and recognize their text on one receipt.

        Args:
            receipt: The receipt image and its metadata.

        Returns:
            The recognized words ordered by PaddleOCR reading order.
        """
        if receipt.image is None:
            return OcrResult(image_id=receipt.image_id, words=[])
        return self._to_result(receipt.image_id, np.asarray(receipt.image))

    def recognize_batch(
        self,
        receipts: Iterable[CordReceipt],
        workers: int | None = None,
    ) -> list[OcrResult]:
        """Recognize a batch of receipts, parallelizing with per-thread engines.

        Each worker thread builds its own PaddleOCR instance, since the
        classic 2.x predictors are not safe to share across threads. When
        parallelism is disabled or the engine targets the GPU, the batch is
        processed sequentially.

        Args:
            receipts: Receipts to recognize.
            workers: Maximum worker threads, None or one disables parallelism.

        Returns:
            The OCR results in the same order as the input receipts.
        """
        chunk = list(receipts)
        if (
            workers is None
            or workers < 2
            or len(chunk) < 2
            or self._config.device != "cpu"
        ):
            return [self.recognize(receipt) for receipt in chunk]
        workers = min(workers, len(chunk))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.recognize, chunk))

    def _to_result(self, image_id: int, image: object) -> OcrResult:
        """Run the active OCR API family on the image and parse its payload.

        Args:
            image_id: Identifier of the source receipt.
            image: The receipt image as a numpy array.

        Returns:
            The OCR result of the active API family.
        """
        if self._uses_classic_api():
            payload = self._engine().ocr(image, cls=False)
            return PaddleOcrEngine.to_result_classic(image_id, payload)
        payload = self._engine().predict(image)[0]
        return PaddleOcrEngine.to_result(image_id, payload)

    def _uses_classic_api(self) -> bool:
        """Resolve the API family to a concrete classic-vs-paddlex choice.

        Returns:
            True for the classic 2.x API, False for the paddlex 3.x API.
        """
        if self._config.api == "classic2":
            return True
        if self._config.api == "paddlex3":
            return False
        return not hasattr(PaddleOCR, "predict")

    def _engine(self) -> object:
        """Build the PaddleOCR pipeline once per thread on first use.

        The classic 2.x path creates one instance per thread, serializing
        creation behind a global lock for safe Paddle initialization and
        capping each engine to one intra-op thread. The paddlex 3.x path
        shares a single pipeline instance.

        Returns:
            The lazily constructed PaddleOCR instance for the calling thread.
        """
        if self._uses_classic_api():
            return self._classic_engine()
        if self._ocr is None:
            self._ocr = PaddleOCR(
                lang=self._config.lang,
                return_word_box=self._config.return_word_box,
                use_doc_orientation_classify=(
                    self._config.use_doc_orientation_classify
                ),
                use_doc_unwarping=self._config.use_doc_unwarping,
                use_textline_orientation=(self._config.use_textline_orientation),
                device=self._config.device,
            )
        return self._ocr

    def _classic_engine(self) -> object:
        """Build a per-thread classic PaddleOCR instance on first use.

        Returns:
            The classic 2.x engine bound to the calling thread.
        """
        engine = getattr(self._local, "engine", None)
        if engine is None:
            with _ENGINE_CREATION_LOCK:
                engine = getattr(self._local, "engine", None)
                if engine is None:
                    kwargs: dict[str, object] = {}
                    if self._config.det_model_dir:
                        kwargs["det_model_dir"] = str(
                            Path(self._config.det_model_dir).expanduser()
                        )
                    engine = PaddleOCR(
                        lang=self._config.lang,
                        use_angle_cls=self._config.use_textline_orientation,
                        show_log=False,
                        **kwargs,
                    )
                    _cap_intraop_threads()
                    self._local.engine = engine
        return engine

    @staticmethod
    def to_result_classic(image_id: int, payload: object) -> OcrResult:
        """Convert a classic 2.x OCR output into an OcrResult.

        Args:
            image_id: Identifier of the source receipt.
            payload: Classic output, a per-page list of line entries or None.

        Returns:
            An OcrResult with one OcrWord per detected line.
        """
        words: list[OcrWord] = []
        lines = (payload or [[]])[0]
        for line_index, line in enumerate(lines):
            if not line or len(line) < 2:
                continue
            box, detail = line[0], line[1]
            words.append(
                OcrWord(
                    text=detail[0],
                    bbox=PaddleOcrEngine._bbox(box),
                    confidence=float(detail[1]),
                    line_index=line_index,
                )
            )
        return OcrResult(image_id=image_id, words=words)

    @staticmethod
    def to_result(image_id: int, payload: dict) -> OcrResult:
        """Convert a PaddleOCR output payload into an OcrResult.

        Args:
            image_id: Identifier of the source receipt.
            payload: One PaddleOCR prediction entry.

        Returns:
            An OcrResult, using word-level boxes when available and falling
            back to line-level boxes otherwise.
        """
        scores = payload.get("rec_scores", [])
        if not scores:
            return OcrResult(image_id=image_id, words=[])
        if (
            "text_word" in payload
            and "text_word_boxes" in payload
            and len(payload["text_word"][0]) > 0
        ):
            return PaddleOcrEngine._parse_word_boxes(image_id, payload)
        return PaddleOcrEngine._parse_line_boxes(image_id, payload)

    @staticmethod
    def _parse_word_boxes(image_id: int, payload: dict) -> OcrResult:
        """Build an OcrResult from per-word box payload entries.

        Args:
            image_id: Identifier of the source receipt.
            payload: PaddleOCR prediction with text_word and text_word_boxes.

        Returns:
            One OcrWord per recognized word with its source line index.
        """
        words: list[OcrWord] = []
        for line_index, (score, line_words, line_boxes) in enumerate(
            zip(
                payload["rec_scores"],
                payload["text_word"],
                payload["text_word_boxes"],
            )
        ):
            for text, box in zip(line_words, line_boxes):
                words.append(
                    OcrWord(
                        text=text,
                        bbox=PaddleOcrEngine._bbox(box),
                        confidence=float(score),
                        line_index=line_index,
                    )
                )
        return OcrResult(image_id=image_id, words=words)

    @staticmethod
    def _parse_line_boxes(image_id: int, payload: dict) -> OcrResult:
        """Build an OcrResult from line-level box payload entries.

        Args:
            image_id: Identifier of the source receipt.
            payload: PaddleOCR prediction with rec_texts and rec_boxes.

        Returns:
            One OcrWord per recognized line with the line as a single word.
        """
        words: list[OcrWord] = []
        for line_index, (score, text, box) in enumerate(
            zip(
                payload["rec_scores"],
                payload["rec_texts"],
                payload["rec_boxes"],
            )
        ):
            words.append(
                OcrWord(
                    text=text,
                    bbox=PaddleOcrEngine._bbox(box),
                    confidence=float(score),
                    line_index=line_index,
                )
            )
        return OcrResult(image_id=image_id, words=words)

    @staticmethod
    def _bbox(box: Sequence[float] | np.ndarray) -> BBox:
        """Build a bounding box from a raw box representation.

        Args:
            box: A flat or nested box representation.

        Returns:
            The axis-aligned bounding box.
        """
        x1, y1, x2, y2 = PaddleOcrEngine._flatten_box(box)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _flatten_box(box: Sequence[float] | np.ndarray) -> list[float]:
        """Convert a box to a flat list of axis-aligned coordinates.

        Args:
            box: A flat 4-length box, a paired-corner box or a polygon box.

        Returns:
            The axis-aligned box as four float coordinates.
        """
        if isinstance(box, np.ndarray):
            box = box.tolist()
        if box and all(isinstance(point, (list, tuple)) for point in box):
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            return [min(xs), min(ys), max(xs), max(ys)]
        return [float(value) for value in box]