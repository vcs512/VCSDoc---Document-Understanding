"""Streaming loader for the CORD dataset."""

import io
import json
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from src.schemas.config import CordConfig
from src.schemas.cord import CordLine, CordReceipt, CordWord

_SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


class CordDataset:
    """Streaming access to a CORD dataset split.

    Args:
        config: CORD dataset configuration.
        split: Dataset split to load (train, validation or test).
        load_images: Whether to decode and keep the receipt images.

    Returns:
        A dataset object ready to iterate over parsed receipts.
    """

    def __init__(self, config: CordConfig, split: str, load_images: bool = True) -> None:
        self._split = _SPLIT_ALIASES.get(split, split)
        self._config = config
        self._load_images = load_images
        self._dataset = load_dataset(
            config.dataset_id,
            split=self._split,
            cache_dir=str(Path(config.cache_dir).resolve()),
        )

    def __len__(self) -> int:
        """Return the number of receipts in the split.

        Returns:
            The split size.
        """
        return len(self._dataset)

    def __iter__(self) -> Iterator[CordReceipt]:
        """Yield the parsed receipts of the split.

        Yields:
            One parsed CORD receipt per dataset row.
        """
        for row in self._dataset:
            yield self._parse_row(row)

    def receipt(self, index: int) -> CordReceipt:
        """Return a single receipt by row index.

        Args:
            index: Zero based row index within the split.

        Returns:
            The parsed receipt at the requested index.
        """
        return self._parse_row(self._dataset[index])

    def _parse_row(self, row: dict) -> CordReceipt:
        """Parse one raw dataset row into a structured receipt.

        Args:
            row: A dataset row with image and ground_truth fields.

        Returns:
            The parsed CORD receipt.
        """
        image = self._decode_image(row.get("image")) if self._load_images else None
        payload = json.loads(row["ground_truth"])
        meta = payload["meta"]
        image_size = meta["image_size"]
        return CordReceipt(
            image_id=int(meta["image_id"]),
            split=meta.get("split", self._split),
            version=meta.get("version"),
            image_size=(int(image_size["width"]), int(image_size["height"])),
            lines=[self._parse_line(line) for line in payload.get("valid_line", [])],
            gt_parse=payload.get("gt_parse", {}),
            dontcare=payload.get("dontcare", []),
            repeating_symbol=payload.get("repeating_symbol", []),
            roi=payload.get("roi"),
            image=image,
        )

    @staticmethod
    def _parse_line(line: dict) -> CordLine:
        """Parse one valid line entry.

        Args:
            line: Raw valid_line entry dict.

        Returns:
            The parsed CORD line.
        """
        return CordLine(
            category=line.get("category", ""),
            group_id=int(line.get("group_id", 0)),
            sub_group_id=int(line.get("sub_group_id", 0)),
            words=[
                CordWord(
                    text=word["text"],
                    quad=word["quad"],
                    is_key=int(word.get("is_key", 0)), row_id=word.get("row_id")
                )
                for word in line.get("words", [])
            ],
        )

    @staticmethod
    def _decode_image(image: object) -> Image.Image | None:
        """Decode the row image into a PIL image regardless of transport.

        Args:
            image: A PIL image, an image dict or raw bytes.

        Returns:
            The decoded PIL image, or None when image is None.
        """
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, dict):
            return Image.open(io.BytesIO(image.get("bytes", b"")))
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        return Image.open(image)