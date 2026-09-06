"""Shared pytest fixtures for the CORD core tests."""

import json
from pathlib import Path

import pytest

from src.schemas.cord import (
    CordLine,
    CordReceipt,
    CordWord,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def parse_ground_truth(payload: dict) -> CordReceipt:
    """Build a parsed receipt from a raw ground truth dict.

    Mirrors the parsing done by the CordDataset loader.

    Args:
        payload: The decoded ground truth dict.

    Returns:
        The parsed CORD receipt.
    """
    meta = payload["meta"]
    image_size = meta["image_size"]
    return CordReceipt(
        image_id=int(meta["image_id"]),
        split=meta["split"],
        version=meta["version"],
        image_size=(image_size["width"], image_size["height"]),
        lines=[
            CordLine(
                category=line.get("category", ""),
                group_id=int(line.get("group_id", 0)),
                sub_group_id=int(line.get("sub_group_id", 0)),
                words=[
                    CordWord(text=word["text"], quad=word["quad"])
                    for word in line.get("words", [])
                ],
            )
            for line in payload.get("valid_line", [])
        ],
        gt_parse=payload.get("gt_parse", {}),
        dontcare=payload.get("dontcare", []),
        repeating_symbol=payload.get("repeating_symbol", []),
        roi=payload.get("roi"),
    )


@pytest.fixture
def ground_truth_payload() -> dict:
    """Load the synthetic CORD ground truth fixture.

    Returns:
        The decoded ground truth dict.
    """
    fixture = FIXTURES_DIR / "ground_truth.json"
    with open(fixture, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def receipt(ground_truth_payload: dict) -> CordReceipt:
    """Build a parsed receipt from the fixture payload.

    Args:
        ground_truth_payload: The decoded ground truth dict.

    Returns:
        The parsed CORD receipt.
    """
    return parse_ground_truth(ground_truth_payload)
