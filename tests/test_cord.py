"""Tests for CORD receipt parsing and gold token extraction."""


from src.schemas.cord import CordReceipt


def test_entity_count(receipt: CordReceipt) -> None:
    """A labeled receipt should produce one entity per labeled line."""
    assert len(receipt.entities()) == 3


def test_token_count(receipt: CordReceipt) -> None:
    """Tokens should be produced only for labeled lines."""
    assert len(receipt.tokens()) == 4


def test_empty_category_excluded_from_entities(receipt: CordReceipt) -> None:
    """Lines without a category must not become entities."""
    categories = {entity.category for entity in receipt.entities()}
    assert all(category != "" for category in categories)


def test_ocr_words_includes_unlabeled_lines(receipt: CordReceipt) -> None:
    """OCR ground truth words should come from all lines regardless of label."""
    assert len(receipt.ocr_words()) == 5


def test_ignore_categories_filters_tokens(receipt: CordReceipt) -> None:
    """Passing ignore categories should exclude matching tokens."""
    tokens = receipt.tokens(ignored_categories={"menu.cnt"})
    assert len(tokens) == 3
    assert all(token.category != "menu.cnt" for token in tokens)


def test_entities_sorted_by_group_id(receipt: CordReceipt) -> None:
    """Entities are returned in line order which tracks group ids."""
    group_ids = [entity.group_id for entity in receipt.entities()]
    assert group_ids[0] == 12
    assert group_ids[1] == 12
    assert group_ids[2] == 5


def test_receipt_metadata(receipt: CordReceipt) -> None:
    """The receipt should faithfully expose its metadata."""
    assert receipt.image_id == 0
    assert receipt.image_size == (100, 100)
    assert receipt.split == "train"
    assert "menu" in receipt.gt_parse
