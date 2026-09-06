"""Tests for the JSON tree flattening utility."""

from src.core.serialization import JsonTreeFlattener
from src.schemas.kie import LabeledSpan


def test_flatten_nested_dict() -> None:
    """Nested dicts should be flattened with dot-joined category paths."""
    tree = {"sub_total": {"subtotal_price": "100", "tax_price": "10"}}
    spans = JsonTreeFlattener.flatten(tree)
    assert spans == [
        LabeledSpan(category="sub_total.subtotal_price", text="100"),
        LabeledSpan(category="sub_total.tax_price", text="10"),
    ]


def test_flatten_list_of_dicts() -> None:
    """List entries should have their zero-based index inserted into the path."""
    tree = {"menu": [{"nm": "Nasi", "price": "5000"}]}
    spans = JsonTreeFlattener.flatten(tree)
    assert spans == [
        LabeledSpan(category="menu.0.nm", text="Nasi"),
        LabeledSpan(category="menu.0.price", text="5000"),
    ]


def test_flatten_multiple_list_entries() -> None:
    """Each list entry should receive its own index segment."""
    tree = {"menu": [{"nm": "A"}, {"nm": "B"}]}
    spans = JsonTreeFlattener.flatten(tree)
    assert len(spans) == 2
    assert spans[0].category == "menu.0.nm"
    assert spans[1].category == "menu.1.nm"


def test_flatten_scalar() -> None:
    """A top-level scalar value should produce a single span."""
    tree = {"total_price": "50000"}
    spans = JsonTreeFlattener.flatten(tree)
    assert spans == [LabeledSpan(category="total_price", text="50000")]


def test_flatten_empty_tree() -> None:
    """An empty dict should produce an empty list."""
    assert JsonTreeFlattener.flatten({}) == []


def test_flatten_empty_list_value() -> None:
    """A dict containing an empty list should produce no spans for that list."""
    tree = {"items": [], "price": "10"}
    spans = JsonTreeFlattener.flatten(tree)
    assert spans == [LabeledSpan(category="price", text="10")]


def test_flatten_ordering() -> None:
    """Spans should be emitted in dict traversal order."""
    tree = {"z": {"x": "1"}, "a": "2"}
    spans = JsonTreeFlattener.flatten(tree)
    assert [s.category for s in spans] == ["z.x", "a"]


def test_flatten_receipt_gt_parse() -> None:
    """The synthetic fixture gt_parse should flatten to the expected paths."""
    tree = {
        "menu": [{"nm": "Nasi Campur", "cnt": "1"}],
        "total": {"total_price": "75,000"},
    }
    spans = JsonTreeFlattener.flatten(tree)
    categories = [span.category for span in spans]
    assert "menu.0.nm" in categories
    assert "menu.0.cnt" in categories
    assert "total.total_price" in categories
    assert len(spans) == 3
