"""Tests for the token-level F1 metric."""

import pytest

from src.core.evaluation.common import (
    normalize_category,
    normalize_text,
)
from src.core.evaluation.token_f1 import (
    token_level_f1,
)
from src.schemas.kie import LabeledSpan


def test_perfect_match() -> None:
    """Identical gold and predicted spans should yield F1 of 1.0."""
    spans = [LabeledSpan(category="menu.nm", text="Nasi")]
    result = token_level_f1(spans, spans)
    assert result.overall.f1 == pytest.approx(1.0)
    assert result.overall.precision == pytest.approx(1.0)
    assert result.overall.recall == pytest.approx(1.0)


def test_partial_match() -> None:
    """Partially matching spans should yield intermediate scores."""
    gold = [
        LabeledSpan(category="sub_total.subtotal_price", text="100"),
        LabeledSpan(category="menu.0.nm", text="Nasi"),
        LabeledSpan(category="menu.0.price", text="5000"),
    ]
    predicted = [
        LabeledSpan(category="sub_total.subtotal_price", text="100"),
        LabeledSpan(category="menu.0.nm", text="Nasi"),
        LabeledSpan(category="menu.0.price", text="5001"),
        LabeledSpan(category="total.total_price", text="100"),
    ]
    result = token_level_f1(gold, predicted)
    assert result.overall.precision == pytest.approx(0.5)
    assert result.overall.recall == pytest.approx(2 / 3)
    assert result.overall.f1 == pytest.approx(4 / 7)


def test_zero_scores_on_total_mismatch() -> None:
    """No overlapping spans should yield all zeros."""
    gold = [LabeledSpan(category="a", text="1")]
    predicted = [LabeledSpan(category="b", text="2")]
    result = token_level_f1(gold, predicted)
    assert result.overall.f1 == pytest.approx(0.0)
    assert result.overall.precision == pytest.approx(0.0)
    assert result.overall.recall == pytest.approx(0.0)


def test_macro_average() -> None:
    """Macro F1 should be the unweighted mean of per-class F1s."""
    gold = [
        LabeledSpan(category="a", text="1"),
        LabeledSpan(category="b", text="2"),
    ]
    predicted = [
        LabeledSpan(category="a", text="1"),
        LabeledSpan(category="b", text="3"),
    ]
    result = token_level_f1(gold, predicted)
    per_class_f1 = (result.per_class["a"].f1 + result.per_class["b"].f1) / 2
    assert result.macro is not None
    assert result.macro.f1 == pytest.approx(per_class_f1)


def test_category_prefix_stripping() -> None:
    """B/I prefixes in categories should be stripped before comparison."""
    gold = [LabeledSpan(category="B-menu.nm", text="Nasi")]
    predicted = [LabeledSpan(category="I-menu.nm", text="Nasi")]
    result = token_level_f1(gold, predicted)
    assert result.overall.f1 == pytest.approx(1.0)


def test_text_whitespace_normalization() -> None:
    """Extra whitespace in text values should be collapsed."""
    gold = [LabeledSpan(category="a", text="Hello  World")]
    predicted = [LabeledSpan(category="a", text="Hello World")]
    result = token_level_f1(gold, predicted)
    assert result.overall.f1 == pytest.approx(1.0)


def test_empty_inputs() -> None:
    """Both inputs empty should yield all zeros without errors."""
    result = token_level_f1([], [])
    assert result.overall.f1 == pytest.approx(0.0)
    assert result.per_class == {}


def test_normalization_helpers() -> None:
    """Helper normalization functions should strip tags and collapse spaces."""
    assert normalize_category("B-menu.nm") == "menu.nm"
    assert normalize_category("I-menu.price") == "menu.price"
    assert normalize_text("  a   b  ") == "a b"


def test_per_class_scores() -> None:
    """Per-class scores should reflect the balance of that category."""
    gold = [
        LabeledSpan(category="a", text="1"),
        LabeledSpan(category="a", text="2"),
        LabeledSpan(category="b", text="3"),
    ]
    predicted = [
        LabeledSpan(category="a", text="1"),
        LabeledSpan(category="b", text="4"),
    ]
    result = token_level_f1(gold, predicted)
    assert result.per_class["a"].f1 == pytest.approx(2 / 3)
    assert result.per_class["b"].f1 == pytest.approx(0.0)
