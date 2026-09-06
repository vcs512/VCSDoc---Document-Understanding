"""Tests for the entity-level SER metric."""

import pytest

from src.core.evaluation.ser import entity_level_f1
from src.schemas.cord import CordEntity
from src.schemas.kie import KieEntity


def make_gold() -> list[CordEntity]:
    """Build a small gold entity set.

    Returns:
        A list of ground truth entities.
    """
    return [
        CordEntity(category="menu.nm", group_id=1, text="Nasi Campur"),
        CordEntity(category="menu.price", group_id=1, text="75,000"),
        CordEntity(category="total.total_price", group_id=0, text="75,000"),
    ]


def test_perfect_match() -> None:
    """Identical gold and predicted entities should yield F1 of 1.0."""
    gold = make_gold()
    predicted = [
        KieEntity(category="menu.nm", text="Nasi Campur"),
        KieEntity(category="menu.price", text="75,000"),
        KieEntity(category="total.total_price", text="75,000"),
    ]
    result = entity_level_f1(gold, predicted)
    assert result.overall.f1 == pytest.approx(1.0)


def test_deduplicated_title_price() -> None:
    """Two entities sharing the same text/category may merge in predictions.

    When a model predicts the total price as a menu price, the micro F1 drops.
    """
    gold = make_gold()
    predicted = [
        KieEntity(category="menu.nm", text="Nasi Campur"),
        KieEntity(category="total.total_price", text="75,000"),
    ]
    result = entity_level_f1(gold, predicted)
    assert result.overall.recall == pytest.approx(2 / 3)
    assert result.overall.precision == pytest.approx(1.0)


def test_missing_and_extra_entities() -> None:
    """Missing gold entities and extra predictions should affect both sides."""
    gold = [
        CordEntity(category="a", group_id=0, text="x"),
        CordEntity(category="a", group_id=0, text="y"),
    ]
    predicted = [
        KieEntity(category="a", text="x"),
        KieEntity(category="a", text="z"),
        KieEntity(category="b", text="w"),
    ]
    result = entity_level_f1(gold, predicted)
    assert result.overall.precision == pytest.approx(1 / 3)
    assert result.overall.recall == pytest.approx(0.5)


def test_empty_inputs() -> None:
    """Empty inputs should produce all-zero scores."""
    result = entity_level_f1([], [])
    assert result.overall.f1 == pytest.approx(0.0)
    assert result.per_class == {}


def test_group_id_does_not_affect_matching() -> None:
    """Group ids on gold entities are not used for matching."""
    gold = [
        CordEntity(category="menu.nm", group_id=5, text="Nasi"),
        CordEntity(category="menu.nm", group_id=9, text="Nasi"),
    ]
    predicted = [KieEntity(category="menu.nm", text="Nasi")]
    result = entity_level_f1(gold, predicted)
    assert result.overall.recall == pytest.approx(0.5)
