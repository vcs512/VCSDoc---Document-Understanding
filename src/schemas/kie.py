"""KIE prediction data transfer objects."""

from pydantic import BaseModel


class LabeledSpan(BaseModel):
    """A text value paired with its semantic category.

    Args:
        category: Semantic category path (e.g. "sub_total.subtotal_price").
        text: The text value associated with the category.

    Returns:
        A validated labeled span.
    """

    category: str
    text: str


class KieEntity(BaseModel):
    """A predicted semantic entity.

    Args:
        category: Semantic category of the entity.
        text: Entity text.
        group_id: Optional group identifier, populated by token-based models.

    Returns:
        A validated KIE entity.
    """

    category: str
    text: str
    group_id: int | None = None


class KiePrediction(BaseModel):
    """The structured prediction of a KIE model for one receipt.

    Args:
        image_id: Identifier of the source receipt.
        raw: Optional raw model output (token logits, JSON tree, etc.).
        spans: Labeled spans used for token-level evaluation.
        entities: Entities used for semantic entity recognition evaluation.

    Returns:
        A validated KIE prediction.
    """

    image_id: int
    raw: object | None = None
    spans: list[LabeledSpan] = []
    entities: list[KieEntity] = []
