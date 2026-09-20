"""CORD dataset data transfer objects."""

from pydantic import BaseModel, Field

from src.schemas.bbox import BBox
from src.schemas.ocr import OcrWord


class CordWord(BaseModel):
    """A ground truth word inside a CORD valid line.

    Args:
        text: Raw word text as annotated.
        quad: The four-corner quad point dict from the dataset.
        is_key: Whether the word was annotated as part of a field key.
        row_id: Raw row identifier assigned by the dataset.

    Returns:
        A validated CORD word.
    """

    text: str
    quad: dict[str, float]
    is_key: int = 0
    row_id: int | None = None

    @property
    def bbox(self) -> BBox:
        """Convert the quad to a minimum enclosing bounding box.

        Returns:
            The derived BBox.
        """
        return BBox.from_quad(self.quad)


class CordLine(BaseModel):
    """A semantically labeled group of words on a receipt.

    Args:
        category: Semantic category (e.g. "menu.nm"), empty when unlabeled.
        group_id: Identifier of the group the words belong to.
        sub_group_id: Optional identifier of the sub group within a group.
        words: The words that form this line in reading order.

    Returns:
        A validated CORD line.
    """

    category: str = ""
    group_id: int
    sub_group_id: int = 0
    words: list[CordWord] = Field(default_factory=list)

    def is_labeled(self) -> bool:
        """Tell whether the line carries a semantic category.

        Returns:
            True when the category string is non-empty.
        """
        return bool(self.category)

    def text(self) -> str:
        """Concatenate the line words into a single string.

        Returns:
            The words joined with single spaces.
        """
        return " ".join(word.text for word in self.words)


class GoldToken(BaseModel):
    """A word with its ground truth label used for token-level evaluation.

    Args:
        text: Word text.
        bbox: Word bounding box.
        category: Semantic category of the source line.
        group_id: Group identifier of the source line.
        sub_group_id: Sub group identifier of the source line.

    Returns:
        A validated gold token.
    """

    text: str
    bbox: BBox
    category: str
    group_id: int
    sub_group_id: int = 0


class CordEntity(BaseModel):
    """A ground truth semantic entity assembled from a CORD line.

    Args:
        category: Semantic category of the entity.
        group_id: Group identifier used to disambiguate repeated values.
        text: Space-joined entity text.

    Returns:
        A validated CORD entity.
    """

    category: str
    group_id: int
    text: str


class CordReceipt(BaseModel):
    """A parsed CORD dataset row.

    Args:
        image_id: Identifier of the receipt.
        split: Dataset split this receipt belongs to.
        version: Annotation format version.
        image_size: (width, height) pixel dimensions.
        lines: Semantically labeled valid lines.
        gt_parse: Hierarchical ground truth key-value tree.
        dontcare: Word indices explicitly marked as don't-care.
        repeating_symbol: Word indices marked as repeating symbols.
        roi: Raw region of interest annotations, optional.
        image: Decoded receipt image, optional.

    Returns:
        A validated CORD receipt.
    """

    image_id: int
    split: str
    version: str | None = None
    image_size: tuple[int, int]
    lines: list[CordLine]
    gt_parse: dict
    dontcare: list[list[dict[str, float]]] = Field(default_factory=list)
    repeating_symbol: list[list[dict[str, object]]] = Field(default_factory=list)
    roi: dict | None = None
    image: object | None = Field(default=None, exclude=True)

    def tokens(
        self, ignored_categories: set[str] | None = None
    ) -> list[GoldToken]:
        """Flatten labeled lines into gold tokens per word.

        Args:
            ignored_categories: Semantic categories to exclude from the output.

        Returns:
            A list of gold tokens, one per word of labeled lines.
        """
        ignored = ignored_categories or set()
        tokens: list[GoldToken] = []
        for line in self.lines:
            if not line.is_labeled() or line.category in ignored:
                continue
            for word in line.words:
                tokens.append(
                    GoldToken(
                        text=word.text,
                        bbox=word.bbox,
                        category=line.category,
                        group_id=line.group_id,
                        sub_group_id=line.sub_group_id,
                    )
                )
        return tokens

    def entities(
        self, ignored_categories: set[str] | None = None
    ) -> list[CordEntity]:
        """Build ground truth entities from labeled lines.

        Args:
            ignored_categories: Semantic categories to exclude from the output.

        Returns:
            A list of entities, one per labeled line excluding ignored ones.
        """
        ignored = ignored_categories or set()
        return [
            CordEntity(
                category=line.category, group_id=line.group_id, text=line.text()
            )
            for line in self.lines
            if line.is_labeled() and line.category not in ignored
        ]

    def ocr_words(self) -> list[OcrWord]:
        """Build the OCR detection and recognition ground truth.

        Returns:
            An OcrWord per raw word, with default confidence of 1.0.
        """
        return [
            OcrWord(text=word.text, bbox=word.bbox, confidence=1.0)
            for line in self.lines
            for word in line.words
        ]
