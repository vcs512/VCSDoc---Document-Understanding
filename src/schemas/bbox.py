"""Axis-aligned bounding box shared by OCR and KIE pipelines."""

from functools import cached_property

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned bounding box in image pixel coordinates.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        A validated bounding box with equal access to coordinate pairs.
    """

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @classmethod
    def from_quad(cls, quad: dict[str, float]) -> "BBox":
        """Build the minimum enclosing box from a CORD quad point dict.

        Args:
            quad: Mapping with x1, y1, x2, y2, x3, y3, x4, y4 point keys.

        Returns:
            A BBox with min/max coordinates over the eight points.
        """
        xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
        ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
        return cls(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

    @cached_property
    def area(self) -> float:
        """Compute the box area.

        Returns:
            The area in squared pixels.
        """
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def intersection(self, other: "BBox") -> float:
        """Compute the intersection area with another box.

        Args:
            other: The other bounding box.

        Returns:
            The overlapping area in squared pixels.
        """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(self, other: "BBox") -> float:
        """Compute the intersection over union with another box.

        Args:
            other: The other bounding box.

        Returns:
            A value in the closed range [0, 1], 1.0 for identical boxes.
        """
        overlap = self.intersection(other)
        union = self.area + other.area - overlap
        if union <= 0:
            return 0.0
        return overlap / union
