"""Flattening of hierarchical KIE outputs into labeled spans."""

from collections.abc import Mapping, Sequence

from src.schemas.kie import LabeledSpan


class JsonTreeFlattener:
    """Flatten a gt_parse-style tree into a flat list of labeled spans.

    The resulting category path joins the tree keys (and list indexes) with
    dots, e.g. "menu.0.nm" or "sub_total.subtotal_price".
    """

    @classmethod
    def flatten(cls, tree: Mapping) -> list[LabeledSpan]:
        """Flatten a hierarchical mapping into labeled spans.

        Args:
            tree: Nested dict/list value tree such as gt_parse.

        Returns:
            A list of labeled spans ordered by tree traversal.
        """
        return cls._flatten_value(tree, [])

    @classmethod
    def _flatten_value(
        cls, value: object, path: list[str]
    ) -> list[LabeledSpan]:
        """Recursively flatten one node of the value tree.

        Args:
            value: Dict, list or scalar node to flatten.
            path: Accumulated category path segments.

        Returns:
            The labeled spans found below this node.
        """
        if isinstance(value, Mapping):
            return [
                span
                for key, child in value.items()
                for span in cls._flatten_value(child, [*path, str(key)])
            ]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [
                span
                for index, child in enumerate(value)
                for span in cls._flatten_value(child, [*path, str(index)])
            ]
        return [LabeledSpan(category=".".join(path), text=str(value))]
