"""Levenshtein edit distance for CER/WER computation."""

from collections.abc import Sequence


def edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    """Compute the Levenshtein edit distance between two sequences.

    Works on character strings and on any sequence of tokens.

    Args:
        source: First sequence.
        target: Second sequence.

    Returns:
        The minimum number of insertions, deletions and substitutions.
    """
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            substitution = previous[target_index - 1] + int(
                source_item != target_item
            )
            deletion = previous[target_index] + 1
            insertion = current[target_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]
