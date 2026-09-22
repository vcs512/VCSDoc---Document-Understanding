"""Flattening and persistence of evaluation reports."""

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import cast

_CSV_ORDER = ("detection", "recognition", "token_f1", "ser")

_CSV_FIELDS = [
    "model",
    "split",
    "metric",
    "category",
    "precision",
    "recall",
    "f1",
    "value",
]

_SCALAR_KEYS = {
    "detection": (
        "average_precision",
        "mean_iou",
        "precision",
        "recall",
        "num_gt",
        "num_pred",
    ),
    "recognition": ("cer", "wer", "num_gt", "num_pred"),
}


def flatten_report(report: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten an evaluation report into spreadsheet-friendly rows.

    Args:
        report: The serialized evaluation report mapping.

    Returns:
        Tidy rows with the columns model, split, metric, category, precision,
        recall, f1 and value.
    """
    rows: list[dict[str, object]] = []
    metrics = cast(Mapping[str, Mapping[str, object]], report["metrics"])
    for name in _CSV_ORDER:
        if name not in metrics:
            continue
        payload = metrics[name]
        if name in ("token_f1", "ser"):
            rows.extend(_flatten_f1(report, name, payload))
        else:
            rows.extend(_flatten_scalar(report, name, payload))
    return rows


def resolve_output(output: str, limit: int | None) -> Path:
    """Resolve the report output path, optionally suffixed with the limit.

    Args:
        output: Output path as configured by the service.
        limit: Optional receipt limit evaluated by the service.

    Returns:
        The resolved output path.
    """
    path = Path(output).resolve()
    if limit is not None:
        stem = f"{path.stem}-limit{limit}"
        path = path.with_name(f"{stem}{path.suffix}")
    return path


def write_csv(output: Path, rows: list[dict[str, object]]) -> None:
    """Write flattened report rows to a CSV file.

    Args:
        output: Destination CSV path.
        rows: Flattened report rows.
    """
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _flatten_f1(
    report: Mapping[str, object],
    name: str,
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    """Flatten an F1 metric into overall, per-class and macro rows.

    Args:
        report: The serialized evaluation report mapping.
        name: The metric identifier.
        payload: The F1 metric payload.

    Returns:
        One row per scope with precision, recall and f1.
    """
    rows = [_f1_row(report, name, "overall", payload["overall"])]
    per_class = cast(Mapping[str, object], payload.get("per_class", {}))
    rows.extend(
        _f1_row(report, name, category, score)
        for category, score in sorted(per_class.items())
    )
    macro = payload.get("macro")
    if macro is not None:
        rows.append(_f1_row(report, name, "macro", macro))
    return rows


def _f1_row(
    report: Mapping[str, object],
    name: str,
    category: str,
    score: object,
) -> dict[str, object]:
    """Build one CSV row from an F1 score triplet.

    Args:
        report: The serialized evaluation report mapping.
        name: The metric identifier.
        category: The row scope, overall, macro or a category name.
        score: The score triplet with precision, recall and f1.

    Returns:
        A row with the score columns populated and value left empty.
    """
    triplet = cast(Mapping[str, float], score)
    return {
        "model": report["model"],
        "split": report["split"],
        "metric": name,
        "category": category,
        "precision": round(triplet["precision"], 6),
        "recall": round(triplet["recall"], 6),
        "f1": round(triplet["f1"], 6),
        "value": None,
    }


def _flatten_scalar(
    report: Mapping[str, object],
    name: str,
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    """Flatten detection or recognition metrics into scalar rows.

    Args:
        report: The serialized evaluation report mapping.
        name: The metric identifier.
        payload: The detection or recognition metric payload.

    Returns:
        One row per scalar metric with the value column populated.
    """
    rows: list[dict[str, object]] = []
    for key in _SCALAR_KEYS[name]:
        value = payload[key]
        rows.append(
            {
                "model": report["model"],
                "split": report["split"],
                "metric": name,
                "category": key,
                "precision": None,
                "recall": None,
                "f1": None,
                "value": round(value, 6) if isinstance(value, float) else value,
            }
        )
    return rows
