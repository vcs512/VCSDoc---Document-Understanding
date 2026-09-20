"""Evaluate the PaddleOCR + LayoutLMv3 pipeline on the CORD dataset."""

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from src.core.config import (
    load_cord_config,
    load_evaluation_config,
    load_layoutlmv3_service_config,
)
from src.core.data.cord import CordDataset
from src.core.kie.layoutlmv3 import LayoutLmv3KieEngine
from src.core.ocr.paddle import PaddleOcrEngine
from src.services.evaluate import Evaluator

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


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR + LayoutLMv3 on the CORD dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/evaluate_layoutlmv3.json",
        help="Service configuration file.",
    )
    parser.add_argument(
        "--cord-config",
        default="configs/cord.json",
        help="CORD dataset configuration file.",
    )
    parser.add_argument(
        "--evaluation-config",
        default="configs/evaluation.json",
        help="Evaluation metrics configuration file.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to evaluate, overriding the service config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many receipts.",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    """Run the evaluation and return the serialized report.

    Args:
        args: Parsed command line arguments.

    Returns:
        The serialized evaluation report mapping.
    """
    service_config = load_layoutlmv3_service_config(args.config)
    cord_config = load_cord_config(args.cord_config)
    evaluation_config = load_evaluation_config(args.evaluation_config)
    split = args.split or service_config.split
    dataset = CordDataset(cord_config, split, load_images=True)
    ocr = PaddleOcrEngine(service_config.ocr)
    kie = LayoutLmv3KieEngine(service_config.kie)
    evaluator = Evaluator(ocr, kie, evaluation_config)
    report = evaluator.evaluate(dataset, split=split, limit=args.limit)
    return report.model_dump()


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation service and persist the report.

    Args:
        argv: Optional command line arguments for testing.
    """
    args = build_parser().parse_args(argv)
    report = run(args)
    output = resolve_output(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output.with_suffix(".csv"), flatten_report(report))
    print(json.dumps(report, indent=2))


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


def resolve_output(args: argparse.Namespace) -> Path:
    """Resolve the report output path.

    Args:
        args: Parsed command line arguments.

    Returns:
        The output path, optionally suffixed with the limit.
    """
    service_config = load_layoutlmv3_service_config(args.config)
    output = Path(service_config.output).resolve()
    if args.limit is not None:
        stem = f"{output.stem}-limit{args.limit}"
        output = output.with_name(f"{stem}{output.suffix}")
    return output


if __name__ == "__main__":
    main()