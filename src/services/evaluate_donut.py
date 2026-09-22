"""Evaluate the Donut image-to-text model on the CORD dataset."""

import argparse
import json

from src.core.config import (
    load_cord_config,
    load_donut_service_config,
    load_evaluation_config,
)
from src.core.data.cord import CordDataset
from src.core.kie.donut import DonutKieEngine
from src.core.reporting import (
    flatten_report,
    resolve_output,
    write_csv,
)
from src.services.evaluate import Evaluator

_MODEL_ID = "donut"


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Donut on the CORD dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/evaluate_donut.json",
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
    service_config = load_donut_service_config(args.config)
    cord_config = load_cord_config(args.cord_config)
    evaluation_config = load_evaluation_config(args.evaluation_config)
    split = args.split or service_config.split
    dataset = CordDataset(cord_config, split, load_images=True)
    kie = DonutKieEngine(service_config.donut)
    evaluator = Evaluator(None, kie, evaluation_config)
    report = evaluator.evaluate(
        dataset,
        split=split,
        limit=args.limit,
        model=_MODEL_ID,
        include_ocr=False,
        tree_based=True,
    )
    return report.model_dump()


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation service and persist the report.

    Args:
        argv: Optional command line arguments for testing.
    """
    args = build_parser().parse_args(argv)
    report = run(args)
    service_config = load_donut_service_config(args.config)
    output = resolve_output(service_config.output, args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output.with_suffix(".csv"), flatten_report(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
