"""Tests for the evaluation report CSV flattening and writing."""

import csv
import json

import pytest

from src.core.kie.base import KieEngine
from src.core.ocr.base import OcrEngine
from src.core.reporting import (
    _CSV_FIELDS,
    flatten_report,
    write_csv,
)
from src.schemas.config import DetectionConfig, EvaluationConfig
from src.schemas.cord import CordReceipt
from src.schemas.kie import KieEntity, KiePrediction, LabeledSpan
from src.schemas.ocr import OcrResult
from src.services.evaluate import Evaluator


class PerfectOcrEngine(OcrEngine):
    """OCR engine returning the gold words as its predictions."""

    def recognize(self, receipt: CordReceipt) -> OcrResult:
        """Return the receipt OCR words.

        Args:
            receipt: The receipt under test.

        Returns:
            The exact ground truth OCR result.
        """
        return OcrResult(
            image_id=receipt.image_id, words=receipt.ocr_words()
        )


class GoldKieEngine(KieEngine):
    """KIE engine predicting the exact gold entities and spans."""

    def predict(
        self, receipt: CordReceipt, ocr: OcrResult | None = None
    ) -> KiePrediction:
        """Return the gold entities from the receipt lines.

        Args:
            receipt: The receipt under test.
            ocr: Unused OCR result.

        Returns:
            Spans and entities matching the gold labels.
        """
        prediction = KiePrediction(image_id=receipt.image_id)
        for entity in receipt.entities():
            prediction.entities.append(
                KieEntity(
                    category=entity.category,
                    text=entity.text,
                    group_id=entity.group_id,
                )
            )
        for token in receipt.tokens():
            prediction.spans.append(
                LabeledSpan(category=token.category, text=token.text)
            )
        return prediction


def build_report(receipt: CordReceipt) -> dict:
    """Build a serialized report with all metric families populated.

    Args:
        receipt: The receipt being evaluated.

    Returns:
        The serialized evaluation report mapping.
    """
    evaluation_config = EvaluationConfig(
        detection=DetectionConfig(iou_threshold=0.5),
        ignore_categories=[],
    )
    evaluator = Evaluator(
        PerfectOcrEngine(), GoldKieEngine(), evaluation_config
    )
    return evaluator.evaluate([receipt], split="test").model_dump()


def test_flatten_report_rows(receipt: CordReceipt) -> None:
    """Flattened rows should cover all four metric families."""
    rows = flatten_report(build_report(receipt))
    assert [row["metric"] for row in rows].count("detection") == 6
    assert [row["metric"] for row in rows].count("recognition") == 4
    assert {row["category"] for row in rows if row["metric"] == "token_f1"} >= {
        "overall",
        "macro",
    }
    assert {row["category"] for row in rows if row["metric"] == "ser"} >= {
        "overall",
        "macro",
    }
    assert all(set(row) == set(_CSV_FIELDS) for row in rows)


def test_flatten_report_scalar_values(receipt: CordReceipt) -> None:
    """Detection and recognition should populate only the value column."""
    rows = flatten_report(build_report(receipt))
    detection = [row for row in rows if row["metric"] == "detection"]
    assert detection[0]["category"] == "average_precision"
    assert detection[0]["value"] == pytest.approx(1.0)
    assert detection[0]["precision"] is None
    counts = [row for row in detection if row["category"] == "num_gt"]
    assert counts[0]["value"] == 5


def test_flatten_report_f1_values(receipt: CordReceipt) -> None:
    """F1 scopes should populate precision, recall and f1."""
    rows = flatten_report(build_report(receipt))
    overall = [
        row for row in rows if row["metric"] == "token_f1" and row["category"] == "overall"
    ]
    assert overall[0]["f1"] == pytest.approx(1.0)
    assert overall[0]["value"] is None


def test_write_csv_rows(tmp_path) -> None:
    """write_csv should persist the header and every flattened row."""
    report = {
        "model": "paddleocr+layoutlmv3",
        "split": "test",
        "metrics": {
            "detection": {
                "average_precision": 0.5,
                "mean_iou": 0.75,
                "precision": 0.5,
                "recall": 1.0,
                "num_gt": 2,
                "num_pred": 2,
            },
            "recognition": {
                "cer": 0.25,
                "wer": 1.0,
                "num_gt": 2,
                "num_pred": 2,
            },
        },
    }
    rows = flatten_report(report)
    output = tmp_path / "report.csv"
    write_csv(output, rows)
    with output.open(encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    cer = [row for row in reader if row["metric"] == "recognition" and row["category"] == "cer"]
    assert cer[0]["value"] == "0.25"
    assert len(reader) == len(rows)


def test_flatten_report_missing_metrics(receipt: CordReceipt) -> None:
    """Flattened rows should skip metric families absent from the report."""
    empty = {
        "overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "per_class": {},
        "macro": None,
    }
    report = {
        "model": "donut",
        "split": "test",
        "metrics": {"token_f1": empty, "ser": empty},
    }
    rows = flatten_report(report)
    assert {row["metric"] for row in rows} == {"token_f1", "ser"}
    assert all(set(row) == set(_CSV_FIELDS) for row in rows)


def test_main_writes_json_and_csv(tmp_path, monkeypatch) -> None:
    """The CLI should persist both the JSON and the CSV report."""
    import src.services.evaluate_layoutlmv3 as reporting

    config_file = tmp_path / "config.json"
    output = tmp_path / "reports" / "report.json"
    config_file.write_text(
        json.dumps(
            {
                "output": str(output),
                "ocr": {"lang": "korean", "api": "classic2"},
                "kie": {
                    "model_dir": "checkpoints/layoutlmv3-finetuned-cord"
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(_args) -> dict:
        """Return a complete but minimal report mapping."""
        empty = {"overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "per_class": {}, "macro": None}
        return {
            "model": "paddleocr+layoutlmv3",
            "split": "test",
            "metrics": {
                "detection": {
                    "average_precision": 0.0,
                    "mean_iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "num_gt": 0,
                    "num_pred": 0,
                },
                "recognition": {
                    "cer": 0.0,
                    "wer": 0.0,
                    "num_gt": 0,
                    "num_pred": 0,
                },
                "token_f1": empty,
                "ser": empty,
            },
        }

    monkeypatch.setattr(reporting, "run", fake_run)
    reporting.main(["--config", str(config_file)])
    assert output.exists()
    assert output.with_suffix(".csv").exists()