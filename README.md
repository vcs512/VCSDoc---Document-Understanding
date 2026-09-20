# VCSDoc - Document Understanding

Document understanding and information retrieval techniques, benchmarked on the
CORD dataset.

CORD is a receipt understanding dataset with 30 semantic entity classes (menu
items, prices, totals, etc.) and represents a key information extraction (KIE)
task.

## Evaluation Protocol

KIE models are scored on the official CORD split, and OCR is scored separately.
All results are recorded per service in the Services section below.

### Token-level F1

Spans are matched as normalized `(category, text)` pairs. Categories are
normalized by stripping BIO/IOB prefixes (`B-`, `I-`, `E-`, `S-`) and text is
whitespace-collapsed. Matching is multiset-based, so repeated values and
multi-word values count as many times as they occur. Reports micro-averaged F1
over all spans, per-category precision/recall/F1, and the macro average across
categories.

### Semantic entity recognition (SER)

Entity-level F1 comparing gold entities (one per labeled CORD line, grouped by
group id) against the predicted model entities. Uses the same multiset matching
as token-level F1 and reports overall, per-category and macro scores.
Categories listed in `ignore_categories` are excluded from the gold entities.

### OCR detection (mAP / IoU)

Word boxes are matched greedily by detection confidence to gold boxes with IoU
at least `detection.iou_threshold` (0.5 by default), each gold box matched
once. Reports average precision (area under the precision-recall curve), mean
IoU of the matched boxes, and micro precision/recall.

### OCR recognition (CER / WER)

Computed over ground truth and predicted text pairs aligned by the detection
matching. CER and WER use the Levenshtein edit distance normalized by the
number of gold characters and words respectively.

### Configuration

`configs/evaluation.json` holds `detection.iou_threshold` and
`ignore_categories`. Aggregated results are reported as an evaluation report
with the model and split identifiers.

## Evaluate PaddleOCR + LayoutLMv3

- Inputs: CORD receipt images (from `data/`), the fine-tuned LayoutLMv3
  checkpoint (`checkpoints/layoutlmv3-finetuned-cord`) and the service config
  (`configs/evaluate_layoutlmv3.json`).

| Key | Description | Value |
| --- | --- | --- |
| `split` | CORD split to evaluate | `test` |
| `output` | Output report path (JSON; CSV written alongside) | `reports/evaluate-layoutlmv3.json` |
| `ocr.lang` | OCR recognition language | `latin` |
| `ocr.api` | OCR API family (`classic2`, `paddlex3`, `auto`) | `classic2` |
| `ocr.det_model_dir` | Explicit detection model directory override | `~/.paddleocr/whl/det/ml/Multilingual_PP-OCRv3_det_infer` |
| `ocr.return_word_box` | Emit word-level boxes instead of line boxes | `false` |
| `ocr.use_doc_orientation_classify` | Normalize document orientation before detection | `false` |
| `ocr.use_doc_unwarping` | Unwarp curled documents before detection | `false` |
| `ocr.use_textline_orientation` | Correct cropped line orientation | `true` |
| `ocr.device` | OCR inference device, `cpu` or `gpu` | `cpu` |
| `kie.model_dir` | Fine-tuned LayoutLMv3 checkpoint | `checkpoints/layoutlmv3-finetuned-cord` |
| `kie.device` | KIE inference device, `cpu`, `gpu` or `auto` | `auto` |
| `kie.max_length` | Maximum sequence length | `512` |

- Usage: `docker compose run --rm evaluate-layoutlmv3`. Extra CLI arguments
  override the compose command, so they must be passed as a full command, e.g.
  `docker compose run --rm evaluate-layoutlmv3 python -m src.services.evaluate_layoutlmv3 --config configs/evaluate_layoutlmv3.json --limit 10`.
- Outputs: `reports/evaluate-layoutlmv3.json` and
  `reports/evaluate-layoutlmv3.csv`.

```bash
reports/
├── evaluate-layoutlmv3.json   # detection (mAP/IoU), recognition (CER/WER),
│                              # token-level F1 and SER metrics
└── evaluate-layoutlmv3.csv    # flattened spreadsheet view
```

The report metric values live in the JSON/CSV artifacts; the report is also
printed to stdout as JSON.

## Evaluate Donut

- Inputs: CORD receipt images.
- Usage: `docker compose run --rm evaluate-donut`
- Outputs: dataset metrics.

## Evaluate Moondream

- Inputs: CORD receipt images.
- Usage: `docker compose run --rm evaluate-moondream`
- Outputs: dataset metrics.

## PEFT in LayoutLMv3

- Inputs: LayoutLMv3 checkpoint and CORD train split.
- Usage: `docker compose run --rm train-layoutlmv3`
- Outputs: LoRA adapter checkpoint and fine-tuned metrics.

## PEFT in Donut

- Inputs: Donut checkpoint and CORD train split.
- Usage: `docker compose run --rm train-donut`
- Outputs: LoRA adapter checkpoint and fine-tuned metrics.

## Roadmap

- [x] Define evaluation protocol
- [x] Evaluate PaddleOCR + LayoutLMv3
- [ ] Evaluate Donut
- [ ] Evaluate Moondream
- [ ] PEFT in LayoutLMv3
- [ ] PEFT in Donut
