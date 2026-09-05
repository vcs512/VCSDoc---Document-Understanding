# VCSDoc - Document Understanding

Document understanding and information retrieval techniques, benchmarked on the
CORD dataset.

CORD is a receipt understanding dataset with 30 semantic entity classes (menu
items, prices, totals, etc.) and represents a key information extraction (KIE)
task.

## Evaluation Protocol

- KIE models (LayoutLMv3, Moondream) are scored with token-level F1 and semantic
    entity recognition (SER) on the official CORD split.
- OCR (PaddleOCR) is scored separately with detection mAP/IOU and recognition
    CER.
- All results are recorded per service in the Services section below.

## Evaluate PaddleOCR + LayoutLMv3:
- Inputs: CORD receipt images.
- Usage: `docker compose run --rm evaluate-layoutlmv3`
- Outputs: dataset metrics.

## Evaluate Donut:
- Inputs: CORD receipt images.
- Usage: `docker compose run --rm evaluate-donut`
- Outputs:  dataset metrics.

## Evaluate Moondream:
- Inputs: CORD receipt images.
- Usage: `docker compose run --rm evaluate-moondream`
- Outputs:  dataset metrics.

## PEFT in LayoutLMv3:
- Inputs: LayoutLMv3 checkpoint and CORD train split.
- Usage: `docker compose run --rm train-layoutlmv3`
- Outputs: LoRA adapter checkpoint and fine-tuned metrics.

## PEFT in Donut:
- Inputs: Donut checkpoint and CORD train split.
- Usage: `docker compose run --rm train-donut`
- Outputs: LoRA adapter checkpoint and fine-tuned metrics.

## Roadmap

- [ ] Define evaluation protocol
- [ ] Evaluate PaddleOCR + LayoutLMv3
- [ ] Evaluate Donut
- [ ] Evaluate Moondream
- [ ] PEFT in LayoutLMv3
- [ ] PEFT in Donut
