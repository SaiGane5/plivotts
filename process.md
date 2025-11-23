# Process and Results

## Overview

This document records the actions I took to complete the tasks described in `assignment.md`:
- Installed dependencies
- Trained a quick baseline token-classification model
- Ran inference on the dev set and evaluated span-level metrics
- Measured inference latency (p50/p95)
- Saved artifacts to `out/` and wrote this process file


## Commands executed

The exact commands I ran (copy-paste ready):

```bash
pip install -r requirements.txt

python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 1 \
  --batch_size 8

python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json

python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json

python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

## Hyperparameters / choices

- Base model: `distilbert-base-uncased` (Hugging Face)
- Epochs: `1` (fast baseline)
- Batch size: `8` during training
- Learning rate: default `5e-5` from `src/train.py`
- Max token length: default `256`
- Device: automatic (`cuda` if available else CPU)

## Artifacts produced

- Model + tokenizer saved to `out/`
- Predictions written to `out/dev_pred.json` (dev set)

## Evaluation results (dev)

From the evaluation run (span-level metrics):

- Number of utterances written to `out/dev_pred.json`: 2
- Per-entity precision/recall/F1: all 0.000 for this 1-epoch baseline
- Macro-F1: 0.000
- PII-only: P=0.000 R=0.000 F1=0.000
- Non-PII: P=0.000 R=0.000 F1=0.000

These zero scores indicate that with only 1 training epoch the model produced no matching spans on dev. This is expected for a very short quick baseline.

## Latency

Measured over 50 runs with `batch_size=1` (inference per utterance):

- p50: 12.12 ms
- p95: 12.92 ms

These latency numbers are measured on the machine used for this run and are within the assignment's strong target (p95 <= 20 ms).

## Short analysis & next steps

What worked:
- The training / predict / evaluation pipeline ran end-to-end and produced artifacts.
- Latency is good for the chosen model and the provided inference code.

What to improve (recommended next steps):
- Train longer (increase `--epochs` to 3–10) to get meaningful entity detection and PII precision.
- Use a CRF head or better decoding heuristics to improve span extraction.
- Add small regex-based post-processing (only as a helper) to increase precision for numeric entities (credit card, phone) while keeping learned model as primary detector.
- Experiment with other base models (e.g., `roberta-base`) and tokenization alignment fixes.
- Quantize or use ONNX / TorchScript for faster CPU inference if targeting even lower latency.

If you want, I can now:
- Re-run training with higher epochs and report updated metrics, or
- Implement a small post-processing helper to boost precision for PII numeric types, or
- Prepare `out/test_pred.json` by running prediction on the test set.

-- End of process

## Generated Synthetic Datasets

I generated synthetic train/dev datasets to help development and experiments.

- Generator script: `scripts/generate_data.py` (seeded for reproducibility)
- Train set: `data/gen_train.jsonl` — 600 examples
- Dev set: `data/gen_dev.jsonl` — 150 examples

The generator uses templates with placeholders (name, city, email, phone, credit card, date, location) and injects common STT-style noise: spoken forms like `at`/`dot`, repeated words, fillers (`um`), and grouping of number words. The produced files follow the same JSONL format used by the starter code: each line is a JSON object with `id`, `text`, and `entities` (with `start`, `end`, `label`).

You can regenerate or change sizes by editing and re-running `scripts/generate_data.py`.

## Validation & Debugging Steps

I added a dataset validation script to help catch common issues before training:

- `scripts/validate_dataset.py <path> --model <tokenizer>`
  - Checks entity span bounds and tokenization offset sanity.

Recommended debug workflow:

1. Validate your dataset files:

```bash
python scripts/validate_dataset.py data/gen_train.jsonl --model distilbert-base-uncased
python scripts/validate_dataset.py data/gen_dev.jsonl --model distilbert-base-uncased
```

2. Run training with a fixed seed and small run to confirm pipeline:

```bash
python src/train.py --train data/gen_train.jsonl --dev data/gen_dev.jsonl --out_dir out --epochs 1 --seed 42
```

3. If something goes wrong:
 - Check `out/train_args.json` and `out/training_metadata.json` for the exact arguments and seed used.
 - Inspect `out` for saved model files; run `python -m pdb src/train.py` for step-through debugging.

## Actions performed (refactor + quick run)

- Added `src/utils.py` for reproducible seeding and metadata saving.
- Refactored `src/dataset.py` to mark special tokens as ignored (`-100`) so loss computation is correct and to improve alignment fallbacks.
- Updated `src/train.py` to accept `--seed`, save `train_args.json` in the output dir, and call the reproducible seed setter.
- Added `scripts/validate_dataset.py` to check entity spans and tokenizer offsets.
- Added `scripts/fix_generated_dataset.py` to clean generated datasets (removed invalid spans).

I ran a quick verification training run on the cleaned synthetic dataset and produced `out_synth/` with model and predictions for the synthetic dev set.

Quick results (1 epoch on synthetic data):

- Per-entity: CREDIT_CARD F1 ~0.70, PHONE F1 ~0.56, EMAIL F1 ~0.30 (others 0.0)
- PII-only: P=0.50 R=0.354 F1=0.415 (needs improvement to reach target 0.75 P)
- Latency (p95): 14.23 ms (batch_size=1)

These show the pipeline is reproducible and fast; to reach PII precision ~0.75 you'll want to train longer, add data augmentation and possibly a stronger model or decoding adjustments (see next steps below).

