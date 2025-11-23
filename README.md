PII Entity Recognition for Noisy STT Transcripts

Setup

1. Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate synthetic data (train/dev/test)

```bash
python data/generate_data.py --out_dir data --train 700 --dev 150 --test 150
```

Baseline training

```bash
python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out --epochs 2 --batch_size 8
```

Predict and evaluate

```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.jsonl
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.jsonl
```

Measure latency

```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

Notes
- This repo uses a learned token classification model (HuggingFace Transformers).
- The synthetic data generator produces lowercased, punctuation-free "noisy STT" style transcripts.
- You can replace the model with a lighter-weight encoder or apply quantization to meet CPU latency targets.
