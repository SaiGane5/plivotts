import argparse
import json
import time
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def measure(model_dir: str, input_path: str, runs: int = 50, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    examples = read_jsonl(input_path)
    times = []
    # warmup
    for _ in range(5):
        ex = examples[0]
        enc = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=512)
        _ = model(**{k: v.to(device) for k, v in enc.items()})

    for i in range(runs):
        ex = examples[i % len(examples)]
        enc = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=512)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**{k: v.to(device) for k, v in enc.items()})
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    arr = np.array(times)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))

    print(json.dumps({"p50_ms": p50, "p95_ms": p95, "all_ms": times}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    measure(args.model_dir, args.input, runs=args.runs, device=args.device)


if __name__ == "__main__":
    main()
