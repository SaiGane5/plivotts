import json
import time
import argparse
import statistics
import platform
import datetime

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import importlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--quantize", action="store_true", help="Apply dynamic quantization to model for CPU inference")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    # Try loading a standard HF token-classification model; if that fails (e.g. we saved a custom
    # CRF model), fall back to the CRF wrapper loader.
    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        if not (hasattr(model, "classifier") or hasattr(model, "crf")):
            raise RuntimeError("Loaded model is not a token-classification model")
    except Exception:
        try:
            crf_mod = importlib.import_module("src.crf_model")
        except Exception:
            try:
                crf_mod = importlib.import_module("crf_model")
            except Exception:
                raise
        TokenClassificationWithCRF = getattr(crf_mod, "TokenClassificationWithCRF")
        model = TokenClassificationWithCRF.from_pretrained(args.model_dir, num_labels=None)
    model.to(args.device)
    model.eval()

    # Optionally apply dynamic quantization for faster CPU inference
    quantized = False
    if args.quantize and args.device == "cpu":
        try:
            from torch.quantization import quantize_dynamic
            # quantize linear layers which are the heavy parts
            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            quantized = True
        except Exception as e:
            print("Quantization failed:", e)

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found in input file.")
        return

    times_ms = []

    # warmup
    for _ in range(5):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))

    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")


if __name__ == "__main__":
    main()
