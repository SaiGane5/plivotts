"""Validate JSONL dataset files for basic issues and tokenizer alignment.

Usage:
  python scripts/validate_dataset.py data/train.jsonl --model distilbert-base-uncased

This script checks:
 - entity spans are within text bounds
 - tokenization offset mapping matches expectation (no negative lengths)
"""
import argparse
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

# ensure repo root is on path so `src` package can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import validate_entity_spans  # now importable


def validate_file(path, tokenizer):
    n = 0
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            obj = json.loads(line)
            text = obj.get("text", "")
            ents = obj.get("entities", [])
            if not validate_entity_spans(text, ents):
                print(f"[BAD SPANS] {path} line {n} id={obj.get('id')}")
                bad += 1
                continue

            enc = tokenizer(text, return_offsets_mapping=True, truncation=True)
            offsets = enc.get("offset_mapping", [])
            for (s, e) in offsets:
                if s > e:
                    print(f"[BAD OFFSET] {path} line {n} id={obj.get('id')} offset {s}>{e}")
                    bad += 1
                    break

    print(f"Validated {n} records in {path}. Issues found: {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="distilbert-base-uncased")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    validate_file(args.path, tokenizer)


if __name__ == "__main__":
    main()
