import argparse
import json
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.labels import LABELS, PII_TRUE, entity_label_from_bio


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def decode_predictions(tokens_offsets, preds, text: str) -> List[Dict]:
    entities = []
    cur_ent = None
    for (off, lab_idx) in zip(tokens_offsets, preds):
        start, end = off
        if start == end:
            # skip special tokens
            continue
        bio = LABELS[lab_idx]
        if bio == "O":
            if cur_ent is not None:
                entities.append(cur_ent)
                cur_ent = None
            continue
        tag, ent = bio.split("_", 1)
        if tag == "B":
            if cur_ent is not None:
                entities.append(cur_ent)
            cur_ent = {"start": start, "end": end, "label": ent}
        elif tag == "I":
            if cur_ent is None:
                # treat as B
                cur_ent = {"start": start, "end": end, "label": ent}
            else:
                # extend
                cur_ent["end"] = end

    if cur_ent is not None:
        entities.append(cur_ent)

    # merge adjacent/duplicate spans
    merged = []
    for e in entities:
        if merged and e["start"] <= merged[-1]["end"] and e["label"] == merged[-1]["label"]:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
        else:
            merged.append(e)

    # add pii flags
    for e in merged:
        e["pii"] = e["label"] in PII_TRUE

    return merged


def predict(model_dir: str, input_path: str, output_path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    examples = read_jsonl(input_path)
    outputs = []
    for ex in examples:
        text = ex["text"]
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0].cpu()
            preds = logits.argmax(-1).tolist()

        offsets = enc["offset_mapping"][0].tolist()

        ents = decode_predictions(offsets, preds, text)

        outputs.append({"id": ex.get("id"), "text": text, "entities": ents})

    with open(output_path, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    predict(args.model_dir, args.input, args.output, device=args.device)


if __name__ == "__main__":
    main()
