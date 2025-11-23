import json
from typing import List, Dict

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.labels import LABELS


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def char_labels_from_entities(text: str, entities: List[Dict]) -> List[str]:
    labels = ["O"] * len(text)
    for ent in entities:
        s = ent["start"]
        e = ent["end"]
        lab = ent["label"]
        if s < 0 or e > len(text):
            continue
        if s < e:
            labels[s] = "B_" + lab
            for i in range(s + 1, e):
                labels[i] = "I_" + lab
    return labels


def align_labels_with_tokens(example: Dict, tokenizer: PreTrainedTokenizerFast) -> Dict:
    text = example["text"]
    entities = example.get("entities", [])
    char_labels = char_labels_from_entities(text, entities)

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    labels = []
    for idx, (offset_start, offset_end) in enumerate(enc["offset_mapping"]):
        if offset_start == offset_end:
            # special token
            labels.append("O")
        else:
            # pick label at first character of token span
            lab = char_labels[offset_start]
            labels.append(lab)

    enc_inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": [LABELS.index(l) if l in LABELS else 0 for l in labels],
        "offset_mapping": enc["offset_mapping"],
        "text": text,
    }
    return enc_inputs


def load_dataset(tokenizer: PreTrainedTokenizerFast, path: str) -> Dataset:
    raw = read_jsonl(path)
    mapped = [align_labels_with_tokens(x, tokenizer) for x in raw]
    return Dataset.from_list(mapped)
