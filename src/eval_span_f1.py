import argparse
import json
from collections import defaultdict
from typing import List, Tuple


def read_jsonl(path: str):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            data[j.get("id")] = j
    return data


def span_list(ex):
    # returns list of tuples (start,end,label)
    return [(e["start"], e["end"], e["label"]) for e in ex.get("entities", [])]


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def evaluate(gold_path: str, pred_path: str):
    gold = read_jsonl(gold_path)
    pred = read_jsonl(pred_path)

    labels = set()
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for id_, g in gold.items():
        p = pred.get(id_, {"entities": []})
        g_spans = set(span_list(g))
        p_spans = set(span_list(p))

        labels.update([lab for (_, _, lab) in g_spans])
        labels.update([lab for (_, _, lab) in p_spans])

        for span in p_spans:
            if span in g_spans:
                per_label[span[2]]["tp"] += 1
            else:
                per_label[span[2]]["fp"] += 1

        for span in g_spans:
            if span not in p_spans:
                per_label[span[2]]["fn"] += 1

    results = {}
    total_tp = total_fp = total_fn = 0
    for lab in sorted(labels):
        stats = per_label[lab]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        results[lab] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p, micro_r, micro_f1 = precision_recall_f1(total_tp, total_fp, total_fn)

    return {"per_label": results, "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    res = evaluate(args.gold, args.pred)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
