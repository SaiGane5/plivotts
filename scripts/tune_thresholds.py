#!/usr/bin/env python3
"""Sweep span_confidence thresholds to find one that improves PII precision.

This script runs `src/predict.py` for each threshold and evaluates using `src/eval_span_f1.py`.
It requires that `src/predict.py` accepts `--span_confidence` and `--output`.

Usage:
  python scripts/tune_thresholds.py --model_dir out_roberta_crf --input data/gen_dev.jsonl --out_dir tune_out

The script writes `tune_out/threshold_report.json` with per-threshold P/R/F1.
"""
import argparse
import subprocess
import json
import os
from pathlib import Path


def run_command(cmd):
    print("RUN:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
    return res.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="tune_out")
    ap.add_argument("--thresholds", default="0.0,0.2,0.4,0.6,0.7,0.8")
    ap.add_argument("--runs_cmd", default="python src/eval_span_f1.py --gold {gold} --pred {pred}")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    report = {}

    for t in thresholds:
        out_pred = os.path.join(args.out_dir, f"pred_{t:.2f}.json")
        cmd = ["python", "src/predict.py", "--model_dir", args.model_dir, "--input", args.input, "--output", out_pred, "--span_confidence", str(t), "--device", "cpu"]
        ok = run_command(cmd)
        if not ok:
            print("Prediction failed for threshold", t)
            continue
        # Evaluate
        eval_cmd = ["python", "src/eval_span_f1.py", "--gold", args.input, "--pred", out_pred]
        res = subprocess.run(eval_cmd, capture_output=True, text=True)
        out = res.stdout
        # parse PII-only precision from stdout
        pii_line = None
        for line in out.splitlines():
            if line.startswith("PII-only metrics:"):
                pii_line = line
                break
        # fallback parse
        lines = out.splitlines()
        # extract P=... R=... F1=...
        prf = {"P": None, "R": None, "F1": None}
        for line in lines:
            if line.startswith("PII-only metrics:"):
                parts = line.split("PII-only metrics:")[-1].strip()
                # parts like P=0.500 R=0.400 F1=0.444
                for part in parts.split():
                    if part.startswith("P="):
                        prf["P"] = float(part[2:])
                    if part.startswith("R="):
                        prf["R"] = float(part[2:])
                    if part.startswith("F1="):
                        prf["F1"] = float(part[3:])
        report[f"{t:.2f}"] = prf
        # save intermediate
        with open(os.path.join(args.out_dir, "threshold_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    print("Tuning finished. Report saved to", os.path.join(args.out_dir, "threshold_report.json"))


if __name__ == "__main__":
    main()
