import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from labels import ID2LABEL, label_is_pii
import os

# digit word mapping
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def find_numeric_spans(text, min_digits=7):
    """Find spans containing at least `min_digits` numeric tokens/digits (spoken or numeric).

    Returns list of (start, end) character indices.
    """
    spans = []
    # find non-space tokens with positions
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    i = 0
    while i < len(tokens):
        total_digits = 0
        j = i
        while j < len(tokens):
            tok = re.sub(r"[^A-Za-z0-9]", "", tokens[j][0].lower())
            if tok.isdigit():
                total_digits += len(tok)
            elif tok in _DIGIT_WORDS:
                total_digits += 1
            else:
                break
            if total_digits >= min_digits:
                # extend j to include this token and mark span
                span_start = tokens[i][1]
                span_end = tokens[j][2]
                spans.append((span_start, span_end))
                break
            j += 1
        i += 1
    return spans


def _digits_from_token(tok):
    t = re.sub(r"[^A-Za-z0-9+]", "", tok.lower())
    # handle plus at start for international phones
    if t.startswith("+"):
        t = t[1:]
    if t.isdigit():
        return t
    if t in _DIGIT_WORDS:
        return _DIGIT_WORDS[t]
    return ""


def luhn_check(number_str: str) -> bool:
    """Return True if `number_str` passes the Luhn checksum (common for CC numbers)."""
    try:
        digits = [int(ch) for ch in number_str if ch.isdigit()]
    except Exception:
        return False
    if len(digits) < 12:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def find_credit_card_spans(text):
    """Find candidate credit-card spans: contiguous digit groups or spoken-digit runs.

    Filters by plausible digit counts (13-19) and Luhn checksum.
    Returns list of (start, end) spans.
    """
    spans = []
    # contiguous digit groups with optional separators
    for m in re.finditer(r"[\d][\d\s\-()]{11,}\d", text):
        seq = re.sub(r"[^0-9]", "", m.group(0))
        # accept 12-19 digits to increase recall slightly
        if 12 <= len(seq) <= 19 and luhn_check(seq):
            spans.append((m.start(), m.end()))

    # spoken-digit sequences (e.g., 'four five six seven ...')
    tokens = [(mo.group(0), mo.start(), mo.end()) for mo in re.finditer(r"\S+", text)]
    i = 0
    while i < len(tokens):
        digits = ""
        j = i
        while j < len(tokens):
            d = _digits_from_token(tokens[j][0])
            if d == "":
                break
            digits += d
            if len(digits) >= 13 and len(digits) <= 19:
                if luhn_check(digits):
                    spans.append((tokens[i][1], tokens[j][2]))
                    break
            if len(digits) > 19:
                break
            j += 1
        i += 1
    return spans


def find_phone_spans(text):
    """Find candidate phone spans with stricter rules:
    - Prefer contiguous digit sequences of length 10-15 (allow separators),
    - Or spoken-digit runs totaling >=10 digits.
    Returns list of (start, end) spans.
    """
    spans = []
    # formatted phone patterns (e.g., 415-555-0123, (415) 555 0123, +1 415 555 0123)
    for m in re.finditer(r"\+?\d[\d\s().-]{6,}\d", text):
        seq = re.sub(r"[^0-9]", "", m.group(0))
        # allow 7-15 digit sequences (increase recall for shorter local numbers)
        if 7 <= len(seq) <= 15:
            spans.append((m.start(), m.end()))

    # spoken-digit sequences
    tokens = [(mo.group(0), mo.start(), mo.end()) for mo in re.finditer(r"\S+", text)]
    i = 0
    while i < len(tokens):
        digits = ""
        j = i
        while j < len(tokens):
            d = _digits_from_token(tokens[j][0])
            if d == "":
                break
            digits += d
            if 7 <= len(digits) <= 15:
                spans.append((tokens[i][1], tokens[j][2]))
                break
            if len(digits) > 15:
                break
            j += 1
        i += 1
    return spans


def spans_overlap(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or b_end <= a_start)


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--span_confidence", type=float, default=0.0,
                    help="Minimum average token probability for a predicted span to be kept (0-1).")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    # Try to load an HF token-classification model; if that fails, load generic AutoModel (for CRF wrapper)
    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        # If the loaded object doesn't behave like a token-classification model
        # (e.g. it's a base model without `classifier` / `logits`), fall back
        # to trying the custom CRF model loader below.
        if not (hasattr(model, "classifier") or hasattr(model, "crf")):
            raise RuntimeError("Loaded model is not a token-classification model")
    except Exception:
        # fallback to CRF wrapped model
        try:
            from src.crf_model import TokenClassificationWithCRF
        except Exception:
            try:
                from crf_model import TokenClassificationWithCRF
            except Exception:
                raise

        model = TokenClassificationWithCRF.from_pretrained(args.model_dir, num_labels=len(ID2LABEL))
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                # Obtain token-level logits (emissions) and predicted ids
                if hasattr(out, "predictions"):
                    # CRF wrapper: decoded predictions available, logits in out.logits
                    pred_ids = out.predictions[0]
                    emissions = out.logits[0]
                else:
                    logits = out.logits[0]
                    emissions = logits
                    pred_ids = logits.argmax(dim=-1).cpu().tolist()

            # compute per-token probabilities for the predicted label using softmax over emissions
            try:
                probs = torch.softmax(emissions, dim=-1).cpu().tolist()
            except Exception:
                probs = None

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            for span_idx, (s, e, lab) in enumerate(spans):
                # apply span confidence threshold if probs available
                keep = True
                if probs is not None and args.span_confidence > 0.0:
                    # map offsets -> token indices by zipping offsets
                    token_probs = []
                    for (tstart, tend), lid_idx in zip(offsets, range(len(offsets))):
                        if tstart == 0 and tend == 0:
                            continue
                        # token lies inside span
                        if tstart >= s and tend <= e:
                            pred_label_id = pred_ids[lid_idx] if lid_idx < len(pred_ids) else None
                            if pred_label_id is None:
                                continue
                            # probability of the predicted label for this token
                            token_prob = probs[lid_idx][int(pred_label_id)]
                            token_probs.append(token_prob)
                    if token_probs:
                        avg_prob = sum(token_probs) / len(token_probs)
                        if avg_prob < args.span_confidence:
                            keep = False
                    else:
                        keep = False

                if not keep:
                    continue

                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })

            # Post-processing: regex/heuristic-based detection for numeric PII types
            # Only add matches that don't overlap existing spans
            existing_spans = [(x["start"], x["end"]) for x in ents]

            # CREDIT_CARD: stricter detection with Luhn check
            cc_spans = find_credit_card_spans(text)
            for s, e in cc_spans:
                if any(spans_overlap(s, e, a, b) for a, b in existing_spans):
                    continue
                ents.append({"start": int(s), "end": int(e), "label": "CREDIT_CARD", "pii": True})
                existing_spans.append((s, e))

            # PHONE: stricter detection (contiguous 10-15 digits or spoken-digit runs >=10)
            phone_spans = find_phone_spans(text)
            for s, e in phone_spans:
                if any(spans_overlap(s, e, a, b) for a, b in existing_spans):
                    continue
                ents.append({"start": int(s), "end": int(e), "label": "PHONE", "pii": True})
                existing_spans.append((s, e))
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
