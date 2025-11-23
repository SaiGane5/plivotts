import json
from pathlib import Path
import sys


def fix_file(src_path, dst_path):
    removed = 0
    total = 0
    with open(src_path, "r", encoding="utf-8") as inf, open(dst_path, "w", encoding="utf-8") as outf:
        for line in inf:
            total += 1
            obj = json.loads(line)
            text = obj.get("text", "")
            ents = obj.get("entities", [])
            clean = []
            for e in ents:
                s = e.get("start")
                ed = e.get("end")
                if s is None or ed is None:
                    removed += 1
                    continue
                if not (0 <= s < ed <= len(text)):
                    removed += 1
                    continue
                clean.append(e)
            obj["entities"] = clean
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Processed {total} records. Removed {removed} invalid entities.")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gen_train.jsonl")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + ".clean.jsonl")
    fix_file(src, dst)
