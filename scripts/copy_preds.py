"""Copy predictions from a model-specific out directory to `out/dev_pred.json`.

This ensures the repository contains the mandatory `out/dev_pred.json` file.
"""
import shutil
from pathlib import Path
import sys


def main(src_dir: str = "out_roberta_crf", dst_dir: str = "out"):
    src = Path(src_dir) / "dev_pred.json"
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"Source predictions not found: {src}")
        return 1
    shutil.copy(src, dst / "dev_pred.json")
    print(f"Copied {src} -> {dst / 'dev_pred.json'}")
    return 0


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "out_roberta_crf"
    sys.exit(main(src))
