import random
import os
import json
import torch
import numpy as np


def set_seed(seed: int):
    """Set seed for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Some ops are nondeterministic; this attempts to reduce variability
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def save_training_metadata(out_dir: str, metadata: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def validate_entity_spans(text: str, entities: list) -> bool:
    """Basic validation that entity spans are well-formed.

    Returns True if OK, False otherwise.
    """
    n = len(text)
    for e in entities:
        s = e.get("start")
        ed = e.get("end")
        lab = e.get("label")
        if s is None or ed is None or lab is None:
            return False
        if not (0 <= s < ed <= n):
            return False
    return True
