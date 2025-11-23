from labels import LABEL2ID, ID2LABEL
from typing import Union

def create_model(model_name: str, use_crf: bool = False):
    """Create a model. If use_crf=True, return a CRF-wrapped model, otherwise HF AutoModelForTokenClassification."""
    if use_crf:
        try:
            from .crf_model import TokenClassificationWithCRF
        except Exception:
            from crf_model import TokenClassificationWithCRF

        model = TokenClassificationWithCRF(model_name, num_labels=len(LABEL2ID))
        return model
    else:
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        return model
