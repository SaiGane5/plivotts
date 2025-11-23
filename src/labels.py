LABELS = [
    "O",
    "B_CREDIT_CARD",
    "I_CREDIT_CARD",
    "B_PHONE",
    "I_PHONE",
    "B_EMAIL",
    "I_EMAIL",
    "B_PERSON_NAME",
    "I_PERSON_NAME",
    "B_DATE",
    "I_DATE",
    "B_CITY",
    "I_CITY",
    "B_LOCATION",
    "I_LOCATION",
]

# PII mapping: which labels should be treated as PII
PII_TRUE = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"}
PII_FALSE = {"CITY", "LOCATION"}

def label_to_index(label: str) -> int:
    return LABELS.index(label)

def index_to_label(idx: int) -> str:
    return LABELS[idx]

def entity_label_from_bio(bio_label: str) -> str:
    # converts B_PERSON_NAME -> PERSON_NAME
    if bio_label == "O":
        return "O"
    return bio_label.split("_", 1)[1]
