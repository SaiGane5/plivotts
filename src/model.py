from transformers import AutoConfig, AutoModelForTokenClassification

def build_model(model_name: str, num_labels: int):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    return model
