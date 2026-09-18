from transformers import AutoModelForTokenClassification
from .labels import label2id, id2label

MODEL_NAME = "google-bert/bert-base-multilingual-cased"

def create_model(model_name: str = MODEL_NAME):
    return AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)
