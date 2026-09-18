import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .labels import label2id


class BERTDataset(Dataset):
    def __init__(self, file, max_length=128, model_name="google-bert/bert-base-multilingual-cased"):
        with open(file, encoding="utf-8") as handle:
            self.data = json.load(handle)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer(item["tokens"], is_split_into_words=True, padding="max_length",
            truncation=True, max_length=self.max_length, return_tensors="pt")
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id == previous_word:
                # Ignore repeated subword pieces so a word contributes once to loss.
                label_ids.append(-100)
            else:
                label_ids.append(label2id[item["labels"][word_id]])
            previous_word = word_id
        return {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label_ids, dtype=torch.long)}
