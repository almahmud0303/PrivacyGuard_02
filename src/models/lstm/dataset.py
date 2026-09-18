import json
import torch
from torch.utils.data import Dataset
from src.models.transformer.labels import label2id


class PIIDataset(Dataset):
    def __init__(self, annotation_file, max_len=32):
        with open(annotation_file, encoding="utf-8") as handle:
            self.data = json.load(handle)
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.label2idx = label2id.copy()
        for item in self.data:
            for word in item["tokens"]:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        token_ids = [self.word2idx.get(word, 1) for word in item["tokens"][:self.max_len]]
        label_ids = [self.label2idx[label] for label in item["labels"][:self.max_len]]
        padding = self.max_len - len(token_ids)
        token_ids.extend([0] * padding)
        # -100 prevents padding from being trained as the O class.
        label_ids.extend([-100] * padding)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)
