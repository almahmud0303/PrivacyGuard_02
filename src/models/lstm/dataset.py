import json
import unicodedata
from collections import Counter

import torch
from torch.utils.data import Dataset

from src.models.transformer.labels import label2id


def normalize_token(token: str) -> str:
    """Reduce avoidable vocabulary fragmentation while preserving token shape."""
    normalized = unicodedata.normalize("NFKC", token).casefold().replace(r"\@", "@")
    if "@" in normalized:
        # Email usernames are nearly always unseen; one learned shape token is
        # substantially more useful than sending every address to <UNK>.
        return "<EMAIL_TOKEN>"
    return "".join("0" if character.isdigit() else character for character in normalized)


class PIIDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        max_len=128,
        word2idx=None,
        min_frequency=2,
        word_dropout=0.0,
        normalize_tokens=True,
    ):
        with open(annotation_file, encoding="utf-8") as handle:
            self.data = json.load(handle)
        self.max_len = max_len
        self.min_frequency = min_frequency
        self.word_dropout = word_dropout
        self.normalize_tokens = normalize_tokens
        self.label2idx = label2id.copy()

        if word2idx is None:
            frequencies = Counter(
                self._normalize(word)
                for item in self.data
                for word in item["tokens"]
            )
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            for word, frequency in frequencies.items():
                if frequency >= min_frequency:
                    self.word2idx[word] = len(self.word2idx)
        else:
            self.word2idx = dict(word2idx)

    def _normalize(self, token):
        return normalize_token(token) if self.normalize_tokens else token

    def token_to_id(self, token):
        return self.word2idx.get(self._normalize(token), self.word2idx["<UNK>"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        token_ids = [self.token_to_id(word) for word in item["tokens"][:self.max_len]]
        if self.word_dropout > 0:
            token_ids = [
                self.word2idx["<UNK>"]
                if token_id > 1 and torch.rand(()) < self.word_dropout
                else token_id
                for token_id in token_ids
            ]
        label_ids = [self.label2idx[label] for label in item["labels"][:self.max_len]]
        padding = self.max_len - len(token_ids)
        token_ids.extend([0] * padding)
        # -100 prevents padding from being trained as the O class.
        label_ids.extend([-100] * padding)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)
