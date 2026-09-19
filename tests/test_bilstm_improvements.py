import json

import torch

from src.models.lstm.dataset import PIIDataset, normalize_token
from src.models.lstm.model import BiLSTM_PII
from src.models.transformer.labels import LABELS, label2id


def test_dataset_normalizes_case_digits_and_trains_unknown_token(tmp_path):
    records = [
        {"tokens": ["EMP-45892", "RareName"], "labels": ["B-EMPLOYEE_ID", "B-PERSON"]},
        {"tokens": ["emp-77123", "common"], "labels": ["B-EMPLOYEE_ID", "O"]},
        {"tokens": ["EMP-99001", "common"], "labels": ["B-EMPLOYEE_ID", "O"]},
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    dataset = PIIDataset(path, min_frequency=2, max_len=8)

    assert normalize_token("EMP-45892") == "emp-00000"
    assert normalize_token(r"New.User99\@Example.com") == "<EMAIL_TOKEN>"
    assert dataset.token_to_id("Emp-12345") == dataset.word2idx["emp-00000"]
    assert dataset.token_to_id("unseen-person") == dataset.word2idx["<UNK>"]
    assert "rarename" not in dataset.word2idx


def test_packed_bilstm_real_tokens_do_not_depend_on_padding_length():
    torch.manual_seed(7)
    model = BiLSTM_PII(vocab_size=10, num_labels=len(LABELS), dropout=0.0).eval()
    short = torch.tensor([[2, 3, 4, 0]])
    long = torch.tensor([[2, 3, 4, 0, 0, 0, 0]])
    with torch.inference_mode():
        short_logits = model(short, lengths=torch.tensor([3]))
        long_logits = model(long, lengths=torch.tensor([3]))
    assert torch.allclose(short_logits[:, :3], long_logits[:, :3], atol=1e-6)


def test_bilstm_inference_scans_tokens_beyond_first_window(monkeypatch):
    import src.inference.model_service as service

    class Dataset:
        max_len = 4
        window_overlap = 2

        @staticmethod
        def token_to_id(token):
            return int(token.removeprefix("token")) + 1

    class Model:
        def __call__(self, tokens, lengths=None):
            logits = torch.zeros((*tokens.shape, len(LABELS)))
            logits[..., 0] = 5.0
            target = tokens == 10  # token9 occurs well after the first four tokens.
            logits[..., label2id["B-PERSON"]][target] = 12.0
            return logits

    monkeypatch.setattr(service, "_lstm_components", lambda: (Dataset(), Model(), torch))
    text = " ".join(f"token{index}" for index in range(12))
    entities = service._predict_lstm(text)

    assert [(entity["entity"], entity["type"]) for entity in entities] == [("token9", "PERSON")]
    assert entities[0]["start"] > len("token0 token1 token2 token3")
