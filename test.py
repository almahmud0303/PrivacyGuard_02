import ast
from pathlib import Path

import pandas as pd

from src.features.embeddings import train_word2vec


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "clean_dataset.csv"
)


df = pd.read_csv(DATA_FILE)


sentences = df["tokens"].apply(ast.literal_eval)


model = train_word2vec(
    sentences
)


print(
    model.wv["phone"]
)