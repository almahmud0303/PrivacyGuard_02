"""One-command training pipeline for PrivacyGuard.

The fast default trains and evaluates all classical baselines. Expensive neural
models are opt-in with --bilstm and --bert.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "classification_dataset.csv"
MODELS = ROOT / "models_saved"; RESULTS = ROOT / "results"

def train_baselines() -> pd.DataFrame:
    frame = pd.read_csv(DATA).dropna(subset=["text", "label"])
    x_train, x_test, y_train, y_test = train_test_split(
        frame.text.astype(str), frame.label, test_size=.2, random_state=42, stratify=frame.label)
    estimators = {"logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                  "naive_bayes": MultinomialNB(), "svm": LinearSVC(class_weight="balanced")}
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True); rows = []
    for name, estimator in estimators.items():
        model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), analyzer="word", max_features=10000)),
                          ("classifier", estimator)])
        model.fit(x_train, y_train); predicted = model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, predicted),
                     "precision": precision, "recall": recall, "f1": f1})
        joblib.dump(model, MODELS / f"{name}.joblib")
    results = pd.DataFrame(rows); results.to_csv(RESULTS / "baseline_results.csv", index=False)
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PrivacyGuard models")
    parser.add_argument("--bilstm", action="store_true", help="also train the BiLSTM")
    parser.add_argument("--bert", action="store_true", help="also fine-tune BERT (slow; may download weights)")
    parser.add_argument("--bert-samples", type=int, default=200)
    args = parser.parse_args()
    if not DATA.exists(): raise SystemExit(f"Missing dataset: {DATA}")
    print(train_baselines().to_string(index=False))
    if args.bilstm: subprocess.run([sys.executable, "-m", "src.models.lstm.train"], cwd=ROOT, check=True)
    if args.bert:
        subprocess.run([sys.executable, "-m", "src.models.transformer.train", "--max-samples", str(args.bert_samples)], cwd=ROOT, check=True)
    print(f"Models: {MODELS}\nResults: {RESULTS}")

if __name__ == "__main__": main()
