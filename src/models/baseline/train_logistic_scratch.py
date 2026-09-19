from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .logistic_regression_scratch import LogisticRegressionScratch


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "classification_dataset.csv"
MODEL_PATH = ROOT / "models_saved" / "logistic_regression_scratch.joblib"
RESULT_PATH = ROOT / "results" / "logistic_regression_scratch_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NumPy Logistic Regression without changing active models"
    )
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(DATA).dropna(subset=["text", "label"])
    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"].astype(str),
        frame["label"].astype(int),
        test_size=0.2,
        random_state=args.seed,
        stratify=frame["label"],
    )

    # TF-IDF remains a library feature extractor. Only the Logistic Regression
    # mathematics is implemented from scratch.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        max_features=args.max_features,
    )
    train_features = vectorizer.fit_transform(x_train)
    test_features = vectorizer.transform(x_test)

    classifier = LogisticRegressionScratch(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        class_weight="balanced",
        tolerance=args.tolerance,
    )
    classifier.fit(train_features, y_train.to_numpy())
    predictions = classifier.predict(test_features)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "model": "logistic_regression_scratch",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "epochs_completed": classifier.n_iter_,
        "initial_loss": classifier.loss_history_[0],
        "final_loss": classifier.loss_history_[-1],
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "features": int(train_features.shape[1]),
        "train_samples": int(train_features.shape[0]),
        "test_samples": int(test_features.shape[0]),
        "seed": args.seed,
    }

    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = classifier.weights_.argsort()
    top_safe = [
        {"feature": str(feature_names[index]), "weight": float(classifier.weights_[index])}
        for index in sorted_indices[:10]
    ]
    top_pii = [
        {"feature": str(feature_names[index]), "weight": float(classifier.weights_[index])}
        for index in sorted_indices[-10:][::-1]
    ]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": {0: "SAFE", 1: "PII"},
            "metadata": metrics,
        },
        MODEL_PATH,
    )
    RESULT_PATH.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "top_safe_features": top_safe,
                "top_pii_features": top_pii,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print("\nTop PII features:")
    for item in top_pii:
        print(f"  {item['feature']:<25} {item['weight']:+.6f}")
    print("\nTop SAFE features:")
    for item in top_safe:
        print(f"  {item['feature']:<25} {item['weight']:+.6f}")
    print(f"\nModel: {MODEL_PATH}\nResults: {RESULT_PATH}")


if __name__ == "__main__":
    main()
