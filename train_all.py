from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "processed" / "classification_dataset.csv"

MODELS = ROOT / "models_saved"
RESULTS = ROOT / "results"


def train_baselines() -> pd.DataFrame:

    frame = pd.read_csv(DATA).dropna(
        subset=["text", "label"]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        frame.text.astype(str),
        frame.label,
        test_size=0.2,
        random_state=42,
        stratify=frame.label
    )


    estimators = {

        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

        "naive_bayes":
            MultinomialNB(),

        "svm":
            LinearSVC(
                class_weight="balanced"
            )
    }


    MODELS.mkdir(
        exist_ok=True
    )

    RESULTS.mkdir(
        exist_ok=True
    )


    rows = []


    for name, estimator in estimators.items():


        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        analyzer="word",
                        max_features=10000
                    )
                ),

                (
                    "classifier",
                    estimator
                )
            ]
        )


        # Train model

        model.fit(
            x_train,
            y_train
        )


        # Prediction

        predicted = model.predict(
            x_test
        )


        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predicted,
            average="weighted",
            zero_division=0
        )


        rows.append(
            {
                "model": name,

                "accuracy": accuracy_score(
                    y_test,
                    predicted
                ),

                "precision": precision,

                "recall": recall,

                "f1": f1
            }
        )


        # Save model

        joblib.dump(
            model,
            MODELS / f"{name}.joblib"
        )


    results = pd.DataFrame(rows)


    results.to_csv(
        RESULTS / "baseline_results.csv",
        index=False
    )


    return results



def main() -> None:


    parser = argparse.ArgumentParser(
        description="Train PrivacyGuard models"
    )


    parser.add_argument(
        "--bilstm",
        action="store_true",
        help="also train the BiLSTM"
    )


    parser.add_argument(
        "--bert",
        action="store_true",
        help="also fine-tune BERT (slow; may download weights)"
    )


    parser.add_argument(
        "--bilstm-epochs",
        type=int,
        default=8
    )


    parser.add_argument(
        "--bilstm-batch-size",
        type=int,
        default=128
    )


    parser.add_argument(
        "--bilstm-max-length",
        type=int,
        default=128,
        help="training window length; inference automatically uses overlapping windows"
    )


    parser.add_argument(
        "--bert-samples",
        type=int,
        default=200
    )


    parser.add_argument(
        "--bert-epochs",
        type=int,
        default=3
    )


    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=8
    )


    parser.add_argument(
        "--bert-gradient-accumulation",
        type=int,
        default=1
    )


    parser.add_argument(
        "--ner-samples",
        type=int,
        default=30000,
        help="number of directly labeled multilingual NER records to generate"
    )


    parser.add_argument(
        "--skip-ner-generation",
        action="store_true",
        help="reuse existing train/validation/test NER JSON files"
    )


    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="train only requested neural models"
    )


    args = parser.parse_args()



    if not DATA.exists():

        raise SystemExit(
            f"Missing dataset: {DATA}"
        )


    # Train classical ML models

    if not args.skip_classical:

        print(
            train_baselines().to_string(
                index=False
            )
        )



    # Generate NER dataset

    if (
        (args.bilstm or args.bert)
        and not args.skip_ner_generation
    ):

        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.dataset.generate_ner_dataset",
                "--samples",
                str(args.ner_samples)
            ],
            cwd=ROOT,
            check=True
        )



    # Train BiLSTM

    if args.bilstm:

        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.lstm.train",
                "--epochs",
                str(args.bilstm_epochs),
                "--batch-size",
                str(args.bilstm_batch_size),
                "--max-length",
                str(args.bilstm_max_length)
            ],
            cwd=ROOT,
            check=True
        )



    # Train BERT

    if args.bert:


        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.transformer.train",

                "--max-samples",
                str(args.bert_samples),

                "--epochs",
                str(args.bert_epochs),

                "--batch-size",
                str(args.bert_batch_size),

                "--gradient-accumulation",
                str(args.bert_gradient_accumulation)
            ],

            cwd=ROOT,

            check=True
        )



        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate_bert",

                "--batch-size",
                str(args.bert_batch_size)
            ],

            cwd=ROOT,

            check=True
        )


    print(
        f"Models: {MODELS}\nResults: {RESULTS}"
    )



if __name__ == "__main__":

    main()
