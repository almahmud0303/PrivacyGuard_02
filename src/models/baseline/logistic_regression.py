import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score


# Dataset path
DATA = "../../../data/processed/classification_dataset.csv"


# Load dataset
df = pd.read_csv(DATA)


print(df.head())


# Input and output

X = df["text"]
y = df["label"]


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# TF-IDF

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


# Model

model = LogisticRegression()


model.fit(
    X_train_tfidf,
    y_train
)


# Prediction

prediction = model.predict(
    X_test_tfidf
)


# Evaluation

accuracy = accuracy_score(
    y_test,
    prediction
)

f1 = f1_score(
    y_test,
    prediction
)


print("Accuracy:", accuracy)

print("F1 Score:", f1)



# Save result

os.makedirs(
    "../../results",
    exist_ok=True
)


results = pd.DataFrame({

    "Model": [
        "Logistic Regression"
    ],

    "Accuracy": [
        accuracy
    ],

    "F1": [
        f1
    ]

})


results.to_csv(
    "../../results/baseline_results.csv",
    index=False
)


print("Results saved")