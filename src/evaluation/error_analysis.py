import pandas as pd


def analyze_errors(
        texts,
        true,
        pred
):

    error_count = 0

    for t, y, p in zip(
        texts,
        true,
        pred
    ):

        if y != p:

            error_count += 1

            print("TEXT:")
            print(t)

            print("TRUE LABEL:")
            print(y)

            print("PREDICTED LABEL:")
            print(p)

            print("----------------")


    print("Total Errors:", error_count)



# -----------------------------
# Load Test Dataset
# -----------------------------

DATA = "../../data/processed/test.csv"


df = pd.read_csv(DATA)


# Assuming columns:
# text  -> input sentence
# label -> actual class

texts = df["text"]

true_labels = df["label"]



# Example prediction output
# Replace this with your model predictions

predicted_labels = [
    1,
    0,
    1,
    0
]



# Run error analysis

analyze_errors(
    texts,
    true_labels,
    predicted_labels
)