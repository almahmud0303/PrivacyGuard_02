import pandas as pd
import pickle


from sklearn.model_selection import train_test_split


from visualization import plot_confusion



# Load dataset

df=pd.read_csv(
"../../data/processed/classification_dataset.csv"
)



# Split data

X_train,X_test,y_train,y_test=train_test_split(

    df["text"],

    df["label"],

    test_size=0.2,

    random_state=42

)



# Load model

model=pickle.load(

open(
"../../models_saved/logistic_model.pkl",
"rb"
)

)



# Load TF-IDF

vectorizer=pickle.load(

open(
"../../models_saved/logistic_vectorizer.pkl",
"rb"
)

)



# Convert text to vectors

X_test=vectorizer.transform(
    X_test
)



# Prediction

prediction=model.predict(
    X_test
)



# CREATE CONFUSION MATRIX

plot_confusion(

    y_test,

    prediction

)


print(
"Confusion matrix created"
)