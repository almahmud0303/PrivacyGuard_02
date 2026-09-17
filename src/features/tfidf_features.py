import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import os



def create_tfidf_features(
        train_text,
        test_text
):


    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=5000
    )


    X_train = vectorizer.fit_transform(
        train_text
    )


    X_test = vectorizer.transform(
        test_text
    )


    os.makedirs(
        "../../models_saved",
        exist_ok=True
    )


    with open(
        "../../models_saved/tfidf_vectorizer.pkl",
        "wb"
    ) as f:

        pickle.dump(
            vectorizer,
            f
        )


    return X_train, X_test



if __name__=="__main__":


    df_train=pd.read_csv(
        "../../data/processed/train.csv"
    )


    df_test=pd.read_csv(
        "../../data/processed/test.csv"
    )


    X_train,X_test=create_tfidf_features(

        df_train["text"],

        df_test["text"]

    )


    print(
        "Training shape:",
        X_train.shape
    )


    print(
        "Testing shape:",
        X_test.shape
    )