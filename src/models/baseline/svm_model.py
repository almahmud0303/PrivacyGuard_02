import pandas as pd
import os


from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.svm import LinearSVC


from sklearn.metrics import accuracy_score, f1_score



DATA = "../../../data/processed/classification_dataset.csv"



df = pd.read_csv(DATA)



X = df["text"]

y = df["label"]




X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



# TF-IDF

tfidf = TfidfVectorizer()



X_train_tfidf = tfidf.fit_transform(

    X_train

)



X_test_tfidf = tfidf.transform(

    X_test

)




# SVM Model

model = LinearSVC()



model.fit(

    X_train_tfidf,

    y_train

)




prediction = model.predict(

    X_test_tfidf

)




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



# Append result

os.makedirs(

    "../../results",

    exist_ok=True

)



results = pd.DataFrame({

    "Model":[
        "SVM"
    ],

    "Accuracy":[
        accuracy
    ],

    "F1":[
        f1
    ]

})



results.to_csv(

    "../../results/baseline_results.csv",

    mode="a",

    header=False,

    index=False

)



print("Results saved")