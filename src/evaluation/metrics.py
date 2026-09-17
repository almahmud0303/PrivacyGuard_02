from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report,

    confusion_matrix

)


import pandas as pd





def calculate_metrics(
        y_true,
        y_pred
):


    results={}


    results["accuracy"]=accuracy_score(
        y_true,
        y_pred
    )


    results["precision"]=precision_score(
        y_true,
        y_pred,
        average="weighted"
    )


    results["recall"]=recall_score(
        y_true,
        y_pred,
        average="weighted"
    )


    results["f1"]=f1_score(
        y_true,
        y_pred,
        average="weighted"
    )


    return results






def print_report(
        y_true,
        y_pred
):


    print(

        classification_report(
            y_true,
            y_pred
        )

    )





def save_results(
        results
):


    df=pd.DataFrame(
        results
    )


    df.to_csv(

        "../../results/model_results.csv",

        index=False

    )


    print(
        "Results saved"
    )