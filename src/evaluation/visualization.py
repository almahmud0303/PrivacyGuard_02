import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix




def plot_confusion(
        y_true,
        y_pred
):


    cm=confusion_matrix(

        y_true,

        y_pred

    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d"

    )


    plt.xlabel(
        "Prediction"
    )


    plt.ylabel(
        "Actual"
    )


    plt.savefig(

        "../../results/confusion_matrix.png"

    )
