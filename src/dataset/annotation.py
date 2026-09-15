import pandas as pd
import re
import json
import os



INPUT_PATH="../../data/raw/pii_dataset.csv"

OUTPUT_PATH="../../data/annotations/bio_labels.json"



def annotate_sentence(text):

    tokens=text.split()

    labels=[]


    for token in tokens:


        clean=token.strip(".,!?")



        if re.match(
            r"01\d{9}",
            clean
        ):

            labels.append("B-PHONE")



        elif re.match(
            r"[\w\.-]+@[\w\.-]+",
            clean
        ):

            labels.append("B-EMAIL")



        elif re.match(
            r"\d{10}",
            clean
        ):

            labels.append("B-NID")



        elif clean in [
            "Rahim",
            "Ahmed",
            "Karim",
            "Hasan",
            "John",
            "Smith",
            "David",
            "Miller"
        ]:

            labels.append("B-PERSON")



        elif clean in [
            "Dhaka",
            "Bangladesh",
            "Chittagong",
            "USA",
            "York"
        ]:

            labels.append("B-LOCATION")



        else:

            labels.append("O")


    return {
        "tokens":tokens,
        "labels":labels
    }





df=pd.read_csv(INPUT_PATH)



annotations=[]



for text in df["text"]:

    annotations.append(
        annotate_sentence(text)
    )



os.makedirs(
    "../../data/annotations",
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        annotations,
        f,
        indent=4,
        ensure_ascii=False
    )



print("BIO Annotation Completed")
print("Samples:",len(annotations))