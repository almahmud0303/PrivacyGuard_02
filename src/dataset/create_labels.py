import pandas as pd
import re


input_file="../../data/raw/pii_dataset.csv"

output_file="../../data/processed/classification_dataset.csv"



df=pd.read_csv(input_file)



def check_pii(text):


    patterns=[

        r"01\d{9}",

        r"[\w\.-]+@[\w\.-]+",

        r"\d{10}"

    ]


    for pattern in patterns:

        if re.search(pattern,text):

            return 1


    return 0




df["label"]=df["text"].apply(
    check_pii
)



df.to_csv(
    output_file,
    index=False
)


print("Labels created")

print(
df["label"].value_counts()
)