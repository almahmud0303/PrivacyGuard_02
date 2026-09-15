import pandas as pd
import json


df=pd.read_csv(
"../../data/raw/pii_dataset.csv"
)


print(df.head())



with open(
"../../data/annotations/bio_labels.json"
) as f:

    data=json.load(f)


print(data[0])