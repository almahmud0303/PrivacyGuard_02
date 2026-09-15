import pandas as pd

from pipeline import preprocessing_pipeline



input_file="../../data/raw/pii_dataset.csv"


output_file="../../data/processed/clean_dataset.csv"



df=pd.read_csv(
    input_file
)



df["tokens"]=df["text"].apply(
    preprocessing_pipeline
)



df.to_csv(
    output_file,
    index=False
)



print(
"Preprocessing completed"
)