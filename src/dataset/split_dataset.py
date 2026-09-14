import pandas as pd


from sklearn.model_selection import train_test_split



df=pd.read_json(

"data/processed/bio_dataset.json"

)



train,temp=train_test_split(

df,

test_size=0.2,

random_state=42

)



validation,test=train_test_split(

temp,

test_size=0.5,

random_state=42

)



train.to_json(

"data/processed/train.json",

orient="records"

)


validation.to_json(

"data/processed/validation.json",

orient="records"

)



test.to_json(

"data/processed/test.json",

orient="records"

)



print("Dataset split completed")

print(
len(train),
len(validation),
len(test)
)