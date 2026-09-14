import pandas as pd



df=pd.read_csv(
"data/raw/pii_dataset.csv"
)



def get_label(token):


    if "@" in token:

        return "B-EMAIL"


    elif token.isdigit() and len(token)==11:

        return "B-PHONE"


    elif token.isdigit():

        return "B-ACCOUNT"


    else:

        return "O"



annotations=[]



for text in df["text"]:


    tokens=text.split()


    labels=[]


    for token in tokens:

        labels.append(
            get_label(token)
        )


    annotations.append(
        {
        "tokens":tokens,
        "labels":labels
        }
    )



output=pd.DataFrame(
annotations
)


output.to_json(

"data/processed/bio_dataset.json",

orient="records"

)


print(
"BIO annotation completed"
)