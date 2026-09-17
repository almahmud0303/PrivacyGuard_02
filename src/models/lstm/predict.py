import torch

from model import BiLSTM_PII


labels=[

"O",

"B-PERSON",

"B-EMAIL",

"B-PHONE",

"B-NID",

"B-LOCATION"

]


def predict(tokens,model,dataset):


    ids=[

        dataset.word2idx.get(
            x,
            1
        )

        for x in tokens

    ]



    ids=ids+[0]*(32-len(ids))


    x=torch.tensor(
        [ids]
    )



    with torch.no_grad():

        output=model(x)


    prediction=torch.argmax(
        output,
        dim=-1
    )



    result=[]


    for word,label in zip(
        tokens,
        prediction[0]
    ):

        result.append(

            (
            word,
            labels[label]
            )

        )


    return result