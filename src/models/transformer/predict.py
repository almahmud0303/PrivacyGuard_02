import torch


from transformers import BertTokenizer,BertForTokenClassification


from labels import id2label



model=BertForTokenClassification.from_pretrained(

"../../../models_saved/bert_pii"

)



tokenizer=BertTokenizer.from_pretrained(

"bert-base-uncased"

)



def predict(text):


    tokens=text.split()



    inputs=tokenizer(

        tokens,

        is_split_into_words=True,

        return_tensors="pt",

        padding=True

    )



    outputs=model(**inputs)



    predictions=torch.argmax(

        outputs.logits,

        dim=2

    )



    result=[]


    for token,label in zip(

        tokens,

        predictions[0][:len(tokens)]

    ):


        result.append(

            (

            token,

            id2label[label.item()]

            )

        )


    return result



print(

predict(

"My phone number is 01712345678"

)

)