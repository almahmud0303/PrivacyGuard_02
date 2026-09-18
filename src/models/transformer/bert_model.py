from transformers import BertForTokenClassification


from .labels import label2id,id2label



MODEL_NAME="bert-base-uncased"



def create_model():

    model=BertForTokenClassification.from_pretrained(

        MODEL_NAME,

        num_labels=len(label2id),

        id2label=id2label,

        label2id=label2id

    )


    return model
