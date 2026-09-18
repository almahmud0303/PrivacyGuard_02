import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification


from .labels import id2label



MODEL_PATH = Path(__file__).resolve().parents[3] / "models_saved" / "bert_pii"



class BERTEntityDetector:


    def __init__(self):

        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(
                f"BERT checkpoint not found at {MODEL_PATH}. "
                "Train the transformer model and save it there before running test.py."
            )

        self.tokenizer=AutoTokenizer.from_pretrained(
            MODEL_PATH
        )


        self.model=AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH
        )


        self.model.eval()



    def predict(self,text):


        tokens=text.split()


        encoded=self.tokenizer(

            tokens,

            is_split_into_words=True,

            return_tensors="pt",

            padding=True,

            truncation=True

        )



        with torch.no_grad():

            output=self.model(

                **encoded

            )



        predictions=torch.argmax(

            output.logits,

            dim=-1

        )



        word_ids=encoded.word_ids()



        entities=[]


        current_entity=None



        for token_id,word_id in zip(

            predictions[0],

            word_ids

        ):



            if word_id is None:

                continue



            label=id2label[
                token_id.item()
            ]



            word=tokens[word_id]



            if label!="O":


                entities.append(

                    {

                    "entity":word,

                    "type":
                    label.replace(
                        "B-",
                        ""
                    )

                    }

                )



        return entities
