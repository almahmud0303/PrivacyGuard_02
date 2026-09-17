import json

import torch

from torch.utils.data import Dataset

from transformers import BertTokenizer



class BERTDataset(Dataset):


    def __init__(

        self,

        file,

        max_length=128

    ):


        with open(file) as f:

            self.data=json.load(f)


        self.tokenizer=BertTokenizer.from_pretrained(

            "bert-base-uncased"

        )


        self.max_length=max_length



    def __len__(self):

        return len(self.data)




    def __getitem__(self,index):


        item=self.data[index]


        tokens=item["tokens"]

        labels=item["labels"]



        encoding=self.tokenizer(

            tokens,

            is_split_into_words=True,

            padding="max_length",

            truncation=True,

            max_length=self.max_length,

            return_tensors="pt"

        )



        word_ids=encoding.word_ids()



        label_ids=[]



        for word_id in word_ids:


            if word_id is None:

                label_ids.append(-100)


            else:

                label_ids.append(

                    self.label_to_id(
                        labels[word_id]
                    )

                )



        return {


            "input_ids":

            encoding["input_ids"].squeeze(),



            "attention_mask":

            encoding["attention_mask"].squeeze(),



            "labels":

            torch.tensor(label_ids)

        }




    def label_to_id(self,label):

        mapping={

            "O":0,

            "B-PERSON":1,

            "B-EMAIL":2,

            "B-PHONE":3,

            "B-NID":4,

            "B-LOCATION":5

        }


        return mapping[label]