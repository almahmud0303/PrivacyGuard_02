import json
import torch

from torch.utils.data import Dataset


class PIIDataset(Dataset):

    def __init__(
            self,
            annotation_file,
            max_len=32
    ):

        with open(
            annotation_file,
            encoding="utf-8"
        ) as f:

            self.data=json.load(f)


        self.max_len=max_len


        self.build_vocab()




    def build_vocab(self):


        self.word2idx={
            "<PAD>":0,
            "<UNK>":1
        }


        self.label2idx={

            "O":0,

            "B-PERSON":1,

            "B-EMAIL":2,

            "B-PHONE":3,

            "B-NID":4,

            "B-LOCATION":5

        }



        for item in self.data:


            for word in item["tokens"]:


                if word not in self.word2idx:

                    self.word2idx[word]=len(
                        self.word2idx
                    )





    def encode_tokens(
            self,
            tokens
    ):


        ids=[]


        for word in tokens:


            ids.append(

                self.word2idx.get(
                    word,
                    self.word2idx["<UNK>"]
                )

            )


        return ids





    def __len__(self):

        return len(self.data)





    def __getitem__(
            self,
            index
    ):


        item=self.data[index]


        token_ids=self.encode_tokens(
            item["tokens"]
        )


        label_ids=[

            self.label2idx[x]

            for x in item["labels"]

        ]



        # Padding

        token_ids=token_ids[:self.max_len]


        label_ids=label_ids[:self.max_len]



        while len(token_ids)<self.max_len:

            token_ids.append(0)

            label_ids.append(0)



        return (

            torch.tensor(
                token_ids,
                dtype=torch.long
            ),

            torch.tensor(
                label_ids,
                dtype=torch.long
            )

        )