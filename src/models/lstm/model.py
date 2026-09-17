import torch
import torch.nn as nn



class BiLSTM_PII(nn.Module):


    def __init__(
            self,
            vocab_size,
            embedding_dim=100,
            hidden_dim=128,
            num_labels=6
    ):


        super().__init__()



        self.embedding=nn.Embedding(

            vocab_size,

            embedding_dim,

            padding_idx=0

        )



        self.lstm=nn.LSTM(

            embedding_dim,

            hidden_dim,

            batch_first=True,

            bidirectional=True

        )



        self.fc=nn.Linear(

            hidden_dim*2,

            num_labels

        )





    def forward(
            self,
            x
    ):


        embedded=self.embedding(x)



        output,(hidden,cell)=self.lstm(
            embedded
        )



        logits=self.fc(
            output
        )



        return logits