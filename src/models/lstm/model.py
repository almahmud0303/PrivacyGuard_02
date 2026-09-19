import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class BiLSTM_PII(nn.Module):


    def __init__(
            self,
            vocab_size,
            embedding_dim=100,
            hidden_dim=128,
            num_labels=6,
            dropout=0.2,
    ):


        super().__init__()



        self.embedding=nn.Embedding(

            vocab_size,

            embedding_dim,

            padding_idx=0

        )

        self.dropout=nn.Dropout(dropout)



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
            x,
            lengths=None,
    ):


        embedded=self.dropout(self.embedding(x))

        if lengths is None:
            output,(hidden,cell)=self.lstm(embedded)
        else:
            # Packing prevents the backward LSTM from treating trailing padding
            # as real context and also makes long padded batches faster.
            packed=pack_padded_sequence(
                embedded,
                lengths.clamp(min=1).to("cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output,(hidden,cell)=self.lstm(packed)
            output,_=pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1),
            )



        logits=self.fc(self.dropout(output))



        return logits
