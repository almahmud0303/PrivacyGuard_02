import torch

import torch.nn as nn

from torch.utils.data import DataLoader


from dataset import PIIDataset

from model import BiLSTM_PII



device="cuda" if torch.cuda.is_available() else "cpu"



dataset=PIIDataset(

"../../../data/annotations/bio_labels.json"

)



loader=DataLoader(

dataset,

batch_size=32,

shuffle=True

)



model=BiLSTM_PII(

vocab_size=len(
    dataset.word2idx
)

).to(device)




criterion=nn.CrossEntropyLoss(

    ignore_index=0

)



optimizer=torch.optim.Adam(

    model.parameters(),

    lr=0.001

)




epochs=10



for epoch in range(epochs):


    total_loss=0



    for tokens,labels in loader:


        tokens=tokens.to(device)

        labels=labels.to(device)



        optimizer.zero_grad()



        output=model(tokens)



        loss=criterion(

            output.view(
                -1,
                6
            ),

            labels.view(
                -1
            )

        )



        loss.backward()


        optimizer.step()



        total_loss+=loss.item()



    print(

        "Epoch",

        epoch+1,

        "Loss",

        total_loss

    )




torch.save(

    model.state_dict(),

    "../../../models_saved/bilstm_pii.pt"

)


print("Model Saved")