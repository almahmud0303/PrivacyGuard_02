import torch
from pathlib import Path

import torch.nn as nn

from torch.utils.data import DataLoader


from dataset import PIIDataset

from model import BiLSTM_PII



device="cuda" if torch.cuda.is_available() else "cpu"

project_root = Path(__file__).resolve().parents[3]
dataset_path = project_root / "data" / "annotations" / "bio_labels.json"
model_dir = project_root / "models_saved"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "bilstm_pii.pt"


dataset=PIIDataset(

dataset_path

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

    model_path

)


print("Model Saved")