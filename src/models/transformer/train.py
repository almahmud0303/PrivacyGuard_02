import torch

from torch.utils.data import DataLoader

from torch.optim import AdamW

from bert_model import create_model

from bert_dataset import BERTDataset



# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("==========================")
print("Using device:", device)
print("==========================")



# =========================
# LOAD DATASET
# =========================


print("Loading dataset...")


dataset = BERTDataset(
    "../../../data/annotations/bio_labels.json"
)


print(
    "Dataset loaded successfully"
)


print(
    "Total samples:",
    len(dataset)
)



# =========================
# DATALOADER
# =========================


loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=True

)
print(
    "DataLoader created"
)
# =========================
# MODEL
# =========================
print(
    "Creating BERT model..."
)
model=create_model()
model.to(device)
print(
    "Model loaded successfully"
)
# =========================
# OPTIMIZER
# =========================


optimizer=AdamW(

    model.parameters(),

    lr=2e-5

)
# =========================
# TRAINING SETTINGS
# =========================
epochs=1
# =========================
# TRAINING LOOP
# =========================


model.train()



for epoch in range(epochs):


    print("\nStarting Epoch:", epoch+1)



    total_loss=0



    for step,batch in enumerate(loader):


        print(
            f"Processing batch {step+1}/{len(loader)}"
        )



        optimizer.zero_grad()



        input_ids=batch["input_ids"].to(device)


        attention_mask=batch["attention_mask"].to(device)


        labels=batch["labels"].to(device)



        output=model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels

        )



        loss=output.loss



        print(
            "Batch loss:",
            loss.item()
        )



        loss.backward()



        optimizer.step()



        total_loss += loss.item()
    avg_loss = total_loss / len(loader)



    print(
        "Epoch",
        epoch+1,
        "Average Loss:",
        avg_loss
    )



# =========================
# SAVE MODEL
# =========================


save_path="../../../models_saved/bert_pii"



model.save_pretrained(
    save_path
)



print("==========================")
print("BERT MODEL SAVED")
print(save_path)
print("==========================")