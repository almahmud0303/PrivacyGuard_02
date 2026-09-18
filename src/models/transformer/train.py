import torch
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from torch.optim import AdamW

from .bert_model import create_model

from .bert_dataset import BERTDataset


parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()



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

project_root = Path(__file__).resolve().parents[3]
dataset_path = project_root / "data" / "annotations" / "bio_labels.json"

print("Loading dataset...")


dataset = BERTDataset(
    dataset_path
)

if args.max_samples > 0:
    dataset.data = dataset.data[:args.max_samples]


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

    batch_size=args.batch_size,

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
epochs=args.epochs
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


save_path = project_root / "models_saved" / "bert_pii"
save_path.mkdir(parents=True, exist_ok=True)



model.save_pretrained(
    save_path
)

dataset.tokenizer.save_pretrained(
    save_path
)



print("==========================")
print("BERT MODEL SAVED")
print(save_path)
print("==========================")
