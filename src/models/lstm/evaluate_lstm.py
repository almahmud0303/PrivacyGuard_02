import torch
from pathlib import Path


from torch.utils.data import DataLoader


from dataset import PIIDataset

from model import BiLSTM_PII

project_root = Path(__file__).resolve().parents[3]
dataset_path = project_root / "data" / "annotations" / "bio_labels.json"
model_path = project_root / "models_saved" / "bilstm_pii.pt"



dataset=PIIDataset(

dataset_path

)



loader=DataLoader(

dataset,

batch_size=32

)




model=BiLSTM_PII(

len(dataset.word2idx)

)



model.load_state_dict(

torch.load(

model_path,

map_location="cpu"

)

)



model.eval()



true=[]

pred=[]



with torch.no_grad():


    for x,y in loader:


        output=model(x)



        prediction=torch.argmax(

            output,

            dim=-1

        )



        true.extend(
            y.flatten().tolist()
        )


        pred.extend(
            prediction.flatten().tolist()
        )





from sklearn.metrics import classification_report



print(

classification_report(

true,

pred,

zero_division=0

)

)