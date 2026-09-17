from dataset import PIIDataset


dataset=PIIDataset(
"../../../data/annotations/bio_labels.json"
)


x,y=dataset[0]


print(x)

print(y)