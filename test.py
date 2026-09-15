import torch
import transformers
import sklearn
import nltk


print("Python environment working")

print(
    "PyTorch version:",
    torch.__version__
)


print(
    "CUDA available:",
    torch.cuda.is_available()
)


print(
    "Transformers:",
    transformers.__version__
)


print(
    "Scikit Learn:",
    sklearn.__version__
)