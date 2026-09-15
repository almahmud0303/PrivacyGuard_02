import os


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_saved"
)


RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)


MAX_LENGTH = 128

RANDOM_SEED = 42


DEVICE = "cuda"