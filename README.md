# PrivacyGuard — AI Privacy Firewall

PrivacyGuard is a local NLP system for detecting, scoring, and redacting Personally
Identifiable Information (PII). It provides a Streamlit model-testing interface,
a FastAPI endpoint, classical machine-learning baselines, a BiLSTM sequence model,
and a BERT token-classification model.

The application supports English, Bengali, and Banglish input. Structured values
are protected by deterministic patterns even while an experimental neural model is
selected, so a model-loading error does not expose obvious PII.

## 1. Main features

- Detect phone numbers, email addresses, NIDs, valid credit-card numbers, and IP addresses.
- Experimentally detect people and locations with BiLSTM or BERT.
- Assign LOW, MEDIUM, HIGH, or CRITICAL risk.
- Replace detected spans with safe placeholders.
- Compare Pattern, BERT, BiLSTM, Logistic Regression, Naive Bayes, and SVM models.
- Process English, Bengali, and Banglish text locally.
- Download protected text from the Streamlit interface.
- Access the core detector through a REST API.
- Train model families through `train_all.py`.
- Fall back safely when a selected artifact cannot be loaded.

## 2. System pipeline

```text
Input text
   ↓
Selected detection model
   ↓
Pattern safety layer
   ↓
PII entities and confidence values
   ↓
Risk scoring
   ↓
Span-safe redaction
   ↓
Protected text
```

There are two model behaviors:

1. **Entity models** — Pattern detector, BiLSTM, and BERT return token or character spans.
2. **Document classifiers** — Logistic Regression, Naive Bayes, and SVM decide whether the
   complete text contains PII. Pattern spans are then used for safe redaction.

## 3. Requirements

- Windows 10/11 with PowerShell (the commands below use PowerShell)
- Python 3.11 recommended
- At least 4 GB free disk space
- Internet access during dependency installation and first BERT download
- Optional CUDA-compatible GPU for faster BERT training

## 4. Clone and open the project

```powershell
git clone <your-repository-url>
cd NLP_PROJECT
```

If the project is already downloaded, open PowerShell in the directory containing
`app.py`, `api.py`, and `train_all.py`.

## 5. Create the Python environment

A clean environment is strongly recommended. It avoids conflicts with global Python,
Spyder, Anaconda, or an older project environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Confirm the interpreter:

```powershell
python --version
python -c "import torch, streamlit, transformers; print('Environment ready')"
```

In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**.

## 6. Dataset overview

The project uses synthetic data so real private information is not required for
training. The generator combines names, Bangladeshi phone numbers, emails, NIDs,
locations, English templates, Banglish templates, and non-PII sentences.

### Current raw dataset

`data/raw/pii_dataset.csv` currently contains 6,000 rows:

- 5,000 template-generated records
- 1,000 normal/non-PII sentences
- Columns: `id`, `text`

Examples:

```text
My name is John Smith and my phone number is 01712345678
amar email holo rahim@gmail.com
ami thaki Dhaka Bangladesh
Today is a beautiful day
```

### Classification dataset

`data/processed/classification_dataset.csv` adds a binary `label` column:

- `1`: a phone, email, or NID pattern occurs in the text
- `0`: none of those three classification patterns occurs
- Current distribution: 3,730 PII rows and 2,270 non-PII rows

Locations and names alone are not considered positive by this binary-label script.
This is why the classical classifier task differs from sequence labeling.

### BIO annotation dataset

`data/annotations/bio_labels.json` contains 6,000 tokenized records. Each record has
parallel `tokens` and `labels` arrays:

```json
{
  "tokens": ["amar", "phone", "01712345678"],
  "labels": ["O", "O", "B-PHONE"]
}
```

Supported training labels are:

| Label | Meaning |
|---|---|
| `O` | Not PII |
| `B-PERSON` | Beginning of a person name |
| `B-EMAIL` | Email address |
| `B-PHONE` | Phone number |
| `B-NID` | National ID number |
| `B-LOCATION` | Location |

Current token-label counts are:

| Label | Count |
|---|---:|
| `O` | 26,013 |
| `B-LOCATION` | 3,872 |
| `B-PERSON` | 3,000 |
| `B-EMAIL` | 1,889 |
| `B-PHONE` | 1,216 |
| `B-NID` | 625 |

The annotation logic is dictionary/template based and is intended for coursework
and model comparison, not as a production ground-truth corpus. In particular, it
does not yet provide complete `I-*` labels for every multi-token entity.

### Checked-in sequence splits

The current JSON split files contain:

| File | Records |
|---|---:|
| `data/processed/train.json` | 10,400 |
| `data/processed/validation.json` | 1,300 |
| `data/processed/test.json` | 1,300 |

These files were produced by an earlier 13,000-record generation run and therefore
do not match the current 6,000-row raw CSV. Do not mix artifacts from different data
generations in a controlled experiment. Regenerate all derived files and retrain all
models together when preparing final research results.

## 7. Recreate the dataset

The legacy dataset scripts currently use paths relative to their own directories.
Run them in this exact order:

```powershell
Push-Location src\dataset
python create_dataset.py
python create_labels.py
python annotation.py
python split_dataset.py
Pop-Location
```

This performs the following operations:

1. `create_dataset.py` creates the raw synthetic CSV.
2. `create_labels.py` creates the binary classification dataset.
3. `annotation.py` creates token-level BIO annotations.
4. `split_dataset.py` creates 70% training, 15% validation, and 15% test CSV splits.

Because generation uses random template selection, counts and examples can change
between runs unless a random seed is added to the generator.

## 8. Preprocessing

The preprocessing pipeline applies:

1. NFKC Unicode normalization.
2. Repeated-character reduction.
3. HTML removal.
4. URL removal.
5. Special-character filtering.
6. Whitespace normalization.
7. Lowercasing.
8. NLTK word tokenization.
9. Noise correction such as `phn → phone`, `eml → email`, and `amar → my`.

Download NLTK tokenizers once:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Generate `data/processed/clean_dataset.csv`:

```powershell
Push-Location src\preprocessing
python apply_preprocessing.py
Pop-Location
```

Note: the legacy cleaner retains Latin letters and structured PII characters but
removes Bengali characters. Runtime detection does not use that cleaner and preserves
Bengali input. Improve the cleaner before using it to train a truly multilingual model.

## 9. Model families

### Pattern detector

The runtime pattern detector is always available and recognizes:

- Bangladeshi phone numbers
- Email addresses
- 10-, 13-, and 17-digit NIDs
- Luhn-valid credit-card numbers
- IPv4 addresses

It returns exact character offsets, confidence, type, source, and risk.

### Classical models

Logistic Regression, Multinomial Naive Bayes, and Linear SVM use word-level TF-IDF
features with unigrams and bigrams. They classify a complete sentence as PII or SAFE.

Saved files:

```text
models_saved/logistic_regression.joblib
models_saved/naive_bayes.joblib
models_saved/svm.joblib
```

### BiLSTM

The BiLSTM uses an embedding layer, bidirectional LSTM, padded token sequences, and
six BIO classes. Its checkpoint is saved as:

```text
models_saved/bilstm_pii.pt
```

The vocabulary is rebuilt from `data/annotations/bio_labels.json` when inference
runs, so that file must remain compatible with the checkpoint.

### BERT

BERT uses `bert-base-uncased`, subword tokenization, word-to-subword label alignment,
and a token-classification head. Its local checkpoint is stored in:

```text
models_saved/bert_pii/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

The current checkpoint is English BERT. For stronger Bengali/Banglish research,
replace it with multilingual BERT and retrain using a properly multilingual corpus.

## 10. Train models

### Train classical baselines

```powershell
python train_all.py
```

This command:

1. Reads `classification_dataset.csv`.
2. Creates a stratified 80/20 train/test split.
3. Builds a TF-IDF pipeline for each classifier.
4. Trains Logistic Regression, Naive Bayes, and SVM.
5. Saves the complete pipelines in `models_saved/`.
6. Writes accuracy, precision, recall, and F1 to `results/baseline_results.csv`.

### Train classical models and BiLSTM

```powershell
python train_all.py --bilstm
```

### Train classical models and a small BERT experiment

```powershell
python train_all.py --bert --bert-samples 200
```

### Train all models on the full available datasets

```powershell
python train_all.py --bilstm --bert --bert-samples 0
```

`--bert-samples 0` disables sample limiting. Full BERT training can take a long time
on CPU. Existing checkpoint files are overwritten by successful neural training.

## 11. Evaluate and test

Run the automated tests:

```powershell
python -m pytest -q
```

The tests verify structured PII detection, multilingual text preservation, risk and
redaction behavior, model registration, and safe fallback for an unavailable model.

Evaluation outputs belong in `results/`. The classical trainer creates a table with:

```text
model, accuracy, precision, recall, f1
```

For a defensible report, evaluate every sequence model against one fixed test split
and report entity-level precision, recall, F1, confusion matrices, and representative
false-positive/false-negative examples.

## 12. Run the Streamlit application

Activate the environment and start the root entry point:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

UI workflow:

1. Choose a model in the sidebar.
2. Check whether its artifact is ready.
3. Keep **Safe fallback** enabled for crash-resistant testing.
4. Set the redaction confidence threshold.
5. Choose an English, Bengali, or Banglish example, or enter custom text.
6. Select **Run model**.
7. Review the model used, entities, overall risk, classification, and protected text.
8. Download the protected text if required.

The first BERT request is slower because the approximately 400 MB checkpoint is
loaded into memory. Models are cached after loading. If a selected model is absent or
incompatible, the UI reports the reason and uses the pattern detector when fallback
is enabled.

## 13. Run the REST API

Start the backend:

```powershell
python api.py
```

Available URLs:

- Health check: `http://127.0.0.1:8000/health`
- Interactive documentation: `http://127.0.0.1:8000/docs`
- Scan endpoint: `POST http://127.0.0.1:8000/scan`

Example PowerShell request:

```powershell
$body = @{
    text = "My phone is 01712345678"
    confidence_threshold = 0.8
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/scan" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Example response fields:

```json
{
  "original": "My phone is 01712345678",
  "entities": [
    {
      "entity": "01712345678",
      "type": "PHONE",
      "start": 12,
      "end": 23,
      "confidence": 0.99,
      "source": "pattern",
      "risk": "HIGH",
      "score": 3
    }
  ],
  "entity_count": 1,
  "overall_risk": "HIGH",
  "safe_text": "My phone is [PHONE_NUMBER]"
}
```

The response includes entity text, type, offsets, confidence, source, risk, and score.

## 14. Risk scoring and redaction

| Entity | Score | Risk | Placeholder |
|---|---:|---|---|
| Location | 1 | LOW | `[LOCATION]` |
| Person / Date | 2 | MEDIUM | `[PERSON]` / `[DATE]` |
| Phone / Email / IP | 3 | HIGH | `[PHONE_NUMBER]` / `[EMAIL]` / `[IP_ADDRESS]` |
| NID / Account / Credit card | 4 | CRITICAL | Type-specific placeholder |

Overall risk is the highest risk among detected entities. Redaction is performed
from right to left using character offsets, preventing earlier replacements from
invalidating later spans.

## 15. Project structure

```text
NLP_PROJECT/
├── app.py                         Streamlit entry point
├── api.py                         FastAPI entry point
├── train_all.py                   Automated model trainer
├── requirements.txt               Python dependencies
├── pytest.ini                     Test discovery configuration
├── data/
│   ├── raw/                       Generated source CSV
│   ├── annotations/               BIO annotation JSON
│   └── processed/                 Classification, cleaned, and split data
├── models_saved/                  BERT, BiLSTM, and classical artifacts
├── results/                       Metrics and plots
├── tests/                         Automated tests
└── src/
    ├── app/                       Streamlit page implementation
    ├── dataset/                   Generation and annotation scripts
    ├── preprocessing/             Cleaning and tokenization
    ├── features/                  TF-IDF, n-grams, and Word2Vec
    ├── models/
    │   ├── baseline/              Original classical experiments
    │   ├── lstm/                  BiLSTM dataset/model/trainer
    │   └── transformer/           BERT dataset/model/trainer
    ├── inference/model_service.py Unified model loading and inference
    ├── evaluation/                Metrics and visualization helpers
    └── redection/                 Detection, risk, and redaction engine
```

The directory name `redection` is retained for compatibility even though
`redaction` is the correct spelling.

## 16. Common problems

### A model is marked unavailable

Train it with `train_all.py` and confirm its expected file exists in `models_saved/`.
The UI remains usable through Safe fallback.

### scikit-learn DLL is blocked

Some Windows Application Control configurations block compiled packages in an older
environment. Delete that environment manually, create a clean `.venv`, and reinstall:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### NLTK tokenizer resource is missing

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### BERT is slow

The first inference loads the complete checkpoint. Later requests reuse the cached
model. Use the Pattern detector for fast functional testing.

### BiLSTM output looks inaccurate

The synthetic vocabulary is small and the current checkpoint is experimental.
Regenerate a consistent dataset, expand the vocabulary, add full BIO labels, retrain,
and evaluate against a held-out split.

### Port 8501 is already in use

```powershell
python -m streamlit run app.py --server.port 8502
```

## 17. Recommended research workflow

1. Freeze a dataset version and random seed.
2. Correct and validate BIO annotations.
3. Create one reproducible train/validation/test split.
4. Train classical baselines.
5. Train and tune BiLSTM.
6. Fine-tune multilingual BERT.
7. Compare entity-level metrics on the same test set.
8. Perform language-wise and entity-wise error analysis.
9. Record hardware, hyperparameters, runtime, and random seeds.
10. Generate result tables, confusion matrices, training curves, and ablations.

## 18. Current limitations

- The dataset is synthetic and has limited names, locations, templates, and linguistic diversity.
- Current raw and checked-in sequence-split files come from different generation runs.
- BIO data primarily uses `B-*` rather than complete multi-token `B-*`/`I-*` spans.
- The preprocessing cleaner removes Bengali characters and needs revision for multilingual training.
- The BERT base model is English rather than multilingual.
- Classical models perform document classification, not entity extraction.
- Pattern detection deliberately favors high precision and does not recognize every possible PII format.
- The API currently exposes the pattern-based privacy firewall; model comparison is provided by Streamlit.

## 19. Quick command reference

```powershell
# Setup
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Tests
python -m pytest -q

# Classical training
python train_all.py

# All model training
python train_all.py --bilstm --bert --bert-samples 0

# Streamlit
python -m streamlit run app.py

# API
python api.py
```

PrivacyGuard is an educational/research project. Do not treat synthetic-data model
scores as evidence that the system is ready to protect real production records
without broader datasets, security review, and independent evaluation.
