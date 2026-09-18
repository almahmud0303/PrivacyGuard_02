# PrivacyGuard

PrivacyGuard is a local English/Bengali/Banglish privacy firewall. It detects PII,
assigns risk, and redacts sensitive spans. The repository also contains classical
ML, BiLSTM, and BERT training/evaluation experiments.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a new `.venv` if an old environment reports DLL or application-control errors.

## Run

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`.

Run the REST backend with `python api.py`. Its interactive documentation is at
`http://127.0.0.1:8000/docs`; `POST /scan` accepts:

```json
{"text": "My phone is 01712345678", "confidence_threshold": 0.8}
```

## Train

```powershell
python train_all.py
python train_all.py --bilstm
python train_all.py --bert --bert-samples 200
python train_all.py --bilstm --bert --bert-samples 0
```

The default trains Logistic Regression, Naive Bayes, and SVM, saves pipelines in
`models_saved/`, and writes metrics to `results/baseline_results.csv`. A value of
`0` for `--bert-samples` uses the full BIO dataset. BERT is slow and ideally uses
a CUDA GPU.

## Test

```powershell
python -m pytest -q
```

## Architecture

```text
Browser -> Streamlit -> PrivacyGuard engine -> detection -> risk -> redaction
Client  -> FastAPI   -> PrivacyGuard engine -> JSON response
Dataset -> train_all.py -> classical/BiLSTM/BERT models -> result CSV files
```

Key paths are `app.py`, `api.py`, `train_all.py`, `src/redection/` (legacy name),
`src/models/`, `data/`, `models_saved/`, and `results/`.

Runtime detection uses high-precision patterns for phone, email, NID, valid
credit-card, and IP values. Experimental BERT/BiLSTM code and saved artifacts are
kept separate so retraining a research checkpoint cannot silently weaken safe
redaction.
