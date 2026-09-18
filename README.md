# PrivacyGuard — Multilingual PII Detection and Redaction

PrivacyGuard is a local NLP system for detecting personally identifiable information (PII), assigning a privacy-risk level, and producing redacted text. It supports English, Bengali, and Banglish text and provides several detectors for experimentation:

- Pattern-based detection
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Bidirectional LSTM (BiLSTM)
- Multilingual BERT (mBERT)

The Streamlit Model Lab lets you test these detectors from one interface. A FastAPI service is also included for local REST integration.

## 1. Main capabilities

- Detect names, locations, organizations, addresses, phone numbers, email addresses, national IDs, account numbers, cards, passports, medical IDs, employee IDs, student IDs, IP addresses, dates, health conditions, occupations, and education information.
- Preserve multi-token entities such as `Sarah Ahmed`, `Dhaka University`, and `TechNova Solution Ltd.` as complete spans through BIO sequence labels.
- Train and compare classical, recurrent neural, and Transformer models.
- Calculate entity-level and overall privacy risk.
- Redact detected character spans without changing unrelated text.
- Test models through Streamlit and download the protected result.
- Use a local FastAPI endpoint for pattern-based scans.

## 2. System architecture

```text
                         TRAINING

Raw classification data ──> TF-IDF ──> LR / NB / SVM artifacts

Synthetic multilingual NER data ──> BIO sequences
                                  ├──> Vocabulary ──> BiLSTM artifact
                                  └──> WordPiece ──> mBERT artifact


                         INFERENCE

User text
   │
   ├── Pattern detector ─────────────────────────────┐
   ├── BiLSTM token classifier ──> BIO spans ────────┤
   ├── mBERT token classifier ──> BIO spans ─────────┤
   └── Classical document classifier ─> patterns ────┤
                                                     ▼
                                      Character-offset entities
                                                     │
                                      Risk scoring and filtering
                                                     │
                                             Span redaction
                                                     ▼
                                      Protected text and report
```

There are two different prediction tasks in this project:

1. **Document classification:** Logistic Regression, Naive Bayes, and SVM decide whether an entire text is sensitive. They do not independently locate entity boundaries. When one of these models is selected, structured patterns provide the spans needed for display and redaction.
2. **Token classification (NER):** BiLSTM and BERT assign a BIO label to each token. They can therefore learn both the entity type and its start/end boundary from annotated examples.

## 3. Project structure

```text
NLP_PROJECT/
├── app.py                         # Streamlit entry point
├── api.py                         # FastAPI entry point
├── train_all.py                   # Main training orchestrator
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── pii_dataset.csv        # Document-classification data
│   └── processed/
│       ├── ner_dataset.json       # Complete NER dataset
│       ├── train.json
│       ├── validation.json
│       └── test.json
├── models_saved/                  # Trained model artifacts
├── results/                       # Metrics and experiment output
├── src/
│   ├── app/app.py                 # Streamlit Model Lab
│   ├── dataset/                   # Dataset generation and labels
│   ├── features/                  # TF-IDF and n-gram features
│   ├── inference/model_service.py # Unified model loading/inference
│   ├── models/
│   │   ├── baseline/              # LR, NB, and SVM training
│   │   ├── lstm/                  # BiLSTM model and training
│   │   └── transformer/           # BERT dataset/model/training
│   ├── evaluation/                # BERT held-out evaluation
│   └── redection/                 # Detection, risk, and redaction
└── tests/
    └── test_privacyguard.py
```

The package name `redection` is used by the current imports and should be kept unchanged when running the project.

## 4. Environment setup

### Requirements

- Python 3.10 or 3.11 is recommended.
- Git
- At least 8 GB RAM for classical models and BiLSTM
- More memory and an NVIDIA CUDA GPU are strongly recommended for full BERT training

### Windows PowerShell

Run every command from the project root:

```powershell
cd "E:\academic\4.1\NLP_lab\NLP_PROJECT"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, allow it for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
cd /path/to/NLP_PROJECT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm that the package imports correctly:

```bash
python -c "from src.inference.model_service import ModelService; print(ModelService().available_models())"
```

## 5. Dataset design

### 5.1 Document-classification dataset

`data/raw/pii_dataset.csv` is used by the classical models. Each row contains text and a document-level class indicating whether the text contains sensitive information. The classical pipeline converts a complete input into a TF-IDF vector and predicts one class for the whole document.

This data is suitable for questions such as:

> Does this message contain PII?

It is not sufficient by itself for questions such as:

> Which exact characters form the person's name?

Exact span detection requires the NER dataset.

### 5.2 Multilingual NER dataset

`data/processed/ner_dataset.json` contains token sequences and a label for every token. It is generated from varied templates and value pools in English, Bengali, and Banglish. Entity values are inserted into labeled slots, so annotation is created from the data-generation structure rather than guessed afterward with regular expressions.

The default generation command creates 30,000 examples and reproducible train, validation, and test splits:

```bash
python -m src.dataset.generate_ner_dataset --samples 30000
```

The default split is:

- Training: 80%, or 24,000 examples
- Validation: 10%, or 3,000 examples
- Test: 10%, or 3,000 examples

The random seed is fixed for reproducibility.

### 5.3 Entity types

The NER models support these entity classes:

| Group | Entity types |
|---|---|
| Identity | `PERSON`, `NID`, `PASSPORT` |
| Contact | `PHONE`, `EMAIL`, `ADDRESS`, `IP_ADDRESS` |
| Place and organization | `LOCATION`, `ORGANIZATION` |
| Financial | `ACCOUNT`, `CREDIT_CARD` |
| Employment and study | `EMPLOYEE_ID`, `STUDENT_ID`, `OCCUPATION`, `EDUCATION` |
| Medical | `MEDICAL_ID`, `HEALTH_CONDITION` |
| General context | `DATE` |

### 5.4 BIO annotation

BIO labels encode both type and boundary:

- `B-TYPE`: first token of an entity
- `I-TYPE`: continuation token of the same entity
- `O`: token outside all entities

Example:

```text
Token:  My      name    is      Sarah       Ahmed       from      Dhaka
Label:  O       O       O       B-PERSON    I-PERSON    O         B-LOCATION
```

Because `Sarah` starts the person and `Ahmed` continues it, inference merges them into one `PERSON` span. Correct BIO consistency is essential: an `I-PERSON` token should not start an unrelated entity.

Multi-token BIO labels are enabled for names, locations, organizations, addresses, cards, dates, health conditions, occupations, and education entities. Single-token identifiers can still contain punctuation or digits as one lexical token.

### 5.5 Improving dataset quality

Maximum accuracy cannot be guaranteed by increasing the sample count alone. For reliable real-world performance, the training data should contain:

- Many names and locations that do not repeat between train and test sets
- English, Bengali script, and natural Banglish spelling variations
- Long messages with several entity types in the same paragraph
- Negative examples containing ordinary numbers and capitalized words
- Hard examples such as phone numbers versus NIDs versus account numbers
- Spelling mistakes, punctuation, honorifics, and code-mixed sentences
- Organization, hospital, university, job, and address contexts
- Manually reviewed real examples with consent and safely anonymized values

Never place real secrets or unapproved personal data in the repository. Synthetic data is useful for bootstrapping, but a manually reviewed, representative validation and test set is necessary to measure real accuracy.

## 6. Preprocessing and token alignment

PrivacyGuard preserves character positions because redaction ultimately operates on the original input. Aggressive cleaning can destroy those positions, so normalization must not silently rewrite the inference text before offsets are calculated.

The main preprocessing responsibilities are:

- Unicode-aware handling for Bengali and Latin text
- Stable token boundaries for words, punctuation, emails, phone numbers, and IDs
- Padding and truncation for fixed-size neural batches
- Label alignment between source words and model tokens
- Ignoring padding and non-supervised subword positions in the loss

For BERT, a word can be divided into several WordPiece tokens. The first subword receives the word's BIO label; later subwords use `-100` so PyTorch excludes them from cross-entropy loss. During inference, WordPiece predictions are mapped back to source character offsets before adjacent BIO tokens are joined.

## 7. Model architectures and operation

### 7.1 Pattern detector

The pattern detector is a deterministic baseline and safety option. It does not require training.

**Architecture**

```text
Original text
   ├── Regular expressions for structured identifiers
   ├── Context rules for selected names and locations
   └── Candidate overlap resolution
                ↓
      Typed character spans
```

Regular expressions are well suited to values with recognizable structure, including emails, Bangladeshi phone numbers, NIDs, cards, accounts, and IP addresses. Context rules use nearby phrases such as “my name is” or location indicators to propose less structured spans. Candidate spans are ranked and overlaps are resolved so the same characters are not redacted twice.

**Strengths:** fast, explainable, deterministic, and useful when no trained artifact is available.

**Limitations:** it cannot learn arbitrary names or locations, and rules require maintenance as formats change. It is a safety layer for structured PII, not a replacement for a trained NER model.

### 7.2 Shared TF-IDF architecture for classical models

Logistic Regression, Naive Bayes, and SVM share the same feature stage:

```text
Document
   ↓
Word unigrams and bigrams
   ↓
TF-IDF weighting (maximum 10,000 features)
   ↓
Sparse document vector
```

Term frequency represents how often a feature occurs in a document. Inverse document frequency reduces the influence of terms that appear in most documents. Word bigrams add short phrases such as `phone number` or `national id` that are more informative than individual words.

#### Logistic Regression

Logistic Regression learns one weight per TF-IDF feature plus an intercept. The dot product between the document vector and learned weights is converted to a probability with a logistic function. Features strongly associated with PII acquire positive weights; features associated with safe text acquire negative weights.

```text
TF-IDF vector x ──> z = w·x + b ──> sigmoid(z) ──> document class
```

It is usually a strong, fast, interpretable text baseline. Its limitation is that it predicts a document class and does not emit token-level entity spans.

#### Multinomial Naive Bayes

Naive Bayes estimates how likely each feature is under each class and combines those likelihoods with the class prior. It makes the simplifying assumption that features are conditionally independent given the class.

```text
TF-IDF features ──> class priors + feature likelihoods ──> most probable class
```

It trains very quickly and works well on many sparse text problems. Correlated phrases and subtle long-range context are weaknesses because the independence assumption does not model sequence structure.

#### Linear SVM

Linear SVM learns a separating hyperplane with the largest possible margin between sensitive and non-sensitive training examples.

```text
TF-IDF vector ──> signed distance from hyperplane ──> document class
```

The large-margin objective often performs strongly with high-dimensional sparse text. The implementation is a linear document classifier, not a sequence labeler; it therefore relies on the structured detector to produce redaction spans in the application.

### 7.3 BiLSTM sequence labeler

The BiLSTM is a neural named-entity recognizer that predicts one BIO label for every input word.

**Architecture**

```text
Whitespace tokens (maximum length 32)
                ↓
Vocabulary lookup and padding
                ↓
100-dimensional trainable embedding
                ↓
Bidirectional LSTM, hidden size 128 per direction
        ┌──────────────┴──────────────┐
        │ forward context             │ backward context
        └──────────────┬──────────────┘
                256-dimensional state
                          ↓
                Linear classification head
                          ↓
              BIO logits for every token
```

The forward LSTM reads left to right and the backward LSTM reads right to left. Their hidden states are concatenated, so a token prediction can use words on both sides. A linear layer maps the 256-dimensional combined state to the BIO label vocabulary.

Training uses cross-entropy loss with padding positions set to `-100`, Adam with a default learning rate of `0.001`, and a default 10 epochs. The saved checkpoint contains the network weights, word-to-index vocabulary, label list, and maximum length needed for identical inference behavior.

At inference time, tokens are converted to vocabulary IDs, padded, processed by the network, and normalized with softmax. BIO predictions are merged into spans and mapped to character offsets. Unknown words use the vocabulary's unknown token; consequently, vocabulary diversity is important. The short maximum sequence length also means long inputs should be expanded through chunking or a larger configured limit in future experiments.

### 7.4 Multilingual BERT token classifier

The Transformer model fine-tunes `google-bert/bert-base-multilingual-cased` for token classification. This multilingual encoder can represent English, Bengali, and mixed-script text in a shared contextual space.

**Architecture**

```text
Original words
    ↓
Multilingual WordPiece tokenizer
    ↓
Token + position + segment embeddings
    ↓
Stack of bidirectional Transformer encoder blocks
    ├── Multi-head self-attention
    ├── Residual connection and normalization
    ├── Feed-forward network
    └── Residual connection and normalization
    ↓
Contextual representation for every subword
    ↓
Linear token-classification head
    ↓
BIO label logits
```

Self-attention allows each token to weigh relevant tokens anywhere in the input. For example, a number can be interpreted differently when preceded by `phone`, `national id`, `account`, or `medical id`. Unlike the BiLSTM's step-by-step recurrence, attention processes relationships between all visible positions directly.

The training dataset uses a maximum sequence length of 256. Only the first subword of each source word is supervised; special tokens, padding, and continuation subwords receive the ignored label `-100`. The loss is class-weighted cross entropy: rare entity labels receive more weight, while the dominant `O` class is capped so it does not overwhelm entity learning.

Optimization uses AdamW, a default learning rate of `2e-5`, weight decay, a linear learning-rate schedule with warmup, gradient clipping, and optional gradient accumulation. Validation runs after each epoch. The checkpoint with the best entity-token F1 is saved, and a confidence threshold is calibrated on validation predictions.

Inference loads the local tokenizer, model, label mapping, and calibrated threshold. Token probabilities come from softmax. Predictions are aligned back to source words, BIO segments are merged, and character offsets are recovered for risk scoring and redaction.

**Strengths:** contextual multilingual representations, subword handling, and the strongest potential for previously unseen names and phrases.

**Limitations:** higher training cost, sensitivity to label quality and class imbalance, and no guarantee of recognizing entity distributions absent from training data. A high synthetic-data score should always be confirmed on manually reviewed real-world text.

### 7.5 Model comparison

| Model | Prediction level | Learns boundaries | Main advantage | Main limitation |
|---|---|---:|---|---|
| Pattern | Character span | Rule-defined | Fast and deterministic | Cannot generalize like NER |
| Logistic Regression | Document | No | Fast, interpretable baseline | No learned spans/context |
| Naive Bayes | Document | No | Extremely fast training | Strong independence assumption |
| Linear SVM | Document | No | Strong sparse-text margin | No learned spans/probability by default |
| BiLSTM | Token | Yes | Learns sequential context | Fixed vocabulary and short sequences |
| mBERT | Subword/token | Yes | Contextual and multilingual | Most computationally expensive |

## 8. Training commands

Run commands from the repository root with the virtual environment active.

### 8.1 Train classical models

```bash
python train_all.py
```

This trains Logistic Regression, Naive Bayes, and Linear SVM and writes their artifacts under `models_saved/`. Baseline metrics are written to `results/baseline_results.csv`.

### 8.2 Train BiLSTM

Generate 30,000 NER examples and train the BiLSTM:

```bash
python train_all.py --skip-classical --bilstm --ner-samples 30000
```

Reuse existing NER splits instead of generating them again:

```bash
python train_all.py --skip-classical --bilstm --skip-ner-generation
```

### 8.3 Train BERT

Full training command:

```bash
python train_all.py --skip-classical --bert --bert-samples 0 --ner-samples 30000 --bert-epochs 3 --bert-batch-size 8
```

`--bert-samples 0` means use the complete generated training split. On a CPU, this can take much longer than a few minutes. Training time depends on the processor/GPU, memory, disk speed, sequence lengths, and whether model files are already cached.

For a quick pipeline test, use a small subset and one epoch:

```bash
python train_all.py --skip-classical --bert --bert-samples 100 --ner-samples 1000 --bert-epochs 1 --bert-batch-size 4
```

This quick command verifies that training and saving work; it is not intended to produce an accurate production model.

If GPU memory is limited, lower the physical batch size and accumulate gradients:

```bash
python train_all.py --skip-classical --bert --bert-samples 0 --skip-ner-generation --bert-epochs 3 --bert-batch-size 4 --bert-gradient-accumulation 2
```

The effective batch size in this example is `4 × 2 = 8` without storing all eight samples in memory simultaneously.

### 8.4 Train every model

```bash
python train_all.py --bilstm --bert --ner-samples 30000 --bert-samples 0 --bert-epochs 3 --bert-batch-size 8
```

### 8.5 Important training options

| Option | Meaning |
|---|---|
| `--skip-classical` | Do not train LR, NB, and SVM |
| `--bilstm` | Train the BiLSTM sequence labeler |
| `--bert` | Fine-tune BERT and evaluate the best checkpoint |
| `--ner-samples N` | Generate `N` NER examples |
| `--skip-ner-generation` | Reuse the existing train/validation/test files |
| `--bert-samples N` | Limit BERT training examples; `0` uses all |
| `--bert-epochs N` | Set the number of BERT epochs |
| `--bert-batch-size N` | Set BERT's physical batch size |
| `--bert-gradient-accumulation N` | Accumulate gradients across `N` batches |

## 9. Evaluation

### Classical results

Classical training records accuracy, precision, recall, and F1 in:

```text
results/baseline_results.csv
```

These are document-level classification metrics and must not be compared directly with token- or entity-level NER metrics without clearly labeling the task.

### BERT validation and test results

During training, BERT records epoch history and selects the best validation checkpoint. The pipeline measures token accuracy and entity-token micro precision, recall, and F1 while excluding the `O` class from entity metrics.

Training history:

```text
results/bert_metrics.json
```

Held-out test evaluation runs automatically after BERT training. It can also be run manually:

```bash
python -m src.evaluation.evaluate_bert --batch-size 8
```

Test metrics:

```text
results/bert_test_metrics.json
```

For research reporting, also evaluate exact entity spans. A prediction should count as correct only when both its type and complete boundary match the reference. Review false positives, false negatives, type confusion, broken multi-word spans, and performance separately for English, Bengali, and Banglish.

Avoid tuning on the test set. Use training data to learn weights, validation data to select settings and confidence thresholds, and the test set once for the final unbiased report.

## 10. Unified inference flow

`src/inference/model_service.py` is the common interface used by the Streamlit app. It checks whether the requested artifact is ready, loads it when needed, and returns a consistent result containing:

- Requested and actually used model
- Detected entity text and type
- Confidence and source
- Start and end character positions
- Per-entity and overall risk
- Protected text
- Warning or fallback information

Behavior differs by model family:

- Pattern mode directly returns structured and contextual spans.
- BiLSTM and BERT return their own learned BIO spans when inference succeeds.
- Classical models classify the entire text; patterns locate the spans used for redaction.
- If an artifact is unavailable or a neural model raises an error, optional safe fallback can run the pattern detector instead of crashing the UI.

The fallback is clearly reported so pattern results are not mistaken for neural predictions.

## 11. Privacy risk scoring

Every detected type has a severity weight. The highest entity severity becomes the overall result.

| Risk | Weight | Examples |
|---|---:|---|
| Low | 1 | Location |
| Medium | 2 | Person, organization, date, occupation, education |
| High | 3 | Phone, email, IP address, address, employee ID, student ID |
| Critical | 4 | NID, passport, account, card, medical ID, health condition |

Risk is a configurable application policy, not a legal conclusion. Adjust the mapping for the jurisdiction and use case before deployment.

## 12. Redaction engine

Only entities at or above the selected confidence threshold are redacted. Each accepted entity is represented by original-text character offsets:

```json
{
  "entity": "Sarah Ahmed",
  "type": "PERSON",
  "start": 11,
  "end": 22,
  "confidence": 0.94
}
```

The engine replaces accepted spans with typed placeholders such as `[PERSON]`, `[PHONE]`, or `[NID]`. Replacements are applied from right to left, preventing an early replacement from shifting the offsets of later entities.

Example:

```text
Before: My name is Sarah Ahmed and my phone is +8801712345678.
After:  My name is [PERSON] and my phone is [PHONE].
```

## 13. Run the Streamlit application

Start the Model Lab:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The sidebar provides:

- Detector selection
- Artifact readiness status
- Minimum redaction confidence
- Safe-fallback control

The main panel accepts custom or example text, runs the selected model, displays the model actually used, overall risk, protected text, and a table of detected spans. Protected text can be downloaded.

When the UI says an artifact is not ready, train that model first and restart Streamlit. Streamlit caches model resources for faster reruns, so restarting is the simplest way to guarantee that a newly saved artifact is loaded.

## 14. Run the FastAPI service

Start the backend:

```bash
python api.py
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Pattern-based scan example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/scan" `
  -ContentType "application/json" `
  -Body '{"text":"My phone is +8801712345678"}'
```

The current REST endpoint uses the privacy firewall's pattern detector. Multi-model comparison is provided by the Streamlit application through the unified model service.

## 15. Testing

Run the automated test suite:

```bash
pytest -q
```

Tests cover structured PII, Bengali input, card validation, name/location/NID handling, model registry behavior, fallback behavior, direct NER-label inference, and lexical span preservation.

Before accepting a newly trained model, also test a manually reviewed collection containing:

- Previously unseen full names
- Multi-word organizations and locations
- Bengali and Banglish input
- Several identifier formats
- Long mixed-entity paragraphs
- Safe text with no PII
- Ambiguous numeric values that must not be mislabeled

## 16. Recommended experiment workflow

1. Review entity definitions and annotation rules.
2. Generate the multilingual NER data.
3. Manually inspect samples and repair systematic template or label errors.
4. Train classical baselines.
5. Train BiLSTM and establish a neural sequence baseline.
6. Fine-tune BERT using the full training split.
7. Select settings using validation F1, not test performance.
8. Evaluate once on the held-out test split.
9. Perform exact-span and language-specific error analysis.
10. Test through Streamlit with realistic long text.
11. Add corrected failure cases to the training distribution without leaking test examples.
12. Retrain, version artifacts, and record the data version and command used.

## 17. Accuracy guidance

For stronger NER accuracy:

- Prefer BERT for the final detector and use BiLSTM as a comparison model.
- Increase diversity before simply increasing duplicate template volume.
- Balance rare entity types and include confusing negative examples.
- Keep entity values disjoint across data splits where possible.
- Correct label boundaries, especially for multi-word names and organizations.
- Track per-class precision, recall, and F1 rather than only overall accuracy.
- Tune the confidence threshold on validation data.
- Keep structured patterns as a defense-in-depth layer for identifiers where recall is critical.
- Measure results on real, manually labeled, consented data from the intended domain.

No model can detect every sensitive value with perfect accuracy. Production privacy protection should combine a trained model, validated structured rules, access controls, secure logging, human review for high-risk workflows, and continuous monitoring.

## 18. Troubleshooting

### BERT reports that its artifact is unavailable

Train it and restart Streamlit:

```bash
python train_all.py --skip-classical --bert --bert-samples 0 --skip-ner-generation --bert-epochs 3 --bert-batch-size 8
streamlit run app.py
```

### CUDA runs out of memory

Use a smaller batch and gradient accumulation:

```bash
python train_all.py --skip-classical --bert --bert-samples 0 --skip-ner-generation --bert-batch-size 2 --bert-gradient-accumulation 4
```

### Training is too slow

First verify the pipeline with a small subset. Then train the full dataset on a CUDA-capable GPU. A ten-minute target is reasonable only for a smoke test or sufficiently fast hardware; it is not a reliable full-training promise.

### A full name is split or includes surrounding words

Check that the training sequence uses `B-PERSON` for the first name token, `I-PERSON` for every continuation token, and `O` for nearby grammar. Add varied corrected examples rather than adding a special-case name function.

### NID, phone, and account are confused

Add negative and contextual examples for all competing number formats. A neural model needs phrases such as `national id`, `phone`, and `account number` across multiple languages and writing styles. Keep structured validation rules for identifiers whose formats are known.

### The application uses the pattern detector unexpectedly

Read the UI warning and the “model used” field. The selected artifact may be missing or inference may have failed. Disable safe fallback while debugging if you want the original exception to be surfaced instead of receiving pattern results.

## 19. End-to-end quick start

```powershell
cd "E:\academic\4.1\NLP_lab\NLP_PROJECT"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.dataset.generate_ner_dataset --samples 30000
python train_all.py
python train_all.py --skip-classical --bilstm --skip-ner-generation
python train_all.py --skip-classical --bert --bert-samples 0 --skip-ner-generation --bert-epochs 3 --bert-batch-size 8
pytest -q
streamlit run app.py
```

In another activated terminal, start the API if needed:

```powershell
cd "E:\academic\4.1\NLP_lab\NLP_PROJECT"
.\.venv\Scripts\Activate.ps1
python api.py
```

The complete local flow is:

```text
Dataset generation
       ↓
Model training and validation
       ↓
Saved artifacts
       ↓
Streamlit or FastAPI input
       ↓
PII detection
       ↓
Risk classification
       ↓
Confidence filtering and redaction
       ↓
Protected output
```
