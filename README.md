# NLP PII Detection Project

This project prepares text data containing personally identifiable information (PII), creates BIO-style labels, applies text preprocessing, and builds classical NLP features and Word2Vec embeddings.

The current repository is at the feature-engineering stage. Dataset preparation, preprocessing, n-gram features, TF-IDF features, and Word2Vec training are implemented. A final trainable PII-recognition model and an inference application are not implemented yet.

## 1. Project Setup

### Requirements

- Windows PowerShell
- Python 3.11 or a compatible Python 3 version
- Internet access for installing Python packages

The project uses a local virtual environment named `venv`. The environment used during development was Python 3.11.13.

### Create and activate the virtual environment

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run PowerShell as the current user and enable local scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

The dependencies are listed in `requirements.txt`.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important packages used by the current code include:

- `pandas` for CSV data handling
- `nltk` for tokenization
- `scikit-learn` for train/test splitting, n-grams, and TF-IDF
- `gensim` for Word2Vec embeddings
- `numpy`, `matplotlib`, `seaborn`, `torch`, `transformers`, and related NLP tools for later stages

`gensim` was added to `requirements.txt` after the Word2Vec test reported `ModuleNotFoundError: No module named 'gensim'`.

### Select the VS Code interpreter

In VS Code, select the interpreter at:

```text
venv\Scripts\python.exe
```

To avoid accidentally using a different system Python, the explicit interpreter command is always safe:

```powershell
e:/academic/4.1/NLP_lab/NLP_PROJECT/venv/Scripts/python.exe script.py
```

## 2. Repository Structure

```text
data/
	raw/
		pii_dataset.csv                 Raw text dataset
	processed/
		clean_dataset.csv               Text plus serialized cleaned tokens
		train.csv                       Training split, when generated
		val.csv                         Validation split, when generated
		test.csv                        Test split, when generated
		bio_dataset.json                Token and label records
		train.json                      JSON training records currently present
		validation.json                 JSON validation records currently present
		test.json                       JSON test records currently present

src/
	config.py                         Shared project paths and settings
	dataset/
		create_dataset.py               Generate synthetic raw records
		annotation.py                   Create BIO-style annotations
		split_dataset.py                Create train/validation/test CSV splits
		test_dataset.py                 Inspect raw data and annotations
	preprocessing/
		normalizer.py                   Unicode and repeated-character normalization
		cleaner.py                     HTML, URL, character, and whitespace cleaning
		tokenizer.py                   NLTK tokenization helpers
		noise_handler.py               Common typo/noise replacement map
		pipeline.py                    Combined preprocessing pipeline
		apply_preprocessing.py         Create clean_dataset.csv
		test_preprocessing.py          Manual preprocessing examples
	features/
		ngram_features.py              Word and character n-gram features
		tfidf_features.py              TF-IDF features and vectorizer persistence
		embeddings.py                  Word2Vec training and sentence embeddings
	evaluation/                      Metric and visualization modules
	models/                           Reserved for model implementations
	inference/                        Reserved for inference code
	training/                         Reserved for training code

test.py                             Root-level Word2Vec smoke test
notebooks/01_EDA.ipynb              EDA notebook scaffold
models_saved/                       Generated Word2Vec and TF-IDF artifacts
saved_models/                       Reserved model directory
```

## 3. Data Creation

`src/dataset/create_dataset.py` creates 5,000 synthetic text records from templates containing names, phone numbers, email addresses, NIDs, and locations. It writes the result to `data/raw/pii_dataset.csv`.

Because the script uses `../../data/...` paths, run it from its own directory:

```powershell
cd src/dataset
python create_dataset.py
cd ../..
```

The raw CSV contains a text column and generated records similar to:

```text
My name is John Smith and my phone number is 01712345678
Contact me at john@yahoo.com
I live at Dhaka Bangladesh
```

## 4. BIO Annotation

`src/dataset/annotation.py` reads the raw text and creates token-level labels. The current recognizers include:

- `B-PHONE` for Bangladeshi-style phone numbers beginning with `01`
- `B-EMAIL` for email-like values
- `B-NID` for ten-digit numeric values
- `B-PERSON` for names in its hard-coded name list
- `B-LOCATION` for locations in its hard-coded location list
- `O` for other tokens

Run it from `src/dataset`:

```powershell
cd src/dataset
python annotation.py
cd ../..
```

The output is written to `data/annotations/bio_labels.json`. Each record contains parallel `tokens` and `labels` arrays, for example:

```json
{
	"tokens": ["email", "me", "john@gmail.com"],
	"labels": ["O", "O", "B-EMAIL"]
}
```

## 5. Dataset Splitting

`src/dataset/split_dataset.py` splits the raw CSV with a fixed random seed:

- 70 percent training data
- 15 percent validation data
- 15 percent test data

Run it from `src/dataset`:

```powershell
cd src/dataset
python split_dataset.py
cd ../..
```

The CSV files are written to `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv`.

## 6. Preprocessing Pipeline

The preprocessing pipeline in `src/preprocessing/pipeline.py` applies these steps in order:

1. Unicode normalization using NFKC normalization.
2. Repeated-character reduction, such as reducing a character repeated three or more times.
3. HTML tag removal.
4. URL removal.
5. Special-character filtering while preserving letters, numbers, spaces, `@`, `.`, and `_`.
6. Whitespace normalization.
7. Lowercasing.
8. NLTK word tokenization.
9. Noise correction using mappings such as `phn` to `phone`, `nmbr` to `number`, and `addr` to `address`.

### Generate the processed CSV

Run the preprocessing script from `src/preprocessing` because its input and output paths are relative to that directory:

```powershell
cd src/preprocessing
python apply_preprocessing.py
cd ../..
```

The output is `data/processed/clean_dataset.csv`. It contains:

- `text`: original input text
- `tokens`: a serialized Python-list representation of the cleaned tokens

For example, a row may contain:

```text
text: my phn no is 01712345678
tokens: ['my', 'phone', 'no', 'is', '01712345678']
```

### Test preprocessing manually

```powershell
cd src/preprocessing
python test_preprocessing.py
cd ../..
```

This prints the original and processed form of several sample sentences.

NLTK tokenizers may require downloaded tokenizer data. If NLTK reports that tokenizer data is missing, run the following once in the project environment:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 7. Feature Engineering

### Word n-grams

`src/features/ngram_features.py` provides word n-gram features with a range of 1 to 3 and character n-gram features with a range of 3 to 5. Character features are capped at 3,000 features.

The word n-gram function returns a sparse feature matrix and its `CountVectorizer`.

### TF-IDF

`src/features/tfidf_features.py` creates word-level TF-IDF features using 1- and 2-grams, limited to 5,000 features. It fits on training text, transforms test text, and saves the vectorizer to:

```text
models_saved/tfidf_vectorizer.pkl
```

The module can be run from `src/features` after the CSV splits have been generated:

```powershell
cd src/features
python tfidf_features.py
cd ../..
```

### Word2Vec embeddings

`src/features/embeddings.py` trains a skip-gram Word2Vec model with:

- vector size: 100
- context window: 5
- minimum token count: 1
- skip-gram mode: enabled
- workers: 4

It also provides `sentence_embedding`, which averages the vectors for tokens known by the model and returns a zero vector when no token is known.

## 8. Current Smoke Test

The root-level `test.py` verifies Word2Vec training against `clean_dataset.csv` and looks up the vector for `phone`.

Run it from the project root with the project interpreter:

```powershell
e:/academic/4.1/NLP_lab/NLP_PROJECT/venv/Scripts/python.exe .\test.py
```

The successful run prints a 100-dimensional NumPy vector. It also creates or updates:

```text
models_saved/word2vec.model
```

Two issues were fixed while reaching this working state:

1. The original data path pointed to `../../data/processed/train.csv`, which was outside the repository when run from the project root. The test now resolves its data path relative to `test.py`.
2. The processed token column contains serialized lists. Calling `.split()` produced malformed tokens such as `"'phone',"`; `ast.literal_eval` now converts each value back into a real token list before Word2Vec training.

## 9. Verification History

The following checks have succeeded in the configured Python 3.11 virtual environment:

```powershell
e:/academic/4.1/NLP_lab/NLP_PROJECT/venv/Scripts/python.exe .\test.py
```

The test completed without an exception and printed the Word2Vec vector for `phone`.

The n-gram smoke test was also run successfully earlier and produced:

```text
(13000, 325)
```

This means the current `clean_dataset.csv` produced 13,000 rows and 325 word n-gram features with the current data and vectorizer settings.

## 10. Known Limitations and Next Steps

- Several source scripts use paths relative to the current working directory. Run them from the directories shown above, or refactor them to use paths based on `Path(__file__)`.
- `annotation.py` uses hard-coded names and locations, so it will not recognize every possible person or location.
- Most entities receive only `B-` labels; multi-token entities are not currently represented with full `B-`, `I-`, `O` spans.
- The current preprocessing output stores token lists as CSV strings. JSON or a dedicated token column format would be safer for larger pipelines.
- The project has feature extraction and embedding code, but no completed supervised model in `src/training` and no completed prediction flow in `src/inference`.
- The `app` directory is currently empty.
- `notebooks/01_EDA.ipynb` is currently an empty notebook scaffold.

Recommended next implementation steps are supervised sequence-labeling model training, evaluation on the held-out test set, an inference API or Streamlit interface, and path handling based on the shared configuration in `src/config.py`.
