# Language Identification System Using NLP, Machine Learning and Transformer-Based Deep Learning Models

A small Flask demonstration that predicts one of ten supported languages, displays a confidence estimate, and shows its normalized text. The interface calls the `/predict` JSON endpoint. Predictions are computed from a saved TF-IDF/logistic-regression baseline, Unicode script analysis, or an optional Hugging Face model.

## Problem statement and objectives

Identifying a language helps route text to suitable translation, search, and NLP tools. This project demonstrates a complete language-identification workflow with a browser UI and API. Its objectives are to normalize Unicode text, extract character and word features, train and evaluate a classical classifier, optionally call a pretrained transformer, and return an understandable result.

## Features

- Supports English, Marathi, Hindi, Gujarati, Bengali, Tamil, Telugu, Kannada, Malayalam, and Punjabi.
- Responsive browser interface; `POST /predict` API.
- Unicode NFC normalization, case folding, punctuation cleanup, and tokenization.
- Character and word TF-IDF with logistic regression as the ML baseline.
- Script-aware identification for scripts unique to a supported language. Hindi and Marathi both use Devanagari, so the trained classifier separates those labels.
- `MODEL_TYPE=ml` (default) and `MODEL_TYPE=transformer` configuration.
- Clear handling for empty input, short input, mixed scripts, numbers, and unsupported-script examples such as Spanish.

## Technologies

Python 3.10+, Flask, HTML5, CSS3, JavaScript, scikit-learn, pandas, NumPy, joblib, Hugging Face Transformers, and PyTorch.

## Architecture and workflow

```text
Browser → Flask POST /predict → Unicode preprocessing → script analysis
                                                ├→ TF-IDF features → Logistic Regression
                                                └→ optional Hugging Face classifier
                           → confidence estimate + JSON → result card
```

Preprocessing retains Unicode letters, combining marks, and digits, replaces punctuation with spaces, and collapses whitespace. Tokenization is a separate function in `preprocessing/text_preprocessor.py`. The ML feature union combines character n-grams (1–5) with word unigrams/bigrams. Logistic Regression returns class probabilities. For unique Indic scripts, script evidence is used directly; for Devanagari, only Hindi and Marathi probabilities are compared. This makes the demo practical and helps avoid confusion between scripts, but script recognition should not be confused with a trained deep neural network.

## Dataset

`data/language_dataset.csv` has `text,language` columns and 120 short, manually curated representative sample sentences (12 per label). It is a demonstration dataset, not a real-world benchmark. Its holdout metrics measure only this small sample and should not be generalized. For a credible evaluation, collect consented and licensed, varied examples and maintain a separate test set. Larger sources to investigate include AI4Bharat IndicCorp, OSCAR, Wikipedia dumps, and FLORES-200; follow each source's access terms, license, and intended use. Deduplicate, balance languages, and avoid train/test sentence leakage.

## Models

### ML baseline

Run `python -m models.train_model` to split the CSV into stratified train/holdout partitions, fit character/word TF-IDF plus Logistic Regression, save `models/saved_models/language_classifier.joblib`, and write actual training holdout metrics to `models/saved_models/training_metrics.json`. Prediction loads that artifact (and trains it once if missing).

### Transformer option

Set `MODEL_TYPE=transformer` to enable the pretrained `papluca/xlm-roberta-base-language-detection` checkpoint. The checkpoint recognizes 20 languages, including English and Hindi, but does not cover the full ten-language set. This project uses it for Latin-script input when its result is English; Devanagari stays with the ML baseline so Hindi and Marathi can be distinguished. Other cases transparently use script analysis or the saved ML baseline. First use downloads model files from Hugging Face and needs internet access and disk space; inference is CPU-only. This is pretrained inference, not a fine-tuned Indic classifier. The model card and coverage: [Hugging Face model](https://huggingface.co/papluca/xlm-roberta-base-language-detection).

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include PyTorch and Transformers for the optional transformer setting, so installation can take time and disk space. For a smaller ML-only install, use `pip install Flask numpy pandas scikit-learn joblib`.

## Run the application

```bash
python app.py
```

Open <http://127.0.0.1:5000>. The default `MODEL_TYPE` is `ml`; override it with `MODEL_TYPE=transformer` in your shell to use the optional transformer path. On first prediction, the small sample ML dataset trains a local model automatically if no saved model exists.

## API

`POST /predict`, JSON request:

```json
{"text":"नमस्कार, तुम्ही कसे आहात?"}
```

Example response shape (the actual confidence is computed at runtime):

```json
{
  "language": "Marathi",
  "confidence": 0.93,
  "preprocessed_text": "नमस्कार तुम्ही कसे आहात",
  "model_used": "Character/word TF-IDF + Logistic Regression (Devanagari classes)",
  "tokens": ["नमस्कार", "तुम्ही", "कसे", "आहात"]
}
```

Confidence is a model/script-based estimate, not a guarantee of correctness. Short input may return `Unknown` and a helpful message. Empty requests receive HTTP 400. `Hola amigo` is outside the supported language set and is returned as unknown rather than intentionally mapped to one of the ten classes.

Try `curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{"text":"Hello, how are you?"}'`.

## Training and evaluation

```bash
python -m models.train_model
python -m evaluation.evaluate
```

The evaluation script computes accuracy, macro precision, macro recall, macro F1, per-class report, and a confusion matrix on a stratified holdout split and saves actual values to `evaluation/evaluation_results.json`. These metrics use the included sample data only; the README intentionally does not report a benchmark score.

## Limitations

- No language detector can promise 100% accuracy for every text. Very short, noisy, transliterated, code-mixed, and closely related text can be ambiguous.
- The included CSV is deliberately small and educational; its evaluation score is not representative of real traffic.
- Devanagari-based Hindi/Marathi distinction depends on this small learned baseline and wording, so use a longer sentence.
- The optional pretrained transformer does not support every required language; fallback model is disclosed in each API response.
- Script recognition identifies script, not necessarily language, for languages that share scripts or for text pasted in an unexpected script.

## Future scope

- Collect a licensed, balanced corpus and train/evaluate an Indic-focused language-ID classifier.
- Code-mixed and word-level language identification.
- Speech language identification.
- Quantized offline/edge inference.
- Add more Indian regional languages, including languages that share scripts.

## Viva explanation (simple English)

1. The user enters a sentence and JavaScript sends it to Flask as JSON.
2. The preprocessing module applies Unicode normalization, case folding, removes punctuation, and tokenizes the words while preserving Indic characters.
3. The system checks the script. A unique supported Indic script identifies its language; Devanagari needs the classifier because Hindi and Marathi share it.
4. The ML model converts character and word patterns into TF-IDF features. Logistic Regression predicts a label and probabilities.
5. In transformer mode, a pretrained Hugging Face XLM-RoBERTa language classifier is used only for its supported labels; the response identifies any fallback.
6. Flask returns language, confidence estimate, cleaned text, and model name. The page displays those fields.
7. The training/evaluation script reports metrics from a held-out part of the included sample CSV. Those scores are for demonstration only, not a real-world accuracy claim.
