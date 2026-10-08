"""Prediction orchestration with script checks and an optional HF transformer."""
from functools import lru_cache
import joblib
import numpy as np

from config import ML_MODEL_PATH, MODEL_TYPE, TRANSFORMER_MODEL, SUPPORTED_LANGUAGES
from preprocessing.text_preprocessor import preprocess_text, tokenize

SCRIPT_LANGUAGES = {
    "Devanagari": {"Hindi", "Marathi"}, "Gujarati": {"Gujarati"},
    "Bengali": {"Bengali"}, "Tamil": {"Tamil"}, "Telugu": {"Telugu"},
    "Kannada": {"Kannada"}, "Malayalam": {"Malayalam"}, "Gurmukhi": {"Punjabi"},
}


def _script(char):
    n = ord(char)
    ranges = [(0x0900,0x097F,"Devanagari"),(0x0A80,0x0AFF,"Gujarati"),
              (0x0980,0x09FF,"Bengali"),(0x0B80,0x0BFF,"Tamil"),
              (0x0C00,0x0C7F,"Telugu"),(0x0C80,0x0CFF,"Kannada"),
              (0x0D00,0x0D7F,"Malayalam"),(0x0A00,0x0A7F,"Gurmukhi")]
    return next((name for lo, hi, name in ranges if lo <= n <= hi), None)


def _load_ml():
    if not ML_MODEL_PATH.exists():
        from models.train_model import train_model
        train_model()
    return joblib.load(ML_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_transformer():
    from transformers import pipeline
    return pipeline("text-classification", model=TRANSFORMER_MODEL, tokenizer=TRANSFORMER_MODEL,
                    top_k=None, device=-1)


def _transformer_predict(text):
    """Use public multilingual HF checkpoint; return None when it cannot cover result."""
    try:
        outputs = _load_transformer()(text, truncation=True, max_length=256)[0]
        best = max(outputs, key=lambda item: item["score"])
        labels = {"en": "English", "hi": "Hindi"}
        # This popular checkpoint covers 20 languages, but only English/Hindi here.
        language = labels.get(best["label"].lower().split("_")[-1])
        if language not in SUPPORTED_LANGUAGES:
            return None
        return language, float(best["score"]), f"Hugging Face {TRANSFORMER_MODEL}"
    except Exception:
        return None


def predict_language(text, model_type=None):
    clean = preprocess_text(text)
    tokens = tokenize(text)
    if not clean or not any(char.isalpha() for char in clean):
        return {"language": "Unknown", "confidence": 0.0, "preprocessed_text": clean,
                "model_used": "Input validation", "message": "Please enter text containing letters."}
    counts = {}
    for ch in text:
        script = _script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    total_script_chars = sum(counts.values())
    if total_script_chars:
        latin_letters = sum(ch.isalpha() and ch.isascii() for ch in text)
        if latin_letters and latin_letters / (latin_letters + total_script_chars) > 0.15:
            return {"language": "Unknown", "confidence": 0.0, "preprocessed_text": clean,
                    "model_used": "Script analysis", "message": "Text appears code-mixed across scripts; enter a single-language sample."}
        script, count = max(counts.items(), key=lambda pair: pair[1])
        share = count / total_script_chars
        if share < 0.65:
            return {"language": "Unknown", "confidence": 0.0, "preprocessed_text": clean,
                    "model_used": "Script analysis + ML baseline", "message": "Text appears mixed across scripts; enter a single-language sample."}
        candidates = SCRIPT_LANGUAGES[script]
        if len(candidates) == 1:
            language = next(iter(candidates))
            confidence = min(0.99, 0.82 + 0.17 * share)
            used = "Unicode script identification"
        else:
            model = _load_ml()
            probs = model.predict_proba([clean])[0]
            classes = list(model.classes_)
            candidates_probs = [(classes[i], float(probs[i])) for i in range(len(classes)) if classes[i] in candidates]
            language, raw = max(candidates_probs, key=lambda item: item[1])
            # Confidence is the model probability conditional on the two Devanagari labels.
            confidence = raw / sum(p for _, p in candidates_probs) if candidates_probs else 0.0
            used = "Character/word TF-IDF + Logistic Regression (Devanagari classes)"
    elif all(ch.isascii() or not ch.isalpha() for ch in text):
        selected = (model_type or MODEL_TYPE).lower()
        transformer_result = _transformer_predict(text) if selected == "transformer" else None
        if transformer_result:
            language, confidence, used = transformer_result
        else:
            # The ten-class training CSV contains no out-of-scope Latin language.
            # Use a general language-ID package as a rejection gate so Spanish
            # is not forced into the English class.
            try:
                from langdetect import DetectorFactory, detect_langs
                DetectorFactory.seed = 0
                detected = detect_langs(text)
                top = detected[0]
                if top.lang != "en":
                    language, confidence = "Unknown", 0.0
                    used = "Latin-language identification (unsupported-language check)"
                    unsupported_message = "This text appears to be outside the 10 supported languages."
                else:
                    language, confidence = "English", float(top.prob)
                    used = "Latin-language identification"
                    unsupported_message = None
            except Exception:
                language, confidence = "Unknown", 0.0
                used = "Latin-language identification"
                unsupported_message = "Could not confidently identify this Latin-script text. Try a longer sample."
    else:
        return {"language": "Unknown", "confidence": 0.0, "preprocessed_text": clean,
                "model_used": "Script analysis", "message": "Could not identify one supported script reliably."}

    message = None
    if confidence < 0.45 and language != "English":
        language = "Unknown"
        message = locals().get("unsupported_message") or "The text is too ambiguous for a reliable result. Try a longer sample."
        confidence = 0.0
    return {"language": language, "confidence": round(float(confidence), 4),
            "preprocessed_text": clean, "model_used": used, "tokens": tokens,
            **({"message": message} if message else {})}
