"""Stratified holdout evaluation of the saved ML baseline."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from config import DATA_PATH, ML_MODEL_PATH, SUPPORTED_LANGUAGES
from preprocessing.text_preprocessor import preprocess_text
from models.train_model import train_model


def evaluate():
    data = pd.read_csv(DATA_PATH).dropna(subset=["text", "language"])
    data["text"] = data.text.astype(str).map(preprocess_text)
    train, test = train_test_split(data, test_size=0.25, random_state=42, stratify=data.language)
    if not ML_MODEL_PATH.exists():
        train_model()
    import joblib
    model = joblib.load(ML_MODEL_PATH)
    pred = model.predict(test.text)
    p, r, f, _ = precision_recall_fscore_support(test.language, pred, average="macro", zero_division=0)
    metrics = {"accuracy": accuracy_score(test.language, pred), "macro_precision": p,
               "macro_recall": r, "macro_f1": f,
               "labels": SUPPORTED_LANGUAGES,
               "confusion_matrix": confusion_matrix(test.language, pred, labels=SUPPORTED_LANGUAGES).tolist(),
               "classification_report": classification_report(test.language, pred, zero_division=0, output_dict=True)}
    out = Path(__file__).with_name("evaluation_results.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(test.language, pred, zero_division=0))
    print("Confusion matrix labels:", ", ".join(SUPPORTED_LANGUAGES))
    print(confusion_matrix(test.language, pred, labels=SUPPORTED_LANGUAGES))
    print("Metrics:", {key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")})
    print(f"Saved: {out}")
    return metrics


if __name__ == "__main__":
    evaluate()
