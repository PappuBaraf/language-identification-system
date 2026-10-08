"""Train the transparent character/word TF-IDF + logistic regression baseline."""
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import DATA_PATH, ML_MODEL_PATH
from preprocessing.text_preprocessor import preprocess_text


def train_model(data_path=DATA_PATH, model_path=ML_MODEL_PATH):
    data = pd.read_csv(data_path).dropna(subset=["text", "language"])
    data["text"] = data["text"].astype(str).map(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        data.text, data.language, test_size=0.25, random_state=42,
        stratify=data.language,
    )
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=1,
                                     sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                     sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ])),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    result = {"holdout_accuracy": float(accuracy), "holdout_rows": len(y_test),
              "report": classification_report(y_test, predictions, output_dict=True,
                                               zero_division=0)}
    out_path = model_path.parent / "training_metrics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Holdout accuracy on this sample dataset: {accuracy:.4f} ({len(y_test)} rows)")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model: {model_path}\nSaved metrics: {out_path}")
    return pipeline, result


if __name__ == "__main__":
    train_model()
