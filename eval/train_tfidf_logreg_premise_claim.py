from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any, List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib


# ---------------- Utility functions: parsing choices/gold exactly as in eval ----------------

def parse_choices(raw: Any) -> List[str]:
    """Parse the `choices` field into a list of string labels."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        return [str(x) for x in raw]

    if isinstance(raw, str):
        s = raw.strip()
        # Try JSON and Python literal parsing
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, (list, tuple)):
                    return [str(x) for x in obj]
            except Exception:
                continue

        # Fallback: split by comma
        s = s.strip("[]")
        parts = [
            p.strip().strip('"').strip("'")
            for p in s.split(",")
            if p.strip()
        ]
        if parts:
            return parts

    raise ValueError(f"Cannot parse choices from value: {raw!r}")


def gold_to_label(gold: Any, choices: List[str]) -> str:
    """
    Map the `gold` field to a string label ("premise" or "claim").

    We handle multiple formats:
      - direct string labels ("premise"/"claim")
      - index into the choices list
      - raw numeric 0/1 labels
    """
    # Case 1: gold is already a string label or index-like string
    if isinstance(gold, str):
        g = gold.strip().lower()
        if g in ("premise", "claim"):
            return g
        try:
            idx = int(g)
            if 0 <= idx < len(choices):
                return str(choices[idx]).strip().lower()
        except Exception:
            pass
    else:
        # Case 2: treat as index
        try:
            idx = int(gold)
            if 0 <= idx < len(choices):
                return str(choices[idx]).strip().lower()
        except Exception:
            pass

    # Case 3: fallback 0/1 encoding
    if gold == 0:
        return "premise"
    if gold == 1:
        return "claim"

    raise ValueError(f"Cannot map gold={gold!r} with choices={choices!r}")


def load_dataset(parquet_path: str) -> Tuple[List[str], List[int]]:
    """
    Load FinLLM/AgentScen Task 1 data from a parquet file and return
    (texts, label_ids) where 0=premise, 1=claim.
    """
    df = pd.read_parquet(parquet_path)

    required = ["text", "choices", "gold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    texts: List[str] = []
    labels: List[int] = []

    for row in df.itertuples(index=False):
        text = str(getattr(row, "text"))
        choices_raw = getattr(row, "choices")
        gold_raw = getattr(row, "gold")

        choices = parse_choices(choices_raw)
        label_str = gold_to_label(gold_raw, choices)
        if label_str not in ("premise", "claim"):
            # Ignore unexpected labels
            continue

        label_id = 0 if label_str == "premise" else 1

        texts.append(text)
        labels.append(label_id)

    return texts, labels


def main():
    parquet_path = "train-00000-of-00001-1fccc37a08c8dcfb (2).parquet"
    parquet_path = str(Path(parquet_path).resolve())
    print(f"📂 Loading dataset from: {parquet_path}")

    texts, labels = load_dataset(parquet_path)
    print(f"Total samples: {len(texts)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF features: unigrams + bigrams, remove extremely rare terms
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=50000,
    )

    print("🔧 Fitting TF-IDF vectorizer...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression classifier
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # data is roughly balanced; this adds a bit of robustness
        n_jobs=-1,
    )

    print("🧠 Training Logistic Regression classifier...")
    clf.fit(X_train_tfidf, y_train)

    print("📊 Evaluating on test set...")
    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["premise", "claim"],
        digits=4,
    )

    print("\n========== TF-IDF + Logistic Regression Result ==========")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nPer-label metrics:")
    print(report)

    # Save model and vectorizer
    out_dir = Path("models_tfidf_logreg")
    out_dir.mkdir(exist_ok=True)
    vec_path = out_dir / "tfidf_vectorizer.joblib"
    clf_path = out_dir / "logreg_premise_claim.joblib"

    joblib.dump(vectorizer, vec_path)
    joblib.dump(clf, clf_path)

    print(f"\n✅ Saved TF-IDF vectorizer to: {vec_path}")
    print(f"✅ Saved Logistic Regression model to: {clf_path}")


if __name__ == "__main__":
    main()

