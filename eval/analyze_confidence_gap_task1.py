from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_tfidf_logreg_premise_claim import load_dataset


def build_confidence_df(
    parquet_path: Path,
    vec_path: Path,
    clf_path: Path,
    max_items: int | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with confidence statistics for each sample.

    1) Use load_dataset() to read texts + gold labels (0=premise, 1=claim)
       from the parquet file.
    2) Load the TF-IDF vectorizer and Logistic Regression model to obtain
       prediction probabilities.
    3) For each sample, compute:
       - predicted label
       - whether the prediction is correct
       - confidence (max class probability)
    """
    print(f"📂 Loading dataset from: {parquet_path}")
    texts, labels = load_dataset(str(parquet_path))

    if max_items is not None:
        texts = texts[:max_items]
        labels = labels[:max_items]

    labels = np.array(labels)
    print(f"Total samples for analysis: {len(texts)}")

    print(f"📦 Loading TF-IDF vectorizer from: {vec_path}")
    vectorizer = joblib.load(vec_path)

    print(f"📦 Loading Logistic Regression model from: {clf_path}")
    clf = joblib.load(clf_path)

    print("🔮 Running classifier to obtain probabilities...")
    X_tfidf = vectorizer.transform(texts)
    proba = clf.predict_proba(X_tfidf)  # shape: (n_samples, 2)

    pred = np.argmax(proba, axis=1)
    correct = (pred == labels).astype(int)
    confidence = np.max(proba, axis=1)

    df = pd.DataFrame(
        {
            "confidence": confidence,
            "gold": labels,
            "pred": pred,
            "correct": correct,
        }
    )
    print("✅ Built confidence DataFrame with shape:", df.shape)
    return df


def bucket_confidence(c: float) -> str:
    """
    Bucket confidence scores into three ranges:
      [0, 0.6)   -> "low"
      [0.6, 0.8) -> "medium"
      [0.8, 1.0] -> "high"

    You can change the thresholds if needed.
    """
    if c < 0.6:
        return "low"
    elif c < 0.8:
        return "medium"
    else:
        return "high"


def compute_confrg(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """
    Compute the Confidence Robustness Gap (ConfRG).

      ConfRG = Acc(high-confidence) - Acc(low-confidence)
    """
    df = df.copy()
    df["bucket"] = df["confidence"].apply(bucket_confidence)

    acc_by_bucket = (
        df.groupby("bucket")["correct"]
        .mean()
        .reindex(["low", "medium", "high"])
    )

    high_acc = acc_by_bucket.get("high", np.nan)
    low_acc = acc_by_bucket.get("low", np.nan)
    confrg = high_acc - low_acc

    return confrg, acc_by_bucket


def plot_conf_bar(acc_by_bucket: pd.Series, out_path: Path) -> None:
    """Plot accuracy by confidence bucket as a bar chart."""
    fig, ax = plt.subplots()

    buckets = acc_by_bucket.index.tolist()
    values = (acc_by_bucket.values * 100.0).tolist()  # convert to %

    ax.bar(buckets, values)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Confidence bucket")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy by confidence bucket")

    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved confidence bar chart to: {out_path}")


def plot_conf_hist(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot the confidence distributions of correct vs. incorrect predictions
    as overlaid histograms.
    """
    fig, ax = plt.subplots()

    correct_conf = df[df["correct"] == 1]["confidence"].values
    wrong_conf = df[df["correct"] == 0]["confidence"].values

    bins = np.linspace(0.0, 1.0, 21)

    ax.hist(correct_conf, bins=bins, alpha=0.5, label="Correct", density=True)
    ax.hist(wrong_conf, bins=bins, alpha=0.5, label="Incorrect", density=True)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distribution: correct vs incorrect predictions")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📊 Saved confidence histogram to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confidence-based robustness analysis for FinLLM Task 1."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="train-00000-of-00001-1fccc37a08c8dcfb (2).parquet",
        help="Parquet file used in train_tfidf_logreg_premise_claim.py",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="tfidf_vectorizer.joblib",
        help="Path to the saved TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg_premise_claim.joblib",
        help="Path to the saved Logistic Regression model.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (set None for all).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="confidence_task1_analysis",
        help="Directory to save plots and metrics.",
    )

    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    vec_path = Path(args.vectorizer)
    clf_path = Path(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_conf = build_confidence_df(
        parquet_path=parquet_path,
        vec_path=vec_path,
        clf_path=clf_path,
        max_items=args.max_items,
    )

    metrics_csv = out_dir / "task1_confidence_metrics.csv"
    df_conf.to_csv(metrics_csv, index=False)
    print(f"💾 Saved confidence metrics to: {metrics_csv}")

    confrg, acc_by_bucket = compute_confrg(df_conf)
    print("\n========== Confidence Robustness Gap (ConfRG) ==========")
    print("Accuracy by confidence bucket:")
    print(acc_by_bucket)
    print(f"\nConfRG = Acc(high) - Acc(low) = {confrg:.4f}")

    plot_conf_bar(acc_by_bucket, out_dir / "confidence_by_bucket.png")
    plot_conf_hist(df_conf, out_dir / "confidence_hist_correct_vs_wrong.png")


if __name__ == "__main__":
    main()



