from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field


# ============================================================
# 1. Data and prediction structures
# ============================================================

class Task1Example(BaseModel):
    """
    One example for FinLLM Task 1 (financial argument classification).

    Here we *do not* read the `choices` column from the dataset.
    We assume the label space is always ["premise", "claim"].
    """
    id: str
    text: str
    query: str
    choices: List[str] = Field(default_factory=lambda: ["premise", "claim"])
    gold_label: str      # e.g. "premise" or "claim"
    gold_index: int      # index in choices

    @classmethod
    def from_raw(
        cls,
        raw: Dict[str, Any],
    ) -> "Task1Example":
        """
        Convert a raw row (e.g. from parquet/json) into a standardized Task1Example.

        EXPECTED FIELDS:
            - raw["id"]
            - raw["query"]
            - raw["text"]
            - raw["answer"]  (gold label string, e.g. "premise")
            - raw["gold"]    (gold label index, e.g. 0)
        We ignore any `choices` field and always use ["premise", "claim"].
        """

        # Fixed label space to avoid any dtype issues
        choices = ["premise", "claim"]

        # ----- parse gold index (may be int/float/NaN) -----
        gold_idx_raw = raw.get("gold", None)
        gold_index: int = -1

        if gold_idx_raw is not None:
            if isinstance(gold_idx_raw, int):
                gold_index = gold_idx_raw
            elif isinstance(gold_idx_raw, float) and not math.isnan(gold_idx_raw):
                gold_index = int(gold_idx_raw)

        # ----- parse gold label -----
        answer = raw.get("answer", None)
        gold_label: Optional[str] = None

        if 0 <= gold_index < len(choices):
            gold_label = choices[gold_index].lower()
        elif isinstance(answer, str):
            gold_label = answer.strip().lower()
            lower_choices = [c.lower() for c in choices]
            if gold_label in lower_choices:
                gold_index = lower_choices.index(gold_label)
            else:
                gold_index = -1
        else:
            raise ValueError(f"Cannot determine gold label from raw: {raw}")

        if gold_label is None:
            raise ValueError(f"Gold label is None after parsing raw: {raw}")

        return cls(
            id=str(raw.get("id", "")),
            text=str(raw.get("text", "")),
            query=str(raw.get("query", "")),
            choices=choices,
            gold_label=gold_label,
            gold_index=gold_index,
        )


class Task1Prediction(BaseModel):
    """
    Model prediction for one Task 1 example.

    You can later extend this with logits, raw output, etc.
    """
    id: str
    pred_label: str          # e.g. "premise" or "claim"
    pred_index: int          # index in example.choices
    scores: Optional[Dict[str, float]] = None  # e.g. {"premise": 0.7, "claim": 0.3}


# Unified prediction function signature: takes one example, returns one prediction
PredictFn = Callable[[Task1Example], Task1Prediction]


# ============================================================
# 2. Metric computation
# ============================================================

@dataclass
class PerLabelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def compute_classification_metrics(
    gold_labels: List[str],
    pred_labels: List[str],
    label_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Accuracy and (macro) F1 for a classification task.
    """
    assert len(gold_labels) == len(pred_labels), "gold/pred length mismatch"

    if label_list is None:
        label_list = sorted(list(set(gold_labels) | set(pred_labels)))

    total = len(gold_labels)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    accuracy = correct / total if total > 0 else 0.0

    stats: Dict[str, PerLabelStats] = {lbl: PerLabelStats() for lbl in label_list}

    for g, p in zip(gold_labels, pred_labels):
        if g not in stats:
            stats[g] = PerLabelStats()
        if p not in stats:
            stats[p] = PerLabelStats()

        if g == p:
            stats[g].tp += 1
        else:
            stats[p].fp += 1
            stats[g].fn += 1

    per_label_f1: Dict[str, float] = {}
    per_label_precision: Dict[str, float] = {}
    per_label_recall: Dict[str, float] = {}

    for lbl, s in stats.items():
        per_label_precision[lbl] = s.precision()
        per_label_recall[lbl] = s.recall()
        per_label_f1[lbl] = s.f1()

    if label_list:
        macro_f1 = sum(per_label_f1[lbl] for lbl in label_list) / len(label_list)
    else:
        macro_f1 = 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_label": {
            lbl: {
                "precision": per_label_precision[lbl],
                "recall": per_label_recall[lbl],
                "f1": per_label_f1[lbl],
            }
            for lbl in label_list
        },
        "total_examples": total,
        "correct_examples": correct,
    }


# ============================================================
# 3. Evaluation core
# ============================================================

def evaluate_task1(
    examples: List[Task1Example],
    predict_fn: PredictFn,
    label_list: Optional[List[str]] = None,
    collect_errors: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate a model on FinLLM Task 1 examples.
    """
    if label_list is None:
        label_list = ["premise", "claim"]

    gold_labels: List[str] = []
    pred_labels: List[str] = []

    errors: List[Dict[str, Any]] = []

    for ex in examples:
        pred = predict_fn(ex)

        gold = ex.gold_label.lower()
        pred_label = pred.pred_label.lower()

        gold_labels.append(gold)
        pred_labels.append(pred_label)

        if collect_errors and gold != pred_label:
            errors.append(
                {
                    "id": ex.id,
                    "text": ex.text,
                    "query": ex.query,
                    "gold": gold,
                    "pred": pred_label,
                    "choices": ex.choices,
                    "scores": pred.scores,
                }
            )

    metrics = compute_classification_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        label_list=label_list,
    )

    return metrics, errors


# ============================================================
# 4. Data loading helper (parquet)
# ============================================================

def load_task1_examples_from_parquet(parquet_path: str) -> List[Task1Example]:
    """
    Load Task 1 examples directly from a parquet file.

    This assumes the parquet file has columns like:
        id, query, text, answer, gold
    exactly as in the FinLLM Task 1 dataset on HuggingFace.
    We *ignore* any `choices` column, to avoid dtype issues.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)

    examples: List[Task1Example] = []

    for _, row in df.iterrows():
        raw = row.to_dict()
        ex = Task1Example.from_raw(raw)
        examples.append(ex)

    return examples


# ============================================================
# 5. Model prediction stub (replace with your real model later)
# ============================================================

def my_model_predict(example: Task1Example) -> Task1Prediction:
    """
    Stub baseline model that always predicts "premise".

    For real usage, you MUST replace the logic inside this function with:
      1. Build a prompt from example.text and/or example.query.
      2. Call your LLM / classifier.
      3. Parse the output into "premise" or "claim".
    """
    # Baseline: always "premise"
    label = "premise"

    lower_choices = [c.lower() for c in example.choices]
    if label not in lower_choices:
        idx = 0
        label = example.choices[idx]
    else:
        idx = lower_choices.index(label)
        label = example.choices[idx]

    return Task1Prediction(
        id=example.id,
        pred_label=label,
        pred_index=idx,
        scores=None,
    )


# ============================================================
# 6. CLI entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a FinLLM Task 1 model (premise/claim classification)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Task 1 evaluation file (parquet format).",
    )

    args = parser.parse_args()

    examples: List[Task1Example] = load_task1_examples_from_parquet(args.data_path)

    metrics, errors = evaluate_task1(
        examples=examples,
        predict_fn=my_model_predict,  # Replace with your real model later
        label_list=["premise", "claim"],
        collect_errors=True,
    )

    print("=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    errors_path = Path(args.data_path).with_suffix(".errors.jsonl")
    with errors_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nNumber of misclassified examples: {len(errors)}")
    print(f"Misclassified examples saved to: {errors_path}")


if __name__ == "__main__":
    main()






