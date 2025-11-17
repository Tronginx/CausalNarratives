from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple

from pydantic import BaseModel, Field

# External metrics libraries (you must install them):
#   pip install rouge-score bert-score
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score


# =========================
# 1. Data and prediction structures
# =========================

class Task2Example(BaseModel):
    """
    One example for FinLLM Task 2 (financial text summarization).
    """
    id: str
    text: str              # source financial news article
    query: str             # prompt / instruction
    gold_summary: str      # reference abstractive summary

    @classmethod
    def from_raw(
        cls,
        raw: Dict,
    ) -> "Task2Example":
        """
        Convert a raw row (e.g. from json/csv) into a standardized Task2Example.

        EXPECTED FIELDS (adapt to your real dataset if needed):
            - raw["id"]
            - raw["query"]
            - raw["text"]
            - raw["answer"]   (gold summary)
        """
        # You can adjust these keys if your dataset uses different names
        return cls(
            id=str(raw.get("id", "")),
            text=str(raw.get("text", "")),
            query=str(raw.get("query", "")),
            gold_summary=str(raw.get("answer", "")),
        )


class Task2Prediction(BaseModel):
    """
    Model prediction for one Task 2 example (generated summary).
    """
    id: str
    pred_summary: str
    # You can add extra fields later if needed, e.g. raw_output, logits, etc.


# Unified prediction function signature
SummarizationPredictFn = Callable[[Task2Example], Task2Prediction]


# =========================
# 2. Metric computation
# =========================

@dataclass
class RougeStats:
    rouge1_f1_sum: float = 0.0
    rouge2_f1_sum: float = 0.0
    rougel_f1_sum: float = 0.0
    count: int = 0

    def update(self, rouge1_f1: float, rouge2_f1: float, rougel_f1: float) -> None:
        self.rouge1_f1_sum += rouge1_f1
        self.rouge2_f1_sum += rouge2_f1
        self.rougel_f1_sum += rougel_f1
        self.count += 1

    def averages(self) -> Dict[str, float]:
        if self.count == 0:
            return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougel_f1": 0.0}
        return {
            "rouge1_f1": self.rouge1_f1_sum / self.count,
            "rouge2_f1": self.rouge2_f1_sum / self.count,
            "rougel_f1": self.rougel_f1_sum / self.count,
        }


def compute_summarization_metrics(
    references: List[str],
    predictions: List[str],
    use_bertscore: bool = True,
    bertscore_lang: str = "en",
) -> Tuple[Dict, List[Dict]]:
    """
    Compute ROUGE-1/2/L and BERTScore for a list of summaries.

    Args:
        references:  list of gold summaries (reference texts)
        predictions: list of predicted summaries (same length as references)
        use_bertscore: whether to compute BERTScore
        bertscore_lang: language code for BERTScore (e.g. "en" for English)

    Returns:
        metrics: overall metrics dict
        per_example_scores: list of dict with per-example ROUGE/BERTScore
    """
    assert len(references) == len(predictions), "references/predictions length mismatch"

    # ROUGE scorer (we use F1 from rouge1, rouge2, and rougeLsum)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True
    )

    rouge_stats = RougeStats()
    per_example_scores: List[Dict] = []

    # First pass: compute ROUGE for each example
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1_f1 = scores["rouge1"].fmeasure
        rouge2_f1 = scores["rouge2"].fmeasure
        rougel_f1 = scores["rougeLsum"].fmeasure

        rouge_stats.update(rouge1_f1, rouge2_f1, rougel_f1)

        per_example_scores.append(
            {
                "rouge1_f1": rouge1_f1,
                "rouge2_f1": rouge2_f1,
                "rougel_f1": rougel_f1,
                # "bertscore_f1" will be filled later (if enabled)
            }
        )

    # Second pass: BERTScore (batch computation for efficiency)
    overall_bertscore_f1: Optional[float] = None
    bertscore_per_example: Optional[List[float]] = None

    if use_bertscore:
        # bertscore_score returns three tensors (P, R, F1)
        P, R, F1 = bertscore_score(
            cands=predictions,
            refs=references,
            lang=bertscore_lang,
            rescale_with_baseline=True,
        )
        # F1 is a 1-D tensor; take mean as overall score
        overall_bertscore_f1 = float(F1.mean().item())
        bertscore_per_example = [float(x.item()) for x in F1]

        # Attach per-example BERTScore to records
        for rec, bs in zip(per_example_scores, bertscore_per_example):
            rec["bertscore_f1"] = bs
    else:
        # If BERTScore is disabled, leave fields absent or set to None
        overall_bertscore_f1 = None
        # Not adding per-example BERTScore fields

    # Aggregate ROUGE averages
    rouge_avgs = rouge_stats.averages()

    metrics: Dict[str, float] = {
        "rouge1_f1": rouge_avgs["rouge1_f1"],
        "rouge2_f1": rouge_avgs["rouge2_f1"],
        "rougel_f1": rouge_avgs["rougel_f1"],
        "num_examples": rouge_stats.count,
    }

    # Final ranking metric is ROUGE-1 F1 (as per task description)
    metrics["ranking_metric"] = "rouge1_f1"

    if overall_bertscore_f1 is not None:
        metrics["bertscore_f1"] = overall_bertscore_f1

    return metrics, per_example_scores


# =========================
# 3. Evaluation core
# =========================

def evaluate_task2(
    examples: List[Task2Example],
    predict_fn: SummarizationPredictFn,
    use_bertscore: bool = True,
    bertscore_lang: str = "en",
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate a summarization model on FinLLM Task 2.

    Args:
        examples:      list of Task2Example
        predict_fn:    function that generates summaries
        use_bertscore: whether to compute BERTScore
        bertscore_lang: language code for BERTScore texts

    Returns:
        metrics: dict with overall scores
        detailed_records: per-example records including ROUGE/BERTScore
    """
    references: List[str] = []
    predictions: List[str] = []
    ids: List[str] = []
    texts: List[str] = []
    queries: List[str] = []

    for ex in examples:
        pred = predict_fn(ex)

        ids.append(ex.id)
        texts.append(ex.text)
        queries.append(ex.query)

        references.append(ex.gold_summary)
        predictions.append(pred.pred_summary)

    metrics, per_example_scores = compute_summarization_metrics(
        references=references,
        predictions=predictions,
        use_bertscore=use_bertscore,
        bertscore_lang=bertscore_lang,
    )

    # Merge meta information with per-example scores
    detailed_records: List[Dict] = []
    for i, score_dict in enumerate(per_example_scores):
        record = {
            "id": ids[i],
            "text": texts[i],
            "query": queries[i],
            "gold_summary": references[i],
            "pred_summary": predictions[i],
        }
        record.update(score_dict)
        detailed_records.append(record)

    return metrics, detailed_records


# =========================
# 4. Data loading helper
# =========================

def load_task2_examples_from_jsonl(jsonl_path: str) -> List[Task2Example]:
    """
    Load Task 2 summarization examples from a JSONL file.

    If your dataset is not JSONL, modify this function accordingly.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    examples: List[Task2Example] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            ex = Task2Example.from_raw(raw)
            examples.append(ex)

    return examples


# =========================
# 5. Model prediction stub (you must replace logic)
# =========================

def my_summarization_predict(example: Task2Example) -> Task2Prediction:
    """
    Stub baseline summarization function.

    This implementation simply truncates the source text, and is ONLY meant
    for testing the evaluation pipeline.

    You MUST replace this with your real summarization model logic:
      1. Build a prompt:
         "Instruction: [task prompt] Context: [input context] Response: "
         using example.query and example.text.
      2. Call your LLM / summarization model.
      3. Parse the output into a clean summary string.
    """
    # Very naive baseline: take the first 200 characters of the input text
    summary = example.text[:200]

    return Task2Prediction(
        id=example.id,
        pred_summary=summary,
    )


# =========================
# 6. CLI entry point
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a FinLLM Task 2 model (financial text summarization)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Task 2 evaluation file (JSONL format by default).",
    )
    parser.add_argument(
        "--no_bertscore",
        action="store_true",
        help="Disable BERTScore computation (faster, but fewer metrics).",
    )
    parser.add_argument(
        "--bertscore_lang",
        type=str,
        default="en",
        help="Language code for BERTScore (default: en).",
    )

    args = parser.parse_args()

    examples: List[Task2Example] = load_task2_examples_from_jsonl(args.data_path)

    use_bertscore = not args.no_bertscore

    metrics, detailed_records = evaluate_task2(
        examples=examples,
        predict_fn=my_summarization_predict,  # Replace with your own function if needed
        use_bertscore=use_bertscore,
        bertscore_lang=args.bertscore_lang,
    )

    print("=== Task 2 Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Save per-example results (including ROUGE/BERTScore)
    output_path = Path(args.data_path).with_suffix(".task2_eval.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for rec in detailed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nPer-example scores saved to: {output_path}")


if __name__ == "__main__":
    main()
