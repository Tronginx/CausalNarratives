from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional, Tuple

from pydantic import BaseModel, Field
import math


# =========================
# 1. Data and prediction structures
# =========================

class Task3Example(BaseModel):
    """
    One example for FinLLM Task 3 (single stock trading).
    Represents the market context at a single time step.
    """
    id: str
    date: str                         # e.g. "2020-10-09"
    price: Dict[str, float]           # mapping from symbol to price (e.g. {"DRIV": 17.52})
    filing_k: Dict[str, Any] = Field(default_factory=dict)
    filing_q: Dict[str, Any] = Field(default_factory=dict)
    news: Dict[str, List[str]] = Field(default_factory=dict)
    # You can add more raw fields from the dataset as needed

    @classmethod
    def from_raw(cls, raw: Dict) -> "Task3Example":
        """
        Convert a raw row (e.g. from json/csv) into a standardized Task3Example.

        EXPECTED FIELDS (adapt to your real dataset if needed):
            - raw["id"]
            - raw["date"]
            - raw["price"]     (dict: symbol -> price)
            - raw["filing_k"]
            - raw["filing_q"]
            - raw["news"]
        """
        return cls(
            id=str(raw.get("id", "")),
            date=str(raw.get("date", "")),
            price=dict(raw.get("price", {})),
            filing_k=dict(raw.get("filing_k", {})),
            filing_q=dict(raw.get("filing_q", {})),
            news={k: list(v) for k, v in dict(raw.get("news", {})).items()},
        )

    def get_main_symbol_and_price(self, explicit_symbol: Optional[str] = None) -> Tuple[str, float]:
        """
        Return (symbol, price) for the main traded asset.

        If `explicit_symbol` is given, use that. Otherwise:
          - If there is exactly one key in `price`, use that symbol.
          - Otherwise raise an error (caller must specify).
        """
        if explicit_symbol is not None:
            if explicit_symbol not in self.price:
                raise KeyError(f"Symbol '{explicit_symbol}' not found in price dict for example {self.id}")
            return explicit_symbol, float(self.price[explicit_symbol])

        if len(self.price) == 1:
            symbol = next(iter(self.price.keys()))
            return symbol, float(self.price[symbol])

        raise ValueError(
            f"Cannot infer main symbol from price dict for example {self.id}: "
            f"multiple symbols present and no explicit_symbol provided."
        )


class TradingDecision(BaseModel):
    """
    Structured trading decision parsed from the model's JSON output.
    """
    investment_decision: str          # "buy", "sell", or "hold"
    summary_reason: str
    short_memory_index: float
    middle_memory_index: float
    long_memory_index: float
    reflection_memory_index: float


class Task3Prediction(BaseModel):
    """
    Model prediction for one Task 3 example.
    """
    id: str
    decision: TradingDecision
    raw_output: Optional[str] = None  # optional raw model output (for debugging)


# Unified prediction function signature
TradingPredictFn = Callable[[Task3Example], Task3Prediction]


# =========================
# 2. Trading metrics and simulation
# =========================

@dataclass
class TradingMetrics:
    sharpe_ratio: float
    cumulative_return: float
    daily_volatility: float
    annualized_volatility: float
    max_drawdown: float
    num_steps: int


def simulate_trading_and_compute_metrics(
    examples: List[Task3Example],
    predictions: List[Task3Prediction],
    asset_symbol: Optional[str] = None,
    annualization_factor: int = 252,
    decision_to_position: Optional[Dict[str, float]] = None,
) -> Tuple[TradingMetrics, List[Dict]]:
    """
    Simulate a simple trading strategy over time and compute performance metrics.

    Args:
        examples: list of Task3Example, one per time step.
        predictions: list of Task3Prediction, same length and aligned with examples.
        asset_symbol: if provided, use this symbol from `price` dict;
                      otherwise infer from each example (must have exactly one symbol).
        annualization_factor: number of trading days per year (default 252).
        decision_to_position: mapping from decision string to position weight.
                              Defaults to {"buy": 1.0, "sell": -1.0, "hold": 0.0}.

    Returns:
        metrics: TradingMetrics with SR, CR, DV, AV, MD.
        detailed_records: list of per-step records including returns and equity curve.
    """
    assert len(examples) == len(predictions), "examples/predictions length mismatch"

    if decision_to_position is None:
        decision_to_position = {
            "buy": 1.0,
            "sell": -1.0,
            "hold": 0.0,
        }

    # Sort by date to ensure chronological order
    paired = list(zip(examples, predictions))
    paired.sort(key=lambda pair: pair[0].date)

    sorted_examples = [p[0] for p in paired]
    sorted_predictions = [p[1] for p in paired]

    # Extract prices, positions, and metadata
    prices: List[float] = []
    positions: List[float] = []
    dates: List[str] = []
    ids: List[str] = []

    for ex, pred in zip(sorted_examples, sorted_predictions):
        symbol, price = ex.get_main_symbol_and_price(asset_symbol)
        decision_str = pred.decision.investment_decision.strip().lower()
        if decision_str not in decision_to_position:
            raise ValueError(f"Unknown investment_decision '{decision_str}' for example {ex.id}")
        pos = decision_to_position[decision_str]

        prices.append(price)
        positions.append(pos)
        dates.append(ex.date)
        ids.append(ex.id)

    # Need at least two time steps to compute returns
    if len(prices) < 2:
        metrics = TradingMetrics(
            sharpe_ratio=0.0,
            cumulative_return=0.0,
            daily_volatility=0.0,
            annualized_volatility=0.0,
            max_drawdown=0.0,
            num_steps=0,
        )
        return metrics, []

    # Compute asset daily returns and portfolio returns
    asset_returns: List[float] = []
    portfolio_returns: List[float] = []

    for t in range(len(prices) - 1):
        p_t = prices[t]
        p_next = prices[t + 1]
        if p_t <= 0:
            asset_ret = 0.0
        else:
            asset_ret = (p_next / p_t) - 1.0
        pos = positions[t]
        port_ret = pos * asset_ret

        asset_returns.append(asset_ret)
        portfolio_returns.append(port_ret)

    # Build equity curve (start with 1.0)
    equity_curve: List[float] = [1.0]
    for r in portfolio_returns:
        equity_curve.append(equity_curve[-1] * (1.0 + r))

    # Basic stats
    n = len(portfolio_returns)
    mean_r = sum(portfolio_returns) / n if n > 0 else 0.0

    if n > 1:
        var_r = sum((r - mean_r) ** 2 for r in portfolio_returns) / (n - 1)
    else:
        var_r = 0.0

    std_r = math.sqrt(var_r)
    daily_vol = std_r
    annualized_vol = std_r * math.sqrt(annualization_factor) if std_r > 0 else 0.0

    if std_r > 0:
        sharpe = (mean_r / std_r) * math.sqrt(annualization_factor)
    else:
        sharpe = 0.0

    cumulative_return = equity_curve[-1] - 1.0

    # Max drawdown (positive number, e.g. 0.25 = 25%)
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak  # <= 0
        if drawdown < max_dd:
            max_dd = drawdown
    max_drawdown = abs(max_dd)

    metrics = TradingMetrics(
        sharpe_ratio=sharpe,
        cumulative_return=cumulative_return,
        daily_volatility=daily_vol,
        annualized_volatility=annualized_vol,
        max_drawdown=max_drawdown,
        num_steps=n,
    )

    # Detailed per-step records (aligned to the time steps where returns are defined)
    detailed_records: List[Dict] = []
    # For returns, we have length n = len(prices) - 1, aligned with days 0..n-1
    for t in range(n):
        rec = {
            "id": ids[t],
            "date": dates[t],
            "price": prices[t],
            "position": positions[t],
            "asset_return": asset_returns[t],
            "portfolio_return": portfolio_returns[t],
            "equity": equity_curve[t + 1],  # equity after applying return of day t
        }
        detailed_records.append(rec)

    return metrics, detailed_records


# =========================
# 3. High-level evaluation API
# =========================

def load_task3_examples_from_jsonl(jsonl_path: str) -> List[Task3Example]:
    """
    Load Task 3 trading examples from a JSONL file.

    If your dataset is not JSONL, modify this function accordingly.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    examples: List[Task3Example] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            ex = Task3Example.from_raw(raw)
            examples.append(ex)

    return examples


def my_trading_predict(example: Task3Example) -> Task3Prediction:
    """
    Stub trading prediction function.

    This implementation always outputs 'hold' with dummy reasons and indices.
    You MUST replace this function with your real LLM-based trading decision model.

    The model output must conform to the JSON format:
      {
        "investment_decision": string,           # "buy", "sell", or "hold"
        "summary_reason": string,
        "short_memory_index": number,
        "middle_memory_index": number,
        "long_memory_index": number,
        "reflection_memory_index": number
      }
    """
    decision = TradingDecision(
        investment_decision="hold",
        summary_reason="Baseline strategy does nothing (hold).",
        short_memory_index=0.0,
        middle_memory_index=0.0,
        long_memory_index=0.0,
        reflection_memory_index=0.0,
    )
    return Task3Prediction(
        id=example.id,
        decision=decision,
        raw_output=None,
    )


def evaluate_task3(
    examples: List[Task3Example],
    predict_fn: TradingPredictFn,
    asset_symbol: Optional[str] = None,
    annualization_factor: int = 252,
    decision_to_position: Optional[Dict[str, float]] = None,
) -> Tuple[TradingMetrics, List[Dict]]:
    """
    High-level evaluation for FinLLM Task 3 (single stock trading).

    Args:
        examples: list of Task3Example
        predict_fn: function that maps Task3Example -> Task3Prediction
        asset_symbol: main traded symbol in the price dict (if None, infer)
        annualization_factor: trading days per year (default 252)
        decision_to_position: mapping from decision string to position weight

    Returns:
        metrics: TradingMetrics
        detailed_records: list of per-step records with returns and equity
    """
    predictions: List[Task3Prediction] = []
    for ex in examples:
        pred = predict_fn(ex)
        predictions.append(pred)

    metrics, detailed_records = simulate_trading_and_compute_metrics(
        examples=examples,
        predictions=predictions,
        asset_symbol=asset_symbol,
        annualization_factor=annualization_factor,
        decision_to_position=decision_to_position,
    )
    return metrics, detailed_records


# =========================
# 4. CLI entry point
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a FinLLM Task 3 model (single stock trading)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Task 3 evaluation file (JSONL format by default).",
    )
    parser.add_argument(
        "--asset_symbol",
        type=str,
        default=None,
        help="Main traded symbol (e.g. 'DRIV'). If omitted and each example "
             "has exactly one symbol in `price`, it will be inferred.",
    )

    args = parser.parse_args()

    examples: List[Task3Example] = load_task3_examples_from_jsonl(args.data_path)

    metrics, detailed_records = evaluate_task3(
        examples=examples,
        predict_fn=my_trading_predict,  # Replace with your model
        asset_symbol=args.asset_symbol,
        annualization_factor=252,
        decision_to_position={"buy": 1.0, "sell": -1.0, "hold": 0.0},
    )

    print("=== Task 3 Trading Metrics ===")
    print(json.dumps({
        "sharpe_ratio": metrics.sharpe_ratio,
        "cumulative_return": metrics.cumulative_return,
        "daily_volatility": metrics.daily_volatility,
        "annualized_volatility": metrics.annualized_volatility,
        "max_drawdown": metrics.max_drawdown,
        "num_steps": metrics.num_steps,
        "ranking_metric": "sharpe_ratio",
    }, indent=2))

    # Save detailed per-step records
    output_path = Path(args.data_path).with_suffix(".task3_eval.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for rec in detailed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nPer-step trading records saved to: {output_path}")


if __name__ == "__main__":
    main()
