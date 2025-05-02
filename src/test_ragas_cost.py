"""
# evaluate_ragas_cost.py

Utility script to **evaluate an existing RAG test‑set** with GPT‑4o _only_ (no data generation)
and print a clean cost / token breakdown.

It follows the official Ragas cost‑tracking recipe:
<https://docs.ragas.io/en/stable/howtos/applications/_cost/#token-usage-for-evaluations>

---
## Quick start
```bash
pip install ragas langchain-openai pandas
export OPENAI_API_KEY=sk‑…
python evaluate_ragas_cost.py \
       --csv my_testset.csv           # required – CSV must contain question, answer, contexts, ground_truth
       --out_csv scored.csv           # optional – saves metric columns next to your data
       --max_samples 20               # optional – pilot run to extrapolate full‑set cost
```

The script **does not** mutate your CSV; it streams scores into memory, writes them only
when `--out_csv` is given.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.evaluation import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_openai import ChatOpenAI

# ------------------------------
# Pricing constants (May 2025)
# ------------------------------
PRICE_4O_IN: float = 5 / 1_000_000     # $/token   – input (prompt) tokens
PRICE_4O_OUT: float = 15 / 1_000_000   # $/token   – output (completion) tokens

# ------------------------------
# Helpers
# ------------------------------

def _parse_contexts(cell: str | List[str]) -> List[str]:
    """Make sure the *contexts* column is a list[str] as Ragas expects."""
    if isinstance(cell, list):
        return cell
    # try JSON first
    try:
        parsed = json.loads(cell)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except Exception:
        pass
    # fall back to splitter on `|||` or line breaks
    if isinstance(cell, str):
        parts = [p.strip() for p in cell.split("|||") if p.strip()]
        return parts if parts else [cell]
    return [str(cell)]


def load_eval_dataset(csv_path: Path, delimiter: str = ",") -> EvaluationDataset:
    """Load CSV → `EvaluationDataset`, validating required columns."""
    df = pd.read_csv(csv_path, delimiter=delimiter)

    # Common column aliases → canonical names
    rename_map = {
        "ground_truth_answer": "ground_truth",
        "generated_answer": "answer",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"question", "answer", "ground_truth", "contexts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {', '.join(sorted(missing))}. "
            "Expected at least question, answer, contexts, ground_truth."
        )

    df["contexts"] = df["contexts"].apply(_parse_contexts)
    return EvaluationDataset.from_pandas(df)


# ------------------------------
# Main
# ------------------------------

def main(args: argparse.Namespace) -> None:
    dataset = load_eval_dataset(Path(args.csv), delimiter=args.delimiter)

    # Optional pilot‑size slice
    if args.max_samples:
        dataset = EvaluationDataset.from_pandas(
            dataset.to_pandas()
            .sample(n=args.max_samples, random_state=42)
            .reset_index(drop=True)
        )

    critic = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    metrics = [faithfulness, answer_relevancy, context_precision]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=critic,
        token_usage_parser=get_token_usage_for_openai,
    )

    usage = result.total_tokens()
    cost = result.total_cost(
        cost_per_input_token=PRICE_4O_IN, cost_per_output_token=PRICE_4O_OUT
    )

    print("\n=== Ragas evaluation complete ===")
    print(f"Samples evaluated : {len(dataset):,}")
    print(f"Prompt tokens     : {usage.input_tokens:,}")
    print(f"Completion tokens : {usage.output_tokens:,}")
    print(f"Total cost        : ${cost:.4f} USD")
    print(f"Per‑sample cost   : ${cost/len(dataset):.6f} USD")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        result.scores.to_csv(args.out_csv, index=False)
        print(f"\nSaved metric columns → {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a CSV test‑set with GPT‑4o critic and print token‑level cost."
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV test‑set")
    parser.add_argument("--out_csv", help="If set, write the scored dataset here")
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Only score N random samples (useful for cheap pilot runs)",
    )
    parser.add_argument(
        "--delimiter", default=",", help="CSV delimiter (default ',')"
    )

    main(parser.parse_args())