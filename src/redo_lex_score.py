#!/usr/bin/env python3
"""
Compute Lexical Similarity uncertainty for every row whose `samples`
column contains a JSON/ repr‑encoded list of generated texts, and store
the result in a new column called `lex_score`.
"""
import ast
import json
import os
import pandas as pd

from uncertainty_estimation.uncertainty_estimator_factory import (
    get_uncertainty_estimator,
    compute_uncertainty,                 # both defined in factory module
)                                       # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}


INPUT  = "output/response_test_data/filt_anno_split_testset_merged.csv"
OUTPUT = INPUT                          # overwrite in‑place; change if you want

# ---------- helpers ----------------------------------------------------------
def to_list(cell):
    """Convert stringified list -> real list, leave lists as‑is."""
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            return ast.literal_eval(cell)
    raise ValueError(f"samples column must be list or str, got {type(cell).__name__}")

# ---------- main -------------------------------------------------------------
df = pd.read_csv(INPUT)

# Parse once so we don’t repeat work inside the apply loop
df["samples_parsed"] = df["samples"].apply(to_list)

# Build the estimator once (default metric = ROUGE‑L)
lex_est = get_uncertainty_estimator("lexical_similarity")     # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

# Map every list of samples → scalar uncertainty
df["lex_score"] = df["samples_parsed"].apply(
    lambda samples: compute_uncertainty(lex_est, samples)
)

df.drop(columns=["samples_parsed"]).to_csv(OUTPUT, index=False)
print(f"Saved {len(df)} rows with lex_score to {OUTPUT}")
