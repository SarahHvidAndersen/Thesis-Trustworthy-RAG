#!/usr/bin/env python
"""
Adds an 'alignscore' column to output/full_testset_with_predictions.csv
Uses the vendored LM-Polygraph AlignScorer (no external AlignScore package).
Works on Python 3.12 with torch 2.x + transformers 4.x on CPU.
"""
import pathlib
import pandas as pd
from tqdm.auto import tqdm                      
from alignscore_utils import AlignScorer

CSV_IN   = pathlib.Path("output/answered_test_data/testset_with_predictions.csv")
CKPT     = pathlib.Path("AlignScore-large.ckpt")   # or -large
BATCH_SZ = 8                                      # safe for CPUs

def main() -> None:
    df = pd.read_csv(CSV_IN)
    df = df[0:8]
    scorer = AlignScorer(model="roberta-large",
                ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",  # or -base.ckpt
                device="cpu",                          
                batch_size=8)

    refs = df["reference"].tolist()
    hyps = df["final_answer"].tolist()
    scores = []
    for i in tqdm(range(0, len(df), BATCH_SZ), desc="AlignScore"):
        scores.extend(scorer.score(refs[i:i+BATCH_SZ], hyps[i:i+BATCH_SZ]))

    df["alignscore"] = scores
    out = "output/quantitative_metrics/alignscore_testset_with_predictions.csv"
    df.to_csv(out, index=False)
    print(f" {len(df)} rows written → {out}")

if __name__ == "__main__":
    main()
