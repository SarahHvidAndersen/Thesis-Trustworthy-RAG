#!/usr/bin/env python
"""
Minimal CLI for running the RAG pipeline.

Examples
--------
$ uv run -m internal.run_cli --provider Ollama --query "Explain what the neocortex is?" --no-save-env
$ uv run -m internal.run_cli -p Huggingface                       # interactive query prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from internal.core import run_rag, get_config
from internal.logging_utils.csv_logger import initialize_csv, log_experiment
from internal.providers.provider_utils import ensure_provider_input

from dotenv import load_dotenv
load_dotenv(override=True)

# add user input arguments
def parse_args() -> argparse.Namespace:
    cfg = get_config()
    default_provider = cfg["model"]["type"]

    parser = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation from the CLI")
    parser.add_argument(
        "-p", "--provider",
        choices=("ChatUI", "Ollama", "Huggingface"),
        default=default_provider,
        help=f"Which backend to use (default: {default_provider})"
    )
    parser.add_argument(
        "-q", "--query",
        help="Prompt to send to the model. If omitted, you will be asked for it."
    )
    parser.add_argument(
        "--save-env",
        dest="save_env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist any entered URLs / API keys to .env (default: true)"
    )
    return parser.parse_args()


def get_query(arg_value: str | None) -> str:
    if arg_value:
        return arg_value
    # interactive fallback
    try:
        return input("Enter your query: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nAborted.")


def persist_path() -> Path:
    out_dir = Path("output/client_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "experiment_results.csv"


# main entry-point
def main() -> None:
    args = parse_args()
    cfg = get_config()
    provider = args.provider.replace("huggingface", "hf")
    query = get_query(args.query)

    # get (or prompt for) the URL / API-key
    api_cred = ensure_provider_input(
        provider,
        persist=args.save_env
    )

    print("\n--- RAG CLI -------------------------------------------")
    print(f"provider : {provider}")
    print(f"persist to .env: {'yes' if args.save_env else 'no'}")
    print(f"query : {query}\n")

    # run the pipeline
    result = run_rag(
        query=query,
        cfg=cfg,
        provider_name_override=provider,
        api_key_override=api_cred,
    )

    if result is None:
        print("No result returned (e.g. empty document collection).")
        return

    print("\n--- RETRIEVED DOCS (top-k) ----------------------------")
    for d in result["retrieved_docs"]:
        snippet = d["text"][:100].replace("\n", " ")
        print(f"[{d['source']}] {d['id']}  score={d['rerank_score']:.3f}  →  {snippet}…")

    if result["raw_uncertainty"] is not None:
        print(f"\nuncertainty (raw)  : {result['raw_uncertainty']:.4f}")
        print(f"confidence (scaled): {result['calibrated_confidence']:.4f}")
    
    # pretty print
    print("\n--- FINAL ANSWER --------------------------------------")
    print(result["final_answer"])

    # log
    gen_cfg = cfg['generation']
    retr_cfg = cfg.get('retrieval', {})
    model_type = provider

    n_samples = gen_cfg['n_samples']
    temperature = gen_cfg['temperature']
    top_p = gen_cfg['top_p']
    uq_method = cfg["uncertainty"]["method"]

    csv_path = persist_path()
    initialize_csv(csv_path)
    log_experiment(csv_path, {
                "query": query,
                "answer": result['final_answer'],
                "samples": json.dumps(result["samples"], ensure_ascii=False),
                "model": model_type,
                "settings": json.dumps({
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": gen_cfg["max_new_tokens"],
                    "top_k": retr_cfg["top_k"],
                    "n_samples": n_samples
                }),
                "uncertainty_method": uq_method,
                "raw_uncertainty": result["raw_uncertainty"],
                "calibrated_confidence": result["calibrated_confidence"],
                "retrieved_documents": result["retrieved_docs"],
            })
    print(f"\nExperiment logged to {csv_path}")


if __name__ == "__main__":
    main()
