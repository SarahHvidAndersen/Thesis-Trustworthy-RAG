
# lex sim isn't working correctly, returns blank spot in csv i think
# samples compute correctly though, so just re-calculating with redo_ue_score.py
# instead of re-running full gpu process

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from internal.core import (
    get_config,
    init_provider,
    init_estimator,
    init_scaler,
    rag_pipeline,)
from internal.uncertainty_estimation.uncertainty_estimator_factory import (
    get_uncertainty_estimator,
    compute_uncertainty,)

from dotenv import load_dotenv
load_dotenv(override=True)

def json_safe(obj):
    """Return *obj* serialised with numpy scalars converted to built‑ins."""

    def default(o):  # called for objects json can’t handle
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    return json.dumps(obj, default=default, ensure_ascii=False)

# Main evaluation loop
def run_eval(
    input_csv: str,
    output_csv: str,
    n_samples: int  = 5,
    save_every: int = 10,            # save intermediate results every N queries
    resume: bool   = True,           # continue where a previous run left off
):
    cfg = get_config()
    provider_name = cfg['model']['type']
    model_id = cfg['model'][f"{provider_name}_model"]
    temperature = cfg['generation']['temperature']
    top_p = cfg['generation']['top_p']
    top_k = cfg['retrieval']['top_k']
    semantic_weight = cfg['retrieval'].get('semantic_weight', 0.5)

    api_url = os.getenv("CHATUI_GPU_API_URL", "").strip() # gpu run
    provider = init_provider(
        model_type="chatui",
        model_id=cfg["model"]["chatui_model"],
        api_key=api_url,
        cfg=cfg,
    )

    # initialise the three chosen UQ estimators once 
    lex_est = init_estimator(cfg, "lexical_similarity")
    deg_est = init_estimator(cfg, "deg_mat")
    ecc_est = init_estimator(cfg, "eccentricity")

    # Load data 
    df_in = pd.read_csv(input_csv)

    # If resuming, load the existing output and mark processed queries
    done = set()
    if resume and Path(output_csv).exists():
        df_done = pd.read_csv(output_csv)
        done.update(df_done["user_input"].tolist())
        print(f" Resuming – {len(done)} queries already processed, {len(df_in) - len(done)} remaining.")
        dfs_out = [df_done]
    else:
        dfs_out = []

    # Iterate
    pending_rows = []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="RAGAS EVAL"):
        prompt = str(row["user_input"]).strip()
        if prompt in done:
            continue

        start = time.time()

        result = rag_pipeline(
            query=prompt,
            top_k=top_k,
            provider=provider,
            n_samples=n_samples,
            estimator=lex_est, # computing with lexical similarity in the reg pipeline
            chat_history=[],
        )

        samples = result.get("samples", [])
        final_answer = result.get("final_answer", "")
        retrieved_docs = result.get("retrieved_docs", [])

        # Compute all three uncertainty scores on the SAME sample list
        try:
            deg_score = compute_uncertainty(deg_est, samples)
        except Exception as e:
            print(f"DegMat error → {e}");   deg_score = None
        try:
            ecc_score = compute_uncertainty(ecc_est, samples)
        except Exception as e:
            print(f"Eccentricity error → {e}");   ecc_score = None

        # simply extracting lex sim score from the regular rag pipeline
        try:
            lex_score = result.get("uncertainty", "")
        except Exception as e:
            print(f"LexSim error → {e}"); lex_score = None

        elapsed = time.time() - start

        pending_rows.append({
            **row.to_dict(), ## copy *all* of the original CSV columns
            #"user_input": prompt, included above
            "samples": json_safe(samples),
            "final_answer": final_answer,
            "retrieved_docs": json_safe(retrieved_docs),
            "deg_score": deg_score,
            "ecc_score": ecc_score,
            "lex_score": lex_score,
            "time_sec": round(elapsed, 3),
            "provider": provider_name,
            "model_id": model_id,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "semantic_weight": semantic_weight,
        })

        # save intermediate results
        if len(pending_rows) % save_every == 0:
            dfs_out.append(pd.DataFrame(pending_rows))
            pd.concat(dfs_out, ignore_index=True).to_csv(output_csv, index=False)
            print(f"Saved checkpoint with {sum(len(df) for df in dfs_out)} rows.")
            pending_rows.clear()

    # final save
    if pending_rows:
        dfs_out.append(pd.DataFrame(pending_rows))
    pd.concat(dfs_out, ignore_index=True).to_csv(output_csv, index=False)
    print(f" Finished – results saved to {output_csv} (total {sum(len(df) for df in dfs_out)} rows).")


if __name__ == "__main__":

    run_eval(
        input_csv="output/raw_test_data/full_f-anno_split_testset.csv", 
        output_csv="output/answered_test_data/testset_with_predictions.csv",
        n_samples=5,
        save_every=10,
        resume=True,
    )