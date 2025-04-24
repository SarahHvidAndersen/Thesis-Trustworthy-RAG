#from rag_pipeline import rag_pipeline
#from core import cfg, provider, estimator
from csv_logger import initialize_csv, log_experiment
import json
from core import run_rag, get_config


def main():
    cfg = get_config()
    model_cfg = cfg['model']
    gen_cfg = cfg['generation']

    query = "What is the role of the prefrontal cortex? answer in one sentence."
    result = run_rag(query, cfg)

    if result is None:
        print("Pipeline returned no result (e.g., an empty document collection).")
        exit(1)

    final_answer, uncertainty, retrieved_docs, samples = (
        result["final_answer"],
        result["uncertainty"],
        result["retrieved_docs"],
        result["samples"],
    )

    print("\n=== Final Answer ===")
    print(final_answer)

    print("\n=== Uncertainty Score ===")
    print(uncertainty)

    #print("\n=== Generated Samples ===")
    #for i, s in enumerate(samples):
    #    print(f"Sample {i+1}: {s}\n")

    ##print("\n=== Retrieved Documents ===")
    #for doc in retrieved_docs:
    #    print(f"ID: {doc['id']} - Metadata: {doc['metadata']}")

    CSV_FILE = "experiment_results.csv"
    initialize_csv(CSV_FILE)

    experiment_data = {
                "query": query,
                "answer": result['final_answer'],
                "samples": json.dumps(result["samples"], ensure_ascii=False),
                "model": model_type,
                "settings": json.dumps({
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": gen_cfg["max_new_tokens"],
                    "top_k": top_k,
                    "n_samples": result["n_samples"],
                }),
                "uncertainty_method": cfg["uncertainty"]["method"],
                "uncertainty_score": result["uncertainty"],
                "retrieved_documents": result["retrieved_docs"],
            }

    log_experiment(CSV_FILE, experiment_data)

    return


if __name__ == "__main__":
    main()