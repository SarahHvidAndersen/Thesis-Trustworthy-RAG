import os
import json
from internal.logging_utils.csv_logger import initialize_csv, log_experiment
from internal.core import run_rag, get_config

def main():
    # Load configuration
    cfg = get_config()
    model_cfg = cfg['model']
    gen_cfg = cfg['generation']
    retr_cfg = cfg.get('retrieval', {})

    # Extract settings for clarity
    model_type   = model_cfg.get('type')
    top_k        = retr_cfg.get('top_k')
    n_samples    = gen_cfg.get('n_samples')
    temperature  = gen_cfg.get('temperature')
    top_p        = gen_cfg.get('top_p')
    uq_method    = cfg['uncertainty']['method']

    # Define your test query here
    query = "What are embeddings in natural language processing? answer in one sentence."

    # Print all inputs for debugging
    print("\n--- Running RAG CLI Debug ---")
    print(f"Query: {query}")
    print(f"Model Type: {model_type}")
    print(f"Top-K docs: {top_k}, N-samples: {n_samples}")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Uncertainty Method: {uq_method}\n")

    # Run the pipeline
    print("Running run_rag()...")
    result = run_rag(
        query=query,
        cfg=cfg,
        top_k_override=top_k,
        n_samples_override=n_samples,
        uq_method_override=uq_method
    )

    # Check for empty result
    if result is None:
        print("Pipeline returned no result (e.g., empty document collection). Exiting.")
        return

    # Unpack the result
    final_answer = result.get("final_answer")
    uncertainty = result.get("uncertainty")
    retrieved_docs = result.get("retrieved_docs")
    samples = result.get("samples")

    # Print raw outputs for inspection
    print("\n--- RAW SAMPLES ---")
    for idx, s in enumerate(samples, start=1):
        print(f"Sample {idx}: {s}")

    print(F"\n--- Retrieved Documents --- length: {len(retrieved_docs)}")
    for doc in retrieved_docs:
        print(f"ID: {doc.get('id')}, Rerank: {doc.get('rerank_score')}, Source: {doc.get('source')}")
        print(f" Text snippet: {doc.get('text')[:100].replace('\n',' ')}...")

    print("\n--- Uncertainty Score ---")
    print(uncertainty)

    print("\n--- Final Answer ---")
    print(final_answer)

    # Persist experiment data
    CSV_FILE = "output/client_run/experiment_results.csv"
    initialize_csv(CSV_FILE)
    experiment_data = {
        "query": query,
        "answer": final_answer,
        "samples": json.dumps(samples, ensure_ascii=False),
        "model": model_type,
        "settings": json.dumps({
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": gen_cfg.get("max_new_tokens"),
            "top_k": top_k,
            "n_samples": n_samples,
        }),
        "uncertainty_method": uq_method,
        "uncertainty_score": uncertainty,
        "retrieved_documents": retrieved_docs,
    }
    log_experiment(CSV_FILE, experiment_data)
    print(f"\nExperiment logged to {CSV_FILE}")

if __name__ == "__main__":
    main()
