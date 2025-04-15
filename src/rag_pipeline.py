import os
import logging

# local modules
from retriever import load_embedding_model, retrieve_documents
from embedding_process.vector_db import init_db, get_collection


def rag_pipeline(query, top_k=5, provider=None, device="cpu", n_samples=2, estimator=None):
    """
    Executes the end-to-end RAG pipeline:
      1. Loads the query embedding model.
      2. Retrieves top_k documents from ChromaDB.
      3. Combines the texts into a context.
      4. Generates an answer (or multiple samples) using the provider.
      5. Optionally calculates an uncertainty score using the estimator.
    
    Parameters:
      query: The user query.
      top_k: Number of documents to retrieve.
      provider: Instance of GeneratorProvider.
      device: 'cpu' or 'cuda' for the embedding model.
      n_samples: Number of answer samples to generate.
      estimator: An uncertainty estimator instance (implementing the __call__ method).
    
    Returns:
      A dictionary with keys:
         "final_answer": The chosen answer (e.g., first sample).
         "samples": The list of generated samples.
         "retrieved_docs": Retrieved document metadata.
         "uncertainty": A float uncertainty score (or None).
         "top_k": The top_k value used.
         "n_samples": The number of generated samples.
    """
    print('Loading query embedding model...')
    model = load_embedding_model(device=device)
    
    print('Loading database...')
    db_client = init_db(db_path="chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")
    print(f"Collection count: {collection.count()}")
    
    if not collection.count():
        print('Empty collection; aborting.')
        return None
    
    print("Retrieving documents...")
    retrieved_docs = retrieve_documents(query, model, collection, top_k=top_k)
    
    context = "\n".join([doc["text"] for doc in retrieved_docs])
    print(f"Context built:\n{context}\n")
    
    samples = []
    if n_samples > 1 and estimator is not None:
        print(f"Generating {n_samples} samples for uncertainty estimation...")
        for i in range(n_samples):
            sample_response = provider.generate(query, context)
            print(f"Sample {i+1}:\n{sample_response}\n")
            samples.append(sample_response)

        #stats = {"sample_texts": [samples]}  # Expecting a list of samples for one query.
        #uncertainty_scores = estimator(stats)
        #uncertainty = float(uncertainty_scores[0])

        # Use the factory helper to compute uncertainty.
        from uncertainty_estimator_factory import compute_uncertainty
        uncertainty_score = compute_uncertainty(estimator, samples)
        print(uncertainty_score)
        final_answer = samples[0]  # we can later choose the sample with the best score.
    else:
        print("Generating a single answer...")
        sample_response = provider.generate(query, context)
        samples.append(sample_response)
        final_answer = sample_response
        uncertainty_score = None
    
    print("Generation complete.")
    
    return {
        "final_answer": final_answer,
        "samples": samples,
        "retrieved_docs": retrieved_docs,
        "uncertainty": uncertainty_score,
        "top_k": top_k,
        "n_samples": n_samples
    }


# doesn't work rn, update
if __name__ == "__main__":
    # Example usage: Using the chatuiprovider.
    from providers.provider import HuggingFaceProvider, ChatUIProvider
    from dotenv import load_dotenv
    load_dotenv()
    
    # configurations
    HF_API_KEY = os.getenv("HF_API_KEY")
    HF_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

    CHATUI_API_URL = os.getenv("CHATUI_API_URL")
    
    # Initialize the provider.
    # test hf header (cache) later
    #provider = HuggingFaceProvider(api_url=HF_MODEL_URL, headers={"Authorization": f"Bearer {HF_API_KEY}"}, "X-use-cache": "false")
    provider = ChatUIProvider(api_url=CHATUI_API_URL)

    # -- Import and initialize the uncertainty estimator.
    from uncertainty_estimation.lexical_similarity import LexicalSimilarity
    lexsim_estimator = LexicalSimilarity(metric="rougeL")
  
    # Example query.
    query = "What is the role of the prefrontal cortex in decision-making? answer in one sentence."
    
   # Run the RAG pipeline with multiple samples to compute uncertainty.
    answer, docs, uncertainty = rag_pipeline(query, top_k=5, provider=provider, device="cpu",
                                              n_samples=5, estimator=lexsim_estimator)
    print("\n=== Retrieved Documents ===")
    for doc in docs:
        print(f"ID: {doc['id']} - Metadata: {doc['metadata']}")

    print("\n=== Final Answer ===")
    print(answer)

    print("\n=== Uncertainty Score ===")
    print(uncertainty)

    from csv_logger import initialize_csv, log_experiment

    # Specify the CSV file path.
    CSV_FILE = "experiment_results.csv"

    # Initialize the CSV (this creates the file with headers if needed).
    initialize_csv(CSV_FILE)

    # Log the experiment data.
    log_experiment(CSV_FILE, {
        "query": query,
        "answer": answer,
        "model": 'experiment_model',
        "settings": 'experiment_settings',
        "uncertainty_score": uncertainty
    })

        
    
