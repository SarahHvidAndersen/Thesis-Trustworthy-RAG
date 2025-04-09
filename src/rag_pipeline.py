import os
import logging

# local modules
from retriever import load_embedding_model, retrieve_documents
from embedding_process.vector_db import init_db, get_collection

def rag_pipeline(query, top_k=5, provider=None, device="cpu", n_samples=1, estimator=None):
    """
    Executes an end-to-end RAG pipeline:
      1. Loads the query embedding model.
      2. Retrieves top_k documents from the ChromaDB collection.
      3. Combines the retrieved texts into a context.
      4. Generates an answer using the provided generator.
      
    Parameters:
      query: The user query.
      top_k: Number of documents to retrieve.
      provider: An instance of GeneratorProvider (HuggingFaceProvider or ChatUIProvider).
      device: "cpu" or "cuda" for the embedding model.
    
    Returns:
      A tuple: (generated answer, list of retrieved documents).
    """
    print('loading model')
    # Load the query embedding model
    model = load_embedding_model(device=device)
    
    print('loading database')
    # Initialize ChromaDB and get the collection.
    db_client = init_db(db_path="chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")
    print(f"Collection count: {collection.count()}")

    if not collection.count():
      print('empty collection')
      return
    
    # Retrieve top_k relevant documents for the query.
    retrieved_docs = retrieve_documents(query, model, collection, top_k=top_k)

    # Combine retrieved text fields to form the context.
    context = "\n".join([doc["text"] for doc in retrieved_docs])
    print(f"Context built from retrieved documents:\n{context}\n")

    # If we're generating multiple samples to compute uncertainty:
    if n_samples > 1 and estimator is not None:
        responses = []
        print(f"Generating {n_samples} samples for uncertainty estimation...")
        for i in range(n_samples):
            sample_response = provider.generate(query, context)
            print(f"Sample {i+1}:\n{sample_response}\n")
            responses.append(sample_response)
        
        # Build stats dictionary expected by the estimator. stats holds a list of samples.
        stats = {"sample_texts": [responses]}
        uncertainty_scores = estimator(stats)
        uncertainty_score = float(uncertainty_scores[0])

        # For the final answer, we could choose the first sample (or do further selection).
        final_answer = responses[0]
    else:
        # Single sample generation; uncertainty will be None.
        print("Generating a single answer...")
        final_answer = provider.generate(query, context)
        uncertainty_score = None
    
    print("Generation complete.")

    return {
        "final_answer": final_answer,
        "samples": responses,
        "retrieved_docs": retrieved_docs,
        "uncertainty": uncertainty_score,
        "top_k": top_k,
        "n_samples": n_samples
    }

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
    from estimators.lexical_similarity import LexicalSimilarity
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

        
    
