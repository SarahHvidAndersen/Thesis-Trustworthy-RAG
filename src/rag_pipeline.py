import os
import logging

# Import your retriever and generate modules.
from retriever import load_query_model, retrieve_documents
from generate import generate_answer
from embedding_process.vector_db import init_db, get_collection

def rag_pipeline(query, top_k=5, api_url="", headers=None, device="cpu"):
    """
    Executes an end-to-end RAG pipeline:
      1. Loads the query embedding model.
      2. Retrieves top_k documents from the ChromaDB collection.
      3. Combines the retrieved texts into a context.
      4. Generates an answer using the Hugging Face serverless API.
      
    Returns the generated answer and the retrieved documents.
    """
    # Load the query embedding model (using SentenceTransformer).
    model = load_query_model(device=device)
    
    # Initialize ChromaDB and get the collection.
    db_client = init_db(db_path="chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")
    
    # Retrieve top_k relevant documents for the query.
    retrieved_docs = retrieve_documents(query, model, collection, top_k=top_k)
    
    # Combine retrieved texts to build context.
    # You might consider ordering them by distance or adding separators.
    context = "\n".join([doc["text"] for doc in retrieved_docs])
    
    # Generate an answer using the Hugging Face API.
    answer = generate_answer(query, context, api_url, headers=headers)
    
    return answer, retrieved_docs

if __name__ == "__main__":
    # Set up basic logging.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get a user query.
    query = input("Enter your query: ")
    
    # Configure your Hugging Face API endpoint and token.
    # Replace with your actual model endpoint and token.
    api_url = "https://api-inference.huggingface.co/models/your-model-name"
    headers = {"Authorization": "Bearer YOUR_HF_API_TOKEN"}
    
    # Choose the device.
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or False else "cpu"
    
    # Run the RAG pipeline.
    logging.info("Running RAG pipeline...")
    answer, docs = rag_pipeline(query, top_k=5, api_url=api_url, headers=headers, device=device)
    
    # Output the final answer.
    print("\n=== Generated Answer ===")
    print(answer)
    
    # Optionally, print information about retrieved documents.
    print("\n=== Retrieved Documents ===")
    for doc in docs:
        print(f"ID: {doc['id']} - Metadata: {doc['metadata']}")
