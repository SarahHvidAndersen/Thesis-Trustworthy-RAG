import torch
from sentence_transformers import SentenceTransformer
from internal.database_setup.chroma_db import get_collection, init_db 
from functools import lru_cache


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = "intfloat/multilingual-e5-large-instruct", device: str = "cpu"):
    """
    Loads and caches the embedding model.
    """
    model = SentenceTransformer(model_name, device=device)
    return model

def embed_query(query, model):
    """
    Embeds the query using the provided SentenceTransformer model.
    Returns a numpy array representing the normalized embedding.
    """
    # The model internally handles tokenization, truncation (max_length=512) and normalization.
    embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    return embedding.cpu().numpy()

def retrieve_documents(query, model, collection, top_k=5):
    """
    Given a natural language query, embeds the query and performs a vector similarity search on
    the provided ChromaDB collection. Returns the top_k results.
    
    Each result will include document id, metadata, text, and similarity distance.
    """
    # Embed the query.
    query_embedding = embed_query(query, model)
    
    # Query the collection.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    # The results dictionary contains lists; convert to a list of dicts.
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    return retrieved 


if __name__ == "__main__":
    # Example usage:
    # Initialize (load or create) the database client and collection.
    db_client = init_db(db_path="data/chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")
    print(f"Collection count: {collection.count()}")  # Should print 13758
    
    # Load your query embedding model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_model = load_embedding_model(device=device)
    
    # Example query:
    query = "What does the prefrontal cortex do?"
    
    # Retrieve top 5 documents.
    results = retrieve_documents(query, query_model, collection, top_k=5)
    
    print("Retrieved documents:")
    for res in results:
        print(f"ID: {res['id']}")
        print(f"Distance: {res['distance']:.4f}")
        print(f"Metadata: {res['metadata']}")
        print(f"Text snippet: {res['text'][:200]}...\n")
