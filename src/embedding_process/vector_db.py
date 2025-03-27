import chromadb
from chromadb.config import Settings

def init_db(persist_directory="chroma_db"):
    """
    Initializes a ChromaDB client. If persist_directory is provided,
    the database is persisted on disk.
    """
    client = chromadb.Client(Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))
    return client

def get_collection(client, collection_name="documents"):
    """
    Retrieves a collection if it exists; otherwise creates a new one.
    """
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        collection = client.create_collection(name=collection_name)
    return collection

def add_documents(collection, docs):
    """
    Inserts documents into the collection.
    
    Each document in docs should be a dict with keys:
      - "id": a unique identifier (string)
      - "embedding": a list or array of floats
      - "metadata": a dict with any additional metadata
      - "text": the original text content
    """
    ids = [doc["id"] for doc in docs]
    embeddings = [doc["embedding"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]
    documents = [doc["text"] for doc in docs]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def query_documents(collection, query_embedding, n_results=5):
    """
    Queries the collection with the given embedding, returning top-n results.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["ids", "metadatas", "documents", "distances"]
    )
    return results
