import chromadb
from chromadb.config import Settings

def init_db(db_path):
    """
    Initializes a ChromaDB client. If persist_directory is provided,
    the database is persisted on disk.
    """
    client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=False))
    return client

def get_collection(client, collection_name="documents"):
    """
    Retrieves a collection if it exists; otherwise creates a new one.
    """
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def add_documents(collection, docs):
    """
    Upserts documents into the collection.
    
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

    print(f"Upserting {len(ids)} documents...")

    if not ids:
        print("⚠️ No documents to upsert!")
        return
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print("✅ Upsert complete.")
    return


# only used for testing, maybe delete later
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
