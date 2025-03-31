import os
import chromadb

db_path = "/work/SarahHvidAndersen#6681/Thesis-Trustworthy-RAG/chroma_db"
print(f"Files in {db_path}: {os.listdir(db_path)}")

client = chromadb.PersistentClient(path=db_path)

# Try to get an existing collection explicitly
collection = client.get_collection("rag_documents")
print(f"Collection count: {collection.count()}")  # Should print 4793
print(collection.peek(limit = 10))