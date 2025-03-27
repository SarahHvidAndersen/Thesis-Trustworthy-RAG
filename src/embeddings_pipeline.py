import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedding_process.preprocessing import clean_text 
from embedding_process.embeddings import load_embedding_model, embed_text
#from embedding_process.vector_db import init_db, get_collection, add_documents

def process_file(filepath, chunk_size=2048, chunk_overlap=200):
    """
    Reads a scraped JSON file (e.g., processed_syllabi/Decision_making/scraped_data/xxx.json),
    extracts its "text" field, cleans and splits it into chunks using a recursive splitter,
    and attaches the metadata from the original file to each chunk.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get the raw text and metadata (everything except the "text" field)
    raw_text = data.get("text", "")
    cleaned_text = clean_text(raw_text)
    metadata = {key: data[key] for key in data if key != "text"}
    
    # Use LangChain's recursive splitter to produce coherent chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(cleaned_text)
    
    # Attach the original metadata to each chunk.
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = metadata.copy()
        chunk_data["chunk_text"] = chunk
        # add a chunk identifier
        chunk_data["chunk_id"] = f"{os.path.basename(filepath)}_chunk_{i+1}"
        processed_chunks.append(chunk_data)
    
    return processed_chunks

def process_directory(directory):
    """
    Processes all scraped JSON files in a given directory and returns a list of all chunks.
    """
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            chunks = process_file(filepath)
            all_chunks.extend(chunks)
    return all_chunks

def main():
    # Directories for your scraped JSON files.
    input_directory = "processed_syllabi/Data_science/scraped_data"
    output_directory = "processed_syllabi/Data_science"
    os.makedirs(output_directory, exist_ok=True)
    
    # Process scraped JSON files into chunks.
    all_chunks = process_directory(input_directory)
    print(f"Processed {len(all_chunks)} chunks.")
    
    # Save processed chunks for inspection (optional).
    chunks_path = os.path.join(output_directory, "processed_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # Load SentenceTransformer model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_embedding_model(device=device)
    
    # Extract the text from each chunk.
    chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    
    # Use tqdm to show progress during embedding.
    print("Embedding chunks...")
    embeddings = []
    # Process in batches to provide a progress bar.
    batch_size = 16
    for i in tqdm(range(0, len(chunk_texts), batch_size)):
        batch_texts = chunk_texts[i:i+batch_size]
        batch_embeddings = embed_text(batch_texts, model)
        embeddings.extend(batch_embeddings)
    
    # Append embeddings to each chunk.
    for i, emb in enumerate(embeddings):
        # Convert embedding to list for JSON serialization.
        all_chunks[i]["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb
    
    # Save chunks with embeddings.
    output_path = os.path.join(output_directory, "processed_chunks_with_embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved processed chunks with embeddings to: {output_path}")
    
    # Initialize ChromaDB and add documents.
    db_client = init_db(persist_directory="chroma_db")
    collection = get_collection(db_client, collection_name="decision_making_documents")
    
    # Prepare documents for ChromaDB.
    # Each document must have an "id", "embedding", "metadata", and "text".
    docs = []
    for chunk in all_chunks:
        # Use the chunk_id (from your preprocessing) as the unique id.
        doc = {
            "id": chunk.get("chunk_id"),
            "embedding": chunk.get("embedding"),
            "metadata": {k: v for k, v in chunk.items() if k != "chunk_text" and k != "embedding"},
            "text": chunk.get("chunk_text")
        }
        docs.append(doc)
    
    add_documents(collection, docs)
    print("Documents added to ChromaDB.")


if __name__ == "__main__":
    main()
