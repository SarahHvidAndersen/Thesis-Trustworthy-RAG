import os
import json
from tqdm import tqdm 
import torch
import torch.nn.functional as F
from langchain.text_splitter import RecursiveCharacterTextSplitter

from internal.database_setup.bm25_indexer import build_index
from internal.database_setup.preprocessing import clean_text 
from internal.database_setup.embeddings import load_embedding_model, embed_text
from internal.database_setup.chroma_db import init_db, get_collection, add_documents
# full chromadb/bm25index length: 13758

def process_file(filepath, course, chunk_size=2048, chunk_overlap=200):
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
        chunk_data["chunk_id"] = f"{course}_{os.path.basename(filepath)}_chunk_{i+1}"
        processed_chunks.append(chunk_data)
    
    return processed_chunks

def process_directory(directory, course):
    """
    Processes all scraped JSON files in a given directory and returns a list of all chunks.
    """
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            chunks = process_file(filepath, course)
            all_chunks.extend(chunks)
    return all_chunks


def process_course(course):
    input_directory = os.path.join("data", "processed_syllabi", course, "scraped_data")
    output_directory = os.path.join("data", "processed_syllabi", course)
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Processing course: {course}")
    # Check for already-processed chunks file
    chunks_path = os.path.join(output_directory, "processed_chunks.json")
    if os.path.exists(chunks_path):
        print(f"Found existing chunks file for {course}, loading…")
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        # If processed chunk doesn't exist, scan & split the scraped_data
        all_chunks = process_directory(input_directory, course)
        
        # add course metadata for the first run
        for chunk in all_chunks:
            chunk["course"] = course

        print(f"Processed {len(all_chunks)} chunks for {course}.")
        # Save for next time
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved processed chunks to: {chunks_path}")
    
    # Check if embeddings file exists; if so, load it instead of recomputing.
    embeddings_file = os.path.join(output_directory, "processed_chunks_with_embeddings.json")
    if os.path.exists(embeddings_file):
        print(f"Embeddings file exists for {course}. Loading...")
        with open(embeddings_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        # Load model and compute embeddings.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_embedding_model(device=device)
        chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]

        print("Embedding chunks...")
        embeddings = []
        batch_size = 16

        for i in tqdm(range(0, len(chunk_texts), batch_size), desc=f"Embedding {course}"):
            batch_texts = chunk_texts[i:i+batch_size]
            batch_embeddings = embed_text(batch_texts, model)
            embeddings.extend(batch_embeddings)
        for i, emb in enumerate(embeddings):
            all_chunks[i]["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb

        # save embeddings
        with open(embeddings_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved processed chunks with embeddings to: {embeddings_file}")
    
    return all_chunks

def main(courses):
    # Process each course.
    all_courses_chunks = []
    for course in courses:
        course_chunks = process_course(course)
        # Add course metadata to each chunk.
        for chunk in course_chunks:
            chunk["course"] = course
        all_courses_chunks.extend(course_chunks)

    print(f"Total chunks processed: {len(all_courses_chunks)}")
    #print("Sample chunk:", all_courses_chunks[:1])  # Print first chunk for inspection

    chunk_ids = [chunk.get("chunk_id") for chunk in all_courses_chunks]
    print(f"Unique chunk IDs: {len(set(chunk_ids))}, Total chunks: {len(chunk_ids)}")

    # initialize ChromaDB
    db_client = init_db(db_path="data/chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")

    print("Checking existing documents in ChromaDB:")
    print(collection.count()) 
    
    # Prepare documents for ChromaDB.
    docs = []
    for chunk in all_courses_chunks:
        doc = {
            "id": chunk.get("chunk_id"),
            "embedding": chunk.get("embedding"),
            "metadata": {k: v for k, v in chunk.items() if k not in ["chunk_text", "embedding", "flag"]},
            "text": chunk.get("chunk_text")
        }
        docs.append(doc)
    
    print(f"Total documents to upsert: {len(docs)}")

    # upsert to chroma method
    add_documents(collection, docs)
    print("Documents added to ChromaDB.")

    return 


if __name__ == "__main__":

    courses = []
    # List of courses to process
    first_courses = ['Human_computer_interaction', 'Natural_language_processing', 'Adv_cog_neuroscience', 
            'Adv_cognitive_modelling', 'Data_science', 'Decision_making']
    second_courses = ['applied_cognitive_science', 'cognition_and_communication', 'cognitive_neuroscience', 
               'intro_to_cognitive_science', 'Methods_1', 'Methods_2', 'Methods_3', 'Methods_4']
    third_courses = ['perception_and_action', 'philosophy_of_cognitive_science', 'social_and_cultural_dynamics', 
                     'applied_cognitive_science']
    
    courses.append(first_courses)
    courses.append(second_courses)
    courses.append(third_courses)
            
    for course_batch in courses:
        main(course_batch)

    # initialize bm25 index, after creating all chunks (shouldn't be in batches)
    build_index()
    print("Documents added to bm25 index.")


