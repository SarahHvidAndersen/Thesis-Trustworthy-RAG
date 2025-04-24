import os
import json
import bm25s
import jax  # optional: speeds up top-k selection
import Stemmer  # optional: for stemming

# Paths (adjust as needed)
BASE_DIR = os.path.dirname(__file__)
INDEX_DIR = os.path.join(BASE_DIR, 'bm25_index')
# expect all_processed_chunks.json collates chunks from all courses
CHUNKS_PATH = os.path.join(BASE_DIR, 'processed_syllabi', 'all_processed_chunks.json')

# Globals to hold loaded index and chunk map
bm25_retriever = None
chunk_map = {}
# Initialize stemmer
stemmer = Stemmer.Stemmer("english")


def build_and_save_bm25_index(
    chunks_path: str = CHUNKS_PATH,
    index_dir: str = INDEX_DIR,
    stopwords: str = "en"
):
    """
    Build a BM25 index from processed chunks and save it to disk.
    Expects a JSON file with a list of {'chunk_id', 'chunk_text', ...}.

    Uses optional stemming and stopword removal.
    """
    with open(chunks_path, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    corpus = [chunk['chunk_text'] for chunk in all_chunks]
    ids = [chunk['chunk_id'] for chunk in all_chunks]

    # Tokenize with stopwords and stemming
    tokenized = bm25s.tokenize(corpus, stopwords=stopwords, stemmer=stemmer)

    global bm25_retriever, chunk_map
    # Use JAX for faster top-k if available
    bm25_retriever = bm25s.BM25(use_jax=True)
    bm25_retriever.index(tokenized)
    # Save index and optionally the corpus
    bm25_retriever.save(index_dir, corpus=corpus)

    # Build map from id to full chunk dict
    chunk_map = {chunk['chunk_id']: chunk for chunk in all_chunks}
    # Also save chunk_map for quick loading
    with open(os.path.join(index_dir, 'chunk_map.json'), 'w', encoding='utf-8') as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)


def load_bm25_index(
    index_dir: str = INDEX_DIR,
    load_corpus: bool = False
):
    """
    Load a saved BM25 index and the chunk map from disk.
    Returns (bm25_retriever, chunk_map).

    If load_corpus=True, reloads the corpus into the retriever for doc-level outputs.
    """
    global bm25_retriever, chunk_map
    if bm25_retriever is None:
        # memory‐efficient load with optional corpus
        bm25_retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=load_corpus)
        with open(os.path.join(index_dir, 'chunk_map.json'), 'r', encoding='utf-8') as f:
            chunk_map = json.load(f)
    return bm25_retriever, chunk_map


def retrieve(
    query: str,
    top_k: int = 5,
    stopwords: str = "en"
):
    """
    Retrieve top_k chunks for the given query.
    Returns a list of dicts: {'id', 'text', 'metadata', 'score'}.

    Applies the same stopword removal and stemming as indexing.
    """
    retriever, cmap = load_bm25_index()
    tokens = bm25s.tokenize([query], stopwords=stopwords, stemmer=stemmer)[0]
    ids_list, scores_list = retriever.retrieve([tokens], k=top_k)
    ids, scores = ids_list[0], scores_list[0]

    results = []
    for doc_id, score in zip(ids, scores):
        chunk = cmap[doc_id]
        results.append({
            'id': doc_id,
            'text': chunk['chunk_text'],
            'metadata': {k: v for k, v in chunk.items() if k not in ['chunk_text']},
            'score': score
        })
    return results
