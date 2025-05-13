# quiet down windows message
import sys
if sys.platform.startswith("win"):
    import types
    # pretend we have a resource module so downstream imports just get a no-op
    sys.modules['resource'] = types.ModuleType('resource')
    
import os, glob, json
import bm25s, Stemmer


def collect_chunks():
    processed_dir = os.path.join('data/processed_syllabi')
    
    chunks = []
    for path in glob.glob(os.path.join(processed_dir, '*', 'processed_chunks.json')):
        with open(path, encoding='utf-8') as f:
            chunks.extend(json.load(f))
    return chunks

def build_index():
    index_dir = os.path.join("data/bm25_index")
    stemmer = Stemmer.Stemmer("english")

    chunks   = collect_chunks()
    if not chunks:
        raise RuntimeError("No chunks found!")
    corpus    = [c['chunk_text'] for c in chunks]
    tokens    = bm25s.tokenize(corpus, stemmer=stemmer, stopwords="en")

    bm25 = bm25s.BM25(corpus=corpus)
    bm25.index(tokens)

    os.makedirs(index_dir, exist_ok=True)
    bm25.save(index_dir, corpus=corpus)
    with open(os.path.join(index_dir, 'chunk_ids.json'), 'w', encoding='utf-8') as f:
        json.dump([c['chunk_id'] for c in chunks], f)

    print("BM25 index built at", index_dir)

if __name__ == "__main__":
    # run to update index
    build_index()
