# quiet down windows message
import sys
if sys.platform.startswith("win"):
    import types
    # pretend we have a resource module so downstream imports just get a no-op
    sys.modules['resource'] = types.ModuleType('resource')

import os, json, glob
import bm25s
import Stemmer



def collect_chunks():
    #base_dir = os.path.dirname(__file__)                  
    #project_root  = os.path.dirname(base_dir)                  
    processed_dir = os.path.join('processed_syllabi')

    chunks = []
    for path in glob.glob(os.path.join(processed_dir, '*', 'processed_chunks.json')):
        with open(path, encoding='utf-8') as f:
            chunks.extend(json.load(f))
    return chunks

def load_index(mmap=True):
    index_dir = os.path.join("bm25_index")

    # load BM25  raw corpus
    retriever = bm25s.BM25.load(index_dir , mmap=mmap, load_corpus=True)

    # reload the chunk-id ordering
    with open(os.path.join(index_dir , 'chunk_ids.json'), encoding='utf-8') as f:
        chunk_ids = json.load(f)

    # rebuild id→full-chunk map
    chunks    = collect_chunks()
    chunk_map = {c['chunk_id']: c for c in chunks}
    return retriever, chunk_ids, chunk_map


def bm25_retrieve(query, top_k=5):
    stemmer= Stemmer.Stemmer("english")

    retriever, chunk_ids, chunk_map = load_index(mmap=True)

    # make sure the reloaded corpus is non-empty
    if not retriever.corpus:
        raise RuntimeError("Reloaded BM25.corpus is empty!")

    # tokenize into a list of list-of-strings
    tokenized = bm25s.tokenize(
        [query],               # note the list here → [[…]]
        lower=True,
        stopwords="english",
        stemmer=stemmer,
        return_ids=False,      # want str tokens, not numeric IDs
        show_progress=False,
        leave=False,
        allow_empty=True
    )

    # retrieve returns (docs_array, scores_array)
    results, scores = retriever.retrieve(tokenized, k=top_k)
    
    # flatten the first (and only) row into a Python list
    results_out = []
    for doc_dict, score in zip(results[0], scores[0]):
        raw_id = doc_dict["id"]
        cid = chunk_ids[raw_id] if isinstance(raw_id, int) else raw_id
        chunk = chunk_map[cid]
        # chunk has keys: chunk_text, chunk_id, course, title, author…etc
        metadata = { k: v for k, v in chunk.items() if k not in ("chunk_text","embedding") }
        results_out.append({
            "id":       cid,
            "score":    float(score),
            "text":     chunk["chunk_text"],
            "metadata": metadata
        })

    return results_out


def main():
    index_dir = os.path.join("bm25_index")

    stemmer= Stemmer.Stemmer("english")

    retriever, chunk_ids, chunk_map = load_index(mmap=True)

    # Tokenize the query into a list-of-list-of-str [["what","is","the","prefrontal","cortex"]]
    query_tokens = bm25s.tokenize(
        "What is the prefrontal cortex?",
        lower=True,
        stopwords="english",
        stemmer=stemmer,
        return_ids=False,      # ← get strings, not IDs
        show_progress=False,
        leave=False,
        allow_empty=True
    )

    results, scores = retriever.retrieve(query_tokens, k=2)

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i} (score: {score:.2f}): {doc}")


if __name__ == "__main__":
    #main()
    results_out = bm25_retrieve("What is the prefrontal cortex?", top_k=2)
    print(results_out)
