from pathlib import Path
import math
import pandas as pd
from tqdm import tqdm
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.schema import Document
# Text cleaning
from embedding_process.preprocessing import clean_text

DATA_DIR      = "processed_syllabi/"
GLOB_PATTERN  = "**/scraped_data/*.json"
CLEANED_PICKLE = Path("cleaned_docs.pkl")

TEST_PILOT_SIZE = 100          # <= size of the first “pilot” list
SPLIT_DIR       = Path("doc_splits")
SPLIT_DIR.mkdir(exist_ok=True, parents=True)


def load_and_clean_documents() -> list[Document]:
    """Load JSON files, clean the text, and return a list[Document]."""
    if CLEANED_PICKLE.exists():
        print(f"  Loading cleaned docs from cache → {CLEANED_PICKLE}")
        return pd.read_pickle(CLEANED_PICKLE)

    print("  Loading raw JSON files…")
    loader = DirectoryLoader(
        DATA_DIR,
        glob=GLOB_PATTERN,
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",
            "content_key": "text",
            "is_content_key_jq_parsable": False,
            "json_lines": False,
            "metadata_func": lambda obj, meta: {
                **meta,
                **{k: v for k, v in obj.items() if k != "text"},
            },
        },
    )
    raw_docs = loader.load()
    print(f"→ {len(raw_docs):,} raw documents")

    cleaned = []
    for doc in tqdm(raw_docs, desc="cleaning"):
        cleaned.append(
            Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
        )

    pd.to_pickle(cleaned, CLEANED_PICKLE)
    print(f"  Cached cleaned docs → {CLEANED_PICKLE}")
    return cleaned


def make_splits(docs: list[Document], pilot_size: int = TEST_PILOT_SIZE):
    """Yield (index, split_docs) for every chunk after the pilot set."""
    pilot = docs[:pilot_size]
    remainder = docs[pilot_size:]

    n_chunks = math.ceil(len(remainder) / pilot_size)
    for i in range(n_chunks):
        start = i * pilot_size
        end = start + pilot_size
        yield i, remainder[start:end], pilot_size

    return pilot


if __name__ == "__main__":
    all_docs = load_and_clean_documents()

    # take the first N docs as your pilot test set
    cleaned_documents_test = all_docs[:TEST_PILOT_SIZE]
    pilot_path = SPLIT_DIR / f"pilot_{TEST_PILOT_SIZE}.pkl"
    pd.to_pickle(cleaned_documents_test, pilot_path)
    print(f"  Saved pilot split → {pilot_path}")

    # save equal-sized splits of the remainder 
    for idx, split, size in make_splits(all_docs, TEST_PILOT_SIZE):
        if not split:
            continue  # guard against an empty tail split
        out_path = SPLIT_DIR / f"split_{idx:02d}_{len(split)}.pkl"
        pd.to_pickle(split, out_path)
        print(f"  Saved split #{idx} ({len(split)} docs) → {out_path}")

    print("\nAll done!  You now have:")
    print(f"• 1 pilot file ({TEST_PILOT_SIZE} docs)")
    print(
        f"• {len(list(SPLIT_DIR.glob('split_*.pkl')))} additional split files "
        f"in {SPLIT_DIR.resolve()}"
    )
