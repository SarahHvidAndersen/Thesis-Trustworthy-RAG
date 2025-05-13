
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

# Ragas imports
from ragas.cache import DiskCacheBackend
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer, # seems to be broken sometimes, can't identify clusters correctly
)

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.schema import Document

# Text cleaning
from internal.database_setup.preprocessing import clean_text

# if we want to see the cache in action, set the logging level to debug
import logging
from ragas.utils import set_logging_level
set_logging_level("ragas.cache", logging.DEBUG)

# Ensure the OpenAI key is set
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configuration
DATA_DIR = "data/processed_syllabi/"
GLOB = "**/scraped_data/*.json"
CLEANED_DOCS_PATH = Path("data/doc_splits/split_00_100.pkl")
KG_PATH = Path("output/knowledge_graphs/knowledge_graph_split_00_100.json") 

#Saved pilot split → doc_splits\pilot_100.pkl
 # Saved split #0 (100 docs) → doc_splits\split_00_100.pkl
  ##Saved split #1 (100 docs) → doc_splits\split_01_100.pkl
 # Saved split #2 (100 docs) → doc_splits\split_02_100.pkl
 # Saved split #3 (22 docs) → doc_splits\split_03_22.pkl

JSON_OUT = Path("output/raw_test_data/json_version/testset_split_00_100.json") 
CSV_OUT = Path("output/raw_test_data/testset_split_00_100.csv")
TESTSET_SIZE = 50
CACHE_DIR = ".cache/ragas" # or data/.cache/ragas

# load stored Documents
def load_and_clean_documents(
    data_dir: str = DATA_DIR,
    glob_pattern: str = GLOB
) -> list[Document]:
    """
    Reads JSON syllabus files, extracts text, and returns cleaned Document objects.
    """
    # If we have a cached cleaned docs file, load it
    if CLEANED_DOCS_PATH.exists():
        print(f"Loading cleaned documents from cache: {CLEANED_DOCS_PATH}")
        return pd.read_pickle(CLEANED_DOCS_PATH)

    loader = DirectoryLoader(
        data_dir,
        glob=glob_pattern,
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",
            "content_key": "text",
            "is_content_key_jq_parsable": False,
            "json_lines": False,
            "metadata_func": lambda obj, meta: {**meta, **{k: v for k, v in obj.items() if k != "text"}}
        }
    )

    docs = loader.load()
    print(len(docs))

    cleaned_documents = []
    for doc in docs:                    # `documents` is a list of langchain.schema.Document
        raw = doc.page_content               
        cleaned = clean_text(raw)            
        # rewrap into a Document, preserving metadata:
        cleaned_doc = Document(
            page_content=cleaned,
            metadata=doc.metadata
        )
        cleaned_documents.append(cleaned_doc)

    pd.to_pickle(cleaned_documents, CLEANED_DOCS_PATH)
    print(f"Cached cleaned docs to {CLEANED_DOCS_PATH}")

    return cleaned_documents


def build_or_load_kg(docs, generator_llm, generator_embeddings, cacher):
    if KG_PATH.exists():
        print(f"Loading existing KG from {KG_PATH}")
        kg = KnowledgeGraph.load(str(KG_PATH))
    else:
        print("Creating new KG and applying transforms...")
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            ))

        transforms = default_transforms(documents=docs, 
                                    llm=generator_llm, 
                                    embedding_model=generator_embeddings)

        apply_transforms(kg, transforms)
        kg.save(str(KG_PATH))
    return kg


def generate_test_data(kg, generator_llm, generator_embeddings, test_size=TESTSET_SIZE):

    # instantiate testsetgenerator with the finished kg
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg
    )

    #query_distribution = default_query_distribution(generator_llm)
    #print(query_distribution)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.6) # 0.5 or 0.6
        ,
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm),0.4) # 0.25 or 0.4
        #,
        #(MultiHopAbstractQuerySynthesizer(llm=generator_llm),0.25) # 0.25 or none
    ]
    print(query_distribution)

    #for synth in [SingleHopSpecificQuerySynthesizer(generator_llm),
    #          MultiHopSpecificQuerySynthesizer(generator_llm),
    #          MultiHopAbstractQuerySynthesizer(generator_llm)]:
    #    try:
    #        count = len(synth.get_node_clusters(kg))
    #        print(f"✅ {synth.name} → {count} cluster(s)")
    #    except Exception as e:
    #        print(f"❌ {synth.name} → error: {e}")


    # we are not using the generate_with_langchain_docs function from documentation
    # because it will create the kg all over. we create it separately, so that we can save it
    
    print("Generating testset from existing KG...")

    dataset = generator.generate(
        testset_size=test_size, 
        query_distribution=query_distribution,
        raise_exceptions=False,      #  don’t crash when one synth fails 
        with_debugging_logs=True)

    # Persist any new KG nodes 
    # (should be none if KG was complete, but in case of re-runs, save the updated version)
    generator.knowledge_graph.save(str(KG_PATH))

    # save the samples in json and csv format
    df = dataset.to_pandas()
    df.to_json(JSON_OUT, orient="records", indent=2)
    df.to_csv(CSV_OUT, index=False)
    return df


def main():
    # Initialize persistent disk cache
    cacher = DiskCacheBackend(cache_dir=CACHE_DIR) # ".cache/ragas"
    print("Cache entries:", len(cacher.cache))

    # Prepare LLM + embedding wrappers with the shared cache
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", 
                                         model_kwargs={"response_format": {"type": "json_object"}}),
                                           cache=cacher)
    
    embedder = LangchainEmbeddingsWrapper(OpenAIEmbeddings(), cache=cacher)

    # Load & clean docs
    docs = load_and_clean_documents()

    # Build or load KG
    kg = build_or_load_kg(docs, llm, embedder, cacher)

    # Generate testset
    df = generate_test_data(kg, llm, embedder)

    print(f"Pipeline complete. KG saved to {KG_PATH}; testset saved to {JSON_OUT} and {CSV_OUT}.")
    return

if __name__ == "__main__":
    main()