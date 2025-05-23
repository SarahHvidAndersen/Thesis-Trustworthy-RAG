import os, yaml, sys
import re
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from joblib import load
from pathlib import Path

from internal.uncertainty_estimation.uncertainty_estimator_factory import get_uncertainty_estimator, compute_uncertainty
from internal.retrievers.semantic_retriever import load_embedding_model, retrieve_documents
from internal.database_setup.chroma_db import init_db, get_collection
from internal.retrievers.bm25_retriever import bm25_retrieve
from internal.providers.provider import GeneratorProvider
from internal.providers.provider_utils import ensure_provider_input

load_dotenv(override=True)


@lru_cache(maxsize=1)
def get_reranker():
    # Loads the cross-encoder for MS MARCO L-6
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache the YAML config file.
    """
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_model_id(model_cfg: dict, provider: str, override: str | None) -> str:
    """
    Derive the model-ID:
       1) explicit override parameter
       2)   model.chatui_model / model.ollama_model / model.hf_model, if present
       3)   providers.<Provider>.default  → match name in providers.<Provider>.options[*].id
    """
    if override:
        return override

    explicit_key = f"{provider.lower()}_model"      # e.g. chatui_model
    if explicit_key in model_cfg:
        return model_cfg[explicit_key]

    block = model_cfg["providers"][provider]        # e.g. providers.ChatUI
    default_name = block["default"]
    for opt in block["options"]:
        if opt["name"] == default_name:
            return opt["id"]
    raise KeyError(f"No ID found for provider {provider}")


def init_provider(model_type: str, model_id: str, api_key: str, cfg: dict):
    gen_cfg = cfg['generation']

    if model_type in ('Huggingface','hf'):
        from internal.providers.provider import HuggingFaceProvider 

        return HuggingFaceProvider(
            model=model_id,
            api_url=api_key,
            #headers={"Authorization": f"Bearer {api_key}"},
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_new_tokens=gen_cfg["max_new_tokens"],
        )
    elif model_type in ('ChatUI','Ollama'):
        from internal.providers.provider import OllamaProvider
        #api_key = os.getenv("CHATUI_API_URL")
        return OllamaProvider(
            api_url=api_key,
            model_id=model_id,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_new_tokens=gen_cfg["max_new_tokens"],
        )
    else:
        raise ValueError(f"Invalid model_type {model_type}")

def init_estimator(cfg: dict, override_method: str = None):
    method = override_method or cfg["uncertainty"]["method"]
    params = cfg["uncertainty"].get(method, {})
    return get_uncertainty_estimator(method, **params)


def init_scaler(cfg):
    method = cfg["uncertainty"]["method"]           # e.g. "lexical_similarity"
    scale_cfg = cfg["uncertainty"]["scaling"]
    scaler_path = scale_cfg["paths"][method]         # picks the right file

    print(f'scaler path found {scaler_path}')
    return load(scaler_path)

def rag_pipeline(
    query: str,
    top_k: int,
    provider,
    device: str = "cpu",
    n_samples: int = 1,
    estimator=None,
    scaler = None,
    chat_history=None,
    semantic_weight: float = None
) -> dict:
    """
    Core RAG pipeline: embed query, retrieve docs, generate answers, and
     compute uncertainty. Combines semantic (vector) and lexical (BM25) retrieval
    using a specified semantic_weight (0.0 to 1.0).

    Returns dict with:
    "final_answer": The chosen answer (e.g., first sample).
    "samples": The list of generated samples.
    "retrieved_docs": Retrieved document metadata.
    "raw_uncertainty": A float raw uncertainty score (or None).
    "calibrated_confidence": A float scaled confidence score (or None).
    "top_k": The top_k value used.
    "n_samples": The number of generated samples.
    """
    # Prepare query embedding text with optional history
    if chat_history:
        hist_txt = " ".join(
            f"User: {turn['user']} Assistant: {turn['assistant']}"
            for turn in chat_history[-6:]
        )
        embed_query_text = f"{hist_txt} {query}"
    else:
        embed_query_text = query
    
    #---- Hybrid Retrieval
    # Determine weights
    cfg = get_config()
    retr_cfg = cfg.get('retrieval', {})
    total_top_k = retr_cfg.get('top_k', 100)

    weight = retr_cfg.get('semantic_weight', 0.5)
    weight = max(0.0, min(1.0, weight))
    sem_k = max(1, int(round(weight * total_top_k)))
    lex_k = max(1, total_top_k - sem_k)

    # semantic retrieval
    model = load_embedding_model(device=device)
    db_client = init_db(db_path="data/chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")
    if collection.count() == 0:
        print('collection count is 0! empty chromadb database')
        return None
    semantic_docs = retrieve_documents(embed_query_text, model, collection, top_k=sem_k)
    for doc in semantic_docs:
        doc['source'] = 'semantic'

    # Lexical retrieval
    raw_bm25 = bm25_retrieve(query, top_k=lex_k)
    bm_docs = []
    for doc in raw_bm25:
        doc['bm25_score'] = doc.pop('score') # rename for clarity
        doc['source'] = 'bm25'
        bm_docs.append(doc)

    # Combine, then remove duplicates *by id* (preserving first occurrence order, sem prevalence)
    seen = set()
    combined = semantic_docs + bm_docs
    retrieved_docs = []
    for doc in combined:
        if doc["id"] not in seen:
            retrieved_docs.append(doc)
            seen.add(doc["id"])

    # reranking
    reranker = get_reranker()
    pairs = [(query, doc['text']) for doc in retrieved_docs]

    # Compute relevance scores for each pair, attach scores and sort docs
    scores = reranker.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc['rerank_score'] = score
    retrieved_docs.sort(key=lambda d: d['rerank_score'], reverse=True)

    # threshold score, from config
    threshold = retr_cfg.get('threshold', 0.3)
    min_docs = retr_cfg.get('min_docs', 0)
    max_docs = retr_cfg.get('max_docs', 10)

    # Filter candidates above threshold and cap at max to not overwhelm llm
    filtered = [d for d in retrieved_docs if d['rerank_score'] > threshold]
    if len(filtered) < min_docs:
        filtered = retrieved_docs[:min_docs]
    retrieved_docs = filtered[:max_docs]

    # Generation
    samples = []
    if n_samples > 1 and estimator is not None:
        for i in range(n_samples):
            sample = provider.generate(query, retrieved_docs, chat_history)
            samples.append(sample)

        # compute raw value
        raw_uncertainty = compute_uncertainty(estimator, samples)

        # apply scaler
        if scaler is None:
            calibrated_confidence = 0
        else:
            calibrated = scaler.transform([[raw_uncertainty]])[0,0]
            calibrated_confidence = 1 - calibrated

        # build & send the “best‐answer” prompt via the provider helper
        selection_prompt = GeneratorProvider.build_selection_prompt(
            original_query=query,
            candidates=samples,
            retrieved_docs=retrieved_docs,
            history=chat_history
        )
        #print(f"SELECTION MODEL, SELECTION_PROMPT IS {selection_prompt}")

        selection = provider.generate_raw(selection_prompt)
        print(F"SELECTION MODEL OUTPUT IS: " + selection)
    
        # parse the reply with regex
        try:
            #choice = int(selection.strip())
            m = re.search(r"\b([0-9]+)\b", selection)
            choice = int(m.group(1)) if m else 0
        except ValueError:
            choice = 0

        #print(f"SELECTION MODEL CHOICE VALUE IS: {choice}")
        # pick the chosen sample, or abstain
        if 1 <= choice <= len(samples):
            final_answer = samples[choice - 1]
        else:
            # simply select the first sample the model generated as a fall-back method
            final_answer = samples[0]

            # reflexive check is more stable for larger models e.g:
            #final_answer = "I’m not sure about the correct response."
             
    else:
        # Single-sample path
        sample = provider.generate(query, retrieved_docs, chat_history)
        samples.append(sample)
        final_answer = sample
        calibrated_confidence = None
        raw_uncertainty = None
    
    return {
        "final_answer": final_answer,
        "samples": samples,
        "retrieved_docs": retrieved_docs,
        "raw_uncertainty": raw_uncertainty,
        "calibrated_confidence": calibrated_confidence,
        "top_k": top_k,
        "n_samples": n_samples,
    }


def run_rag(
    query: str,
    cfg: dict,
    chat_history: list = None,
    top_k_override: int = None,
    n_samples_override: int = None,
    uq_method_override: str = None,
    provider_name_override=None,
    model_id_override=None,
    api_key_override=None
) -> dict:
    """
    Master entry point: initializes provider & estimator from cfg (with overrides),
    then runs the retrieval-augmented generation pipeline.
    """

    model_cfg = cfg["model"]
    gen_cfg   = cfg["generation"]
    retr_cfg  = cfg.get("retrieval", {})

    top_k = top_k_override or retr_cfg['top_k']
    n_samples = n_samples_override or gen_cfg['n_samples']

    provider_name = provider_name_override or model_cfg["type"]          # "ChatUI" / "Ollama" / "Huggingface"
    model_id = _resolve_model_id(model_cfg, provider_name, model_id_override)

    # -------------------------------- credentials --------------------------------
    if provider_name == "ChatUI":
        api_key = api_key_override or ensure_provider_input("ChatUI")
    elif provider == "Ollama":
        api_key = api_key_override or ensure_provider_input("Ollama")
    elif provider in ("Huggingface", "HF"):
        api_key = api_key_override or ensure_provider_input("Huggingface")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    

    provider = init_provider(provider_name, model_id, api_key, cfg)
    estimator = init_estimator(cfg, override_method=uq_method_override)
    scaler = init_scaler(cfg)

    return rag_pipeline(
        query=query,
        top_k=top_k,
        provider=provider,
        device=cfg.get("device", "cpu"),
        n_samples=n_samples,
        estimator=estimator,
        scaler = scaler,
        chat_history=chat_history
    )


if __name__ == "__main__":
    # Test hybrid retrieval (dummy LLM)
    cfg = get_config()
    class DummyProvider:
        def generate(self, query, context, history=None):
            return f"Dummy answer for '{query}'"
    # Example run
    result = rag_pipeline(
        query="what is the purpose of the philosophy of cognitive science course?",
        top_k=100,
        provider=DummyProvider(),
        device="cpu",
        n_samples=1,
        estimator=None,
        chat_history=[],
        semantic_weight=0.5
    )
    print("Retrieved docs (hybrid+reranker):")
    for i, doc in enumerate(result["retrieved_docs"], 1):
        snippet = doc["text"][:80].replace("\n", " ") + ("…" if len(doc["text"]) > 80 else "")
        score = doc.get("rerank_score", doc.get("distance", 0.0))
        print(f"{i:2d}. ID={doc['id']}, rerank={score:.4f}, src={doc.get('source','N/A')}")
        print(f"    {snippet}")
