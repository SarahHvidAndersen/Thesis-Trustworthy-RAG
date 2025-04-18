import os, yaml
from functools import lru_cache
from dotenv import load_dotenv
from uncertainty_estimator_factory import get_uncertainty_estimator
from retriever import load_embedding_model, retrieve_documents
from embedding_process.vector_db import init_db, get_collection
from uncertainty_estimator_factory import compute_uncertainty

load_dotenv()

@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load and cache the YAML config file."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def init_provider(model_type: str, model_id: str, api_key: str, cfg: dict):
    if model_type == "hf":
        from providers.provider import HuggingFaceProvider # fix cache
        # add cache

        return HuggingFaceProvider(
            api_url=model_id,
            headers={"Authorization": f"Bearer {api_key}"},
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"],
        )
    elif model_type == "chatui":
        from providers.provider import ChatUIProvider
        #CHATUI_API_URL = os.getenv("CHATUI_API_URL")
        return ChatUIProvider(
            api_url=api_key,
            model_id=model_id,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"],
        )
    else:
        raise ValueError(f"Invalid model_type {cfg['model_type']}")

def init_estimator(cfg: dict, override_method: str = None):
    method = override_method or cfg["uncertainty"]["method"]
    params = cfg["uncertainty"].get(method, {})
    return get_uncertainty_estimator(method, **params)


def rag_pipeline(
    query: str,
    top_k: int,
    provider,
    device: str = "cpu",
    n_samples: int = 1,
    estimator=None,
    chat_history=None
) -> dict:
    """
    Core RAG pipeline: embed query, retrieve docs, generate answers, and
    optionally compute uncertainty.

    Returns dict with:
    "final_answer": The chosen answer (e.g., first sample).
    "samples": The list of generated samples.
    "retrieved_docs": Retrieved document metadata.
    "uncertainty": A float uncertainty score (or None).
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

    # Retrieval
    model = load_embedding_model(device=device)
    db_client = init_db(db_path="chroma_db")
    collection = get_collection(db_client, collection_name="rag_documents")
    if collection.count() == 0:
        return None

    retrieved_docs = retrieve_documents(embed_query_text, model, collection, top_k=top_k)
    context = "\n".join(doc["text"] for doc in retrieved_docs)

    # Generation
    samples = []
    if n_samples > 1 and estimator is not None:
        for i in range(n_samples):
            sample = provider.generate(query, context, chat_history)
            samples.append(sample)
        uncertainty = compute_uncertainty(estimator, samples)
        final_answer = samples[0]
    else:
        # Single-sample path
        sample = provider.generate(query, context, chat_history)
        samples.append(sample)
        final_answer = sample
        uncertainty = None
    
    return {
        "final_answer": final_answer,
        "samples": samples,
        "retrieved_docs": retrieved_docs,
        "uncertainty": uncertainty,
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
    model_type_override=None,
    model_id_override=None,
    api_key_override=None,
) -> dict:
    """
    Master entry point: initializes provider & estimator from cfg (with overrides),
    then runs the retrieval-augmented generation pipeline.
    """
    model_cfg = cfg['model']
    gen_cfg = cfg['generation']

    top_k = top_k_override or gen_cfg['top_k']
    n_samples = n_samples_override or gen_cfg['n_samples']
    model_type = model_type_override or cfg_model['type']
    model_id = model_id_override or model_cfg[f"{model_type}_model"]
    api_key = api_key_override or (
        os.getenv('HF_API_KEY') if model_type=='hf' else os.getenv('CHATUI_API_URL')
    )

    provider = init_provider(model_type, model_id, api_key, cfg)
    estimator = init_estimator(cfg, override_method=uq_method_override)

    return rag_pipeline(
        query=query,
        top_k=top_k,
        provider=provider,
        device=cfg.get("device", "cpu"),
        n_samples=n_samples,
        estimator=estimator,
        chat_history=chat_history,
    )
