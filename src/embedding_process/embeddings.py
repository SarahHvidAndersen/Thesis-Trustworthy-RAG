import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="intfloat/multilingual-e5-large-instruct", device="cpu"):
    """
    Loads the embedding model.
    
    device: e.g. "cpu" or "cuda" if available.
    """
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_text(texts, model):
    """
    Given a list of texts, returns normalized embeddings using the SentenceTransformer model.
    The model internally handles tokenization, truncation (max_length=512) and normalization.
    """
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings.cpu().numpy() if hasattr(embeddings, "cpu") else embeddings


if __name__ == "__main__":
    # Example texts (queries and documents)
    texts = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: 南瓜的家常做法",
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "1. 清炒南瓜丝 原料: 嫩南瓜半个 调料: 葱、盐、白糖、鸡精 做法: 南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤."
    ]
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using SentenceTransformer wrapper.
    model = load_embedding_model(device=device)
    embeddings = embed_text(texts, model)
    print("embeddings shape:", embeddings.shape)
