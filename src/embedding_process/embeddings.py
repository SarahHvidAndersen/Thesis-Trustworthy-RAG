import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="intfloat/multilingual-e5-large-instruct", device="cpu"):
    """
    Loads the embedding model.
    
    device: "cpu" or "cuda" if available.
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


# testing
if __name__ == "__main__":
    # Example texts (queries and documents)
    texts = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat",
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
    ]
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using SentenceTransformer wrapper.
    model = load_embedding_model(device=device)
    embeddings = embed_text(texts, model)
    print("embeddings shape:", embeddings.shape)
