import streamlit as st
import os
import yaml
from ui_helpers import render_chat_history, show_result_block

from dotenv import load_dotenv
# Import the RAG pipeline function.
from rag_pipeline import rag_pipeline
# Instantiate the uncertainty estimator via the factory function.
from uncertainty_estimator_factory import get_uncertainty_estimator

#############

# Load sensitive settings from .env and configuration from config.yaml
load_dotenv()
with open(r"C:\Users\au644610\OneDrive - Aarhus universitet\Desktop\Thesis-Trustworthy-RAG\src\config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Depending on the model_type specified, instantiate the right provider.
if config["model_type"] == "hf":
    from providers.provider import HuggingFaceProvider
    # Retrieve HF_API_KEY from environment variables
    HF_API_KEY = os.getenv("HF_API_KEY")

    provider = HuggingFaceProvider(
        api_url=config["hf_model"],
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_new_tokens=config["max_new_tokens"]
    )
elif config["model_type"] == "chatui":
    from providers.provider import ChatUIProvider
    # Retrieve CHATUI_API_URL from .env
    CHATUI_API_URL = os.getenv("CHATUI_API_URL")

    provider = ChatUIProvider(api_url=CHATUI_API_URL,
                              model_id=config["chatui_model"],
    temperature=config["temperature"],
    top_p=config["top_p"],
    max_new_tokens=config["max_new_tokens"]
)
else:
    raise ValueError("Invalid model_type specified in configuration.")

uncertainty_config = config.get("uncertainty", {})
uncertainty_method = uncertainty_config.get("method", "lexical_similarity")
params = uncertainty_config.get(uncertainty_method, {})
estimator = get_uncertainty_estimator(uncertainty_method, **params)

########

st.set_page_config(page_title="Trustworthy RAG Chatbot", layout="wide")

st.title("💬 Cognitive Science Chatbot (RAG + UQ)")

# initialize chat state
if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.chat_input("Ask your question here...")

if query:
    with st.spinner("Thinking..."):
        result = rag_pipeline(
            query,
            top_k=5,
            provider=provider,
            n_samples=2,
            estimator=estimator,
            chat_history=st.session_state["history"]
        )

        assistant_answer = result["final_answer"]

        st.session_state["history"].append({
            "user": query,
            "assistant": assistant_answer,
            "uncertainty": result["uncertainty"],
            "docs": result["retrieved_docs"],
        })

# === UI rendering ===

render_chat_history(st.session_state["history"])