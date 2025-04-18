#silence the watcher’s log noise
import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from app.ui_helpers import render_chat_history
from core import run_rag, get_config
from csv_logger import initialize_csv, log_experiment
import json, os


# streamlit interface
st.set_page_config(page_title="Trustworthy RAG Chatbot", layout="wide")
st.title("💬 Cognitive Science Chatbot")

# Inject custom bubble outline styles\ nst.markdown(
st.markdown(
    """
    <style>
    .chat-bubble { border: 2px solid #ccc; border-radius: 8px; padding: 8px; margin-bottom: 12px; }
    .conf-high { border-color: #ccc !important; }
    .conf-med { border-color: #FFEB3B !important; }  /* light yellow */
    .conf-low { border-color: #FFCDD2 !important; }  /* light red */
    </style>
    """,
    unsafe_allow_html=True
)


# === controls ===
cfg = get_config()
model_cfg = cfg['model']
gen_cfg = cfg['generation']

# Sidebar controls
st.sidebar.header("Cognitive Science Chatbot")
with st.sidebar:
    st.markdown(
        """
        This assistant uses retrieval‑augmented generation over your course syllabi and quantifies its confidence in each answer.  
        Feel free to ask any question about Cognitive Science at AU and see how certain the model is in its response.
        To use this chatbot, input your provider details in the settings below!
        """
    )

st.sidebar.header("🔧 Settings")
demo_mode = st.sidebar.checkbox("Fast demo mode (no real LLM calls)", value=False)

# Reset chat history
if st.sidebar.button("🔄 Reset chat"):
    st.session_state["history"] = []

# Provider Settings (expanded)
with st.sidebar.expander("🔧 Provider Settings", expanded=True):
    option = st.selectbox("Choose LLM Provider:", ["ChatUI", "Huggingface"], index=0)
    if option == 'ChatUI':
        api_key = st.text_input("ChatUI API URL", value=os.getenv("CHATUI_API_URL", ""), type="password")
        model_id = st.selectbox("Model ID", [model_cfg['chatui_model']], index=0)
        model_type = 'chatui'
    else:
        api_key = st.text_input("HF API Key", value=os.getenv("HF_API_KEY", ""), type="password", placeholder="hf_xxx…")
        model_id = st.text_input("Model URL", value=model_cfg['hf_model'], placeholder="https://api-inference.huggingface.co/models/…")
        model_type = 'hf'

# Advanced Settings (collapsed by default)
with st.sidebar.expander("📖 Advanced Settings", expanded=False):
    # Map display names to internal config keys
    METHOD_MAP = {
        'Lexical Similarity': 'lexical_similarity',
        'Degree Matrix NLI': 'deg_mat',
        'Eccentricity NLI': 'eccentricity'
    }
    # Determine default index from config setting
    default_key = cfg['uncertainty']['method']
    default_display = next(k for k,v in METHOD_MAP.items() if v == default_key)
    default_idx = list(METHOD_MAP.keys()).index(default_display)
    # Provide help tooltip with method descriptions and configured parameters
    method_help = (
        "Lexical Similarity (metric=rougeL): measures lexical overlap among samples.  \n"

        "Degree Matrix NLI (affinity=entail, batch_size=10): uses NLI entailment distances.  \n"

        "Eccentricity NLI (similarity_score=NLI_score, thres=0.9): uses NLI-based embedding distances."
    )
    display = st.selectbox(
        "Uncertainty Method", list(METHOD_MAP.keys()),
        index=default_idx,
        help=method_help
    )
    uq_method = METHOD_MAP[display]
    # Standard generation controls
    top_k = st.slider("Documents to retrieve", 1, 20, gen_cfg['top_k'])
    n_samples = st.slider("Samples to generate", 1, 10, gen_cfg['n_samples'])
    temperature = st.slider("Temperature", 0.0, 1.0, gen_cfg['temperature'], 0.01)
    top_p = st.slider("Top-p", 0.0, 1.0, gen_cfg['top_p'], 0.01)


# Initialize chat history
st.session_state.setdefault("history", [])

# Initialize CSV logging once
CSV_FILE = "experiment_results.csv"
initialize_csv(CSV_FILE)

# input question to start rag process
query = st.chat_input("Ask your question here...")

if query:
    with st.spinner("Thinking..."):
        if demo_mode:
            result = {
                "final_answer": "This is a demo answer.",
                "samples": [],
                "retrieved_docs": [],
                "uncertainty": 0.0,
                "top_k": top_k,
                "n_samples": 0
            }
        else:
            # Override config for temperature/top_p
            run_cfg = cfg.copy()
            run_cfg['generation']['top_k'] = top_k
            run_cfg['generation']['n_samples'] = n_samples
            run_cfg['generation']['temperature'] = temperature
            run_cfg['generation']['top_p'] = top_p

            result = run_rag(
                query=query,
                cfg=run_cfg,
                chat_history=st.session_state["history"],
                top_k_override=top_k,
                n_samples_override=n_samples,
                uq_method_override=uq_method,
                model_type_override=model_type,
                model_id_override=model_id,
                api_key_override=api_key,
            )

        st.session_state["history"].append({
            "user": query,
            "assistant": result['final_answer'],
            "uncertainty": result["uncertainty"],
            "docs": result["retrieved_docs"],
        })
        
        # Log data
        experiment_data = {
                "query": query,
                "answer": result['final_answer'],
                "samples": json.dumps(result["samples"], ensure_ascii=False),
                "model": model_type,
                "settings": json.dumps({
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": gen_cfg["max_new_tokens"],
                    "top_k": top_k,
                    "n_samples": result["n_samples"],
                }),
                "uncertainty_method": cfg["uncertainty"]["method"],
                "uncertainty_score": result["uncertainty"],
                "retrieved_documents": result["retrieved_docs"],
            }
        log_experiment(CSV_FILE, experiment_data)

# render chat 
render_chat_history(st.session_state["history"])