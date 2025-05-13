
#silence the watcher’s log noise
import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from ui_helpers import render_chat_history
from internal.core import run_rag, get_config
from internal.logging_utils.csv_logger import initialize_csv, log_experiment
import json, os
import copy
from internal.retrievers.semantic_retriever import load_embedding_model as _load

#streamlit cache add-on
@st.cache_resource
def load_embedding_model(model_name: str, device: str):
    # add Streamlit cache
    return _load(model_name, device)

# streamlit interface
st.set_page_config(page_title="Trustworthy RAG Chatbot", layout="wide")
st.title("💬 Cognitive Science Chatbot")
st.text("I am a Cognitive Science assistant with information about the entire Cognitive Science syllabus at Aarhus University.  " \
"  \n Ask a question from your program to get started! 🧠")


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
retr_cfg = cfg.get('retrieval', {})
# Reset chat history
if st.sidebar.button("🔄 Reset chat "):
    st.session_state["history"] = []


# Sidebar controls
st.sidebar.header("Cognitive Science Chatbot")
with st.sidebar: # this text is currently too long, maybe add a pop-up on start-up?
    st.markdown(
        """
        This assistant uses retrieval‑augmented generation over your course syllabi and quantifies its confidence in each answer.  
        Feel free to ask any question about Cognitive Science at AU and see how confident the model is in its response.  \n
        ⍟ 0-30% = red bubble  \n
        ⍟ 30-70% = yellow bubble  \n
        ⍟ >70% = black bubble  \n
        If an answer has lower confidence, we recommmend checking directly with the sources used for the response. 
        Remember to always verify important information!   \n\n
        To use this chatbot, input your provider details in the settings below.
        """
    )

st.sidebar.header("🔧 Settings")
demo_mode = st.sidebar.checkbox("Fast demo mode (no real LLM calls)", value=False)


# Provider Settings (expanded)
prov_expander = st.sidebar.expander("🔧 Provider Settings", expanded=True)
# Choose provider
provider = prov_expander.selectbox(
    "LLM Provider",
    ["chatui", "hf"],
    index=["chatui", "hf"].index(model_cfg.get('type', 'chatui'))
)

# set API Key / URL
if provider == "chatui":
    api_key = prov_expander.text_input(
        "ChatUI API URL",
        value=os.getenv("CHATUI_API_URL", model_cfg.get('chatui_api_url', '')),
        type="password"
    )
else:
    api_key = prov_expander.text_input(
        "Huggingface API Key",
        value=os.getenv("HF_API_KEY", model_cfg.get('hf_api_key', '')),
        type="password", placeholder="hf_xxx..."
    )

# Model selection
prov = model_cfg.get('providers', {}).get(provider, {})
options = prov.get('options', [])  # list of dicts with 'name' and 'id'
names = [opt['name'] for opt in options]
default = prov.get('default', names[0] if names else '')
default_idx = names.index(default) if default in names else 0
selected_name = prov_expander.selectbox("Model", names, index=default_idx)
model_id = options[names.index(selected_name)]['id'] if options else ''
model_type = provider


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

        "Eccentricity NLI (similarity_score=NLI_score, thres=0.5): uses NLI-based embedding distances."
    )
    display = st.selectbox(
        "Uncertainty Method", list(METHOD_MAP.keys()),
        index=default_idx,
        help=method_help
    )
    uq_method = METHOD_MAP[display]
    # Standard generation controls
    #top_k = st.slider("Documents to retrieve", 10, 100, retr_cfg['top_k'], help="Documents to retrieve with hybrid search")
    n_samples = st.slider("Samples to generate", 1, 10, gen_cfg['n_samples'])
    temperature = st.slider("Temperature", 0.0, 1.0, gen_cfg['temperature'], 0.01)
    top_p = st.slider("Top-p", 0.0, 1.0, gen_cfg['top_p'], 0.01)


# Initialize chat history
st.session_state.setdefault("history", [])

# Initialize CSV logging once
CSV_FILE = "output/streamlit_run/experiment_results.csv" # never overwrites, just appends new experiment data
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
                "raw_uncertainty": 0.5,
                "calibrated_confidence": 0.5,
                "top_k": 100,
                "n_samples": 0
            }
        else:
            # Override config for temperature/top_p
            run_cfg = copy.deepcopy(cfg)
            #run_cfg['retrieval']['top_k'] = top_k
            run_cfg['generation']['n_samples'] = n_samples
            run_cfg['generation']['temperature'] = temperature
            run_cfg['generation']['top_p'] = top_p

            run_cfg["uncertainty"]["method"] = uq_method

            result = run_rag(
                query=query,
                cfg=run_cfg,
                chat_history=st.session_state["history"],
                #top_k_override=top_k,
                n_samples_override=n_samples,
                uq_method_override=uq_method,
                model_type_override=model_type,
                model_id_override=model_id,
                api_key_override=api_key,
            )

        st.session_state["history"].append({
            "user": query,
            "assistant": result['final_answer'],
            "calibrated_confidence": result["calibrated_confidence"],
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
                    "top_k": retr_cfg["top_k"],
                    "n_samples": result["n_samples"],
                }),
                "uncertainty_method": uq_method,
                "raw_uncertainty": result["raw_uncertainty"],
                "calibrated_confidence": result["calibrated_confidence"],
                "retrieved_documents": result["retrieved_docs"],
            }
        log_experiment(CSV_FILE, experiment_data)

# render chat 
render_chat_history(st.session_state["history"])