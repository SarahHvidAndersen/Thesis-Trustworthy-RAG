# Thesis-Trustworthy-RAG
This repository contains a prototype implementation of a Retrieval‑Augmented Generation (RAG) chatbot with uncertainty estimation.  The codebase is structured as a proper Python package (internal) and a Streamlit front‑end app interface. 

## repository layout
.
├─ src/
│  └─ internal/                # main Python package (import internal.*)
│     ├─ core.py               # package entry‑point for CLI experiments
│     ├─ retrievers/
│     ├─ uncertainty_estimator/
│     └─ ...
│  └─ rag_chatbot/ 
│     ├─ streamlit_app.py           # Streamlit UI entry‑point
│  └─ scripts/ 
│
├─ data/                       # manually added
├─ pyproject.toml              # build/dependency metadata
├─ uv.lock                     # dependency versions
└─ README.md 

```shell
# 1. Clone and enter the project
git clone https://github.com/SarahHvidAndersen/Thesis-Trustworthy-RAG.git
$ cd thesis-trustworthy-rag

# 2. Create an environment (built on pyproject.toml + uv.lock)
$ uv venv .venv --python 3.12
$ uv sync                       # installs exact, locked deps

# 3. Activate the environment
# Windows
$ .venv\Scripts\activate
# macOS / Linux
$ source .venv/bin/activate

# 4. Launch the Streamlit UI
$ streamlit run src/rag_chatbot/streamlit_app.py
```
