# Thesis-Trustworthy-RAG

### 1. Clone and enter the project
git clone https://github.com/SarahHvidAndersen/Thesis-Trustworthy-RAG.git
cd thesis‑trustworthy‑rag

### 2. Create an isolated Python 3.10+ environment
python -m venv .venv                 
source .venv/bin/activate            # Windows: .venv\Scripts\activate

### 3. Install the project *in editable mode*
python -m pip install --upgrade pip
pip install -e .[dev]                # pulls runtime + dev/test deps

### 4. Initialise local vector store (optional, but avoids empty‑results)
export STORAGE_PATH=$PWD/data/chroma_db        # Linux/macOS
setx STORAGE_PATH "%cd%\data\chroma_db"        # Windows PowerShell
python -m internal.database_setup.create_demo_index

### 5. Run either interface
# CLI test
python -m internal.core --query "Hello, world"

# Streamlit UI
streamlit run streamlit_app.py


---
# Fast path – if they have uv (best for exact locks)
uv sync                                # reads uv.lock

# Vanilla path – no uv needed
python -m pip install -e .             # pip reads the same build backend
