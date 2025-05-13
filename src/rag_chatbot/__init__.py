"""
RAG‑Chatbot: simple API surface.
"""

from importlib.metadata import version
from .streamlit_app import main as run_app    # one‑liner launcher

__all__ = ["run_app", "__version__"]
__version__ = version("rag-chatbot")
