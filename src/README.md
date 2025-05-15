## Source-Code Guide for `src/`
The `src` folder is organised as a package-style workspace. Production code lives in the `internal` package and in the small `rag_chatbot` front‑end folder. Utility or one-off scripts are collected in `scripts`. 
Most python scripts contain a main block meant to quickly debug that file alone. 
Additionally, every file in internal/uncertainty_estimation can be imported autonomously; they share the same call signature and are hot-swappable via get_uncertainty_estimator().

### Structural Overview
<pre> ``` 
src/
├─ archive/                  # legacy one-time scripts and experiments
├─ internal/                 # core library code (installable, use uv sync)
│  ├─ database_setup/        # DB initialisation + indexing helpers
│  ├─ logging_utils/         # CSV logfile and scraping loggers
│  ├─ metrics/               # evaluation / calibration utilities
│  ├─ providers/             # LLM provider wrappers
│  ├─ retrievers/            # BM25 + dense retrieval
│  ├─ scraping/              # syllabus web and pdf scraping functionalities
│  └─ uncertainty_estimation/# UQ algorithms + factory
├─ rag_chatbot/              # Streamlit user interface
├─ scripts/                  # CLI utilities
└─ thesis_trustworthy_rag/   # package entry-point
``` </pre>

### File-level description
| Path                                                               | Description                                                                               | Entrypoints & Main Functions                                  |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `internal/core.py`                                                 | Orchestrates the RAG pipeline: retrieval → re-rank → generation → uncertainty → calibration. | `run_rag`, `rag_pipeline`, `get_config`                      |
| `internal/course_pipeline.py`                                      | Scrapes the raw syllabus PDFs/HTML into json files                                           | CLI `__main__` block `process_course_syllabi()`               |
| `internal/embeddings_pipeline.py`                                  | Creates sentence-transformer embeddings to Chroma & builds the BM25 index.                    | CLI: `main()`                                                     |
| `internal/run_cli.py`                                              | Minimal terminal chat interface.                                                             |  CLI: `main()`                                                   |
| `internal/database_setup/bm25_indexer.py`                          | Builds a BM25 index with the already processed chunks.                                       | `build_index`                                                |
| `internal/database_setup/chroma_db.py`                             | Chroma database helper (init, collections).                                                  | `init_db`, `get_collection`, `add_documents`                  |
| `internal/database_setup/embeddings.py`                            | Shared embedding helpers.                                                                    | `load_embedding_model`, `embed_text`                |
| `internal/database_setup/preprocessing.py`                         | Text cleaning of the scraped files.                                                           | `clean_text`                                                 |
| `internal/logging_utils/csv_logger.py`                             | Logs experiment metadata to CSV.                                                             | `initialize_csv`, `log_experiment`                           |
| `internal/logging_utils/scraping_logger.py`                        | Structured logger for the web-scraping pipeline.                                             | `scraping_courses_logger`                                             |
| `internal/metrics/alignscore_utils.py`                             | AlignScore wrapper to compare answers.                                                       | `AlignScorer`                                         |
| `internal/metrics/fit_alignscore.py`                               | Fits an AlignScore large regression model.                                                   | CLI `main()`                                                 |
| `internal/metrics/fit_scaler.py`                                   | Fits quantile, isotonic and sigmoid scalers for confidence calibration.                        | CLI `main()`                                                 |
| `internal/providers/provider.py`                                   | Abstract and concrete LLM provider wrappers. Also builds prompt templates.                  | `GeneratorProvider`, `ChatUIProvider`, `HuggingFaceProvider` |
| `internal/retrievers/bm25_retriever.py`                            | Lexical retrieval over BM25 index.                                                           | `bm25_retrieve`                                              |
| `internal/retrievers/semantic_retriever.py`                        | Dense retrieval using multilingual `e5` + Chroma.                                            | `load_embedding_model`, `retrieve_documents`                 |
| `internal/scraping/html_scraper.py`                                | Scrapes html sites such as course pages or online syllabus material                          | `scrape_html`, `scrape_au_course`, `scrape_html_standard`               |
| `internal/scraping/metadata_handler.py`                            | Setup of backward updating of metadata file, so that this can be manually improved over time   | `update_metadata_corrections`                                             |
| `internal/scraping/pdf_processor.py`                               | Splits and extracts text from PDFs via PyMuPDF.                                              | `process_pdf`                                                |
| `internal/scraping/utils.py`                                       | Creates stable filenames for repeat scraped files                                            | `get_stable_filename`                                            |
| `internal/uncertainty_estimation/common.py`                        | Math helpers shared by UE methods.                                                           | `compute_sim_score`                                          |
| `internal/uncertainty_estimation/deberta.py`                       | DeBERTa-MNLI entailment logits.                                                              | `Deberta`                                                    |
| `internal/uncertainty_estimation/deg_mat.py`                       | Degree-Matrix uncertainty.                                                                   | `DegMat`                                                     |
| `internal/uncertainty_estimation/eccentricity.py`                  | Eccentricity uncertainty.                                                                    | `Eccentricity`                                               |
| `internal/uncertainty_estimation/lexical_similarity.py`            | Lexical-Similarity uncertainty.                                                              | `LexicalSimilarity`                                          |
| `internal/uncertainty_estimation/estimator.py`                     | Base class for pluggable UQ estimators - dummy decorator                                     | `Estimator`                                                 |
| `internal/uncertainty_estimation/uncertainty_estimator_factory.py` | Factory & wrapper for UQ estimators.                                                         | `get_uncertainty_estimator`, `compute_uncertainty`           |
| `rag_chatbot/streamlit_app.py`                                     | Streamlit UI (sidebar, chat bubbles, logging).                                               | Streamlit run                                               |
| `rag_chatbot/ui_helpers.py`                                        | Helper functions to render chat history and display                                          | `render_chat_history`                                        |
| `scripts/nbs/merge_splits.ipynb`                                   | Notebook: merge the output splits from Ragas                                                | —                                                                 |
| `scripts/nbs/ragas_inspect.ipynb`                                  | Notebook: inspect Ragas output, identifies the issues with kg size and ´MultiHopAbstractQuerySynthesizer´| —                                                   |
| `scripts/nbs/survey_results.ipynb`                                 | Notebook: gather and save results from the user test survey                                  | —                                                              |
| `scripts/nbs/ue_results.ipynb`                                     | Notebook: gather and save quantitative results on UE method and scalers                      | —                                                               |
| `scripts/generate_ragas_dataset.py`                                | Builds a silver Q\&A dataset via Ragas.                                                      | CLI `main()`                                                 |
| `scripts/generate_testdata_samples.py`                             | Generates answers & raw UQ scores.                                                           | CLI `main()`                                                 |
| `scripts/redo_ue_score.py`                                         | Re-computes uncertainty scores for an answer file.                                           | CLI `main()`                                                 |
| `scripts/split_documents.py`                                       | Splits corpus into shards for Ragas limits.                                                  | CLI `main()`                                                 |
| `archive/`                                                         | Historic experiments & notebooks.                                                            | —                                                            |


❕ Note that some of the scripts require the raw data to run out-of-the-box, which is not available on GitHub

