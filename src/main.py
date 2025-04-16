import os
import yaml
from dotenv import load_dotenv

# Load sensitive settings from .env
load_dotenv()

# Load configuration from the YAML file.
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


# Instantiate the uncertainty estimator
from uncertainty_estimator_factory import get_uncertainty_estimator

uncertainty_config = config.get("uncertainty", {})
method = uncertainty_config.get("method", "lexical_similarity")
params = uncertainty_config.get(method, {})

estimator = get_uncertainty_estimator(method, **params)


# Import the RAG pipeline function.
from rag_pipeline import rag_pipeline

# Example query
query = "What is the role of the prefrontal cortex? answer in one sentence."

# Run the pipeline using parameters from the config.
result = rag_pipeline(
    query,
    top_k=config["top_k"],
    provider=provider,
    device="cpu",
    n_samples=config["n_samples"],
    estimator=estimator
)

if result is None:
    print("Pipeline returned no result (e.g., an empty document collection).")
    exit(1)

final_answer = result["final_answer"]
samples = result["samples"]
retrieved_docs = result["retrieved_docs"]
uncertainty = result["uncertainty"]

print("\n=== Final Answer ===")
print(final_answer)

print("\n=== Uncertainty Score ===")
print(uncertainty)

#print("\n=== Generated Samples ===")
#for i, s in enumerate(samples):
#    print(f"Sample {i+1}: {s}\n")

##print("\n=== Retrieved Documents ===")
#for doc in retrieved_docs:
#    print(f"ID: {doc['id']} - Metadata: {doc['metadata']}")

# Log experiment details to CSV.
from csv_logger import initialize_csv, log_experiment
import json

CSV_FILE = "experiment_results.csv"
initialize_csv(CSV_FILE)

experiment_data = {
    "query": query,
    "answer": final_answer,
    "samples": json.dumps(samples, ensure_ascii=False),
    "model": config["model_type"],
    "settings": f"API URL: {config.get('hf_model') if config['model_type']=='hf' else config.get('chatui_model')}, "
                f"temperature={config['temperature']}, top_p={config['top_p']}, max_new_tokens={config['max_new_tokens']}, "
                f"top_k={config['top_k']}, n_samples={config['n_samples']}",
    "uncertainty_method": method, #config['uncertainty_method'],
    "uncertainty_score": uncertainty,
    "retrieved_documents": retrieved_docs
}

log_experiment(CSV_FILE, experiment_data)
