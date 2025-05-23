import csv
import os

# Define the header for the CSV file.
CSV_FIELDS = ["query", "answer", "samples", "model",
               "settings", "uncertainty_method", "raw_uncertainty",
                 "calibrated_confidence", "retrieved_documents"]

def initialize_csv(csv_filename: str):
    """
    Initializes the CSV file by creating it with headers if it does not exist.
    """
    if not os.path.exists(csv_filename):
        print(f"[CSV Logger] Creating new CSV file: {csv_filename}")
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
    else:
        print(f"[CSV Logger] CSV file {csv_filename} already exists.")

def log_experiment(csv_filename: str, data: dict):
    """
    Appends a new row to the CSV file.
    
    data: a dictionary containing keys as defined in CSV_FIELDS.
    """
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(data)
    #print(f"[CSV Logger] Logged experiment data: {data}")
    print(f"[CSV Logger] Logged experiment data.")


if __name__ == "__main__":
    # test: initialize and log a test row.
    test_csv = "output/streamlit_run/test_experiment_results.csv"
    initialize_csv(test_csv)
    
    test_data = {
        "query": "Tell me a joke.",
        "answer": "Why did the chicken cross the road? To get to the other side!",
        "samples": "[sample response]",
        "model": "hf",
        "settings": "API_URL=https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct, temperature=0.9, top_p=0.95",
        "uncertainty_method": "lexsim",
        "raw_uncertainty": 0.05,
        "calibrated_confidence": 0.9,
        "retrieved_documents": 'test'
    }
    log_experiment(test_csv, test_data)
