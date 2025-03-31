import requests
from dotenv import load_dotenv
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")


def build_prompt(query: str, context: str) -> str:
    """
    Constructs a prompt by combining retrieved context with the user query.
    You can customize this prompt structure to best suit your use case.
    
    Example prompt:
    
    Context: <retrieved context here>
    
    Question: <user query here>
    
    Answer:
    """
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    return prompt

def generate_answer(query: str, context: str, api_url: str, headers: dict = None) -> str:
    """
    Generates an answer by sending a prompt to the Hugging Face serverless API.
    
    Parameters:
      - query: The user query.
      - context: The retrieved context passages.
      - api_url: The Hugging Face API endpoint URL for the generative model.
      - headers: Optional headers for authentication (e.g., {"Authorization": "Bearer YOUR_TOKEN"}).
      
    Returns:
      The generated answer as a string.
    """
    prompt = build_prompt(query, context)
    payload = {"inputs": prompt}
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        # The structure of the response may vary between models.
        # Here we assume the response is a list with a dict containing the generated text.
        generated_text = result[0].get("generated_text", "")
        return generated_text
    else:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")

if __name__ == "__main__":
    # Test the generate_answer function independently.
    
    # Sample query and context (you can replace these with your own test data).
    test_query = "What are the key steps in Bayesian workflow?"
    test_context = (
        "1. Data Cleaning: Removing noise and normalizing data. "
        "2. Model Building: Constructing a probabilistic model. "
        "3. Inference: Drawing conclusions using Bayesian methods. "
        "4. Evaluation: Assessing model performance."
    )
    
    # Replace with your actual Hugging Face inference endpoint for a generative model.
    # For example, you might use: "https://api-inference.huggingface.co/models/your-model-name"
    api_url = "https://api-inference.huggingface.co/models/your-model-name"
    
    # If you require authentication, include your API token in the headers.

    headers = {"Authorization": "Bearer HF_API_KEY"}
    
    try:
        answer = generate_answer(test_query, test_context, api_url, headers=headers)
        print("Generated answer:")
        print(answer)
    except Exception as e:
        print(f"Error generating answer: {e}")
