import requests
import os
from huggingface_hub import InferenceClient,  model_info
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
    payload = {"inputs": prompt,
        "parameters": {
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_new_tokens": 150
        }
    }
    
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

    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    client = InferenceClient(
        #provider="together",
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        api_key=HF_API_KEY,
        #headers=
    )

    #info = model_info("meta-llama/Llama-3.1-8B-Instruct", expand="inferenceProviderMapping")
    #print(info.inference)
    #print(info.inference_provider_mapping)

    completion = client.chat_completion(messages, temperature=0.9, top_p=0.9, max_tokens=100)

    print(completion)
