import os
import requests
from dotenv import load_dotenv
load_dotenv()


class GeneratorProvider:
    #TODO improve prompt template
    """
    Abstract base class for generator providers.
    Any concrete provider must override the generate() method.
    """
    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        """
        Builds a prompt string given a context and a query.
        Both providers will use the same prompt format.
        """
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        return prompt

    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class HuggingFaceProvider(GeneratorProvider):
    """
    Provider that uses the Hugging Face Inference API.
    Generation parameters (temperature, top_p, max_new_tokens) are passed during initialization.
    """
    def __init__(self, api_url: str, headers: dict, temperature: float = 0.9,
                 top_p: float = 0.95, max_new_tokens: int = 150):
        self.api_url = api_url
        self.headers = headers
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        print(f"[HuggingFaceProvider] Initialized with API URL: {self.api_url}")
        print(f"[HuggingFaceProvider] Generation settings: temperature={self.temperature}, "
              f"top_p={self.top_p}, max_new_tokens={self.max_new_tokens}")
        print(f"[HuggingFaceProvider] Using headers: {self.headers}")

    def generate(self, query: str, context: str) -> str:
        # Use the common prompt builder.
        prompt = GeneratorProvider.build_prompt(query, context)
        print(f"[HuggingFaceProvider] Built prompt:\n{prompt}\n")

        # Build payload with generation parameters from config.
        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens
            }
        }
        print(f"[HuggingFaceProvider] Sending request with payload:\n{payload}\n")
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            generated_text = result[0].get("generated_text", "")
            print(f"[HuggingFaceProvider] Received generated text:\n{generated_text}\n")
            return generated_text
        else:
            error_msg = f"[HuggingFaceProvider] API request failed: {response.status_code}, {response.text}"
            print(error_msg)
            raise Exception(error_msg)


class ChatUIProvider(GeneratorProvider):
    """
    Provider that uses the ChatUI API.
    Accepts generation parameters including model_id, temperature, top_p, max_new_tokens, and optionally seed.
    """
    def __init__(self, api_url: str, model_id: str, temperature: float = 0.9,
                 top_p: float = 0.95, max_new_tokens: int = 150, seed=None):
        self.api_url = api_url
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        print(f"[ChatUIProvider] Initialized with API URL: {self.api_url}")
        print(f"[ChatUIProvider] Model ID: {self.model_id}")
        print(f"[ChatUIProvider] Generation settings: temperature={self.temperature}, top_p={self.top_p}, "
              f"max_new_tokens={self.max_new_tokens}, seed={self.seed}")

    def generate(self, query: str, context: str) -> str:
        # Create the prompt using our shared function.
        prompt = GeneratorProvider.build_prompt(query, context)
        print(f"[ChatUIProvider] Built prompt:\n{prompt}\n")
        
        # Build payload; note that ChatUI may expect generation settings under "options"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "max_new_tokens": self.max_new_tokens
            }
        }
        print(f"[ChatUIProvider] Sending request with payload:\n{payload}\n")
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get("response", "").strip()
            print(f"[ChatUIProvider] Received generated text:\n{generated_text}\n")
            return generated_text
        else:
            error_msg = f"[ChatUIProvider] API request failed: {response.status_code}, {response.text}"
            print(error_msg)
            raise Exception(error_msg)


if __name__ == "__main__":
    # Demo usage of the provider classes:
    print("Running provider.py demo...\n")
    from dotenv import load_dotenv
    load_dotenv()
    
    # Uncomment the following block to test HuggingFaceProvider.
    
    #HF_API_KEY = os.getenv("HF_API_KEY", "your_api_key_here")
    #HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    #hf_provider = HuggingFaceProvider(
    #    api_url=HF_API_URL,
    #    headers={"Authorization": f"Bearer {HF_API_KEY}"},
    #    temperature=0.9,
    #    top_p=0.95,
    #    max_new_tokens=150)
    
    #hf_response = hf_provider.generate("Tell me a joke.", "Test context for Hugging Face.")
    #print("HuggingFaceProvider generated response:")
    #print(hf_response)
    
    #Uncomment the following block to test ChatUIProvider.
    
    CHATUI_API_URL = os.getenv("CHATUI_API_URL", "")
    chatui_provider = ChatUIProvider(
        api_url=CHATUI_API_URL,
        model_id="llama3.2:1b",
        temperature=0.9,
        seed=None
    )
    chatui_response = chatui_provider.generate("Tell me a joke.", "Test context for ChatUI.")
    print("ChatUIProvider generated response:")
    print(chatui_response)
