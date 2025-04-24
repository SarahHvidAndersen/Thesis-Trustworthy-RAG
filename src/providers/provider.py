import os
import requests
from dotenv import load_dotenv
load_dotenv()


class GeneratorProvider:
    """
    Abstract base class for generator providers.
    Any concrete provider must override the generate() method.
    """
    @staticmethod
    def build_prompt(query: str, retrieved_docs: list[dict], history=None) -> str:
        # System instruction
        system_prompt = (
            "SYSTEM: You are a knowledgeable cognitive science tutor.\n"
            "Provide concise, factual answers using only the supplied CONTEXT.\n"
            'If the answer is not contained in CONTEXT, respond: "I\'m not sure."\n\n'
        )

        # History from chat
        hist_block = ""
        if history:
            for turn in history[-6:]:
                hist_block += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            hist_block += "\n"

        # Context from the retriever with explicit delimiters + metadata
        # Numbered context section
        context_section = "--- CONTEXT BEGIN ---\n"
        for idx, doc in enumerate(retrieved_docs, start=1):
            try:
                title = doc.get("metadata", {}).get("title", "Unknown")
                # label each snippet with the index
                context_section += f"[{idx}] [Title: {title}] {doc['text']}\n"
            except:
                pass

        context_section += "--- CONTEXT END ---\n\n"

        # Question directive with citation instruction
        question_block = (
            f"QUESTION: {query}\n"
            "Answer using citation markers ([1], [2], etc.) to reference the context sources."
        )

        return system_prompt + hist_block + context_section + question_block

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

    def generate(self, query: str, context: str, history=None) -> str:
        # Use the common prompt builder.
        prompt = GeneratorProvider.build_prompt(query, context, history)
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

    def generate(self, query: str, context: str, history=None) -> str:
        # Create the prompt using our shared function.
        prompt = GeneratorProvider.build_prompt(query, context, history)
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
    chatui_response = chatui_provider.generate("Tell me what kind of chatbot you are.", ["sample1", "sample2"])
    print("ChatUIProvider generated response:")
    print(chatui_response)
