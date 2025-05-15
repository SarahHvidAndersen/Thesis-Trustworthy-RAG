import requests
import os
from dotenv import load_dotenv
load_dotenv()


# Configuration
API_URL = os.getenv("CHATUI_API_URL")
#API_URL = 'overwrite?' 
MODEL_ID = "llama3:8b"
#MODEL_ID = "llama3.2:1b"

def generate_chatui_response(prompt: str) -> str:
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.9,  # Increase variability if needed
            "seed": None
        }
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()

def main():
    prompt = input("Enter your prompt: ")

    print("[*] Generating response via ChatUI...")
    response = generate_chatui_response(prompt)
    print("\n=== Response ===")
    print(response)

if __name__ == "__main__":
    main()
