#!/usr/bin/env python3
"""
quick_hf_chat_test.py
---------------------
Run a single chat-completion request against Hugging Face’s new
Inference-Providers endpoint and print the assistant’s reply.

Prerequisites
* `pip install requests`
* export HF_API_KEY="hf_XXXXXXXXXXXXXXXXXXXXXXXX"  # must have “Inference Providers” scope
"""

import os
import sys
import json
import argparse
import requests

BASE_URL = "https://router.huggingface.co"


def chat_complete(
    provider: str,
    model: str,
    user_message: str,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_tokens: int = 150,
    timeout: int = 60,
) -> str:
    """Call the OpenAI-compatible chat/completions route and return the text."""
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        sys.exit("❌  Set the HF_API_KEY environment variable first")

    url = f"{BASE_URL.rstrip('/')}/{provider}/v3/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    print(f"→ POST {url}\n{json.dumps(payload, indent=2)}\n")
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()        # raise if HTTP error
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    #parser = argparse.ArgumentParser(description="Quick test for HF chat completion")
    #parser.add_argument("--provider", default="novita", help="Inference provider ID")
    #parser.add_argument("--model",    default="google/gemma-2-2b-it", help="Model ID")
    #parser.add_argument("message",    nargs="?", default="Hello, world!",
    #                    help="Prompt to send")
    #args = parser.parse_args()

    

    reply = chat_complete(
        provider="fireworks",
        model="meta-llama/Llama-3.1-8B-Instruct",
        user_message="what is an apple?",
    )
    print("\nAssistant →", reply)


if __name__ == "__main__":
    main()
