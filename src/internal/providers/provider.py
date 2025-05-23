import os
import requests
from typing import Any, List, Dict
import json
from huggingface_hub import InferenceClient, model_info
from dotenv import load_dotenv
load_dotenv(override=True)

# extract context and optional history
def build_context_and_hist(retrieved_docs: list[dict], history: list[dict] | None = None) -> str:
    """
    Returns only the chat history (last turns) and the CONTEXT block,
    without any system or example prompts.
    """
    # chat history
    hist_block = "HISTORY:"
    if history:
        for turn in history[-4:]:
            hist_block += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        hist_block += "\n"

    # Context section
    context_section = " CONTEXT BEGIN \n"
    for idx, doc in enumerate(retrieved_docs, start=1):
        md = doc.get("metadata", {})
        course = md.get("course", "Unknown Course")
        title  = md.get("title",  "Unknown Title")
        author = md.get("author", "Unknown Author")
        snippet = doc["text"].replace("\n", " ").strip()
        context_section += (
            f"[{idx}] [Course: {course}] [Title: {title}] [Author: {author}]\n"
            f"{snippet}\n\n"
        )
    context_section += " CONTEXT END \n\n"
    return hist_block + context_section

class GeneratorProvider:
    """
    Abstract base class for generator providers.
    Any concrete provider must override the generate() method.
    """

    @staticmethod
    def build_prompt(
        query: str,
        retrieved_docs: list[dict],
        history: list[dict] | None = None
    ) -> str:
        # system instructions
        system_prompt = (
            "SYSTEM: You are a knowledgeable cognitive science tutor with information "
            "about the entire Cognitive Science syllabus at Aarhus University.\n"
            "Always provide concise, factual answers to the users QUESTION using the supplied CONTEXT if any was provided.\n"
            "If information from a specific course is requested (e.g. Human Computer "
            "Interaction), only use CONTEXT where the metadata matches that course. "
            "If the answer for a question is not contained in CONTEXT or none was supplied, add that information and ALLWAYS try your best to answer anyway: \"The CONTEXT did not include specific information but...\""
            "For normal questions like: \"hi! who are you?\", respond as a friendly tutor and ignore any irrelevant context.  \n\n"
        )

        # One-shot example
        example_prompt = (
            "EXAMPLE:\n\n"
            "CONTEXT:\n"
            "[1] [Course: Cognitive Neuroscience] “The prefrontal cortex (PFC) is "
            "involved in high-order executive functions like planning and decision making.”\n\n"
            "QUESTION: What does the PFC do?\n"
            "ANSWER: It supports executive functions such as planning and decision making. [1]\n"
            "-----\n\n"
        )

        # add chat history, if any
        hist_block = "HISTORY:"
        if history:
            for turn in history[-4:]:
                hist_block += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            hist_block += "\n"

        # format the retrieved context
        context_section = " CONTEXT BEGIN \n"
        for idx, doc in enumerate(retrieved_docs, start=1):
            md = doc.get("metadata", {})
            course = md.get("course", "Unknown Course")
            title  = md.get("title",  "Unknown Title")
            author = md.get("author", "Unknown Author")
            snippet = doc["text"].replace("\n", " ").strip()
            context_section += (
                f"[{idx}] [Course: {course}] [Title: {title}] [Author: {author}] \n"
                f"{snippet}\n\n"
            )
        context_section += " CONTEXT END  \n\n"

        # Real question
        question_block = (
            f"QUESTION: {query} \n"
        )

        return (system_prompt + example_prompt + hist_block + context_section + question_block)

    @staticmethod
    def build_selection_prompt(
        original_query: str,
        candidates: list[str],
        retrieved_docs: list[dict],
        history: list[dict] | None = None
    ) -> str:
        """
        Build a prompt that:
          1) new system prompt
          2) presents one example of selection,
          3) lists `candidates`,
          4) asks the model to reply exactly with the index (1,2,…) or 0 to abstain.
        """
        system = (
        "SYSTEM: You are a numeric selection assistant. Your job is to pick, by index, "
        "the single best answer from the list of candidates, using ONLY the CONTEXT.\n\n"
        )

        # One-shot selection example
        example = (
            "EXAMPLE:\n\n"
            "CONTEXT:\n"
            "[1] [Course: Neuroscience] “The hippocampus is critical for forming new episodic memories.”\n\n"
            "1. “The hippocampus regulates emotion.”\n"
            "2. “It’s involved in long-term memory formation.”\n"
            "3. “It processes sensory input.”\n\n"
            "QUESTION: Of the above choices, which numbered answer best addresses the question:\n"
            "What is the hippocampus involved in?\n"
            "your answer: 2\n"
            "--\n\n"
        )

        # get system+context via helper
        context_and_hist = build_context_and_hist(retrieved_docs, history)

        # list the generated samples
        choices = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(candidates))
        print(f"CHOICES SUPPLIED ARE {choices}")

        # Instruction using the real user query
        instruction = (
            f"QUESTION: Of the above choices, which numbered answer best addresses the question:\n"
            f"“{original_query}” using only the CONTEXT?  \n"
            "Reply *only* with the index digit: 1, 2, 3, ...) of the best answer.  \n"
            "If none of them are supported, reply exactly with the digit: 0.\n"
            "Never explain the choice - just return the number."
        )

        return (system + example + context_and_hist + choices + "\n\n"+ instruction)


    def generate(self, query: str, context: any, history: any = None) -> str:
        raise NotImplementedError("Subclasses must implement this method.")



class HuggingFaceProvider(GeneratorProvider):
    """
    Provider that uses the Huggingface Inference API.
    Generation parameters (temperature, top_p, max_new_tokens) are passed during initialization.
    """
    def __init__(self, model:str, api_url: str, provider: str | None = None, 
                 #headers: dict, 
                 temperature: float = 0.9,
                 top_p: float = 0.9, max_new_tokens: int = 150):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # `InferenceClient` handles routing + auth
        self.client = InferenceClient(
            model=model,
            provider=provider,     # e.g. "together", "hf-inference", "novita", or None/"auto"
            api_key=api_url,
            headers={"X-use-cache": "false"}
            #timeout=timeout,
        )

        #print(f"[HuggingFaceProvider] Initialized with API URL: {self.api_url}")
        print(f"[HuggingFaceProvider] Generation settings: temperature={self.temperature}, "
              f"top_p={self.top_p}, max_new_tokens={self.max_new_tokens}")
    
    @staticmethod
    def _wrap(prompt: str) -> List[Dict[str, str]]:
        """Turn a plain prompt into the new style messages array."""
        return [{"role": "user", "content": prompt}]
    
    @staticmethod
    def _split_prompt(full: str) -> tuple[str | None, str]:
        """
        Split `full` at the '-----' marker.
        Returns (system_part_or_None, user_part).
        """
        marker = "-----" # split after example
        idx = full.find(marker)
        if idx == -1:                       # fallback – nothing to split
            return None, full
        system = full[:idx].strip()
        user   = full[idx:].lstrip()
        return system, user
    
    def _as_messages(self,
                 user_prompt: str,
                 system_prompt: str | None = None) -> list[dict[str, str]]:
        if system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        return [{"role": "user", "content": user_prompt}]
    
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Run `chat_completion` and return the assistant’s text only."""
        debug_payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "top_p":       self.top_p,
            "max_tokens":  self.max_new_tokens,
        }
        print("[Huggingface] Payload:\n" + json.dumps(debug_payload, indent=2))
        
        resp = self.client.chat_completion(
            messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        # `resp` is a ChatCompletion object (attr & item access both work)
        return resp.choices[0].message.content.strip()

    def generate_raw(self, full_prompt: str) -> str:
        """Send `full_prompt` exactly as given."""
        return self._chat(self._as_messages(full_prompt))
    
    def generate(self, query: str, context: Any, history=None) -> str:
        prompt = GeneratorProvider.build_prompt(query, context, history)
        system_part, user_part = self._split_prompt(prompt)
        return self._chat(self._as_messages(user_part, system_part))
    

class OllamaProvider(GeneratorProvider):
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

        print(f"[OllamaProvider] Initialized with API URL: {self.api_url} and Model ID: {self.model_id}")
        print(f"[OllamaProvider] Generation settings: temperature={self.temperature}, top_p={self.top_p}, "
              f"max_new_tokens={self.max_new_tokens}, seed={self.seed}")

    def _call_api(self, payload: dict) -> str:
        print(f"[OllamaProvider] Sending request with payload:\n{payload}\n")
        #print(f"[OllamaProvider] Sending request.")
        response = requests.post(self.api_url, json=payload, verify=False)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")
        
        print("[OllamaProvider] Received response:")
        print(response.json().get("response", "").strip())
        return response.json().get("response", "").strip()
    
    def generate_raw(self, full_prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "max_new_tokens": self.max_new_tokens
            }
        }
        return self._call_api(payload)

    def generate(self, query: str, context: any, history=None) -> str:
        prompt = GeneratorProvider.build_prompt(query, context, history)
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
        return self._call_api(payload)


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from internal.providers.provider import OllamaProvider, GeneratorProvider

    load_dotenv()
    print("Running provider.py demo…\n")

    #  Build a toy context for debugging 
    retrieved_docs = [
        {
            "id": "chunk1",
            "metadata": {
                "course": "Cognitive Neuroscience",
                "title": "Intro to PFC",
                "author": "Dr. Smith"
            },
            "text": "The prefrontal cortex (PFC) is involved in high-order executive functions."
        },
        {
            "id": "chunk2",
            "metadata": {
                "course": "Cognitive Psychology",
                "title": "Memory Systems",
                "author": "Dr. Jones"
            },
            "text": "The hippocampus is critical for forming new episodic memories."
        }
    ]
    history = [
        {"user": "Hello", "assistant": "Hi there! How can I help?"}
    ]
    user_query = "What does the PFC do?"

    #  Inspect the QA prompt 
    qa_prompt = GeneratorProvider.build_prompt(
        query=user_query,
        retrieved_docs=retrieved_docs,
        history=history
    )
    print("=== QA PROMPT ===")
    print(qa_prompt)
    print("\n" + "="*80 + "\n")

    # Inspect the selection prompt 
    samples = [
        "It supports planning and decision making.",
        "It helps in memory consolidation.",
        "It regulates emotional responses."
    ]
    sel_prompt = GeneratorProvider.build_selection_prompt(
        original_query=user_query,
        candidates=samples,
        retrieved_docs=retrieved_docs,
        history=history
    )
    print("=== SELECTION PROMPT ===")
    print(sel_prompt)
    print("\n" + "="*80 + "\n")

    #  send selection prompt through OllamaProvider 
    chatui_api = os.getenv("CHATUI_API_URL", "")
    if chatui_api:
        chatui = OllamaProvider(
            api_url=chatui_api,
            model_id="llama3:8b", #llama3.2:1b
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=50,
            seed=None
        )
        print("=== Ollama.generate(selection) ===")
        response = chatui.generate(sel_prompt, retrieved_docs, history)
        print(response)
    else:
        print("Set CHATUI_API_URL in your .env to test the live call.")

