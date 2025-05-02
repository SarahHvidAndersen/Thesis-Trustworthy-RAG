import os
import requests
from dotenv import load_dotenv
load_dotenv(override=True)

# extract context and optional history
def build_context_and_hist(retrieved_docs: list[dict], history: list[dict] | None = None) -> str:
    """
    Returns only the chat history (last turns) and the CONTEXT block,
    without any system or example prompts.
    """
    # Optional chat history
    hist_block = ""
    if history:
        for turn in history[-6:]:
            hist_block += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        hist_block += "\n"

    # Context section
    context_section = "--- CONTEXT BEGIN ---\n"
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
    context_section += "--- CONTEXT END ---\n\n"
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
            "Always provide concise, factual answers using the supplied CONTEXT if any was provided.\n"
            "If information from a specific course is requested (e.g. Human Computer "
            "Interaction), only use CONTEXT where the metadata matches that course. "
            "If the answer for an educational question is not contained in CONTEXT, respond: \"I'm not sure about that, sorry!\""
            "For normal questions like: \"hi! who are you?\", respond as a friendly cognitive science tutor and ignore any provided irrelevant context.  \n\n"
        )

        # One-shot example
        example_prompt = (
            "EXAMPLE:\n\n"
            "CONTEXT:\n"
            "[1] [Course: Cognitive Neuroscience] “The prefrontal cortex (PFC) is "
            "involved in high-order executive functions like planning and decision making.”\n\n"
            "QUESTION: What does the PFC do?\n"
            "ANSWER: It supports executive functions such as planning and decision making. [1]\n"
            "----------------\n\n"
        )

        # add chat history, if any
        hist_block = ""
        if history:
            for turn in history[-6:]:
                hist_block += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            hist_block += "\n"

        # format the retrieved context
        context_section = "--- CONTEXT BEGIN ---\n"
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
        context_section += "--- CONTEXT END --- \n\n"

        # Real question
        question_block = (
            f"QUESTION: {query} \n"
            #"You may answer using citation markers ([1], [2], etc.) to reference the context sources when you deem it highly relevant. Otherwise, just respond normally."
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
          1) re-shows system+context (no question line),
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
            "-----------------\n\n"
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
    Provider that uses the Hugging Face Inference API.
    Generation parameters (temperature, top_p, max_new_tokens) are passed during initialization.
    """
    def __init__(self, api_url: str, headers: dict, temperature: float = 0.9,
                 top_p: float = 0.9, max_new_tokens: int = 150):
        self.api_url = api_url
        self.headers = headers
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        print(f"[HuggingFaceProvider] Initialized with API URL: {self.api_url}")
        print(f"[HuggingFaceProvider] Generation settings: temperature={self.temperature}, "
              f"top_p={self.top_p}, max_new_tokens={self.max_new_tokens}")
        print(f"[HuggingFaceProvider] Using headers: {self.headers}")

    def _call_api(self, payload: dict) -> str:
        print(f"[HuggingFaceProvider] Sending request with payload:\n{payload}\n")
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")
        data = response.json()[0]
        return data.get("generated_text", "").strip()
    
    def generate_raw(self, full_prompt: str) -> str:
        """
        Send `full_prompt` verbatim to the LLM, bypassing build_prompt().
        """
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens
            }
        }
        return self._call_api(payload)


    def generate(self, query: str, context: any, history=None) -> str:
        prompt = GeneratorProvider.build_prompt(query, context, history)
        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens
            }
        }
        return self._call_api(payload)
    

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

    def _call_api(self, payload: dict) -> str:
        print(f"[ChatUIProvider generate_raw] Sending request with payload:\n{payload}\n")
        response = requests.post(self.api_url, json=payload)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")
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
    from provider import ChatUIProvider, GeneratorProvider

    load_dotenv()
    print("Running provider.py demo…\n")

    # --- 1) Build a toy context for debugging ---
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

    # --- 2) Inspect the QA prompt ---
    qa_prompt = GeneratorProvider.build_prompt(
        query=user_query,
        retrieved_docs=retrieved_docs,
        history=history
    )
    print("=== QA PROMPT ===")
    print(qa_prompt)
    print("\n" + "="*80 + "\n")

    # --- 3) Inspect the selection prompt ---
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

    # --- 4) send selection prompt through ChatUIProvider ---
    #chatui_api = os.getenv("CHATUI_API_URL", "")
    chatui_api = os.getenv("CHATUI_GPU_API_URL")
    if chatui_api:
        chatui = ChatUIProvider(
            api_url=chatui_api,
            model_id="llama3:8b", #llama3.2:1b
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=50,
            seed=None
        )
        print("=== ChatUIProvider.generate(selection) ===")
        response = chatui.generate(sel_prompt, retrieved_docs, history)
        print(response)
    else:
        print("Set CHATUI_API_URL in your .env to test the live call.")

