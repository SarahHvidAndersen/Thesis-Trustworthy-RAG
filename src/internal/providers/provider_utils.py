from pathlib import Path
from urllib.parse import urlparse, urlunparse
import warnings
from dotenv import load_dotenv, set_key
import os, sys

load_dotenv(override=True)
_ENV_FILE = Path('.') / '.env'                     # repo root → “.env”


def format_url(raw: str) -> str:
    """
    Normalise user input into a proper ChatUI (HTTPS) or Ollama-style localhost
    (HTTP) endpoint, always ending with /api/generate.

    Examples
    -------
    ChatUI
        cool-bot                          → https://app-cool-bot.cloud.aau.dk/api/generate
        app-cool-bot.cloud.aau.dk         → https://app-cool-bot.cloud.aau.dk/api/generate
        https://app-cool-bot.cloud.aau.dk → https://app-cool-bot.cloud.aau.dk/api/generate

    Ollama / localhost
        localhost:8000          → http://localhost:8000/api/generate
        http://localhost        → warning
        http://localhost:8000   → http://localhost:8000/api/generate
    """
    DEFAULT_PREFIX        = "app-"
    DEFAULT_DOMAIN_SUFFIX = ".cloud.aau.dk"
    REQUIRED_PATH         = "/api/generate"

    if not raw:
        return ""
    raw = raw.strip()

    # break out any scheme/host/path if a URL was given
    if raw.startswith(("http://", "https://")):
        parts  = urlparse(raw)
        netloc = parts.netloc or ""
        path   = parts.path or ""
    else:
        netloc = raw               # treat input as host/bot name
        path   = ""

    #  check if localhost (keep port if present)
    host_only = netloc.split(":")[0]
    is_local  = host_only == "localhost"
    has_port  = ":" in netloc

    if is_local and not has_port:
        warnings.warn("No port specified for localhost; the request may fail",
            stacklevel=2,
        )

    # Build / fix netloc when the input wasn’t a full URL
    if not raw.startswith(("http://", "https://")) and not is_local:
        if "." not in netloc:  # bare app name → add prefix/suffix
            netloc = f"{DEFAULT_PREFIX}{netloc}{DEFAULT_DOMAIN_SUFFIX}"

    # Pick appropriate scheme 
    scheme = "http" if is_local else "https"

    # Ensure /api/generate suffix
    if not path.rstrip("/").endswith(REQUIRED_PATH):
        path = path.rstrip("/") + REQUIRED_PATH

    # Compose final URL 
    return urlunparse((scheme, netloc.rstrip("/"), path, "", "", ""))


def ensure_provider_input(provider: str,
                          var_name: str | None = None,
                          prompt_text: str | None = None,
                          persist: bool = True) -> str:
    """
    Return a normalised URL (ChatUI / Ollama) **or** key (Hugging Face) for the chosen provider.

    • Checks the .env first.  
    • Falls back to interactive `input()` if running in a TTY session.  
    • Persists the value into `.env` under `var_name` (created if needed).

    Arguments
    ---------
    provider    : one of {"ChatUI", "Ollama", "Huggingface"}
    var_name    : override the default env-var name (optional)
    prompt_text : override the default prompt (optional)
    """
    # Canonicalise provider label
    p = provider
    #if p in ("Huggingface"):
    #    p = "Huggingface"
    #elif p in ("ChatUI",):
    #    p = "ChatUI"
    #elif p in ("Ollama",):
    #    p = "Ollama"
    #else:
    #    raise ValueError(f"Unsupported provider: {provider}")

    # Default env var names + prompts
    defaults = {
        "ChatUI": ("CHATUI_API_URL",
                "ChatUI hostname OR full link "
                "(e.g. cool-bot | app-cool-bot.cloud.aau.dk): "),
        "Ollama": ("OLLAMA_HOST",
                "Ollama localhost:PORT or full link "
                "(e.g. localhost:11434 | http://localhost:11434): "),
        "Huggingface":     ("HF_API_KEY",
                "Enter your Huggingface API key (starts with hf_…): "),
    }

    env_name, prompt = defaults[p]
    var_name= var_name or env_name
    prompt_text = prompt_text or prompt

    value = os.getenv(var_name)
    if value:
        return format_url(value) if p in ("ChatUI", "Ollama") else value.strip()

    # --- interactive fallback ---
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{var_name} is not set and no TTY is available for a prompt."
        )

    # get user input in terminal 
    value = input(prompt_text).strip()

    if not value:
        raise RuntimeError("Empty value entered – aborting.")

    # normalise before saving
    final = format_url(value) if p in ("ChatUI", "Ollama") else value

    if persist == True:
        try:
            set_key(_ENV_FILE, var_name, final)
            print(f"[config] Saved {var_name} to {_ENV_FILE}")
        except Exception:
            print("An exception occurred, failed to save key to .env")
            pass
    else:
        pass

    return final