import os
import time
import requests
from openai import OpenAI
from google import genai as google_genai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY")

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

if GOOGLE_API_KEY:
    gemini_client = google_genai.Client(api_key=GOOGLE_API_KEY)
else:
    gemini_client = None

if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    anthropic_client = None


# ==============================
# MODEL TYPES
# ==============================

OPENAI_MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

ANTHROPIC_MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-4-5-20251001",
]

OLLAMA_MODELS = [
    "llama3",
    "mistral",
    "phi3",
    "gemma",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-3-flash-preview",
]


# ==============================
# MAIN FUNCTION
# ==============================

def query_llm(prompt, model="gpt-4.1-mini"):

    if model in OPENAI_MODELS:
        return query_openai(prompt, model)

    elif model in OLLAMA_MODELS:
        return query_ollama(prompt, model)

    elif model in GEMINI_MODELS:
        return query_gemini(prompt, model)

    elif model in ANTHROPIC_MODELS:
        return query_anthropic(prompt, model)

    else:
        raise ValueError(f"Unknown model: {model}")


# ==============================
# OPENAI QUERY
# ==============================

def query_openai(prompt, model):

    if openai_client is None:
        raise ValueError("OPENAI_API_KEY not set")

    try:

        response = openai_client.responses.create(
            model=model,
            input=prompt,
        )

        return response.output_text

    except Exception as e:

        print("OpenAI error:", e)
        return "ERROR"


# ==============================
# GEMINI QUERY
# ==============================

def query_gemini(prompt, model, max_retries=5):

    if gemini_client is None:
        raise ValueError("GOOGLE_API_KEY not set")

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text

        except Exception as e:
            error_str = str(e)
            is_retryable = "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
                print(f"Gemini retryable error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                print("Gemini error:", e)
                return "ERROR"


# ==============================
# ANTHROPIC QUERY
# ==============================

def query_anthropic(prompt, model):

    if anthropic_client is None:
        raise ValueError("ANTHROPIC_API_KEY not set")

    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if hasattr(block, "text"):
                return block.text
        return "ERROR"

    except Exception as e:
        print("Anthropic error:", e)
        return "ERROR"


# ==============================
# OLLAMA QUERY
# ==============================

def query_ollama(prompt, model):

    try:

        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        if r.status_code != 200:
            print("Ollama error:", r.text)
            return "ERROR"

        return r.json()["response"]

    except Exception as e:

        print("Ollama connection error:", e)
        return "ERROR"