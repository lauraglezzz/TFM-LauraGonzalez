import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None


# ==============================
# MODEL TYPES
# ==============================

OPENAI_MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

OLLAMA_MODELS = [
    "llama3",
    "mistral",
    "phi3",
    "gemma",
]


# ==============================
# MAIN FUNCTION
# ==============================

def query_llm(prompt, model="gpt-4.1-mini"):

    if model in OPENAI_MODELS:
        return query_openai(prompt, model)

    elif model in OLLAMA_MODELS:
        return query_ollama(prompt, model)

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