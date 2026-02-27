import os
from dotenv import load_dotenv
from openai import OpenAI

# load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def query_llm(prompt, model="gpt-4o-mini"):

    try:

        response = client.responses.create(
            model=model,
            input=prompt
        )

        return response.output_text

    except Exception as e:

        print("LLM error:", e)
        return "ERROR"