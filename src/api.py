import requests

from . import config


def get_embedding(text: str) -> list[float]:
    """
    Generates vector embedding for the input text using embedding API.
    """
    resp = requests.post(
        f"{config.LLM_API_URI}/{config.LLM_EMBED_PATH}",
        json={
            "model": config.LLM_MODEL,
            "input": text,
        },
    )

    if resp.status_code != 200:
        raise ValueError(f"Failed to get embedding: {resp.status_code} - {resp.text}")

    return resp.json()["embeddings"][0]


def generate_normalized_fields(prompt: str) -> str:
    """
    Generates normalized SIEM fields using LLM API
    and return normalized fields as JSON string.
    """
    resp = requests.post(
        f"{config.LLM_API_URI}/{config.LLM_GENERATE_PATH}",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
        },
    )

    if resp.status_code != 200:
        raise ValueError(
            f"Failed to generate normalized fields: {resp.status_code} - {resp.text}"
        )

    return resp.json()["response"]
