from typing import Any

import aiohttp
import numpy as np
import requests

from . import config


async def get_embedding(text: str) -> np.ndarray[Any, Any]:
    """
    Generates vector embedding for the input text using embedding API.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{config.LLM_API_URI}/{config.LLM_EMBED_PATH}",
            json={
                "model": config.LLM_MODEL,
                "input": text,
            },
        ) as resp:
            data = await resp.json()
            embed_array = np.array(data["embeddings"][0], dtype=np.float32)
            return embed_array / np.linalg.norm(embed_array)


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
            "stream": False,
        },
    )

    if resp.status_code != 200:
        raise ValueError(
            f"Failed to generate normalized fields: {resp.status_code} - {resp.text}"
        )

    return resp.json()["response"]
