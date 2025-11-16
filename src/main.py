import asyncio
from pathlib import Path

import numpy as np
from tqdm.asyncio import tqdm_asyncio

from . import api, config, data, prompt


async def saturate_with_embedding(
    ri: data.ReferenceItem, sem: asyncio.Semaphore
) -> None:
    async with sem:
        # Generate embedding vector for the event text
        # and normalize embedding for efficient cosine similarity calculation
        ri.embed = await api.get_embedding(ri.event_text)


async def embeding_generation_semaphore(
    references: list[data.ReferenceItem], sem_limit: int
) -> None:
    sem = asyncio.Semaphore(sem_limit)
    # Schedule tasks for current batch
    tasks = [saturate_with_embedding(ri, sem) for ri in references]
    await tqdm_asyncio.gather(*tasks, desc="Embedding generation", total=len(tasks))


def load_reference_examples(
    base_path: str, sem_limit: int = 5
) -> list[data.ReferenceItem]:
    """
    Loads reference examples from directory structure by finding test directories and processing event files.

    Searches for pattern: base_path/*/tests/events_*.json and corresponding norm_fields_*.json files.
    Generates embeddings for each event and creates ReferenceItem objects.
    """
    base = Path(base_path)
    references: list[data.ReferenceItem] = []

    # Iterate through all test directories matching pattern: base_path/*/tests
    for test_dir in base.glob("*/*/tests"):
        if not test_dir.is_dir():
            continue

        # Scan for event files in current test directory
        for event_file in test_dir.glob("events_*.json"):
            norm_file = event_file.with_name(
                event_file.name.replace("events_", "norm_fields_")
            )
            # Skip if normalized file doesn't exist
            if not norm_file.exists():
                continue

            # Read event and normalized text files
            event_text = event_file.read_text()
            norm_text = norm_file.read_text()

            references.append(
                data.ReferenceItem(
                    event_text=event_text,
                    norm_text=norm_text,
                )
            )

    print(f"Found {len(references)} references")

    # Generate embeddings in batches
    asyncio.run(embeding_generation_semaphore(references, sem_limit))

    return references


def find_most_similar_reference(
    embed: np.ndarray, ref_exs: list[data.ReferenceItem], top_k: int = 3
) -> list[data.ReferenceItem]:
    """
    Finds the top-k most similar reference examples by comparing cosine similarity of embeddings.
    """
    if not ref_exs:
        raise ValueError("Reference examples list is empty")

    if len(ref_exs) < top_k:
        top_k = len(ref_exs)

    # Calculate cosine similarity with all reference embeddings
    similarities = []
    for ref in ref_exs:
        if ref.embed is None:
            continue
        similarity = np.dot(embed, ref.embed)
        similarities.append(similarity)

    # Find indices of top-k most similar references
    similarities_array = np.array(similarities)
    top_k_indices = np.argpartition(similarities_array, -top_k)[-top_k:]

    # Sort by similarity in descending order
    top_k_indices = top_k_indices[np.argsort(similarities_array[top_k_indices])[::-1]]

    return [ref_exs[idx] for idx in top_k_indices]


def normalize_fields(
    base_path: str,
    ref_exs: list[data.ReferenceItem],
    top_k_number: int = 1,
    sem_limit: int = 5,
) -> None:
    """
    Normalizes SIEM events by finding similar reference examples and using LLM to generate normalized fields.
    """
    base = Path(base_path)
    references = []

    # Iterate through all test directories matching pattern: base_path/*/tests
    for test_dir in base.glob("*/tests"):
        if not test_dir.is_dir():
            continue

        # Scan for event files in current test directory
        for event_file in test_dir.glob("events_*.json"):
            event_text = event_file.read_text()

            references.append(
                data.ReferenceItem(
                    event_text=event_text,
                )
            )
            break  # TODO
        break  # TODO

    print(f"Found {len(references)} references")

    # Generate embeddings in batches
    asyncio.run(embeding_generation_semaphore(references, sem_limit))

    for r in references:
        if r.embed is None:
            continue

        # Find top-3 most similar reference examples by comparing embeddings
        most_similar_refs = find_most_similar_reference(
            r.embed, ref_exs, top_k=top_k_number
        )

        # Prepare examples list with (event, normalized) pairs
        examples = [(ref.event_text, ref.norm_text) for ref in most_similar_refs]

        # Generate prompt with event and 3 similar examples
        prompt_text = prompt.render_prompt_correlation(
            event=event_text,
            examples=examples,
        )

        # Generate normalized fields using LLM
        normalized_fields = api.generate_normalized_fields(prompt_text)

        # Save normalized fields to file in the same directory as event file
        norm_file = event_file.with_name(
            event_file.name.replace("events_", "norm_fields_")
        )
        norm_file.write_text(normalized_fields)

        print(
            f"Normalized: {event_file.name} -> {norm_file.name} (saved to {norm_file.parent})"
        )


def main():
    # Load all reference examples from the training data directory
    # and generate embedings for each
    ref_exs = load_reference_examples(config.TRAIN_DATA_PATH)
    print(f"Processed {len(ref_exs)} reference examples")

    # Generate normalized SIEM fields
    normalize_fields(config.TEST_DATA_PATH, ref_exs)
    print("SIEM field generation complete")


if __name__ == "__main__":
    main()
