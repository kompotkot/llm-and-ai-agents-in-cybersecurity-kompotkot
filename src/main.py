import asyncio
import json
from pathlib import Path

import numpy as np
from tqdm.asyncio import tqdm_asyncio

from . import api, config, data, prompt


async def saturate_with_embedding(r: data.Event, sem: asyncio.Semaphore) -> None:
    async with sem:
        # Generate embedding vector for the event text
        # and normalize embedding for efficient cosine similarity calculation
        r.embed = await api.get_embedding(r.event_text)


async def embeding_generation_semaphore(
    references: list[data.Event], sem_limit: int
) -> None:
    sem = asyncio.Semaphore(sem_limit)
    # Schedule tasks for current batch
    tasks = [saturate_with_embedding(r, sem) for r in references]
    await tqdm_asyncio.gather(*tasks, desc="Embedding generation", total=len(tasks))


def load_reference_examples(
    base_path: Path, sem_limit: int = 5
) -> list[data.EventPack]:
    """
    Loads reference examples from directory structure by finding test directories and processing event files.

    Searches for pattern: base_path/*/tests/events_*.json and corresponding norm_fields_*.json files.
    Generates embeddings for each event and creates EventPack list.
    """
    reference_packs: list[data.EventPack] = []
    references_cnt = 0

    # Iterate through all test directories matching pattern: base_path/*/tests
    for test_dir in base_path.glob("*/*/tests"):
        if not test_dir.is_dir():
            continue

        pack = data.EventPack(pack_path=test_dir.parent, events=[])

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

            pack.events.append(
                data.Event(
                    event_file_name=str(event_file.relative_to(pack.pack_path)),
                    event_text=event_text,
                    norm_text=norm_text,
                )
            )

            references_cnt += 1

        reference_packs.append(pack)

    print(f"Found {references_cnt} references in {len(reference_packs)} packs")

    # Generate embeddings in batches
    asyncio.run(
        embeding_generation_semaphore(
            [e for pack in reference_packs for e in pack.events], sem_limit
        )
    )

    return reference_packs


def find_most_similar_reference(
    embed: np.ndarray, references: list[data.Event], top_k: int = 3
) -> list[data.Event]:
    """
    Finds the top-k most similar reference examples by comparing cosine similarity of embeddings.
    """
    if not references:
        raise ValueError("Reference examples list is empty")

    if len(references) < top_k:
        top_k = len(references)

    # Calculate cosine similarity with all reference embeddings
    similarities = []
    for ref in references:
        if ref.embed is None:
            continue
        similarity = np.dot(embed, ref.embed)
        similarities.append(similarity)

    # Find indices of top-k most similar references
    similarities_array = np.array(similarities)
    top_k_indices = np.argpartition(similarities_array, -top_k)[-top_k:]

    # Sort by similarity in descending order
    top_k_indices = top_k_indices[np.argsort(similarities_array[top_k_indices])[::-1]]

    return [references[idx] for idx in top_k_indices]


async def saturate_with_normalized_files(
    event: data.Event,
    pack_path: Path,
    references: list[data.Event],
    sem: asyncio.Semaphore,
    top_k_number: int,
    system_prompt: str,
) -> None:
    async with sem:
        if event.embed is None:
            return

        # Find top-3 most similar reference examples by comparing embeddings
        most_similar_refs = find_most_similar_reference(
            event.embed, references, top_k=top_k_number
        )

        # Prepare examples list with (event, normalized) pairs
        examples = [(r.event_text, r.norm_text) for r in most_similar_refs]

        # Generate prompt with event and 3 similar examples
        prompt_text = prompt.render_prompt_correlation(
            event=event.event_text,
            examples=examples,
        )

        event_file_path = pack_path / event.event_file_name

        if config.IS_DEBUG:
            prompt_file = event_file_path.with_name(
                event_file_path.name.replace("events_", "prompt_").replace(
                    ".json", ".txt"
                )
            )
            prompt_file.write_text(prompt_text, encoding="utf-8")

        normalized_fields = await api.generate_normalized_fields(
            system_prompt, prompt_text
        )

        # Save normalized fields to file in the same directory as event file
        norm_file = event_file_path.with_name(
            event_file_path.name.replace("events_", "norm_fields_")
        )
        norm_file.write_text(normalized_fields, encoding="utf-8")

        if config.IS_DEBUG:
            print(
                f"Normalized: {norm_file.parent.relative_to(config.BASE_DIR)}/{norm_file.name}"
            )


async def normalize_fields_semaphore(
    event_packs: list[data.EventPack],
    references: list[data.Event],
    sem_limit: int,
    top_k_number: int,
    system_prompt: str,
) -> None:
    sem = asyncio.Semaphore(sem_limit)
    # Schedule tasks for current batch
    tasks = []
    for pack in event_packs:
        for event in pack.events:
            tasks.append(
                saturate_with_normalized_files(
                    event, pack.pack_path, references, sem, top_k_number, system_prompt
                )
            )
    await tqdm_asyncio.gather(*tasks, desc="Fields normalization", total=len(tasks))


def normalize_fields(
    base_path: Path,
    reference_packs: list[data.EventPack],
    system_prompt: str,
    top_k_number: int = 2,
    sem_limit: int = 5,
) -> list[data.EventPack]:
    """
    Normalizes SIEM events by finding similar reference examples and using LLM to generate normalized fields.
    """
    event_packs: list[data.EventPack] = []
    events_cnt = 0

    # Iterate through all test directories matching pattern: base_path/*/tests
    for test_dir in base_path.glob("*/tests"):
        if not test_dir.is_dir():
            continue

        pack = data.EventPack(pack_path=test_dir.parent, events=[])

        # Scan for event files in current test directory
        for event_file in test_dir.glob("events_*.json"):
            event_text = event_file.read_text()

            pack.events.append(
                data.Event(
                    event_file_name=str(event_file.relative_to(pack.pack_path)),
                    event_text=event_text,
                )
            )
            events_cnt += 1

        event_packs.append(pack)

    print(f"Found {events_cnt} events in {len(event_packs)} packs to normalize")

    # Generate embeddings in batches
    asyncio.run(
        embeding_generation_semaphore(
            [e for pack in event_packs for e in pack.events], sem_limit
        )
    )

    asyncio.run(
        normalize_fields_semaphore(
            event_packs,
            [e for pack in reference_packs for e in pack.events],
            sem_limit,
            top_k_number,
            system_prompt,
        )
    )

    return event_packs


def _strip_markdown_fence(content: str) -> str:
    """
    Remove surrounding markdown ``` fences (optionally with language hints).
    """
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content

    lines = stripped.splitlines()

    # Drop opening fence with optional language suffix
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    cleaned = "\n".join(lines).strip()
    return cleaned or content


def clean_norm_fields(event_packs: list[data.EventPack]) -> None:
    """
    Clean JSON file structure from artifacts.
    """
    for pack in event_packs:
        for event in pack.events:
            event_file_path = pack.pack_path / event.event_file_name
            norm_file_path = event_file_path.with_name(
                event_file_path.name.replace("events_", "norm_fields_")
            )

            if not norm_file_path.exists():
                continue

            raw_text = norm_file_path.read_text(encoding="utf-8")
            cleaned_text = _strip_markdown_fence(raw_text)

            try:
                json.loads(cleaned_text)
            except json.JSONDecodeError:
                rel_path = norm_file_path
                try:
                    rel_path = norm_file_path.relative_to(config.BASE_DIR)
                except ValueError:
                    pass
                print(f"Invalid JSON: {rel_path}")
                continue

            if cleaned_text != raw_text:
                norm_file_path.write_text(cleaned_text, encoding="utf-8")


def main():
    # Load all reference examples from the training data directory
    # and generate embedings for each
    reference_packs = load_reference_examples(config.TRAIN_DATA_PATH)

    # Prepare taxonomy system prompt
    taxonomy_prompt = prompt.load_taxonomy_system_prompt(
        config.TAXONOMY_RU_PATH, config.TAXONOMY_EN_PATH
    )
    if config.IS_DEBUG:
        system_prompt_file = config.TEST_DATA_PATH / "system_prompt.txt"
        system_prompt_file.write_text(taxonomy_prompt, encoding="utf-8")
        print(
            f"System prompt dumped: {system_prompt_file.relative_to(config.BASE_DIR)}"
        )

    # Generate normalized SIEM fields
    event_packs = normalize_fields(
        config.TEST_DATA_PATH, reference_packs, taxonomy_prompt
    )

    # Verify new file is correct JSON
    clean_norm_fields(event_packs)


if __name__ == "__main__":
    main()
