from pathlib import Path
from typing import Iterable, Optional, Tuple

from langchain_core.messages import SystemMessage

from . import config

# Prompt template for LLM to normalize SIEM events
PROMPT_CORRELATION_TEMPLATE = """
You already know the complete SIEM taxonomy schema from the system instructions.
Always:
1. Read the raw event JSON.
2. Extract only the important facts required by the taxonomy.
3. Map every fact to the canonical SIEM field names and allowed values from the taxonomy.
4. Produce a flat JSON object with only normalized fields. No free-form comments. Do NOT wrap the result in Markdown fences or triple backticks.

=== Reference few-shot pairs ===
{examples}

=== Normalize the following event JSON using the taxonomy rules and examples above ===
{event}

Output ONLY a valid JSON object. No explanations.
""".strip()


def render_prompt_correlation(
    event: str,
    examples: Iterable[Tuple[str, Optional[str]]],
    event_file_path: str,
    debug: bool = False,
) -> str:
    """
    Renders a prompt template with event and multiple example pairs.
    """
    # Format examples section
    examples_text_parts = []
    for i, (example_event, example_normalized) in enumerate(examples, 1):
        if example_normalized is None:
            continue

        examples_text_parts.append(
            f"Example {i} - NOT normalized event JSON:\n{example_event}"
        )
        examples_text_parts.append(
            f"Example {i} - normalized SIEM fields JSON:\n{example_normalized}"
        )

    examples_text = "\n\n".join(examples_text_parts)

    prompt = PROMPT_CORRELATION_TEMPLATE.format(
        event=event, examples=examples_text.strip()
    )

    if debug:
        prompt_file = event_file_path.with_name(
            event_file_path.name.replace("events_", "prompt_").replace(".json", ".txt")
        )
        prompt_file.write_text(prompt, encoding="utf-8")
        print(f"Prompt dumped at: {prompt_file.relative_to(config.BASE_DIR)}")

    return prompt


# System prompt
PROMPT_SYSTEM_TEMPLATE = """
You are a SIEM normalization assistant.
""".strip()


def load_system_prompt(debug: bool = False) -> SystemMessage:
    prompt = PROMPT_SYSTEM_TEMPLATE

    if debug:
        system_prompt_file = config.TEST_DATA_PATH / "system_prompt.txt"
        system_prompt_file.write_text(prompt, encoding="utf-8")
        print(
            f"System prompt dumped at: {system_prompt_file.relative_to(config.BASE_DIR)}"
        )

    return SystemMessage(content=PROMPT_SYSTEM_TEMPLATE)


# Taxonomy RU template
# TODO(kompotkot): Try to re-write it in RU
PROMPT_TAXONOMY_RU_TEMPLATE = """
You are a SIEM normalization assistant. Always follow the taxonomy schema
provided below when converting raw events into normalized SIEM JSON.

Guidelines:
- Use only the field names and enumerations defined in the taxonomy.
- Omit fields that are not relevant.
- Prefer English field names when both RU/EN are provided.
- Preserve JSON validity and keep the output flat.

=== Taxonomy (RU) ===
{ru_schema}
""".strip()

# Taxonomy EN template
PROMPT_TAXONOMY_EN_TEMPLATE = """
You are a SIEM normalization assistant. Always follow the taxonomy schema
provided below when converting raw events into normalized SIEM JSON.

Guidelines:
- Use only the field names and enumerations defined in the taxonomy.
- Omit fields that are not relevant.
- Prefer English field names when both RU/EN are provided.
- Preserve JSON validity and keep the output flat.

=== Taxonomy (EN) ===
{en_schema}
""".strip()


def load_taxonomy_prompt(
    ru_path: Path = None,
    en_path: Path = None,
    debug: bool = False,
) -> set[str, str]:
    """
    Compose a system prompt that embeds the SIEM taxonomy in RU and EN.
    """
    ru_schema = ru_path.read_text(encoding="utf-8")
    taxonomy_ru_prompt = PROMPT_TAXONOMY_RU_TEMPLATE.format(ru_schema=ru_schema.strip())

    en_schema = en_path.read_text(encoding="utf-8")
    taxonomy_en_prompt = PROMPT_TAXONOMY_EN_TEMPLATE.format(en_schema=en_schema.strip())

    if debug:
        taxonomy_ru_prompt_file = config.TEST_DATA_PATH / "taxonomy_ru_prompt.txt"
        taxonomy_ru_prompt_file.write_text(taxonomy_ru_prompt, encoding="utf-8")
        print(
            f"Taxonomy RU prompt dumped at: {taxonomy_ru_prompt_file.relative_to(config.BASE_DIR)}"
        )

        taxonomy_en_prompt_file = config.TEST_DATA_PATH / "taxonomy_en_prompt.txt"
        taxonomy_en_prompt_file.write_text(taxonomy_en_prompt, encoding="utf-8")
        print(
            f"Taxonomy EN prompt dumped at: {taxonomy_en_prompt_file.relative_to(config.BASE_DIR)}"
        )

    return taxonomy_ru_prompt, taxonomy_en_prompt
