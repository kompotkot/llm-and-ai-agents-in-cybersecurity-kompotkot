import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import yaml
from langchain_core.messages import SystemMessage

from . import config

logger = logging.getLogger(__name__)


# System prompt
PROMPT_SYSTEM_TEMPLATE = """
You are a SIEM normalization assistant. Always follow the taxonomy schema 
when converting raw events into normalized SIEM JSON.
""".strip()


def load_system_prompt() -> str:
    prompt = PROMPT_SYSTEM_TEMPLATE

    if logger.isEnabledFor(logging.DEBUG):
        system_prompt_file = config.TEST_DATA_PATH / "system_prompt.txt"
        system_prompt_file.write_text(prompt, encoding="utf-8")
        logger.debug(
            f"System prompt dumped at: {system_prompt_file.relative_to(config.BASE_DIR)}"
        )

    return prompt


# Prompt template for LLM to normalize SIEM events
PROMPT_CORRELATION_TEMPLATE = """
You already know the complete SIEM taxonomy schema from the system instructions.
Always:
1. Read the raw event JSON.
2. Extract only the important facts required by the taxonomy, ignore null fields.
3. Map every fact to the canonical SIEM field names and allowed values from the taxonomy.
4. Concatenate the top-level key with a dot.
5. Produce a flat JSON object with only normalized fields. No free-form comments. Do NOT wrap the result in Markdown fences or triple backticks.

=== Reference few-shot pairs ===
{examples}

=== Normalize the following event JSON using the taxonomy rules and examples above ===
{event}

Output ONLY flat valid JSON object. No explanations.
""".strip()


def render_prompt_correlation(
    event: str,
    examples: Iterable[Tuple[str, Optional[str]]],
    event_file_path: Path,
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

    if logger.isEnabledFor(logging.DEBUG):
        prompt_file = event_file_path.with_name(
            event_file_path.name.replace("events_", "prompt_").replace(".json", ".txt")
        )
        prompt_file.write_text(prompt, encoding="utf-8")
        logger.debug(f"Prompt dumped at: {prompt_file.relative_to(config.BASE_DIR)}")

    return prompt


# Taxonomy RU template
# TODO(kompotkot): Try to re-write it in RU
PROMPT_TAXONOMY_RU_TEMPLATE = """
Guidelines:
- Use only the field names and enumerations defined in the taxonomy.
- Omit fields that are not relevant.
- Preserve JSON validity and keep the output flat.

=== Taxonomy (RU) ===
{ru_schema}
""".strip()

# Taxonomy EN template
PROMPT_TAXONOMY_EN_TEMPLATE = """
Guidelines:
- Use only the field names and enumerations defined in the taxonomy.
- Omit fields that are not relevant.
- Preserve JSON validity and keep the output flat.

=== Taxonomy (EN) ===
{en_schema}
""".strip()


def load_taxonomy_prompt(
    ru_path: Path = None,
    en_path: Path = None,
) -> set[str, str]:
    """
    Compose a system prompt that embeds the SIEM taxonomy in RU and EN.
    """
    ru_schema = ru_path.read_text(encoding="utf-8")
    taxonomy_ru_prompt = PROMPT_TAXONOMY_RU_TEMPLATE.format(ru_schema=ru_schema.strip())

    en_schema = en_path.read_text(encoding="utf-8")
    taxonomy_en_prompt = PROMPT_TAXONOMY_EN_TEMPLATE.format(en_schema=en_schema.strip())

    if logger.isEnabledFor(logging.DEBUG):
        taxonomy_ru_prompt_file = config.TEST_DATA_PATH / "taxonomy_ru_prompt.txt"
        taxonomy_ru_prompt_file.write_text(taxonomy_ru_prompt, encoding="utf-8")
        logger.debug(
            f"Taxonomy RU prompt dumped at: {taxonomy_ru_prompt_file.relative_to(config.BASE_DIR)}"
        )

        taxonomy_en_prompt_file = config.TEST_DATA_PATH / "taxonomy_en_prompt.txt"
        taxonomy_en_prompt_file.write_text(taxonomy_en_prompt, encoding="utf-8")
        logger.debug(
            f"Taxonomy EN prompt dumped at: {taxonomy_en_prompt_file.relative_to(config.BASE_DIR)}"
        )

    return taxonomy_ru_prompt, taxonomy_en_prompt


PROMPT_TAXONOMY_FIELDS_TEMPLATE = """
Guidelines:
- Fields separated by comma.
- Use only the field names and enumerations defined in the taxonomy.

=== Taxonomy Fields ===
{field_names}
""".strip()


def load_taxonomy_fields_prompt():
    data = yaml.safe_load(Path(config.TAXONOMY_EN_PATH).read_text(encoding="utf-8"))

    fields = data.get("Fields", {})
    field_names = list(fields.keys())

    tax_fields_prompt = PROMPT_TAXONOMY_FIELDS_TEMPLATE.format(
        field_names=", ".join(field_names)
    )

    if logger.isEnabledFor(logging.DEBUG):
        tax_fields_prompt_file = config.TEST_DATA_PATH / "taxonomy_fields_prompt.txt"
        tax_fields_prompt_file.write_text(tax_fields_prompt, encoding="utf-8")
        logger.debug(
            f"Taxonomy fields prompt dumped at: {tax_fields_prompt_file.relative_to(config.BASE_DIR)}"
        )

    return tax_fields_prompt


PROMPT_SYSTEM_CLEANUP_TEMPLATE = """
You are a JSON converter.
Output ONLY a valid JSON object.
""".strip()


def load_system_clean_prompt() -> SystemMessage:
    prompt = PROMPT_SYSTEM_CLEANUP_TEMPLATE

    if logger.isEnabledFor(logging.DEBUG):
        system_prompt_file = config.TEST_DATA_PATH / "system_clean_prompt.txt"
        system_prompt_file.write_text(prompt, encoding="utf-8")
        logger.debug(
            f"System clean prompt dumped at: {system_prompt_file.relative_to(config.BASE_DIR)}"
        )

    return SystemMessage(content=prompt)


PROMPT_CLEANUP_TEMPLATE = """
Remove Markdown quotes from JSON object:
{norm_fields}
""".strip()


def load_clean_prompt(
    norm_fields: str,
    event_file_path: Path,
) -> str:
    prompt = PROMPT_CLEANUP_TEMPLATE.format(norm_fields=norm_fields.strip())

    if logger.isEnabledFor(logging.DEBUG):
        prompt_file = event_file_path.with_name(
            event_file_path.name.replace("norm_fields_", "clean_prompt_").replace(
                ".json", ".txt"
            )
        )
        prompt_file.write_text(prompt, encoding="utf-8")
        logger.debug(
            f"Clean prompt dumped at: {prompt_file.relative_to(config.BASE_DIR)}"
        )

    return prompt
