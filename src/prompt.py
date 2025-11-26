from pathlib import Path
from typing import Iterable, Optional, Tuple

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
    event: str, examples: Iterable[Tuple[str, Optional[str]]]
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

    return PROMPT_CORRELATION_TEMPLATE.format(
        event=event, examples=examples_text.strip()
    )


# System taxonomy template
PROMPT_TAXONOMY_SYSTEM_TEMPLATE = """
You are a SIEM normalization assistant. Always follow the taxonomy schema
provided below when converting raw events into normalized SIEM JSON.

Guidelines:
- Use only the field names and enumerations defined in the taxonomy.
- Omit fields that are not relevant.
- Prefer English field names when both RU/EN are provided.
- Preserve JSON validity and keep the output flat.

=== Taxonomy (RU) ===
{ru_schema}

=== Taxonomy (EN) ===
{en_schema}
""".strip()


def load_taxonomy_system_prompt(
    ru_path: Path,
    en_path: Path,
) -> str:
    """
    Compose a system prompt that embeds the SIEM taxonomy in RU and EN.
    """
    ru_schema = ru_path.read_text(encoding="utf-8")
    en_schema = en_path.read_text(encoding="utf-8")

    return PROMPT_TAXONOMY_SYSTEM_TEMPLATE.format(
        ru_schema=ru_schema.strip(), en_schema=en_schema.strip()
    )
