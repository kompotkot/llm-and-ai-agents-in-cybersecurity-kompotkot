# Prompt template for LLM to normalize SIEM events
PROMPT_CORRELATION_TEMPLATE = """
You are an Information Security engineer. You are working on correlation rules.

You have normalized SIEM fields examples below.

{examples}

Generate normalized SIEM fields from next event:
{event}

Output ONLY a valid JSON object. No explanations.
"""


def render_prompt_correlation(event: str, examples: list) -> str:
    """
    Renders a prompt template with event and multiple example pairs.
    """
    # Format examples section
    examples_text = ""
    for i, (example_event, example_normalized) in enumerate(examples, 1):
        examples_text += f"Example {i} of NOT normalized event:\n{example_event}\n\n"
        examples_text += (
            f"Example {i} of normalized SIEM fields:\n{example_normalized}\n\n"
        )

    return PROMPT_CORRELATION_TEMPLATE.format(
        event=event, examples=examples_text.strip()
    )
