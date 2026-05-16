"""Generate a document summary from extracted text using an LLM."""

import logging

from openai import OpenAI

from naive_reporter.config import settings

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT_TEMPLATE = (
    "Summarize the following document in 3 to 5 sentences.\n"
    "The summary must be written in the same language as the document content.\n"
    "Be concise and capture the main topic and key findings.\n"
    "\n"
    "Text:\n"
    "---\n"
    "{text}\n"
    "---\n"
    "\n"
    "Summary:\n"
)


def generate_summary(text: str) -> str:
    """Return a non-empty summary string from the LLM.

    Raises
    ------
    RuntimeError
        If the LLM response is empty or whitespace-only.
    """
    client = OpenAI(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
    )

    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        text=text[:50000]
    )  # hard cap to avoid huge prompts

    logger.debug("Sending summary-generation prompt (%d chars)", len(prompt))

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "You generate document summaries."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content or ""
    summary = raw.strip()

    if not summary:
        logger.error("LLM returned empty summary. Response:\n%s", raw)
        raise RuntimeError("LLM returned empty summary")

    logger.debug("Got summary from LLM (%d chars)", len(summary))
    return summary
