"""Generate a document summary from extracted text using an LLM."""

import logging

from naive_reporter.config import settings
from naive_reporter.llm_client import chat

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
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        text=text[: settings.text_cap]
    )  # hard cap to avoid huge prompts

    logger.debug("Sending summary-generation prompt (%d chars)", len(prompt))

    raw = chat(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        system="You generate document summaries.",
        user=prompt,
    )

    summary = raw.strip()

    if not summary:
        logger.error("LLM returned empty summary. Response:\n%s", raw)
        raise RuntimeError("LLM returned empty summary")

    logger.debug("Got summary from LLM (%d chars)", len(summary))
    return summary
