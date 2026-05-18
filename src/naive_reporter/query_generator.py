"""Generate search queries from extracted text using an LLM."""

import logging

from naive_reporter.config import settings
from naive_reporter.llm_client import chat

logger = logging.getLogger(__name__)

_QUERY_PROMPT_TEMPLATE = (
    "Below is the full text extracted from a PDF document.\n"
    "Generate exactly 5 short search queries (one per line, no numbering) "
    "that a user might type when trying to find this document.\n"
    "\n"
    "Text:\n"
    "---\n"
    "{text}\n"
    "---\n"
    "\n"
    "Search queries:\n"
)


def generate_queries(text: str) -> list[str]:
    """Return exactly 5 non-empty query strings from the LLM.

    Raises
    ------
    RuntimeError
        If the LLM response does not contain exactly 5 non-empty lines.
    """
    prompt = _QUERY_PROMPT_TEMPLATE.format(text=text[: settings.text_cap])

    logger.debug("Sending query-generation prompt (%d chars)", len(prompt))

    raw = chat(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        system="You generate search queries.",
        user=prompt,
    )

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    if len(lines) != 5:
        logger.error("LLM returned %d queries. Response:\n%s", len(lines), raw)
        raise RuntimeError(f"LLM returned {len(lines)} queries instead of 5")

    logger.debug("Got 5 queries from LLM")
    return lines
