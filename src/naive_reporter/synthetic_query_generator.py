"""Generate synthetic search queries from a user prompt."""

import logging

from naive_reporter.config import settings
from naive_reporter.llm_client import chat
from naive_reporter.report_generator_prompts import SYNTHETIC_QUERY_PROMPT

logger = logging.getLogger(__name__)


def generate_queries(prompt: str) -> list[str]:
    """Ask the LLM for search queries based on a user prompt.

    Returns exactly ``settings.number_of_synthetic_user_queries`` lines.

    Raises
    ------
    ValueError
        If ``prompt`` is empty or whitespace-only.
    RuntimeError
        If the LLM does not return the expected number of queries.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    prompt_text = SYNTHETIC_QUERY_PROMPT.format(prompt=prompt)
    logger.debug("Sending synthetic query prompt (%d chars)", len(prompt_text))

    raw = chat(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        system="You generate search queries.",
        user=prompt_text,
    )

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    expected = settings.number_of_synthetic_user_queries

    if len(lines) != expected:
        logger.error(
            "LLM returned %d queries instead of %d. Response:\n%s",
            len(lines),
            expected,
            raw,
        )
        raise RuntimeError(f"LLM returned {len(lines)} queries instead of {expected}")

    logger.debug("Got %d synthetic queries", expected)
    return lines
