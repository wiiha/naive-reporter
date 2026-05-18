"""LLM report/bullet generation and validation."""

import logging

from naive_reporter.config import settings
from naive_reporter.llm_client import chat
from naive_reporter.report_generator_prompts import (
    BULLET_GENERATION_PROMPT,
    BULLET_RETRY_PROMPT,
    BULLET_VALIDATION_PROMPT,
    REPORT_GENERATION_PROMPT,
    REPORT_RETRY_PROMPT,
    REPORT_VALIDATION_PROMPT,
)
from naive_reporter.types import DocContent

logger = logging.getLogger(__name__)


def _format_docs_text(documents: list[DocContent]) -> str:
    """Join capped document texts for prompt context."""
    cap = settings.text_cap
    return "\n\n".join(
        f"Document: {d.stem}\n---\n{d.text[:cap]}\n---" for d in documents
    )


def generate_report(
    prompt: str,
    documents: list[DocContent],
    previous: str = "",
    feedback: str = "",
) -> str:
    """Generate a report. On retry, ``previous`` and ``feedback`` guide the LLM."""
    docs_text = _format_docs_text(documents)

    if previous and feedback:
        user_text = REPORT_RETRY_PROMPT.format(
            prompt=prompt,
            documents=docs_text,
            previous_report=previous,
            feedback=feedback,
        )
    else:
        user_text = REPORT_GENERATION_PROMPT.format(
            prompt=prompt,
            documents=docs_text,
        )

    config = settings.get_reporting_client_config()
    raw = chat(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        system="You are a research analyst who writes detailed reports.",
        user=user_text,
    )

    report = raw.strip()
    if not report:
        raise RuntimeError("LLM returned empty report")
    return report


def validate_report(
    prompt: str,
    documents: list[DocContent],
    report: str,
) -> tuple[bool, str]:
    """Validate a report. Returns (is_valid, reason)."""
    docs_text = _format_docs_text(documents)
    user_text = REPORT_VALIDATION_PROMPT.format(
        prompt=prompt,
        documents=docs_text,
        report=report,
    )

    config = settings.get_reporting_client_config()
    raw = chat(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        system="You are a critical reviewer who validates reports.",
        user=user_text,
    )

    return _parse_validation_response(raw)


def generate_bullets(
    report: str,
    previous: str = "",
    feedback: str = "",
) -> str:
    """Summarize a report into bullets. On retry, uses previous + feedback."""
    if previous and feedback:
        user_text = BULLET_RETRY_PROMPT.format(
            report=report,
            previous_bullets=previous,
            feedback=feedback,
        )
    else:
        user_text = BULLET_GENERATION_PROMPT.format(report=report)

    config = settings.get_reporting_client_config()
    raw = chat(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        system="You are an editor who creates concise bullet summaries.",
        user=user_text,
    )

    bullets = raw.strip()
    if not bullets:
        raise RuntimeError("LLM returned empty bullet summary")
    return bullets


def validate_bullets(report: str, bullets: str) -> tuple[bool, str]:
    """Validate a bullet summary. Returns (is_valid, reason)."""
    user_text = BULLET_VALIDATION_PROMPT.format(
        report=report,
        bullets=bullets,
    )

    config = settings.get_reporting_client_config()
    raw = chat(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        system="You are a critical reviewer who validates summaries.",
        user=user_text,
    )

    return _parse_validation_response(raw)


def _parse_validation_response(raw: str) -> tuple[bool, str]:
    """Parse <VALID|INVALID>:reason from LLM response.

    Returns
    -------
    (is_valid, reason)
        ``is_valid`` is ``True`` when the status prefix is ``VALID``.
    """
    text = raw.strip()

    valid_words = ["VALID", "<VALID>"]
    if text.upper() in valid_words:
        return True, ""

    if ":" not in text:
        return False, f"Malformed validation response (no colon): {text[:200]}"

    status, reason = text.split(":", 1)
    status = status.strip().upper()
    is_valid = status in valid_words
    return is_valid, reason.strip()
