"""Thin wrapper around OpenAI chat completions."""

import logging

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


def chat(base_url: str, api_key: str, model: str, system: str, user: str) -> str:
    """Send a chat completion request and return the content string.

    Raises
    ------
    RuntimeError
        If the LLM call fails or returns empty content.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)

    logger.debug("Sending prompt (%d chars) to %s", len(user), base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    content = response.choices[0].message.content or ""
    logger.debug("Got response (%d chars)", len(content))
    return content
