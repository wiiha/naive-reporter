"""Tests for summary_generator."""

from unittest.mock import MagicMock, patch

import pytest

from naive_reporter.summary_generator import generate_summary


def test_generate_summary_returns_stripped_response() -> None:
    """LLM returns a string with whitespace — we strip it and return it."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "  This is a summary.  \n"

    with patch("naive_reporter.summary_generator.OpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = generate_summary("some text")
        assert result == "This is a summary."

        # Verify the prompt was sent with the correct text
        call_args = mock_client.chat.completions.create.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        assert any("Summarize" in msg["content"] for msg in messages)


def test_generate_summary_raises_on_empty_response() -> None:
    """LLM returns whitespace-only — we raise RuntimeError."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "   \n  "

    with patch("naive_reporter.summary_generator.OpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="empty summary"):
            generate_summary("some text")
