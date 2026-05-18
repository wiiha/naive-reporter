"""Tests for summary_generator."""

from unittest.mock import patch

import pytest

from naive_reporter.summary_generator import generate_summary


def test_generate_summary_returns_stripped_response() -> None:
    """LLM returns a string with whitespace — we strip it and return it."""
    raw = "  This is a summary.  \n"

    with patch("naive_reporter.summary_generator.chat") as mock_chat:
        mock_chat.return_value = raw

        result = generate_summary("some text")
        assert result == "This is a summary."

        call_args = mock_chat.call_args
        assert call_args is not None
        assert "Summarize" in call_args.kwargs["user"]


def test_generate_summary_raises_on_empty_response() -> None:
    """LLM returns whitespace-only — we raise RuntimeError."""
    with patch("naive_reporter.summary_generator.chat") as mock_chat:
        mock_chat.return_value = "   \n  "

        with pytest.raises(RuntimeError, match="empty summary"):
            generate_summary("some text")
