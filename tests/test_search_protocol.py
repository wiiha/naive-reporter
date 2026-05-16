"""Tests for the Searcher ABC contract."""

import pytest

from naive_reporter.search_protocol import Searcher
from naive_reporter.types import Document, SearchResult


class FakeSearcher(Searcher):
    """Minimal concrete implementation for testing the base class."""

    def index(self, documents: list[Document]) -> None:
        pass

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self._validate_k(k)
        return []


def test_validate_k_accepts_positive() -> None:
    s = FakeSearcher()
    s._validate_k(5)  # should not raise


def test_validate_k_accepts_zero() -> None:
    s = FakeSearcher()
    s._validate_k(0)  # should not raise


def test_validate_k_accepts_minus_one() -> None:
    s = FakeSearcher()
    s._validate_k(-1)  # should not raise


def test_validate_k_rejects_other_negative() -> None:
    s = FakeSearcher()
    with pytest.raises(ValueError, match="k must be -1 or >= 0"):
        s._validate_k(-2)
