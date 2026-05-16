"""Pluggable search algorithm protocol."""

from abc import ABC, abstractmethod

from naive_reporter.types import Document, SearchResult


class Searcher(ABC):
    """Pluggable search algorithm."""

    def _validate_k(self, k: int) -> None:
        """Validate the ``k`` parameter before searching.

        - ``k == -1``  => return all results sorted by score.
        - ``k >= 0``   => return top ``k`` results.
        - Any other negative value raises ``ValueError``.
        """
        if k == -1:
            return
        if k < 0:
            raise ValueError(f"k must be -1 or >= 0, got {k}")

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Build the index from a list of documents."""
        ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search the index.

        ``k=-1`` returns all results sorted by score descending.
        """
        ...
