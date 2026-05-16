"""Search engine: reads query files and orchestrates search."""

import logging
from pathlib import Path

from naive_reporter.config import settings
from naive_reporter.search_protocol import Searcher
from naive_reporter.types import Document, SearchResult

logger = logging.getLogger(__name__)


def load_documents(data_dir: str | None = None) -> list[Document]:
    """Read ``*.txt`` files from ``data/queries_txt/`` and return Document objects.

    Each file's stem becomes the document ID. Lines are stripped and empty
    lines are dropped. Unreadable files are skipped with a warning.
    """
    root = Path(data_dir) if data_dir else Path(settings.data_dir)
    queries_dir = root / "queries_txt"
    docs: list[Document] = []
    if not queries_dir.exists():
        logger.warning("Queries directory does not exist: %s", queries_dir)
        return docs

    for path in sorted(queries_dir.glob("*.txt")):
        try:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            docs.append(Document(stem=path.stem, queries=lines))
        except (OSError, UnicodeDecodeError):
            logger.warning(
                "Skipping unreadable query file %s", path.name, exc_info=True
            )
    return docs


class SearchEngine:
    """High-level search engine that loads documents and delegates to a ``Searcher``.

    .. note::

        Indexes are in-memory and static after ``build_index()``.
        Callers must create a new ``SearchEngine`` to pick up new files.

    .. note::

        Not thread-safe. External synchronisation is required for
        concurrent ``search()`` calls on the same instance.
    """

    def __init__(self, searcher: Searcher, data_dir: str | None = None) -> None:
        self._searcher = searcher
        self._data_dir = data_dir or settings.data_dir

    def build_index(self) -> None:
        """Load documents from disk and build the search index."""
        docs = load_documents(self._data_dir)
        self._searcher.index(docs)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Run a search query and return ranked results."""
        return self._searcher.search(query, k=k)
