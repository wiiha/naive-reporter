"""BM25S search implementation."""

import bm25s

from naive_reporter.search_protocol import Searcher
from naive_reporter.types import Document, SearchResult


class BM25Searcher(Searcher):
    """Lexical search via bm25s.

    .. note::

        Not thread-safe. External synchronisation is required if
        ``index()`` or ``search()`` are called concurrently.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._retriever: bm25s.BM25 | None = None

    def index(self, documents: list[Document]) -> None:
        """Build the BM25 index from documents."""
        self._documents = documents
        if not documents:
            self._retriever = None
            return

        corpus = [" ".join(doc.queries) for doc in documents]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search the BM25 index."""
        self._validate_k(k)
        if not self._documents:
            return []

        top_n = len(self._documents) if k == -1 else min(k, len(self._documents))
        if top_n == 0:
            return []

        if self._retriever is None:
            return []

        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self._retriever.retrieve(query_tokens, k=top_n)

        # results and scores are 2-D: one row per input query.
        # We always pass a single query, so we unwrap the first row.
        out: list[SearchResult] = []
        for idx_arr, score_arr in zip(results, scores):
            for idx, score in zip(idx_arr, score_arr):
                score_f = float(score)
                if score_f > 0:
                    out.append(
                        SearchResult(
                            stem=self._documents[idx].stem,
                            score=score_f,
                        )
                    )
        return out
