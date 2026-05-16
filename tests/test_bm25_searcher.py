"""Tests for BM25Searcher."""

import pytest

from naive_reporter.bm25_searcher import BM25Searcher
from naive_reporter.types import Document


def _make_searcher(docs: list[Document]) -> BM25Searcher:
    s = BM25Searcher()
    s.index(docs)
    return s


def test_search_happy_path() -> None:
    searcher = _make_searcher(
        [
            Document(stem="doc_a", queries=["python tutorial", "learn python"]),
            Document(stem="doc_b", queries=["java tutorial", "learn java"]),
        ]
    )
    results = searcher.search("python", k=1)
    assert len(results) == 1
    assert results[0].stem == "doc_a"
    assert results[0].score > 0


def test_search_k_minus_one_returns_all() -> None:
    searcher = _make_searcher(
        [
            Document(stem="doc_a", queries=["aaa"]),
            Document(stem="doc_b", queries=["bbb"]),
        ]
    )
    results = searcher.search("aaa bbb", k=-1)
    assert len(results) == 2


def test_search_empty_corpus() -> None:
    searcher = BM25Searcher()
    searcher.index([])
    assert searcher.search("anything") == []


def test_search_before_index_returns_empty() -> None:
    searcher = BM25Searcher()
    assert searcher.search("anything") == []


def test_search_bad_k_raises() -> None:
    searcher = _make_searcher([Document(stem="x", queries=["python tutorial"])])
    with pytest.raises(ValueError, match="k must be -1"):
        searcher.search("python", k=-3)


def test_search_ranking() -> None:
    searcher = _make_searcher(
        [
            Document(stem="doc_a", queries=["python machine learning"]),
            Document(
                stem="doc_b", queries=["java web framework machine learning tutorial"]
            ),
        ]
    )
    results = searcher.search("machine learning", k=-1)
    assert len(results) == 2
    assert results[0].stem == "doc_a"
    assert results[0].score > results[1].score


def test_search_k_zero_returns_empty() -> None:
    searcher = _make_searcher(
        [
            Document(stem="doc_a", queries=["python tutorial"]),
            Document(stem="doc_b", queries=["java tutorial"]),
        ]
    )
    assert searcher.search("python", k=0) == []


def test_search_empty_query_returns_empty() -> None:
    searcher = _make_searcher(
        [
            Document(stem="doc_a", queries=["python tutorial"]),
        ]
    )
    assert searcher.search("", k=5) == []


def test_index_replaces_previous() -> None:
    searcher = _make_searcher(
        [
            Document(stem="doc_a", queries=["aaa"]),
            Document(stem="doc_b", queries=["bbb"]),
        ]
    )
    assert len(searcher.search("aaa", k=-1)) == 1
    searcher.index([Document(stem="doc_c", queries=["ccc"])])
    results = searcher.search("ccc", k=-1)
    assert len(results) == 1
    assert results[0].stem == "doc_c"
