"""Tests for SearchEngine and load_documents."""

from pathlib import Path

from naive_reporter.bm25_searcher import BM25Searcher
from naive_reporter.search_engine import SearchEngine, load_documents


def test_load_documents_reads_query_files(tmp_path: Path) -> None:
    queries_dir = tmp_path / "queries_txt"
    queries_dir.mkdir()
    (queries_dir / "doc_a.txt").write_text("query one\nquery two\n")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].stem == "doc_a"
    assert docs[0].queries == ["query one", "query two"]


def test_load_documents_skips_unreadable_oserror(tmp_path: Path) -> None:
    queries_dir = tmp_path / "queries_txt"
    queries_dir.mkdir()
    (queries_dir / "good.txt").write_text("hello world")
    # Create a directory named like a .txt file to trigger an OSError
    (queries_dir / "bad.txt").mkdir()
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].stem == "good"


def test_load_documents_skips_unreadable_decode_error(tmp_path: Path) -> None:
    queries_dir = tmp_path / "queries_txt"
    queries_dir.mkdir()
    (queries_dir / "good.txt").write_text("hello world")
    # Write binary garbage to trigger UnicodeDecodeError
    (queries_dir / "bad.txt").write_bytes(b"\xff\xfe\x00\x00")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].stem == "good"


def test_load_documents_missing_dir_returns_empty(tmp_path: Path) -> None:
    docs = load_documents(str(tmp_path))
    assert docs == []


def test_load_documents_ignores_empty_lines(tmp_path: Path) -> None:
    queries_dir = tmp_path / "queries_txt"
    queries_dir.mkdir()
    (queries_dir / "doc.txt").write_text("first\n\n  \nsecond\n")
    docs = load_documents(str(tmp_path))
    assert docs[0].queries == ["first", "second"]


def test_engine_end_to_end(tmp_path: Path) -> None:
    queries_dir = tmp_path / "queries_txt"
    queries_dir.mkdir()
    (queries_dir / "doc_a.txt").write_text("python machine learning")
    (queries_dir / "doc_b.txt").write_text("java web framework")
    engine = SearchEngine(BM25Searcher(), data_dir=str(tmp_path))
    engine.build_index()
    results = engine.search("machine learning", k=1)
    assert len(results) == 1
    assert results[0].stem == "doc_a"
    assert results[0].score > 0


def test_engine_k_minus_one(tmp_path: Path) -> None:
    queries_dir = tmp_path / "queries_txt"
    queries_dir.mkdir()
    (queries_dir / "doc_a.txt").write_text("aaa bbb")
    (queries_dir / "doc_b.txt").write_text("ccc ddd")
    engine = SearchEngine(BM25Searcher(), data_dir=str(tmp_path))
    engine.build_index()
    results = engine.search("aaa", k=-1)
    # Only doc_a matches; doc_b gets score 0 and is filtered out
    assert len(results) == 1
    assert results[0].stem == "doc_a"
