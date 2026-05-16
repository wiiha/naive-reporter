"""Tests for pdf_watcher name collision resolution."""

from pathlib import Path

import pytest

from naive_reporter.pdf_watcher import scan


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data dir with source/processed/seen_hashes subdirectories."""
    source = tmp_path / "source"
    processed = tmp_path / "processed"
    seen_hashes = tmp_path / "seen_hashes"
    source.mkdir()
    processed.mkdir()
    seen_hashes.mkdir()
    return tmp_path


def test_scan_finds_pdfs(mock_data_dir: Path) -> None:
    """Two PDFs in source, none in processed or hashes → both returned."""
    (mock_data_dir / "source" / "a.pdf").write_text("pdf1")
    (mock_data_dir / "source" / "b.pdf").write_text("pdf2")

    results = scan(str(mock_data_dir))
    stems = [stem for _, stem, _ in results]
    assert stems == ["a", "b"]


def test_scan_resolves_collision(mock_data_dir: Path) -> None:
    """PDF already in processed → source PDF gets a suffix."""
    (mock_data_dir / "processed" / "a.pdf").write_text("old")
    (mock_data_dir / "source" / "a.pdf").write_text("new")

    results = scan(str(mock_data_dir))
    stems = [stem for _, stem, _ in results]
    assert stems == ["a_1"]


def test_scan_skips_seen_hash(mock_data_dir: Path) -> None:
    """PDF whose hash was already recorded → skipped."""
    pdf_path = mock_data_dir / "source" / "a.pdf"
    pdf_path.write_text("deadbeef")

    # pre-computed SHA-256 of b"deadbeef"
    hash_name = "2baf1f40105d9501fe319a8ec463fdf4325a2a5df445adf3f572f626253678c9.txt"
    (mock_data_dir / "seen_hashes" / hash_name).write_text("a.pdf")

    results = scan(str(mock_data_dir))
    assert results == []
