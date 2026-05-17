"""Tests for pdf_watcher name collision resolution."""

from pathlib import Path

import pytest

from naive_reporter.pdf_watcher import _hash_file, recover_orphaned_pdfs, scan


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

# ── Recovery tests ──


def test_recover_moves_orphan_without_hash(mock_data_dir: Path) -> None:
    """PDF in processed with no hash file → moved back to source."""
    pdf = mock_data_dir / "processed" / "paper.pdf"
    pdf.write_text("orphan")

    recover_orphaned_pdfs(str(mock_data_dir))

    assert not pdf.exists()
    assert (mock_data_dir / "source" / "paper.pdf").exists()


def test_recover_leaves_healthy_pdf_in_place(mock_data_dir: Path) -> None:
    """PDF in processed with matching hash file → stays in processed."""
    pdf = mock_data_dir / "processed" / "paper.pdf"
    pdf.write_text("healthy")
    pdf_hash = _hash_file(pdf)
    (mock_data_dir / "seen_hashes" / f"{pdf_hash}.txt").write_text("paper.pdf")

    recover_orphaned_pdfs(str(mock_data_dir))

    assert pdf.exists()
    assert not (mock_data_dir / "source" / "paper.pdf").exists()


def test_recover_respects_existing_source_file(mock_data_dir: Path) -> None:
    """If source already has a file with the same name, recovered PDF gets a suffix."""
    (mock_data_dir / "source" / "paper.pdf").write_text("already here")
    orphan = mock_data_dir / "processed" / "paper.pdf"
    orphan.write_text("orphan")

    recover_orphaned_pdfs(str(mock_data_dir))

    assert not orphan.exists()
    assert (mock_data_dir / "source" / "paper.pdf").exists()
    assert (mock_data_dir / "source" / "paper_1.pdf").exists()


def test_recover_logs_and_skips_unreadable_pdf(mock_data_dir: Path, caplog) -> None:
    """If a PDF in processed can't be read (directory with .pdf suffix), skip it."""
    bad_pdf = mock_data_dir / "processed" / "bad.pdf"
    bad_pdf.mkdir()  # a directory, not a file — _hash_file will fail

    with caplog.at_level("WARNING", logger="naive_reporter.pdf_watcher"):
        recover_orphaned_pdfs(str(mock_data_dir))

    assert "Cannot read" in caplog.text
    assert bad_pdf.exists()  # stays in place


def test_recover_is_idempotent(mock_data_dir: Path) -> None:
    """Running recovery twice on already-recovered state does nothing."""
    (mock_data_dir / "source" / "paper.pdf").write_text("already there")
    # No PDFs in processed, nothing to do
    recover_orphaned_pdfs(str(mock_data_dir))
    recover_orphaned_pdfs(str(mock_data_dir))
    assert (mock_data_dir / "source" / "paper.pdf").exists()


def test_recover_then_scan_finds_rescued_pdf(mock_data_dir: Path) -> None:
    """Recovery happens before scan: rescued PDF is re-ingested in the same run."""
    orphan = mock_data_dir / "processed" / "paper.pdf"
    orphan.write_text("rescue me")

    recover_orphaned_pdfs(str(mock_data_dir))
    results = scan(str(mock_data_dir))

    stems = [stem for _, stem, _ in results]
    assert "paper" in stems


def test_recover_with_processed_empty(mock_data_dir: Path, caplog) -> None:
    """Empty processed dir → no-op, no crash."""
    with caplog.at_level("DEBUG", logger="naive_reporter.pdf_watcher"):
        recover_orphaned_pdfs(str(mock_data_dir))

    assert "No PDFs" in caplog.text


def test_recover_cleans_up_stale_artifacts(mock_data_dir: Path) -> None:
    """When an orphan is recovered, its txt/summary/queries files are deleted."""
    stem = "paper"
    pdf = mock_data_dir / "processed" / f"{stem}.pdf"
    pdf.write_text("orphan")

    for subdir in ("txt", "summary_txt", "queries_txt"):
        (mock_data_dir / subdir).mkdir(parents=True, exist_ok=True)
        (mock_data_dir / subdir / f"{stem}.txt").write_text("stale")

    recover_orphaned_pdfs(str(mock_data_dir))

    assert not pdf.exists()
    assert (mock_data_dir / "source" / "paper.pdf").exists()
    for subdir in ("txt", "summary_txt", "queries_txt"):
        assert not (mock_data_dir / subdir / f"{stem}.txt").exists()


def test_scan_skips_unreadable_source_file(mock_data_dir: Path, caplog) -> None:
    """scan() logs a warning and skips a .pdf in source that cannot be read."""
    bad = mock_data_dir / "source" / "bad.pdf"
    bad.mkdir()  # directory masquerading as a PDF
    (mock_data_dir / "source" / "good.pdf").write_text("good")

    with caplog.at_level("WARNING", logger="naive_reporter.pdf_watcher"):
        results = scan(str(mock_data_dir))

    stems = [stem for _, stem, _ in results]
    assert stems == ["good"]
    assert "Cannot read bad.pdf" in caplog.text


def test_scan_uses_is_file_for_hash_check(mock_data_dir: Path) -> None:
    """A directory named {hash}.txt is not treated as healthy."""
    pdf = mock_data_dir / "source" / "a.pdf"
    pdf.write_text("unique")
    pdf_hash = _hash_file(pdf)

    # Create a *directory* with the hash name — must NOT match
    hash_dir = mock_data_dir / "seen_hashes" / f"{pdf_hash}.txt"
    hash_dir.mkdir(parents=True, exist_ok=True)

    results = scan(str(mock_data_dir))
    stems = [stem for _, stem, _ in results]
    assert "a" in stems


def test_recover_uses_is_file_for_hash_check(mock_data_dir: Path) -> None:
    """A directory named {hash}.txt is not treated as healthy."""
    pdf = mock_data_dir / "processed" / "a.pdf"
    pdf.write_text("unique")
    pdf_hash = _hash_file(pdf)

    hash_dir = mock_data_dir / "seen_hashes" / f"{pdf_hash}.txt"
    hash_dir.mkdir(parents=True, exist_ok=True)

    recover_orphaned_pdfs(str(mock_data_dir))

    assert not pdf.exists()
    assert (mock_data_dir / "source" / "a.pdf").exists()


def test_recover_idempotent_with_real_orphan(mock_data_dir: Path) -> None:
    """Running recovery twice on the same orphan state is a no-op the second time."""
    pdf = mock_data_dir / "processed" / "x.pdf"
    pdf.write_text("orphan")

    recover_orphaned_pdfs(str(mock_data_dir))
    assert not pdf.exists()
    assert (mock_data_dir / "source" / "x.pdf").exists()

    # second run — source has x.pdf, processed is empty
    recover_orphaned_pdfs(str(mock_data_dir))
    assert (mock_data_dir / "source" / "x.pdf").exists()
