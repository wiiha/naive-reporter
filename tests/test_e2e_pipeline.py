"""End-to-end pipeline test with mocked external services."""

from pathlib import Path
from unittest.mock import patch

from naive_reporter.pipeline import process_one


def test_pipeline_happy_path(tmp_path: Path) -> None:
    """Full flow: extract text, generate summary, generate queries, move PDF."""
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = source_dir / "hello.pdf"
    pdf_path.write_text("fake pdf content")

    data_dir = tmp_path

    with (
        patch(
            "naive_reporter.pipeline.convert_pdf",
            return_value="Extracted PDF text here.",
        ),
        patch(
            "naive_reporter.pipeline.generate_summary",
            return_value="This is a concise summary.",
        ),
        patch(
            "naive_reporter.pipeline.generate_queries",
            return_value=[
                "query one",
                "query two",
                "query three",
                "query four",
                "query five",
            ],
        ),
    ):
        process_one(pdf_path, "hello", data_dir, "testhash123")

    txt_path = data_dir / "txt" / "hello.txt"
    summary_path = data_dir / "summary_txt" / "hello.txt"
    queries_path = data_dir / "queries_txt" / "hello.txt"
    processed_pdf = data_dir / "processed" / "hello.pdf"
    hash_path = data_dir / "seen_hashes" / "testhash123.txt"

    assert txt_path.exists()
    assert txt_path.read_text() == "Extracted PDF text here."

    assert summary_path.exists()
    assert summary_path.read_text() == "This is a concise summary.\n"

    assert queries_path.exists()
    lines = [
        line.strip() for line in queries_path.read_text().splitlines() if line.strip()
    ]
    assert lines == [
        "query one",
        "query two",
        "query three",
        "query four",
        "query five",
    ]

    assert processed_pdf.exists()
    assert not pdf_path.exists()

    assert hash_path.exists()
    assert hash_path.read_text() == "hello.pdf"
