"""Orchestrate processing of a single PDF end-to-end."""

import logging
import shutil
from pathlib import Path

from naive_reporter.docling_client import convert_pdf
from naive_reporter.query_generator import generate_queries
from naive_reporter.summary_generator import generate_summary

logger = logging.getLogger(__name__)


def process_one(
    source_pdf: Path,
    stem: str,
    data_dir: Path,
    pdf_hash: str,
) -> None:
    """Process one PDF file: extract, generate summary + queries, move to processed.

    Parameters
    ----------
    source_pdf
        Path to the PDF in ``data/source/``.
    stem
        Resolved unique name (no extension) for all downstream files.
    data_dir
        Root data directory.
    pdf_hash
        SHA-256 hex digest of the PDF contents.
    """
    txt_path = data_dir / "txt" / f"{stem}.txt"
    summary_path = data_dir / "summary_txt" / f"{stem}.txt"
    queries_path = data_dir / "queries_txt" / f"{stem}.txt"
    processed_pdf = data_dir / "processed" / f"{stem}.pdf"

    # 1. Extract text via docling
    logger.info("Converting %s via docling …", source_pdf.name)
    text = convert_pdf(source_pdf)

    # 2. Write text file (INV-002)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", txt_path)

    # 3. Generate summary from text
    logger.info("Generating summary for %s …", stem)
    summary = generate_summary(text)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary + "\n", encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    # 4. Generate queries from text
    logger.info("Generating queries for %s …", stem)
    queries = generate_queries(text)

    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text("\n".join(queries) + "\n", encoding="utf-8")
    logger.info("Wrote %s", queries_path)

    # 5. Move PDF to processed (INV-001, INV-005: only after success)
    processed_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_pdf), str(processed_pdf))
    logger.info("Moved %s → %s", source_pdf.name, processed_pdf)

    # 6. Record hash
    hash_path = data_dir / "seen_hashes" / f"{pdf_hash}.txt"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(source_pdf.name, encoding="utf-8")
    logger.info("Recorded hash for %s", source_pdf.name)
