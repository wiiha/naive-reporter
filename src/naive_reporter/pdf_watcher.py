"""Scan source directory and resolve name collisions."""

import hashlib
import logging
from pathlib import Path

from naive_reporter.config import settings

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_unique_stem(name: str, processed_dir: Path) -> str:
    """Find a unique stem that does not collide in processed_dir.

    If ``name`` already exists, append ``_1``, ``_2``, etc. until unique.
    """
    base = name
    counter = 1
    candidate = base
    while (processed_dir / f"{candidate}.pdf").exists():
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def scan(data_dir: str | None = None) -> list[tuple[Path, str, str]]:
    """Return a list of (source_pdf_path, resolved_stem, sha256) tuples.

    The ``resolved_stem`` is the file name *without* extension, guaranteed
    unique in ``data/processed/``.

    PDFs whose SHA-256 already exists in ``data/seen_hashes/`` are skipped.
    """
    root = Path(data_dir) if data_dir else Path(settings.data_dir)
    source_dir = root / "source"
    processed_dir = root / "processed"
    seen_hashes_dir = root / "seen_hashes"
    processed_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(source_dir.glob("*.pdf"))
    result: list[tuple[Path, str, str]] = []
    for pdf in pdfs:
        pdf_hash = _hash_file(pdf)
        hash_file = seen_hashes_dir / f"{pdf_hash}.txt"
        if hash_file.exists():
            logger.info("Skipping %s — hash already seen", pdf.name)
            continue

        stem = pdf.stem
        unique = _resolve_unique_stem(stem, processed_dir)
        result.append((pdf, unique, pdf_hash))
    return result
