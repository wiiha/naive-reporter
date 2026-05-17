"""Scan source directory and resolve name collisions."""

import hashlib
import logging
import shutil
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


def recover_orphaned_pdfs(data_dir: str | None = None) -> None:
    """Move PDFs from ``processed/`` back to ``source/`` if their hash is missing.

    Recomputes the SHA-256 of every ``*.pdf`` in ``data/processed/`` and checks
    whether a matching file exists in ``data/seen_hashes/``.  If the hash file
    is absent the PDF is moved back to ``data/source/`` so it will be picked up
    by the next :func:`scan` and re-ingested.

    Before moving, any existing text artifacts (``txt/``, ``summary_txt/``,
    ``queries_txt/``) belonging to the same stem are removed so they do not
    pollute downstream search after the PDF is re-ingested under a possibly
    different stem.

    Errors for individual files are logged and swallowed so that one bad file
    does not abort recovery.
    """
    root = Path(data_dir) if data_dir else Path(settings.data_dir)
    processed_dir = root / "processed"
    source_dir = root / "source"
    seen_hashes_dir = root / "seen_hashes"
    processed_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(processed_dir.glob("*.pdf"))
    if not pdfs:
        logger.debug("No PDFs in %s to recover", processed_dir)
        return

    logger.info(
        "Checking %d PDF(s) in %s for missing hashes …",
        len(pdfs),
        processed_dir,
    )
    moved = 0
    for pdf in pdfs:
        try:
            pdf_hash = _hash_file(pdf)
        except OSError as exc:
            logger.warning("Cannot read %s — skipping recovery: %s", pdf.name, exc)
            continue

        hash_file = seen_hashes_dir / f"{pdf_hash}.txt"
        if hash_file.is_file():
            continue

        stem = pdf.stem
        # Delete stale artifacts so they don't become ghost documents in search
        for subdir in ("txt", "summary_txt", "queries_txt"):
            artifact = root / subdir / f"{stem}.txt"
            if artifact.exists():
                try:
                    artifact.unlink()
                    logger.info("Removed stale artifact: %s", artifact)
                except OSError:
                    logger.warning("Could not remove stale artifact: %s", artifact)

        try:
            target = source_dir / pdf.name
            # If the same filename already exists in source (shouldn't happen
            # in normal operation, but defensively resolve a unique name).
            counter = 1
            original_target = target
            while target.exists():
                s = original_target.stem
                suffix = original_target.suffix
                target = source_dir / f"{s}_{counter}{suffix}"
                counter += 1
            shutil.move(str(pdf), str(target))
            logger.warning(
                "Moved orphaned PDF back to source (hash missing): %s → %s",
                pdf.name,
                target.name,
            )
            moved += 1
        except OSError as exc:
            logger.error("Failed to move orphaned PDF %s: %s", pdf.name, exc)

    if moved:
        logger.info("Recovered %d orphaned PDF(s) back to %s", moved, source_dir)
    else:
        logger.info(
            "All %d PDF(s) in %s have recorded hashes",
            len(pdfs),
            processed_dir,
        )


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
        try:
            pdf_hash = _hash_file(pdf)
        except OSError as exc:
            logger.warning("Cannot read %s — skipping: %s", pdf.name, exc)
            continue

        hash_file = seen_hashes_dir / f"{pdf_hash}.txt"
        if hash_file.is_file():
            logger.info("Skipping %s — hash already seen", pdf.name)
            continue

        stem = pdf.stem
        unique = _resolve_unique_stem(stem, processed_dir)
        result.append((pdf, unique, pdf_hash))
    return result
