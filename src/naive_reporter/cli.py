"""CLI entry point for naive-reporter."""

import argparse
import logging
import sys
from pathlib import Path

from naive_reporter.config import settings
from naive_reporter.pdf_watcher import scan
from naive_reporter.pipeline import process_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="naive-reporter",
        description="Ingest PDFs, convert to text, and generate search queries.",
    )
    parser.parse_args(argv)

    data_dir = Path(settings.data_dir)
    (data_dir / "source").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "txt").mkdir(parents=True, exist_ok=True)
    (data_dir / "summary_txt").mkdir(parents=True, exist_ok=True)
    (data_dir / "queries_txt").mkdir(parents=True, exist_ok=True)
    (data_dir / "seen_hashes").mkdir(parents=True, exist_ok=True)

    files = scan()
    if not files:
        logger.info("No new PDFs found in %s", data_dir / "source")
        return 0

    logger.info("Found %d PDF(s) to process", len(files))
    for pdf_path, stem, pdf_hash in files:
        try:
            process_one(pdf_path, stem, data_dir, pdf_hash)
        except Exception:
            logger.exception(
                "Failed to process %s — left in source for retry", pdf_path.name
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
