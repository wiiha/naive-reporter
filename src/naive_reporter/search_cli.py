"""CLI entry point for document search."""

import argparse
import logging
import sys
from pathlib import Path

from naive_reporter.bm25_searcher import BM25Searcher
from naive_reporter.config import settings
from naive_reporter.search_engine import SearchEngine
from naive_reporter.semantic_searcher import SemanticSearcher

logger = logging.getLogger(__name__)


def _read_file(data_dir: Path, subdir: str, stem: str) -> str | None:
    """Read a text file under ``data_dir/subdir/stem.txt`` if it exists."""
    path = data_dir / subdir / f"{stem}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        logger.warning("Cannot read %s", path, exc_info=True)
        return None


def _make_searcher(name: str) -> BM25Searcher | SemanticSearcher:
    """Return a Searcher instance by name."""
    if name == "semantic":
        return SemanticSearcher()
    return BM25Searcher()


def main(argv: list[str] | None = None) -> int:
    """Run the search CLI."""
    parser = argparse.ArgumentParser(
        prog="naive-reporter-search",
        description="Search documents by pre-generated queries.",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "-k", type=int, default=5, help="Number of results (-1 for all)"
    )
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument(
        "--show-text", action="store_true", help="Print first 500 chars of full text"
    )
    parser.add_argument(
        "--show-summary", action="store_true", help="Print document summary"
    )
    parser.add_argument(
        "--searcher",
        choices=["bm25", "semantic"],
        default="bm25",
        help="Search backend (default: bm25)",
    )
    args = parser.parse_args(argv)

    if not args.query.strip():
        print("Error: query cannot be empty.", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    engine = SearchEngine(_make_searcher(args.searcher), data_dir=args.data_dir)
    engine.build_index()
    results = engine.search(args.query, k=args.k)

    if not results:
        print("No results found.")
        return 0

    data_dir = Path(args.data_dir)
    for r in results:
        print(f"\n{r.stem}  (score: {r.score:.3f})")

        if args.show_summary:
            text = _read_file(data_dir, "summary_txt", r.stem)
            if text is not None:
                print(f"  Summary: {text[:200]}")
            else:
                print("  Summary: [unreadable]")

        if args.show_text:
            text = _read_file(data_dir, "txt", r.stem)
            if text is not None:
                print(f"  Text: {text[:500]}")
            else:
                print("  Text: [unreadable]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
