"""CLI entry point for the report component."""

import argparse
import logging
import sys

from naive_reporter.config import settings
from naive_reporter.report_pipeline import NoMatchError, run_report

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the report CLI."""
    parser = argparse.ArgumentParser(
        prog="naive-reporter-report",
        description="Generate a report from documents matching a user prompt.",
    )
    parser.add_argument(
        "prompt",
        help="The user prompt to base the report on",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results per synthetic query (default: 5)",
    )
    parser.add_argument(
        "--data-dir",
        default=settings.data_dir,
        help="Data directory",
    )
    args = parser.parse_args(argv)

    if not args.prompt.strip():
        print("Error: prompt cannot be empty.", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    try:
        report_dir = run_report(
            prompt=args.prompt,
            k=args.k,
            data_dir=args.data_dir,
        )
        print(f"Report saved to: {report_dir}")
        return 0
    except NoMatchError:
        logger.warning("No documents matched your prompt")
        return 0
    except RuntimeError as exc:
        logger.error("Report generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
