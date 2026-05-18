"""File I/O helpers for report generation."""

import logging
from datetime import date
from pathlib import Path

from naive_reporter.types import MatchedDocInfo

logger = logging.getLogger(__name__)


def create_report_dir(data_dir: Path) -> Path:
    """Create and return the next report directory for today.

    Uses atomic mkdir to avoid race conditions when multiple CLI
    invocations run concurrently.
    """
    today = date.today().isoformat()
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    next_id = _next_report_id(data_dir)
    while True:
        report_dir = reports_dir / f"{today}-{next_id}"
        try:
            report_dir.mkdir(parents=True, exist_ok=False)
            return report_dir
        except FileExistsError:
            next_id += 1


def _next_report_id(data_dir: Path) -> int:
    """Find the next incremental ID for today's date."""
    today = date.today().isoformat()
    reports_dir = data_dir / "reports"
    if not reports_dir.exists():
        return 0

    existing = [
        d.name for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith(today)
    ]
    if not existing:
        return 0

    # Extract numeric suffixes and take max + 1 for robustness
    suffixes: list[int] = []
    for name in existing:
        try:
            suffix = int(name.rsplit("-", 1)[1])
            suffixes.append(suffix)
        except (ValueError, IndexError):
            continue

    return max(suffixes) + 1 if suffixes else 0


def write_artifacts(
    report_dir: Path,
    prompt: str,
    queries: list[str],
    docs: list[MatchedDocInfo],
    report: str,
    bullets: str,
) -> None:
    """Write the 5 numbered files to the report directory."""
    # 1. Prompt
    (report_dir / "01_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    # 2. Synthetic queries
    (report_dir / "02_queries.txt").write_text(
        "\n".join(queries) + "\n", encoding="utf-8"
    )

    # 3. Documents with query scores
    lines: list[str] = []
    for d in docs:
        lines.append(f"stem: {d.stem}")
        lines.append(f"summary: {d.summary}")
        lines.append("matches:")
        for q, score in d.query_scores:
            lines.append(f"  - query: {q}")
            lines.append(f"    score: {score:.6f}")
        lines.append("")
    (report_dir / "03_documents.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # 4. Report
    (report_dir / "04_report.txt").write_text(report + "\n", encoding="utf-8")

    # 5. Bullets
    (report_dir / "05_bullets.txt").write_text(bullets + "\n", encoding="utf-8")

    logger.info("Wrote artifacts to %s", report_dir)
