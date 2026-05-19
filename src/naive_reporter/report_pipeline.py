"""Report generation pipeline: prompt -> queries -> search -> report -> bullets."""

import logging
from collections.abc import Callable
from pathlib import Path

from naive_reporter.bm25_searcher import BM25Searcher
from naive_reporter.config import settings
from naive_reporter.report_artifacts import create_report_dir, write_artifacts
from naive_reporter.report_generation import (
    generate_bullets,
    generate_report,
    validate_bullets,
    validate_report,
)
from naive_reporter.search_engine import SearchEngine
from naive_reporter.search_protocol import Searcher
from naive_reporter.synthetic_query_generator import (
    generate_queries as generate_synthetic_queries,
)
from naive_reporter.types import DocContent, MatchedDocInfo, SearchResult

logger = logging.getLogger(__name__)


class NoMatchError(RuntimeError):
    """Raised when no documents match the synthetic queries."""


def run_report(
    prompt: str,
    k: int = 5,
    data_dir: str | None = None,
    searcher: Searcher | None = None,
) -> Path:
    """Run the full report pipeline and return the report directory path.

    Raises
    ------
    ValueError
        If ``prompt`` is empty or whitespace-only.
    RuntimeError
        If report or bullet validation fails after 3 attempts.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    root = Path(data_dir) if data_dir else Path(settings.data_dir)
    _searcher = searcher if searcher is not None else BM25Searcher()

    # 1. Generate synthetic queries from the prompt
    logger.info("Generating synthetic queries ...")
    queries = generate_synthetic_queries(prompt)
    logger.info("Got %d synthetic queries", len(queries))

    # 2. Build search engine and collect matched documents
    logger.info("Building search index ...")
    search_engine = SearchEngine(_searcher, data_dir=str(root))
    search_engine.build_index()

    matched_docs = _collect_documents(queries, search_engine, root, k)
    if not matched_docs:
        logger.warning("No documents matched the synthetic queries")
        raise NoMatchError("No documents matched")

    logger.info("Matched %d unique document(s)", len(matched_docs))

    # 3. Read full text for each matched document
    doc_contents = _read_document_texts(matched_docs, root)
    if not doc_contents:
        logger.warning(
            "All matched documents are missing text files; " "cannot generate report"
        )
        raise NoMatchError("No readable document texts found")

    # Create report directory early so artifacts can be written even on failure
    report_dir = create_report_dir(root)

    # 4. Generate and validate report
    report = _generate_with_validation(
        generate_fn=lambda prev, fb: generate_report(
            prompt, doc_contents, previous=prev, feedback=fb
        ),
        validate_fn=lambda r: validate_report(prompt, doc_contents, r),
        max_attempts=3,
        artifact_name="report",
        on_attempt=lambda attempt, result, is_valid, reason: write_artifacts(
            report_dir, prompt, queries, matched_docs, result, "[not validated]"
        ),
    )
    logger.info("Report validated")

    # 5. Generate and validate bullets
    bullets = _generate_with_validation(
        generate_fn=lambda prev, fb: generate_bullets(
            report, previous=prev, feedback=fb
        ),
        validate_fn=lambda b: validate_bullets(report, b),
        max_attempts=3,
        artifact_name="bullet summary",
        on_attempt=lambda attempt, result, is_valid, reason: write_artifacts(
            report_dir, prompt, queries, matched_docs, report, result
        ),
    )
    logger.info("Bullet summary validated")

    # 6. Write final artifacts (overwrites the last on_attempt write)
    write_artifacts(report_dir, prompt, queries, matched_docs, report, bullets)

    return report_dir


def _collect_documents(
    queries: list[str],
    search_engine: SearchEngine,
    data_dir: Path,
    k: int,
) -> list[MatchedDocInfo]:
    """Run each query through SearchEngine, union results, dedup by stem."""
    stem_to_info: dict[str, MatchedDocInfo] = {}

    for query in queries:
        results: list[SearchResult] = search_engine.search(query, k=k)
        for r in results:
            if r.stem not in stem_to_info:
                summary_path = data_dir / "summary_txt" / f"{r.stem}.txt"
                try:
                    summary = (
                        summary_path.read_text(encoding="utf-8").strip()
                        if summary_path.exists()
                        else "[no summary]"
                    )
                except (OSError, UnicodeDecodeError):
                    logger.warning(
                        "Cannot read summary for %s, using placeholder",
                        r.stem,
                    )
                    summary = "[no summary]"
                stem_to_info[r.stem] = MatchedDocInfo(
                    stem=r.stem,
                    summary=summary,
                    query_scores=[],
                )
            stem_to_info[r.stem].query_scores.append((query, r.score))

    return list(stem_to_info.values())


def _read_document_texts(
    matched_docs: list[MatchedDocInfo],
    data_dir: Path,
) -> list[DocContent]:
    """Read full text for each matched document from data/txt/."""
    contents: list[DocContent] = []
    for doc in matched_docs:
        txt_path = data_dir / "txt" / f"{doc.stem}.txt"
        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8")
                contents.append(DocContent(stem=doc.stem, text=text))
            except (OSError, UnicodeDecodeError):
                logger.warning(
                    "Cannot read text file for %s, skipping",
                    doc.stem,
                )
        else:
            logger.warning(
                "Missing text file for %s, skipping from report context",
                doc.stem,
            )
    return contents


def _generate_with_validation(
    generate_fn: Callable[[str, str], str],
    validate_fn: Callable[[str], tuple[bool, str]],
    max_attempts: int,
    artifact_name: str,
    on_attempt: Callable[[int, str, bool, str], None] | None = None,
) -> str:
    """Generate -> validate -> retry up to max_attempts.

    On attempt 1 ``generate_fn`` is called with ``("", "")``.
    On retries it receives ``(previous_result, feedback_reason)`` so the
    generator can include the feedback in the next prompt.

    Parameters
    ----------
    generate_fn
        ``f(previous, feedback) -> str``.
    on_attempt
        Optional callback invoked after every validation attempt with
        ``(attempt_number, result, is_valid, reason)``.
    """
    previous = ""
    last_reason = ""
    for attempt in range(1, max_attempts + 1):
        result = generate_fn(previous, last_reason)
        is_valid, reason = validate_fn(result)
        if on_attempt is not None:
            on_attempt(attempt, result, is_valid, reason)
        if is_valid:
            return result
        previous = result
        last_reason = reason
        logger.warning(
            "%s invalid (attempt %d/%d): %s",
            artifact_name,
            attempt,
            max_attempts,
            reason,
        )

    raise RuntimeError(
        f"{artifact_name} validation failed after {max_attempts} attempts: "
        f"{last_reason}"
    )
