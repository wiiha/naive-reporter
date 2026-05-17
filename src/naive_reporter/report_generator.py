"""Report generation pipeline: prompt -> queries -> search -> report -> bullets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from naive_reporter.config import settings
from naive_reporter.search_engine import SearchEngine
from naive_reporter.types import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYNTHETIC_QUERY_PROMPT = (
    "You are given a user request. Generate exactly 5 short search queries "
    "(one per line, no numbering) that would help find documents relevant to "
    "this request.\n"
    "\n"
    "User request:\n"
    "---\n"
    "{prompt}\n"
    "---\n"
    "\n"
    "Search queries:\n"
)

_REPORT_GENERATION_PROMPT = (
    "You are a research analyst. Write a detailed report that answers the "
    "user's request based solely on the provided documents. You must not use "
    "any outside knowledge, assumptions, or information not explicitly "
    "present in the documents. If the documents do not contain enough "
    "information to answer part of the request, state that explicitly.\n"
    "\n"
    "User request:\n"
    "---\n"
    "{prompt}\n"
    "---\n"
    "\n"
    "Documents:\n"
    "{documents}\n"
    "\n"
    "Report:\n"
)

_REPORT_RETRY_PROMPT = (
    "You are a research analyst. A previous version of this report was "
    "rejected by a reviewer. You must now produce a corrected report that "
    "addresses the feedback. Use only the provided documents — no outside "
    "knowledge.\n"
    "\n"
    "User request:\n"
    "---\n"
    "{prompt}\n"
    "---\n"
    "\n"
    "Documents:\n"
    "{documents}\n"
    "---\n"
    "\n"
    "Your previous report:\n"
    "---\n"
    "{previous_report}\n"
    "---\n"
    "\n"
    "Reviewer's feedback:\n"
    "---\n"
    "{feedback}\n"
    "---\n"
    "\n"
    "Please write a corrected report that fully addresses the feedback.\n"
    "\n"
    "Corrected report:\n"
)

_REPORT_VALIDATION_PROMPT = (
    "You are a critical reviewer. Determine whether the report is true, "
    "fulfills the user's request, and uses ONLY knowledge from the provided "
    "documents with no outside information or hallucination. Respond with "
    "exactly: VALID or INVALID:<reason>\n"
    "\n"
    "User request:\n"
    "---\n"
    "{prompt}\n"
    "---\n"
    "\n"
    "Documents:\n"
    "{documents}\n"
    "\n"
    "Report:\n"
    "---\n"
    "{report}\n"
    "---\n"
    "\n"
    "Your verdict:\n"
)

_BULLET_GENERATION_PROMPT = (
    "Summarize this report into at most 3 bullet points. Each bullet point "
    "must be at most 2 sentences and as concise as possible.\n"
    "\n"
    "Report:\n"
    "---\n"
    "{report}\n"
    "---\n"
    "\n"
    "Bullet points:\n"
)

_BULLET_RETRY_PROMPT = (
    "You are an editor. A previous version of this bullet summary was "
    "rejected by a reviewer. Produce a corrected bullet summary that "
    "fully addresses the feedback. Keep at most 3 bullet points, "
    "each at most 2 sentences.\n"
    "\n"
    "Report:\n"
    "---\n"
    "{report}\n"
    "---\n"
    "\n"
    "Your previous bullet summary:\n"
    "---\n"
    "{previous_bullets}\n"
    "---\n"
    "\n"
    "Reviewer's feedback:\n"
    "---\n"
    "{feedback}\n"
    "---\n"
    "\n"
    "Please write a corrected bullet summary.\n"
    "\n"
    "Corrected bullet points:\n"
)

_BULLET_VALIDATION_PROMPT = (
    "You are a critical reviewer. Determine whether this bullet summary "
    "accurately captures the key points of the report and follows the "
    "constraint of at most 3 bullets with at most 2 sentences each. Respond "
    "with exactly: VALID or INVALID:<reason>\n"
    "\n"
    "Report:\n"
    "---\n"
    "{report}\n"
    "---\n"
    "\n"
    "Bullet summary:\n"
    "---\n"
    "{bullets}\n"
    "---\n"
    "\n"
    "Your verdict:\n"
)

# Hard cap on document text to avoid huge prompts (same as query_generator.py)
_TEXT_CAP = 50000

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MatchedDocInfo:
    """A document matched by one or more synthetic queries."""

    stem: str
    summary: str
    query_scores: list[tuple[str, float]]


@dataclass
class DocContent:
    """Full text content of a document for LLM context."""

    stem: str
    text: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NoMatchError(RuntimeError):
    """Raised when no documents match the synthetic queries."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_report(
    prompt: str,
    k: int = 5,
    data_dir: str | None = None,
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

    # 1. Generate 5 synthetic queries from the prompt
    logger.info("Generating synthetic queries ...")
    queries = _generate_queries_from_prompt(prompt)
    logger.info("Got %d synthetic queries", len(queries))

    # 2. Build search engine and collect matched documents
    logger.info("Building search index ...")
    from naive_reporter.bm25_searcher import BM25Searcher

    search_engine = SearchEngine(BM25Searcher(), data_dir=str(root))
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
    report_dir = _create_report_dir(root)

    # 4. Generate and validate report
    report = _generate_with_validation(
        generate_fn=lambda prev, fb: _generate_report(
            prompt, doc_contents, previous=prev, feedback=fb
        ),
        validate_fn=lambda r: _validate_report(prompt, doc_contents, r),
        max_attempts=3,
        artifact_name="report",
        on_attempt=lambda attempt, result, is_valid, reason: _write_artifacts(
            report_dir, prompt, queries, matched_docs, result, "[not validated]"
        ),
    )
    logger.info("Report validated")

    # 5. Generate and validate bullets
    bullets = _generate_with_validation(
        generate_fn=lambda prev, fb: _generate_bullets(
            report, previous=prev, feedback=fb
        ),
        validate_fn=lambda b: _validate_bullets(report, b),
        max_attempts=3,
        artifact_name="bullet summary",
        on_attempt=lambda attempt, result, is_valid, reason: _write_artifacts(
            report_dir, prompt, queries, matched_docs, report, result
        ),
    )
    logger.info("Bullet summary validated")

    # 6. Write final artifacts (overwrites the last on_attempt write)
    _write_artifacts(report_dir, prompt, queries, matched_docs, report, bullets)

    return report_dir


# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------


def _generate_queries_from_prompt(prompt: str) -> list[str]:
    """Ask the query-generation LLM for 5 search queries based on a user prompt."""
    client = OpenAI(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
    )

    prompt_text = _SYNTHETIC_QUERY_PROMPT.format(prompt=prompt)
    logger.debug("Sending synthetic query prompt (%d chars)", len(prompt_text))

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You generate search queries."},
                {"role": "user", "content": prompt_text},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise RuntimeError(f"LLM query generation failed: {exc}") from exc

    raw = response.choices[0].message.content or ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    if len(lines) != 5:
        logger.error(
            "LLM returned %d queries instead of 5. Response:\n%s",
            len(lines),
            raw,
        )
        raise RuntimeError(f"LLM returned {len(lines)} queries instead of 5")

    logger.debug("Got 5 synthetic queries")
    return lines


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


def _generate_report(
    prompt: str,
    documents: list[DocContent],
    previous: str = "",
    feedback: str = "",
) -> str:
    """Ask llm_reporting to write a report based on prompt + documents.

    On retry, ``previous`` and ``feedback`` are included so the LLM
    can produce a corrected version.
    """
    config = settings.get_reporting_client_config()
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )

    docs_text = "\n\n".join(
        f"Document: {d.stem}\n---\n{d.text[:_TEXT_CAP]}\n---" for d in documents
    )

    if previous and feedback:
        prompt_text = _REPORT_RETRY_PROMPT.format(
            prompt=prompt,
            documents=docs_text,
            previous_report=previous,
            feedback=feedback,
        )
    else:
        prompt_text = _REPORT_GENERATION_PROMPT.format(
            prompt=prompt,
            documents=docs_text,
        )

    logger.debug("Sending report generation prompt (%d chars)", len(prompt_text))

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a research analyst who writes "
                    "detailed reports.",
                },
                {"role": "user", "content": prompt_text},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise RuntimeError(f"LLM report generation failed: {exc}") from exc

    report = (response.choices[0].message.content or "").strip()
    if not report:
        raise RuntimeError("LLM returned empty report")

    logger.debug("Got report (%d chars)", len(report))
    return report


def _validate_report(
    prompt: str,
    documents: list[DocContent],
    report: str,
) -> tuple[bool, str]:
    """Ask llm_reporting to validate the report. Returns (is_valid, reason)."""
    config = settings.get_reporting_client_config()
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )

    docs_text = "\n\n".join(
        f"Document: {d.stem}\n---\n{d.text[:_TEXT_CAP]}\n---" for d in documents
    )
    prompt_text = _REPORT_VALIDATION_PROMPT.format(
        prompt=prompt,
        documents=docs_text,
        report=report,
    )
    logger.debug("Sending report validation prompt (%d chars)", len(prompt_text))

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a critical reviewer who " "validates reports.",
                },
                {"role": "user", "content": prompt_text},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise RuntimeError(f"LLM report validation failed: {exc}") from exc

    raw = response.choices[0].message.content or ""
    return _parse_validation_response(raw)


def _generate_bullets(
    report: str,
    previous: str = "",
    feedback: str = "",
) -> str:
    """Ask llm_reporting to summarize the report into 3 short bullets.

    On retry, ``previous`` and ``feedback`` are included so the LLM
    can produce a corrected version.
    """
    config = settings.get_reporting_client_config()
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )

    if previous and feedback:
        prompt_text = _BULLET_RETRY_PROMPT.format(
            report=report,
            previous_bullets=previous,
            feedback=feedback,
        )
    else:
        prompt_text = _BULLET_GENERATION_PROMPT.format(report=report)

    logger.debug("Sending bullet generation prompt (%d chars)", len(prompt_text))

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are an editor who creates concise "
                    "bullet summaries.",
                },
                {"role": "user", "content": prompt_text},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise RuntimeError(f"LLM bullet generation failed: {exc}") from exc

    bullets = (response.choices[0].message.content or "").strip()
    if not bullets:
        raise RuntimeError("LLM returned empty bullet summary")

    logger.debug("Got bullets (%d chars)", len(bullets))
    return bullets


def _validate_bullets(report: str, bullets: str) -> tuple[bool, str]:
    """Ask llm_reporting to validate the bullet summary."""
    config = settings.get_reporting_client_config()
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )

    prompt_text = _BULLET_VALIDATION_PROMPT.format(
        report=report,
        bullets=bullets,
    )
    logger.debug("Sending bullet validation prompt (%d chars)", len(prompt_text))

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a critical reviewer who "
                    "validates summaries.",
                },
                {"role": "user", "content": prompt_text},
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise RuntimeError(f"LLM bullet validation failed: {exc}") from exc

    raw = response.choices[0].message.content or ""
    return _parse_validation_response(raw)


def _parse_validation_response(raw: str) -> tuple[bool, str]:
    """Parse <VALID|INVALID>:reason from LLM response.

    Returns
    -------
    (is_valid, reason)
        ``is_valid`` is ``True`` when the status prefix is ``VALID``.
    """
    text = raw.strip()

    # The prompt explicitly instructs the LLM to return just <VALID> (no colon)
    # when the content is valid.
    valid_words = ["VALID", "<VALID>"]
    if text.upper() in valid_words:
        return True, ""

    if ":" not in text:
        return False, f"Malformed validation response (no colon): {text[:200]}"

    status, reason = text.split(":", 1)
    status = status.strip().upper()
    is_valid = status in valid_words
    return is_valid, reason.strip()


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


def _create_report_dir(data_dir: Path) -> Path:
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


def _write_artifacts(
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
