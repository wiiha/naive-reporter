"""Tests for the report generator pipeline."""

from pathlib import Path

import pytest

from naive_reporter.report_generator import (
    MatchedDocInfo,
    NoMatchError,
    _collect_documents,
    _create_report_dir,
    _generate_with_validation,
    _next_report_id,
    _parse_validation_response,
    _read_document_texts,
    _write_artifacts,
    run_report,
)
from naive_reporter.search_engine import SearchEngine
from naive_reporter.types import SearchResult

# ---------------------------------------------------------------------------
# Fake Searcher for testing document collection
# ---------------------------------------------------------------------------


class FakeSearcher:
    """Fake Searcher that returns preconfigured results per query."""

    def __init__(self, results_map: dict[str, list[SearchResult]]) -> None:
        """results_map: {query_text: [SearchResult, ...]}."""
        self._results_map = results_map
        self._calls: list[tuple[str, int]] = []

    def index(self, documents: list) -> None:
        pass

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self._calls.append((query, k))
        return self._results_map.get(query, [])


# ---------------------------------------------------------------------------
# Tests for _collect_documents
# ---------------------------------------------------------------------------


class TestCollectDocuments:
    """Test union + dedup of search results across multiple queries."""

    def test_union_deduplicates_by_stem(self, tmp_path: Path) -> None:
        """Same doc matched by two queries appears once with both scores."""
        # Arrange: create summary file
        summary_dir = tmp_path / "summary_txt"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "doc1.txt").write_text("Summary of doc1", encoding="utf-8")

        fake = FakeSearcher(
            {
                "query_a": [SearchResult(stem="doc1", score=0.9)],
                "query_b": [SearchResult(stem="doc1", score=0.8)],
            }
        )
        engine = SearchEngine(fake, data_dir=str(tmp_path))
        engine.build_index()

        # Act
        docs = _collect_documents(
            queries=["query_a", "query_b"],
            search_engine=engine,
            data_dir=tmp_path,
            k=5,
        )

        # Assert
        assert len(docs) == 1
        assert docs[0].stem == "doc1"
        assert docs[0].summary == "Summary of doc1"
        assert len(docs[0].query_scores) == 2
        assert docs[0].query_scores[0] == ("query_a", 0.9)
        assert docs[0].query_scores[1] == ("query_b", 0.8)

    def test_empty_results_returns_empty_list(self, tmp_path: Path) -> None:
        """No matches -> empty list."""
        fake = FakeSearcher({})
        engine = SearchEngine(fake, data_dir=str(tmp_path))
        engine.build_index()

        docs = _collect_documents(
            queries=["q1", "q2"],
            search_engine=engine,
            data_dir=tmp_path,
            k=5,
        )

        assert docs == []

    def test_missing_summary_uses_placeholder(self, tmp_path: Path) -> None:
        """When summary_txt file is missing, use '[no summary]'."""
        fake = FakeSearcher(
            {"query_a": [SearchResult(stem="doc1", score=0.5)]}
        )
        engine = SearchEngine(fake, data_dir=str(tmp_path))
        engine.build_index()

        docs = _collect_documents(
            queries=["query_a"],
            search_engine=engine,
            data_dir=tmp_path,
            k=5,
        )

        assert len(docs) == 1
        assert docs[0].summary == "[no summary]"

    def test_different_docs_from_different_queries(self, tmp_path: Path) -> None:
        """Each query returns a different doc; union contains both."""
        summary_dir = tmp_path / "summary_txt"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "doc1.txt").write_text("Summary 1", encoding="utf-8")
        (summary_dir / "doc2.txt").write_text("Summary 2", encoding="utf-8")

        fake = FakeSearcher(
            {
                "query_a": [SearchResult(stem="doc1", score=0.9)],
                "query_b": [SearchResult(stem="doc2", score=0.8)],
            }
        )
        engine = SearchEngine(fake, data_dir=str(tmp_path))
        engine.build_index()

        docs = _collect_documents(
            queries=["query_a", "query_b"],
            search_engine=engine,
            data_dir=tmp_path,
            k=5,
        )

        assert len(docs) == 2
        stems = {d.stem for d in docs}
        assert stems == {"doc1", "doc2"}


# ---------------------------------------------------------------------------
# Tests for _parse_validation_response
# ---------------------------------------------------------------------------


class TestParseValidationResponse:
    """Test parsing of <VALID|INVALID>:reason strings."""

    @pytest.mark.parametrize(
        "raw,expected_valid,expected_reason",
        [
            ("VALID:looks good", True, "looks good"),
            ("INVALID:missing section", False, "missing section"),
            ("  VALID  :  reason  ", True, "reason"),
            ("INVALID", False, "Malformed validation response"),
            ("maybe: unsure", False, "unsure"),
            ("VALID", True, ""),
        ],
    )
    def test_parsing(
        self,
        raw: str,
        expected_valid: bool,
        expected_reason: str,
    ) -> None:
        is_valid, reason = _parse_validation_response(raw)
        assert is_valid == expected_valid
        assert expected_reason in reason


# ---------------------------------------------------------------------------
# Tests for _next_report_id and _create_report_dir
# ---------------------------------------------------------------------------


class TestNextReportId:
    """Test incremental ID assignment for report directories."""

    def test_first_id_is_zero(self, tmp_path: Path) -> None:
        data_dir = tmp_path
        report_id = _next_report_id(data_dir)
        assert report_id == 0

    def test_increments_existing(self, tmp_path: Path) -> None:
        data_dir = tmp_path
        reports_dir = data_dir / "reports"
        reports_dir.mkdir()
        from datetime import date

        today = date.today().isoformat()
        (reports_dir / f"{today}-0").mkdir()
        (reports_dir / f"{today}-1").mkdir()

        report_id = _next_report_id(data_dir)
        assert report_id == 2


class TestCreateReportDir:
    """Test report directory creation."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        data_dir = tmp_path
        report_dir = _create_report_dir(data_dir)
        assert report_dir.exists()
        assert report_dir.is_dir()
        from datetime import date

        today = date.today().isoformat()
        assert report_dir.name.startswith(today)

    def test_increments_when_directory_exists(self, tmp_path: Path) -> None:
        data_dir = tmp_path
        dir1 = _create_report_dir(data_dir)
        dir2 = _create_report_dir(data_dir)
        assert dir1 != dir2


# ---------------------------------------------------------------------------
# Tests for _generate_with_validation
# ---------------------------------------------------------------------------


class TestGenerateWithValidation:
    """Test retry loop for report and bullet generation."""

    def test_succeeds_first_try(self) -> None:
        def generate_fn(prev, fb):
            return "result"

        def validate_fn(r):
            return (True, "")

        result = _generate_with_validation(
            generate_fn, validate_fn, max_attempts=3, artifact_name="test"
        )
        assert result == "result"

    def test_succeeds_on_second_try(self) -> None:
        calls = []

        def generate_fn(prev, fb):
            calls.append(("gen", prev, fb))
            return "result"

        def validate_fn(r):
            if len(calls) == 1:
                return (False, "first attempt bad")
            return (True, "")

        result = _generate_with_validation(
            generate_fn, validate_fn, max_attempts=3, artifact_name="test"
        )
        assert result == "result"
        assert len(calls) == 2
        # Verify feedback was passed to retry
        assert calls[1][1] == "result"  # previous
        assert calls[1][2] == "first attempt bad"  # feedback

    def test_fails_after_max_attempts(self) -> None:
        def generate_fn(prev, fb):
            return "result"

        def validate_fn(r):
            return (False, "always bad")

        with pytest.raises(RuntimeError, match="validation failed after 3 attempts"):
            _generate_with_validation(
                generate_fn, validate_fn, max_attempts=3, artifact_name="test"
            )


# ---------------------------------------------------------------------------
# Tests for _read_document_texts
# ---------------------------------------------------------------------------


class TestReadDocumentTexts:
    """Test reading full text files for matched documents."""

    def test_reads_existing_texts(self, tmp_path: Path) -> None:
        txt_dir = tmp_path / "txt"
        txt_dir.mkdir()
        (txt_dir / "doc1.txt").write_text("Full text of doc1", encoding="utf-8")

        matched = [MatchedDocInfo(stem="doc1", summary="s", query_scores=[])]
        contents = _read_document_texts(matched, tmp_path)

        assert len(contents) == 1
        assert contents[0].stem == "doc1"
        assert contents[0].text == "Full text of doc1"

    def test_skips_missing_texts(self, tmp_path: Path) -> None:
        matched = [MatchedDocInfo(stem="doc1", summary="s", query_scores=[])]
        contents = _read_document_texts(matched, tmp_path)
        assert contents == []


# ---------------------------------------------------------------------------
# Tests for _write_artifacts
# ---------------------------------------------------------------------------


class TestWriteArtifacts:
    """Test writing the 5 numbered files."""

    def test_writes_all_files(self, tmp_path: Path) -> None:
        report_dir = tmp_path / "report"
        report_dir.mkdir()

        docs = [
            MatchedDocInfo(
                stem="doc1",
                summary="Summary 1",
                query_scores=[("q1", 0.9), ("q2", 0.8)],
            )
        ]

        _write_artifacts(
            report_dir=report_dir,
            prompt="test prompt",
            queries=["q1", "q2", "q3"],
            docs=docs,
            report="test report",
            bullets="- bullet 1",
        )

        assert (
            (report_dir / "01_prompt.txt").read_text(encoding="utf-8")
            == "test prompt\n"
        )
        assert (
            (report_dir / "02_queries.txt").read_text(encoding="utf-8")
            == "q1\nq2\nq3\n"
        )
        assert (
            (report_dir / "04_report.txt").read_text(encoding="utf-8")
            == "test report\n"
        )
        assert (
            (report_dir / "05_bullets.txt").read_text(encoding="utf-8")
            == "- bullet 1\n"
        )

        # Check documents file contains stem, summary, query, score
        docs_text = (report_dir / "03_documents.txt").read_text(encoding="utf-8")
        assert "stem: doc1" in docs_text
        assert "summary: Summary 1" in docs_text
        assert "query: q1" in docs_text
        assert "score: 0.900000" in docs_text


# ---------------------------------------------------------------------------
# Tests for run_report entry point (with mocked LLM)
# ---------------------------------------------------------------------------


class TestRunReport:
    """Integration-level tests for run_report with mocked dependencies."""

    def test_raises_no_match_when_no_documents(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When no documents match, NoMatchError is raised."""
        fake = FakeSearcher({})
        engine = SearchEngine(fake, data_dir=str(tmp_path))
        engine.build_index()

        # Monkeypatch _collect_documents to return empty via empty searcher
        # Actually we need to patch the build_index / search inside run_report
        # Since run_report instantiates its own SearchEngine, we mock
        # _generate_queries_from_prompt and the search engine
        def fake_generate_queries(prompt):
            return ["q1", "q2", "q3", "q4", "q5"]

        monkeypatch.setattr(
            "naive_reporter.report_generator._generate_queries_from_prompt",
            fake_generate_queries,
        )

        with pytest.raises(NoMatchError):
            run_report("test prompt", k=5, data_dir=str(tmp_path))
