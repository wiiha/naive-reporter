"""Tests for the report CLI."""

import pytest

from naive_reporter.report_cli import main


class TestReportCli:
    """Tests for naive-reporter-report CLI."""

    def test_empty_prompt_exits_with_error(self, capsys) -> None:
        """Empty prompt should print error to stderr and exit 1."""
        result = main([""])
        assert result == 1
        captured = capsys.readouterr()
        assert "empty" in captured.err

    def test_help_prints_usage(self, capsys) -> None:
        """--help should print usage and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_whitespace_only_prompt_exits_with_error(self, capsys) -> None:
        """Whitespace-only prompt should be rejected."""
        result = main(["   "])
        assert result == 1
        captured = capsys.readouterr()
        assert "empty" in captured.err

    def test_valid_prompt_calls_run_report(self, monkeypatch, capsys) -> None:
        """Valid prompt should call run_report and print directory path."""
        from pathlib import Path

        fake_dir = Path("/fake/report/dir")

        def fake_run_report(prompt, k, data_dir):
            assert prompt == "What is the weather?"
            assert k == 5
            return fake_dir

        monkeypatch.setattr(
            "naive_reporter.report_cli.run_report",
            fake_run_report,
        )

        result = main(["What is the weather?"])
        assert result == 0
        captured = capsys.readouterr()
        assert str(fake_dir) in captured.out

    def test_no_match_prints_warning(
        self, monkeypatch, caplog
    ) -> None:
        """When no documents match, print warning and exit 0."""
        from naive_reporter.report_pipeline import NoMatchError

        def fake_run_report(*args, **kwargs):
            raise NoMatchError("No documents matched")

        monkeypatch.setattr(
            "naive_reporter.report_cli.run_report",
            fake_run_report,
        )

        with caplog.at_level("WARNING"):
            result = main(["some prompt"])

        assert result == 0
        assert "No documents matched" in caplog.text

    def test_runtime_error_prints_error(
        self, monkeypatch, caplog
    ) -> None:
        """When report generation fails, print error and exit 1."""
        def fake_run_report(*args, **kwargs):
            raise RuntimeError("LLM failed")

        monkeypatch.setattr(
            "naive_reporter.report_cli.run_report",
            fake_run_report,
        )

        with caplog.at_level("ERROR"):
            result = main(["some prompt"])

        assert result == 1
        assert "failed" in caplog.text
