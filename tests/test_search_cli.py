"""Tests for the search CLI."""

from naive_reporter.search_cli import main


def test_cli_no_results(capsys) -> None:
    rv = main(["xyz_nonexistent_term", "-k", "1", "--data-dir", "./data"])
    captured = capsys.readouterr()
    assert rv == 0
    assert "No results found." in captured.out


def test_cli_empty_query(capsys) -> None:
    rv = main(["", "--data-dir", "./data"])
    captured = capsys.readouterr()
    assert rv == 1
    assert "cannot be empty" in captured.err


def test_cli_happy_path(capsys) -> None:
    rv = main(["python tutorial", "-k", "1", "--data-dir", "./data"])
    captured = capsys.readouterr()
    assert rv == 0
    assert "python_intro" in captured.out


def test_cli_show_summary(capsys) -> None:
    rv = main(
        ["python tutorial", "-k", "1", "--data-dir", "./data", "--show-summary"]
    )
    captured = capsys.readouterr()
    assert rv == 0
    assert "Summary:" in captured.out


def test_cli_show_text(capsys) -> None:
    rv = main(
        ["python tutorial", "-k", "1", "--data-dir", "./data", "--show-text"]
    )
    captured = capsys.readouterr()
    assert rv == 0
    assert "Text:" in captured.out


def test_cli_k_minus_one(capsys) -> None:
    # Both mock documents contain "tutorial" in their query sets
    rv = main(["tutorial", "-k", "-1", "--data-dir", "./data"])
    captured = capsys.readouterr()
    assert rv == 0
    assert "python_intro" in captured.out
    assert "java_web" in captured.out
