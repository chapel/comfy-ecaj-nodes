"""Structural regression checks for the repository's CI workflow."""

from pathlib import Path


# AC: @ci-pipeline ac-1
def test_ci_runs_on_unfiltered_push_and_pull_request():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    before_jobs, separator, _ = workflow.partition("\njobs:\n")
    assert separator, "Expected the CI jobs block"
    _, separator, triggers = before_jobs.partition("\non:\n")
    assert separator, "Expected the CI event block"
    # Pin this small, literal block: neither event may acquire branch/path filters.
    # This is intentionally not a general YAML parser and needs no YAML dependency.
    assert [line for line in triggers.splitlines() if line.strip()] == [
        "  push:",
        "  pull_request:",
    ], "CI must run on pushes to any branch and on pull requests without filters"
