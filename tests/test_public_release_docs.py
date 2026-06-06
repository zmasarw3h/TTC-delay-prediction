from pathlib import Path


def test_public_release_docs_exist():
    for path in [
        "README.md",
        ".env.example",
        "Makefile",
        "docs/README.md",
        "docs/model_card.md",
        "docs/technical_report.md",
        "docs/architecture.md",
        "docs/final_qa_checklist.md",
    ]:
        assert Path(path).exists(), path


def test_readme_prediction_example_uses_basic_fields_only():
    readme = Path("README.md").read_text(encoding="utf-8")
    example_start = readme.index("Example prediction request")
    example_end = readme.index("When `timestamp` is provided", example_start)
    example = readme[example_start:example_end]

    assert '"timestamp"' in example
    assert '"Min Delay"' not in example
    assert '"Min Gap"' not in example
    assert "prior_route_mean_delay" not in example
