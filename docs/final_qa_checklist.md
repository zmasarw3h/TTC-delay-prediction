# Final QA Checklist

Use this before making the repo public or linking it from a resume.

- [ ] `pytest` passes.
- [ ] API starts locally with `uvicorn src.api.app:app --reload`.
- [ ] Frontend loads at `http://127.0.0.1:8000/`.
- [ ] `/predict-delay` works with basic fields only: `mode`, `Route`, `Direction`, `Incident`, `Location`, `timestamp`.
- [ ] `/historical-lookup-info` returns the configured local historical CSV status.
- [ ] No generated data, raw data, model artifacts, or generated reports are committed.
- [ ] README metrics match the latest clean scripted reports.
- [ ] Docs do not mention stale notebook-only workflow as the primary implementation.
- [ ] Docs do not mention stale pre-normalization metrics as final results.
- [ ] Screenshots under `docs/images/` are updated if available.
- [ ] `git status --short` is clean before publishing.
- [ ] Public repo privacy check is complete: no secrets, local usernames in generated outputs, raw TTC files, or private scratch notes.

Useful commands:

```bash
pytest
python3 - <<'PY'
from pathlib import Path
from src.api.app import app
assert app.title
for path in ["README.md", "docs/model_card.md", "docs/technical_report.md"]:
    assert Path(path).exists(), path
PY
git status --short
```
