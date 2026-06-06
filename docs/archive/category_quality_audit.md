# Category Quality Audit

This audit exists because frontend option lists exposed polluted category values, including direction options that looked like incident or location fragments. Phase 10C checks whether the modeling data categories are healthy enough for UI options and API input validation decisions before any further frontend polishing.

Run the audit from the repository root:

```bash
python -m src.data.audit_categories --input data/processed/modeling/modeling_dataset.csv --output-dir reports/category_audit
```

Optional arguments:

- `--input`: modeling CSV path. Defaults to `data/processed/modeling/modeling_dataset.csv`.
- `--output-dir`: report directory. Defaults to `reports/category_audit`.
- `--top-n`: number of top values to include per column. Defaults to `25`.

The script checks:

- `mode`
- `Route`
- `Direction`
- `Incident`
- `Location`

It writes:

- `category_summary.json`
- `category_top_values.csv`
- `category_suspicious_values.csv`
- `direction_value_audit.csv`
- `route_value_audit.csv`
- `incident_value_audit.csv`
- `location_value_audit.csv`

Suspicious values are column-specific. `Direction` treats normalized `N`, `E`, `S`, `W`, `B`, and `Unknown` values as healthy and flags long values, digits, lowercase variants, and incident/location words in raw or unnormalized data. `Route` accepts numeric route IDs and route variants such as `29`, `501`, `32A`, `504B`, and `RAD`, while flagging long non-route text. `Incident` treats the curated operational categories from `docs/categorical_normalization.md` as legitimate and flags route-like, location-like, very rare, and fragmented labels outside that policy. `Location` is expected to remain high-cardinality, so the audit focuses on missingness, top values, too-short values, route-like values, and incident-like values after safe deterministic cleanup.

This audit does not retrain models, modify model artifacts, or make performance claims. The reports should guide later cleanup decisions and verify that categorical normalization is reducing obvious category pollution without hiding high-cardinality location risk.
