# Categorical Normalization

This patch adds deterministic categorical normalization before modeling feature generation. Phase 10 category audit found healthy `mode` values, mostly healthy but occasionally polluted `Route` values, dirty `Direction` variants, fragmented `Incident` labels, and very high-cardinality messy `Location` text.

Normalization is applied by `src.data.categorical_normalization` and called from `src.features.build_features` before time and historical features are created. It does not use `Min Delay`, `Min Gap`, future rows, target encodings, geocoding, or model outputs.

## Direction

`Direction` is strictly normalized to:

- `N`
- `E`
- `S`
- `W`
- `B`
- `Unknown`

Examples such as `n`, `NB`, `N/B`, and `north` become `N`. `B/W`, `BW`, `Both`, `Bothways`, and mixed bidirectional values become `B`. Garbage values, station names, incident labels, long text, and missing values become `Unknown`.

## Incident

`Incident` variants are mapped to curated operational categories, including:

- `Mechanical`
- `Utilized Off Route`
- `General Delay`
- `Late Leaving Garage`
- `Investigation`
- `Operations`
- `Operations - Operator`
- `Diversion`
- `Emergency Services`
- `Security`
- `Collision - TTC`
- `Collision - TTC Involved`
- `Road Blocked - NON-TTC Collision`
- `Held By`
- `Cleaning`
- `Cleaning - Unsanitary`
- `Vision`
- `Overhead`
- `Overhead - Pantograph`
- `Rail/Switches`
- `Unknown`
- `Other`

Missing/null-like labels become `Unknown`. Recognized variants such as `Mech`, `Ops`, `Securitty`, and `EMS` map to the closest curated category. Truly unrecognized non-missing labels become `Other` so the model sees one auditable bucket rather than many rare fragments.

## Route

`Route` normalization strips whitespace, converts integer-like values such as `29` and `29.0` to `29`, preserves route variants such as `32A`, `504B`, and `RAD`, and maps null-like values to `Unknown`.

Obvious non-route long text, location-like values, and incident-like values are mapped to `Unknown`. The rule is intentionally conservative: it avoids broad fuzzy matching and does not infer a route from location text.

## Location

`Location` receives safe deterministic text cleanup only:

- whitespace is stripped and repeated spaces are collapsed
- null-like values become `Unknown`
- `&`, `@`, and `/` become `AND`
- common abbreviations such as `STN`, `ST`, `AVE`, `RD`, `BLVD`, `DR`, `CRES`, `PKWY`, `HWY`, `W`, `E`, `N`, and `S` are expanded as standalone tokens
- output is uppercase for deterministic matching

The training pipeline does not fuzzy-snap locations, geocode, or canonicalize one location to another.

## Auditability

When feature building runs, normalized modeling columns are written as `mode`, `Route`, `Direction`, `Incident`, and `Location`. If present, raw source values are preserved as `Route_raw`, `Direction_raw`, `Incident_raw`, and `Location_raw`.

## Rebuild

After this patch, rebuild downstream generated files from the repository root:

```bash
python3 -m src.features.build_features \
  --input data/processed/ttc_delays_cleaned.csv \
  --output-dir data/processed/modeling \
  --max-delay-minutes 240
```

Then rerun audits and later modeling scripts as separate steps when needed. Do not commit generated datasets, reports, or model artifacts unless a later task explicitly requests that.
