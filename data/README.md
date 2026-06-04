# Data

Raw TTC delay workbooks are not committed to this repository. Place downloaded files under mode-specific folders:

```text
data/raw/
  bus/
    *.xlsx
  streetcar/
    *.xlsx
```

The cleaner can read `.xlsx`, `.xlsm`, `.xls`, `.xlsb`, and `.csv` files where the required reader engine is installed. Multi-sheet Excel workbooks are combined automatically.

Run the cleaning pipeline from the repository root:

```bash
python -m src.data.clean_data \
  --bus-raw-dir data/raw/bus \
  --streetcar-raw-dir data/raw/streetcar \
  --processed-dir data/processed
```

You can also run only one mode by passing only `--bus-raw-dir` or only `--streetcar-raw-dir`.

Generated files:

- `data/processed/bus_delays_cleaned.csv`
- `data/processed/streetcar_delays_cleaned.csv`
- `data/processed/ttc_delays_cleaned.csv`

Raw and processed data files are gitignored. Weather enrichment and leakage-safe modeling features will be handled in later phases. `Min Gap` is retained in the cleaned data for auditing but is leakage-sensitive and should not be used in the main model unless later documentation justifies that it is available at incident report time.

