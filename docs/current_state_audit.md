# Current State Audit

Source inspected: `TTC_Delays_Cleaned.ipynb`

## What The Notebook Currently Does

The notebook is an exploratory end-to-end prototype. It currently:

- Imports data science, modeling, weather, plotting, and explainability libraries.
- Reads TTC streetcar delay workbooks from a local downloads folder.
- Reads TTC bus delay workbooks from a separate local downloads folder.
- Normalizes some drifting column names across years.
- Parses report dates and times into a timestamp column named `ts`.
- Derives time features such as `hour`, `day_of_week`, `month`, and `is_holiday`.
- Fetches weather data with `meteostat`.
- Merges weather features into the bus and streetcar delay data by `Date` and `hour`.
- Creates separate bus and streetcar modeling frames.
- Engineers cyclical time features and a 7-day route-hour historical delay feature.
- Builds XGBoost pipelines with numeric, binary, one-hot categorical, and target-encoded categorical features.
- Runs Optuna tuning with `TimeSeriesSplit`.
- Evaluates model output with MAE, RMSE, and R2.
- Computes a route-hour 7-day rolling baseline.
- Generates plots for predictions, residuals, MAE comparison, feature importance, and SHAP summaries.

The notebook also contains some unrelated or unfinished exploratory work, including a local JSON events file read near the end.

## Datasets That Appear To Be Used

The notebook appears to use:

- TTC Streetcar Delays workbooks from `/Users/zmasarweh/Downloads/TTC Streetcar Delays Data`.
- TTC Bus Delays workbooks from `/Users/zmasarweh/Downloads/TTC Bus Delays Data`.
- Weather data fetched through `meteostat`, including hourly and daily data.
- Ontario statutory holidays through the `holidays` package.
- A local events JSON file from `/Users/zmasarweh/Downloads/testest/events_data.json`, although this appears unrelated to the current modeling flow.

Observed stored notebook output shows:

- Streetcar data: 147,225 rows from `2014-01-02 06:31:00` to `2024-12-31 22:32:00`.
- Bus data: 707,397 rows from `2014-01-01 00:23:00` to `2024-12-31 23:39:00`.

## Columns And Features That Appear To Be Used

Observed streetcar columns after weather merge include:

- `Date`
- `Route`
- `Day`
- `Location`
- `Incident`
- `Min Delay`
- `Min Gap`
- `Direction`
- `Vehicle`
- `ts`
- `hour`
- `day_of_week`
- `month`
- `is_holiday`
- `TOTAL_PRECIP`
- `snowfall_flag`
- `TOTAL_SNOW`
- `temp`

Additional engineered columns include:

- `day_of_year`
- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`
- `delay_ma7`

The model pipeline appears to use these feature groups:

- Numeric: `Min Gap`, `TOTAL_PRECIP`, `TOTAL_SNOW`, `temp`, `delay_ma7`.
- Binary: `snowfall_flag`, `is_holiday`.
- Cyclical: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`.
- One-hot categorical: `Route`, `Direction`.
- Target-encoded categorical: `Incident`, `Location`.

`Vehicle` is noted as optional in the notebook but does not appear to be included in the active `ColumnTransformer`.

## Target Variable

The target variable is `Min Delay`.

The notebook correctly drops rows with missing `Min Delay` before modeling. Stored outputs show zero missing `Min Delay` values in the streetcar modeling frame at the checked point.

## Modeling Approach Currently Present

The main active modeling approach is XGBoost regression using:

- `XGBRegressor(objective="reg:squarederror")`.
- `ColumnTransformer` preprocessing.
- `OneHotEncoder` for lower-cardinality categorical variables.
- `category_encoders.TargetEncoder` for `Incident` and `Location`.
- `TimeSeriesSplit` cross-validation.
- Optuna hyperparameter tuning.

There are also imports or fragments for:

- LightGBM.
- CatBoost.
- `train_test_split`.

These do not appear to be part of the clean final modeling path.

## Metrics Currently Present

The notebook includes:

- MAE.
- RMSE.
- R2.
- Baseline route-hour 7-day rolling MAE.
- Manual improvement calculations versus baseline.

Stored notebook outputs include:

- Streetcar Optuna CV best MAE around `3.03`.
- Streetcar route-hour 7-day baseline MAE around `15.34`.
- Bus Optuna CV best MAE around `3.17`.
- Bus test MAE around `3.33`, RMSE around `16.61`, and R2 around `0.891`.
- Bus route-hour 7-day baseline MAE around `16.38`.
- Manual improvement calculations of roughly `80.4%` and `84.2%` better than baseline.

These numbers should be treated as provisional until the pipeline is made reproducible and leakage-safe.

## Reproducibility And Execution Issues

The notebook has several reproducibility problems:

- Hard-coded absolute local paths point to `/Users/zmasarweh/Downloads/...`.
- Output filenames are written directly into the project root or current working directory, such as `consolidated_with_time_features4.xlsx`, `short_with_time_features.xlsx`, `w_strcar_df.csv`, and `mae_compare.png`.
- The configured `OUTFILE` says `ttc_delays_2014_2024.parquet`, but the streetcar block writes an Excel file instead.
- Bus and streetcar data-loading logic is duplicated with only small changes.
- Weather-fetching logic appears in multiple cells with overlapping approaches.
- Some cells depend on variables created many cells earlier, making clean restart execution fragile.
- There are duplicate checks, such as repeated missing-value checks.
- `df = None` appears mid-notebook and can break later assumptions if execution order changes.
- A placeholder `feature_cols = ['feature1', 'feature2', 'feature3']` appears in an unused split block.
- The notebook mixes streetcar and bus variable names, including a bus SHAP cell that transforms `X_test_b` with `best_pipe` instead of `best_pipe_b`.
- The final cells read an unrelated local events JSON file.
- Optuna uses deprecated `suggest_loguniform`; future implementation should use `suggest_float(..., log=True)`.

## Leakage And Modeling Risks

The notebook has some good leakage-aware intent, especially using `shift(1)` before rolling route-hour delay features. However, the current implementation still has risks:

- `Min Gap` is included as a numeric model input. It should be excluded unless it is confirmed to be known at incident report time.
- Target encoding for `Incident` and `Location` must be fit inside training folds only. The pipeline helps, but this must be verified after the data split design is finalized.
- The rolling `delay_ma7` feature is computed on the full sorted dataset before final train/test slicing. `shift(1)` prevents same-row leakage, but split-aware historical feature generation still needs to be formalized.
- Fallback filling for `delay_ma7` uses group means computed across the full dataset, which can leak future target information.
- Last-90-days testing is useful for exploration but does not match the recommended 2014-2022 / 2023 / 2024 split.
- Any preprocessing or imputation must be fit only on training data in the final pipeline.

## Inconsistencies With Resume-Style Claims

The notebook contains strong performance claims, including roughly 80%+ MAE improvement over baseline. These claims are not yet resume-safe because:

- The execution is not fully reproducible from a clean checkout.
- The data comes from hard-coded local paths.
- The evaluation split is not yet the planned fixed chronological split.
- Some target-derived feature fallback logic may leak future information.
- The model includes `Min Gap`, which may not be known at prediction time.
- There are manual metric values in cells that could drift from actual rerun results.

Future resume claims should be generated only after the pipeline is scripted, rerunnable, leakage-audited, and evaluated once on an untouched test set.

## What Should Be Kept

Keep and adapt:

- The goal of modeling `Min Delay`.
- The separate bus and streetcar data ingestion logic, but refactor it into shared reusable functions.
- Column normalization across historical TTC workbook formats.
- Timestamp parsing and chronological sorting.
- Time features such as hour, day, month, holiday, and cyclical encodings.
- Weather feature idea, after making the fetch/cache process reproducible.
- XGBoost as the main model candidate.
- MAE as the main metric.
- Baseline comparison using route-hour historical averages.
- SHAP explainability, after the final model and feature pipeline are stable.

## What Should Be Rewritten

Rewrite:

- Data loading into scripts with configurable input paths.
- Weather fetching into a reproducible cache step.
- Feature engineering into split-aware functions.
- Baseline calculation with explicit historical-only fallbacks.
- Train, validation, and test splitting into fixed chronological partitions.
- Model training and evaluation into reproducible Python modules.
- Plots and explainability into stable reporting scripts.

## What Should Be Removed

Remove or isolate:

- Hard-coded local user paths.
- Placeholder feature-list cells.
- Duplicate bus/streetcar code blocks.
- Duplicate missing-value checks.
- Unused LightGBM and CatBoost fragments unless they become explicit comparison models.
- Manual metric cells that hard-code results.
- Unrelated local events JSON exploration.
- Any root-level generated artifacts that are not part of the planned repo structure.
