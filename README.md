# godi-ml

A reproducible evaluation pipeline for binary classifiers. You give it a CSV or Parquet file and a target column. It trains five models, compares them, checks calibration, and writes a versioned JSON report plus plots.

**Status: alpha (0.1.0a1).** The deterministic pipeline works and is tested. LLM sub-agents that review the report are planned but **not built yet**, see the roadmap below.

## What it does today

1. **Loads** CSV or Parquet and validates the target column.
2. **Profiles the data** (EDA): row and column counts, missing values, target balance, mutual information with the target, and a list of columns where every value is unique (typical IDs that should be excluded).
3. **Splits** the data 60/20/20 (train, validation, test), stratified, with a fixed seed. Scaling and encoding are fitted on the training split only.
4. **Trains five models:** logistic regression, random forest, histogram gradient boosting, XGBoost and LightGBM.
5. **Evaluates** each one: ROC AUC, PR AUC, Brier score, log loss, accuracy, precision, recall and the confusion matrix.
6. **Writes a report**: `metrics.json` (schema version 1.0.0) and PNG plots for ROC, precision-recall, calibration and the confusion matrix of the best model. Feature importance is added when the best model exposes it (tree models do, logistic regression does not).

## Quick start

It is not on PyPI yet, so run it from source:

```bash
git clone https://github.com/santiagopatinoserna/godi-ml
cd godi-ml
uv run godi-ml churn --data tests/fixtures/telco_churn.csv --target Churn
```

Options: `--runs-dir` (output folder, default `godi-ml-runs`), `--seed` (default 42), `--n-jobs` (default all cores).

### Example output

This is a real run on the 1,000-row slice of the IBM Telco churn dataset that ships in `tests/fixtures`, seed 42:

```
Leaderboard:
  logistic_regression        AUC=0.811  Brier=0.150  <- best
  xgboost                    AUC=0.790  Brier=0.174
  hist_gradient_boosting     AUC=0.778  Brier=0.190
  lightgbm                   AUC=0.773  Brier=0.202
  random_forest              AUC=0.759  Brier=0.182
```

The test split here has only 200 rows, so differences of a couple of points between models are within noise. Treat this as a demo of the output, not a benchmark.

## Design choices

- **Reproducible:** every random step uses the same seed, and a test checks that two trainings with the same seed give identical predictions.
- **No leakage from preprocessing:** transformers are fitted on the training split only.
- **Versioned output:** `metrics.json` carries a `schema_version`, so other tools can check what they are reading.
- **Checked:** 45 tests (unit and an end-to-end run on the Telco data), ruff, mypy, and CI on Python 3.11 to 3.13 for Linux and macOS.

## Known limitations

These are things I know are wrong or missing, listed on purpose:

- **The "best model" is chosen on the test split.** The pipeline creates a validation split but does not use it for ranking yet. That makes the leaderboard slightly optimistic. Selecting on validation and reporting test once is the next fix.
- **Only binary classification.** The one recipe is `churn`.
- **No hyperparameter search.** The five models use fixed settings.
- **The leakage check is basic.** It lists all-unique columns and reports mutual information, nothing more.

## Roadmap

- [ ] Select the best model on the validation split, report the test split once
- [ ] Stronger leakage checks (near-duplicate rows across splits, features that predict the target too well alone)
- [ ] Calibration and subgroup checks with thresholds that can fail a CI run
- [ ] LLM sub-agents that read `metrics.json` and write a plain-language review (the report schema is versioned for this purpose)
- [ ] MLflow export

## License

MIT, see [LICENSE](./LICENSE).
