# D90 Cohort Revenue Forecasting

**A reproducible time-series ML case study: maturity-aware historical multipliers, Ridge regression, hybrid forecasting, chronological evaluation, and a daily prediction pipeline.**

> **Portfolio / demonstration repository.** The included `examples/synthetic_cohorts.csv` is **entirely synthetic**. It contains no real app, advertiser, customer, or company revenue data. Real-world performance metrics shown in `docs/RESULTS.md` were supplied from a separate private analysis; they are illustrative of that completed experiment and **cannot be reproduced from the synthetic CSV**. Check permission before publishing even aggregate private business results.

## TL;DR

- **Goal:** Predict cumulative day-90 (`D90`) revenue separately for each **install date × platform** using information available by day 1 (`D1`).
- **Baseline:** `D90/D1 = (D7/D1) × (D28/D7) × (D90/D28)`, with revenue-weighted, maturity-aware historical windows of 30, 60, and 120 cohort days.
- **ML:** A pooled Ridge regression (`alpha=500`) learns log corrections to each revenue-multiplier component from D0/D1 revenue, engagement, and ad-monetization features.
- **Evaluation:** Monthly walk-forward model development in April–November 2025; a separately reserved March–June 2026 holdout with 244 date/platform cohorts.
- **Reported private-data result:** Three-component Ridge had **5.720% D90 WAPE**, versus **6.447%** for the historical baseline (**11.3% lower WAPE relative to baseline**) on the holdout. See caveats and platform results in [`docs/RESULTS.md`](docs/RESULTS.md).
- **Operationalization:** The daily script predicts for **two calendar days ago**, separately for Android and iOS, and can optionally upsert a BigQuery forecast table.

## Repository layout

```text
d90-revenue-forecasting/
├── README.md
├── requirements.txt                 # One dependency file for all included scripts
├── .gitignore                       # Excludes credentials, raw data, logs, predictions
├── .env.example                     # Example environment-variable names (not auto-loaded)
├── run_daily_d90.bat                # Optional Windows Task Scheduler launcher
├── examples/
│   └── synthetic_cohorts.csv        # Fictional, reproducible demonstration dataset
├── scripts/
│   ├── generate_synthetic_data.py   # Rebuilds synthetic demo CSV
│   ├── ridge_model_search_calendar.py # Historical factor feature engineering; early Ridge experiments
│   ├── ridge_weighted_experiment.py # Alpha 100/200/500/1000 + alternative training weights
│   ├── ridge_orgshare_ablation.py   # Tests incremental organic-install share
│   ├── compare_component_hybrids.py # A/B/C/D/E models; development comparison
│   ├── final_holdout_test.py        # Frozen five-model Mar–Jun 2026 evaluation
│   └── daily_d90_forecast.py        # Production-oriented two-day-lag forecasts
├── tests/
│   └── test_pipeline.py             # Standard-library unittest smoke/leakage checks
└── docs/
    ├── DATA_DICTIONARY.md           # Inputs, feature definitions, temporal availability
    ├── EXPERIMENT_LOG.md            # Research sequence; script-by-script purpose
    └── RESULTS.md                   # Private-data aggregate results and limitations
```

`outputs/` is created at runtime; it is ignored by Git. The repo deliberately excludes private cohort exports, original ingestion CSVs, scripts for one-off source-table mutations, notebooks, debug output, caches, and transient experiment results.

## Quick start — runs entirely offline

Tested as a Python project with Python 3.11+ (including local execution with Python 3.13).

### 1. Create an environment

**Windows PowerShell**:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS/Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from the repository **root**, not from inside `scripts/`.

### 2. Generate or use the bundled synthetic data

```bash
python scripts/generate_synthetic_data.py
```

The committed `examples/synthetic_cohorts.csv` is already ready to use and contains one fictional row per day/platform from February 2024 through June 2026. All figures are generated from a fixed random seed; they are not derived from private input files.

### 3. Compare the five forecasting architectures on the development window

```bash
python scripts/compare_component_hybrids.py --csv examples/synthetic_cohorts.csv
```

Outputs are created in `outputs/component_hybrids_*.csv`. These summarize overall, per-platform, monthly, daily-cohort, fold-cutoff, and per-component performance.

### 4. Evaluate the frozen architectures on the holdout

```bash
python scripts/final_holdout_test.py --csv examples/synthetic_cohorts.csv
```

Reports WAPE/Bias on March–June 2026 and seven-day paired block-bootstrap intervals for **improvement versus the historical baseline**. Files appear as `outputs/FINAL_HOLDOUT_2026_*.csv`.

### 5. Reproduce a daily forecast

```bash
python scripts/daily_d90_forecast.py --as-of 2026-06-30 --csv examples/synthetic_cohorts.csv
```

This forecasts the cohorts **installed June 28, 2026**. Two rows are produced: Android and iOS. Results are saved in `outputs/daily_forecasts/` and displayed on screen. It does **not** require the target cohort's D7, D28, or D90 revenue.

### 6. Run tests

```bash
python -m unittest discover -s tests -v
```

### Optional: repeat feature-selection experiments

These are slower than the quick-start steps and are part of the historical development process, not production retraining:

```bash
python scripts/ridge_model_search_calendar.py --csv examples/synthetic_cohorts.csv
python scripts/ridge_weighted_experiment.py --csv examples/synthetic_cohorts.csv
python scripts/ridge_orgshare_ablation.py --csv examples/synthetic_cohorts.csv
```

All offline commands use synthetic data and produce **synthetic benchmark results**, not the private-data metrics in the results document.

## Data schema and what is predicted

One row corresponds to one installation **date** and one **platform** (`Android` or `iOS`). The minimal research dataset includes:

```text
date, platform, installs,
rev_d0, rev_d1, rev_d7, rev_d28, rev_d90,
retention_d1, playtime_d0, ecpm_d0
```

`org_share` is optional for general models and is required only for `ridge_orgshare_ablation.py`. See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for field definitions, engineered features, expected units, and chronological safeguards.

`rev_dX` is **cumulative cohort revenue from install through day X**, not revenue occurring on day X alone. In the daily pipeline, only D0/D1 metrics are required for the target cohort. Later cumulative revenues are used **only as mature historical training labels**.

## Modeling methodology

### 1. Historical baseline

For each platform separately and each forecast date `t`:

```text
F_7_1(t)    = SUM(R7)  / SUM(R1)  over 30 eligible historical cohorts
F_28_7(t)   = SUM(R28) / SUM(R7)  over 60 eligible historical cohorts
F_90_28(t)  = SUM(R90) / SUM(R28) over 120 eligible historical cohorts
baseline(t) = F_7_1(t) * F_28_7(t) * F_90_28(t)
R90_base(t) = R1(t) * baseline(t)
```

Historical windows end at `t-7`, `t-28`, and `t-90`, respectively. They are **not** rolling windows of immature revenue. Revenue-weighted ratios of sums reduce undue influence from small cohorts.

### 2. Ridge correction targets

For a training cohort with mature outcome, each model targets a log correction relative to its historical component:

```text
y_7_1    = log( (R7/R1)   / F_7_1 )
y_28_7   = log( (R28/R7)  / F_28_7 )
y_90_28  = log( (R90/R28) / F_90_28 )
y_28_1   = log( (R28/R1)  / (F_7_1 * F_28_7) )
y_90_1   = log( (R90/R1)  / baseline )
```

The models use `StandardScaler` **fitted on training rows only**, then `Ridge(alpha=500)` with one pooled iOS indicator. The 13 numeric R4 predictors include historical multipliers; D1 revenue per install; D0 revenue share; cohort installs; D1 retention; D0 playtime; D0 ad eCPM; and prior-30-day relative changes in three KPIs.

### 3. Five forecasting alternatives

| Name | ML learns | Other components |
|---|---|---|
| A — Baseline | None | All historical |
| B — End-to-end Ridge | One D90/D1 correction | Three-stage historical baseline |
| C — Three-component Ridge | D7/D1, D28/D7, D90/D28 corrections independently | Three historical components before ML adjustments |
| D — Ridge to D7 | Only D7/D1 correction | Historical D28/D7 and D90/D28 |
| E — Ridge to D28 | One D28/D1 correction | Historical D90/D28 |

All models predict **final D90 cumulative revenue for each daily cohort**. Monthly scores aggregate **daily absolute errors before taking their sum**; they do *not* measure only error on the sum of monthly revenue.

### 4. Preventing leakage with label maturity

During each evaluation month, Ridge is refitted using only cohorts whose label would have matured at the **start of that month**:

- D7 targets: training cutoff = month start minus **8 days**.
- D28 targets: training cutoff = month start minus **29 days**.
- D90 targets: training cutoff = month start minus **91 days**.

Input features for the predicted cohort come from D0/D1 and historical cohorts available at the prediction date. The dev evaluation spans **April 1 – November 30, 2025** (488 date/platform cohorts); the separately held-out evaluation spans **March 1 – June 30, 2026** (244 cohorts). Intermediate December–February cohorts may be used for training when mature but are not part of those reported evaluation scores. The holdout was used for final evaluation after model selection; don't continue tuning against it.

### 5. Metrics

```text
WAPE = SUM(ABS(predicted_D90_i - actual_D90_i)) / SUM(actual_D90_i)
Bias = SUM(predicted_D90_i - actual_D90_i) / SUM(actual_D90_i)
```

Both aggregate across individual day/platform cohorts `i`. Lower WAPE is better; zero signed Bias means no *aggregate* over/underestimate, but can conceal offsets across platforms. Bootstrap CIs describe paired holdout-period differences **conditional on the fitted forecasts**; they do not establish future regime stability or adjust for all prior model exploration.

## Using a real BigQuery table (private/local deployment)

The project is designed to run **offline** with `--csv`. To connect to your own Google Cloud project, provide environment variables. These variables contain identifiers, not credentials; keep your actual project details out of a public repository.

**PowerShell (current terminal session only):**

```powershell
$env:GOOGLE_CLOUD_PROJECT = "your-project-id"
$env:BQ_SOURCE_TABLE = "your-project-id.your_dataset.cohort_table"
$env:BQ_OUTPUT_TABLE = "your-project-id.your_dataset.d90_forecasts"
```

**bash (current shell session only):**

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export BQ_SOURCE_TABLE="your-project-id.your_dataset.cohort_table"
export BQ_OUTPUT_TABLE="your-project-id.your_dataset.d90_forecasts"
```

Authenticate **locally**, without storing service-account files in the repo:

```bash
gcloud auth application-default login
```

Install the Google Cloud CLI separately if needed. Ensure the identity has BigQuery read permission on the source table, and write permissions on the output dataset **only** if you explicitly enable writes. The code reads standard BigQuery tables; it does not create or update the source.

**Read-only daily run** (uses today's date in `Asia/Jerusalem` and forecasts `today - 2`):

```bash
python scripts/daily_d90_forecast.py
```

**Opt-in BigQuery result-table upsert:**

```bash
python scripts/daily_d90_forecast.py --write-bigquery
```

This creates or `MERGE`s into the *separate* table specified by `BQ_OUTPUT_TABLE`, keyed by `(cohort_date, platform, model_version)`; reruns update the existing forecast. It is not necessary for the portfolio demo. **Never use `--write-bigquery` with a synthetic CSV.** The script rejects that combination.

### Daily production semantics

- Forecast date is **two calendar days before the run date**, calculated in `Asia/Jerusalem`; e.g. a November 24 run forecasts November 22.
- It loads both Android and iOS cohorts and D1 features. Missing target rows or incomplete D1 metrics cause a clear failure instead of a stale prediction.
- Historical multipliers and lagged D0/D1 features update by cohort date. Ridge coefficients are refitted each run against training cohorts mature by the **first day of the target cohort's month**, preserving the evaluated monthly cutoff convention.
- Output includes baseline, corrected component multipliers, final D90 revenue and D90 revenue per install, model version, and training cutoffs. Results save to `outputs/daily_forecasts/`.
- A non-null D1 feature is not necessarily **fully ingested**; run only after the upstream D1 ETL has completed. Production data-quality checks and alerting are still needed.
- Windows users can schedule `run_daily_d90.bat` in Task Scheduler, after D1 ETL. The batch file **defaults to local CSV output only** (no BigQuery writes). A machine must remain available for Task Scheduler to run. A cloud deployment is a future extension and is not included.

## Every script, in research order

| Script | Why it exists | Reads / writes |
|---|---|---|
| `scripts/generate_synthetic_data.py` | Generates an independent public demonstration dataset with the exact required columns | Writes `examples/synthetic_cohorts.csv` |
| `scripts/ridge_model_search_calendar.py` | Builds maturity-aware historical factors and D1 features; explores Ridge feature groups, regularization and calendar variants in monthly folds | Source CSV or read-only BigQuery → `outputs/ridge_calendar_cv_*` |
| `scripts/ridge_weighted_experiment.py` | Tests pooled full-feature Ridge with `alpha ∈ {100,200,500,1000}`, with and without revenue-proxy sample weights | Source CSV or read-only BigQuery → `outputs/ridge_weighted_cv_*` |
| `scripts/ridge_orgshare_ablation.py` | Measures whether the share of organic installs adds accuracy beyond the existing D1 predictors | Source CSV or read-only BigQuery → `outputs/ridge_orgshare_ablation_*` |
| `scripts/compare_component_hybrids.py` | Implements A–E component/hybrid methods on April–November 2025, training each component with the appropriate maturity cutoff | Source CSV or read-only BigQuery → `outputs/component_hybrids_*` |
| `scripts/final_holdout_test.py` | Repeats frozen A–E architectures on March–June 2026; calculates platform/month metrics and paired block-bootstrap intervals | Source CSV or read-only BigQuery → `outputs/FINAL_HOLDOUT_2026_*` |
| `scripts/daily_d90_forecast.py` | Runs chosen production candidate C for the cohort installed two days ago; writes two platform forecasts; optional output-table upsert | Source CSV or read-only BigQuery → local CSV and optional BigQuery forecast table |
| `tests/test_pipeline.py` | Tests target-maturity handling, daily prediction consistency, no required immature target labels, and basic data quality | Synthetic CSV; no cloud access |
| `run_daily_d90.bat` | Optional Windows daily-task launcher and local log redirection | Runs daily script; writes `outputs/logs/` |

For a deeper account of **what was tried and why**, see [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## Public GitHub / employment-portfolio guidelines

1. Create a **new private repository** first, review all files, and verify you have permission to disclose any results and implementation details. Then change its visibility to public only if authorized.
2. Never upload real cohort-level revenue, UA spend, company-owned SQL, BigQuery project/table identifiers, API credentials, service-account JSON files, or `outputs/`. The sample CSV here is fictitious.
3. Decide whether `docs/RESULTS.md` (aggregate private-data performance) is approved for public sharing. If unsure, **remove that file and the real-data metric claims in this README** before publishing; the code and synthetic demo still stand alone.
4. Add a short repository description, e.g. **“Maturity-aware D90 cohort revenue forecasting: historical multipliers, Ridge ML, walk-forward validation, production batch pipeline.”**
5. Put a GitHub link in a job application, resume, or LinkedIn project section; highlight the independent validation, forecast-maturity logic, and daily automation rather than claiming access to private data.

**Basic Git commands (from project root):**

```bash
git init
git add .
git status
git commit -m "Prepare D90 forecasting portfolio project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Create an **empty** repository on GitHub before the last two commands. Replace the URL with its real address. You must authenticate Git on your own machine. The GitHub repository is **not** created by this package.

## Limitations and next steps

- Only two platforms and one geography in the original experiment; performance and learned coefficients need not transfer to other markets or products.
- Original revenue outcomes must be fully mature before inclusion in component training; ETL lags can exceed nominal cohort maturity.
- New product features, pricing changes, and shifts in organic/paid composition can create unmodeled structural changes.
- The reported best pooled model was not always the best per platform; aggregate Bias can hide offsetting errors.
- The final holdout was used once after the model-search process. All further performance claims should use **new future cohorts**, not retuning to March–June 2026.
- Daily script is a single batch run, not a managed cloud service: monitoring, retrying, upstream freshness checks, model registry, and alerting should be added for production-critical usage.

---

**Review before publication:** This repository is a technical portfolio package, not proof of authorization to publish business data or a claim that the fictional demo reproduces the private-data results.
