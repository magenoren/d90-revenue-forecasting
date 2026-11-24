"""Daily D90 revenue forecast for the cohort installed TWO DAYS AGO.

Production candidate C, frozen after the March-June 2026 holdout:
  - 3 independent pooled Ridge(alpha=500) corrections for
    D7/D1, D28/D7, D90/D28
  - original R4 D1-available features (13 numeric + platform_iOS)
  - historical factor windows (30, 60, 120) with maturity shifts (7, 28, 90)
  - Ridge training uses the SAME monthly training cutoff convention as the
    completed walk-forward test: month_start - (8, 29, 91) days, respectively.
  - historical baseline factors and prior-30 KPI features update for each
    target cohort date.

No immature D7/D28/D90 outcomes are required for the target cohort.
Default mode reads BigQuery, prints and writes local CSV; no BigQuery writes.
Optional --write-bigquery creates/merges the forecast into a separate output
BigQuery table (never touches the source cohort table).

Examples:
  python daily_d90_forecast.py
  python daily_d90_forecast.py --write-bigquery
  python daily_d90_forecast.py --as-of 2026-07-02 --csv cohorts.csv

Scheduling: run once each morning AFTER D1 reporting for the cohort at t-2
has finished. Time zone for default date: Asia/Jerusalem.

Requirements:
  pip install pandas numpy scikit-learn google-cloud-bigquery db-dtypes pyarrow tzdata
  gcloud auth application-default login  # once, locally
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
SOURCE_TABLE = os.environ.get("BQ_SOURCE_TABLE", "")
OUTPUT_TABLE = os.environ.get("BQ_OUTPUT_TABLE", "")
MODEL_VERSION = "C_ThreeComponent_Ridge_R4_a500_v1"
TIMEZONE = "Asia/Jerusalem"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "daily_forecasts"

ALPHA = 500.0
FIRST_TRAIN_DATE = pd.Timestamp("2024-08-28")
DATA_COLUMNS = [
    "date", "platform", "installs", "rev_d0", "rev_d1", "rev_d7",
    "rev_d28", "rev_d90", "retention_d1", "playtime_d0", "ecpm_d0",
]
FEATURES = [
    "baseline_factor", "factor_7_1", "factor_28_7", "factor_90_28",
    "rpi_d1", "d0_share_of_d1", "installs", "retention_d1",
    "playtime_d0", "retention_d1_vs_prev30",
    "playtime_d0_vs_prev30", "ecpm_d0", "ecpm_d0_vs_prev30",
]
COMPONENTS = [
    ("part_7_1", 7, 8, "factor_7_1"),
    ("part_28_7", 28, 29, "factor_28_7"),
    ("part_90_28", 90, 91, "factor_90_28"),
]
FACTOR_WINDOWS = {
    "factor_7_1": ("rev_d7", "rev_d1", 7, 30),
    "factor_28_7": ("rev_d28", "rev_d7", 28, 60),
    "factor_90_28": ("rev_d90", "rev_d28", 90, 120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="Override today's date for offline replay")
    parser.add_argument("--csv", type=Path, help="Local source CSV instead of BigQuery")
    parser.add_argument("--write-bigquery", action="store_true",
                        help="Opt-in: upsert predictions into a separate BigQuery table")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_raw(target: date, csv_path: Path | None) -> pd.DataFrame:
    """Fetch only install cohorts <= the D1-known target date."""
    if csv_path:
        raw = pd.read_csv(csv_path)
        print(f"Source: {csv_path}")
    else:
        if not PROJECT or not SOURCE_TABLE:
            raise ValueError("Set GOOGLE_CLOUD_PROJECT and BQ_SOURCE_TABLE to read BigQuery; or pass --csv")
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT)
        sql = f"SELECT {', '.join(DATA_COLUMNS)} FROM `{SOURCE_TABLE}` WHERE date <= @target_date"
        config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", target),
        ])
        raw = client.query(sql, job_config=config).to_dataframe()
        print(f"Source: {SOURCE_TABLE} (read-only)")

    missing = sorted(set(DATA_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Source missing required columns: {missing}")
    raw = raw[DATA_COLUMNS].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    if raw["date"].isna().any():
        raise ValueError("Invalid source dates")
    raw = raw.loc[raw.date <= pd.Timestamp(target)].copy()
    if raw.empty:
        raise ValueError("No source cohorts on or before target date")
    if raw.duplicated(["date", "platform"]).any():
        raise ValueError("Duplicate install date/platform keys in input")
    if set(raw.platform.unique()) != {"Android", "iOS"}:
        raise ValueError("Source must contain Android and iOS cohort history")
    for col in DATA_COLUMNS[2:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # D1-available input fields must be present for historical rows AND target.
    available_by_d1 = ["installs", "rev_d0", "rev_d1", "retention_d1", "playtime_d0", "ecpm_d0"]
    if not np.isfinite(raw[available_by_d1].to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite D0-D1 features; ingest D1 reporting before running")
    if (raw[["installs", "rev_d1"]] <= 0).any().any():
        raise ValueError("Installs and D1 revenue must be >0 to calculate ratios")
    if (raw["rev_d0"] < 0).any() or (raw["playtime_d0"] < 0).any() or (raw["ecpm_d0"] < 0).any():
        raise ValueError("Negative D0 input values")
    if not raw.retention_d1.between(0, 1).all():
        raise ValueError("D1 retention must be a 0..1 fraction")

    for platform in ["Android", "iOS"]:
        dates = pd.DatetimeIndex(raw.loc[raw.platform == platform, "date"])
        if pd.Timestamp(target) not in dates:
            raise ValueError(f"Missing target cohort {target} for {platform}; latest available: {dates.max().date()}")
        expected = pd.date_range(dates.min(), target, freq="D")
        if len(expected) != len(dates) or not expected.isin(dates).all():
            raise ValueError(f"Missing daily cohort date(s) for {platform}; historical factors require contiguous days")
    return raw


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Replicate the frozen R4 feature engineering from the final holdout.

    Unlike the research make_dataset(), allow rev_d7/rev_d28/rev_d90 to be
    NULL/immature for recent cohorts. These never become target predictors.
    """
    blocks = []
    for platform, group in raw.groupby("platform"):
        g = group.set_index("date").sort_index().asfreq("D")
        for name, (num, den, age, window) in FACTOR_WINDOWS.items():
            numer = g[num].rolling(window, min_periods=window).sum().shift(age)
            denom = g[den].rolling(window, min_periods=window).sum().shift(age)
            g[name] = numer / denom.replace(0, np.nan)
        g["baseline_factor"] = g["factor_7_1"] * g["factor_28_7"] * g["factor_90_28"]
        g["rpi_d1"] = g["rev_d1"] / g["installs"]
        g["d0_share_of_d1"] = g["rev_d0"] / g["rev_d1"]
        for kpi in ["retention_d1", "playtime_d0", "ecpm_d0"]:
            avg_prev30 = g[kpi].rolling(30, min_periods=30).mean().shift(1)
            g[f"{kpi}_vs_prev30"] = g[kpi] / avg_prev30.replace(0, np.nan) - 1
        g["platform"] = platform
        blocks.append(g.reset_index())
    result = pd.concat(blocks, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    result = result.loc[result.date >= FIRST_TRAIN_DATE].sort_values(["date", "platform"]).reset_index(drop=True)
    return result


def model_fit_predict(train: pd.DataFrame, target_rows: pd.DataFrame,
                      y_col: str) -> np.ndarray:
    x_train = train[FEATURES].copy()
    x_test = target_rows[FEATURES].copy()
    x_train["platform_iOS"] = (train.platform == "iOS").astype(float)
    x_test["platform_iOS"] = (target_rows.platform == "iOS").astype(float)
    pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])
    pipeline.fit(x_train, train[y_col])
    out = pipeline.predict(x_test)
    if not np.isfinite(out).all():
        raise ValueError(f"Nonfinite Ridge correction in {y_col}")
    return out


def forecast_for_day(data: pd.DataFrame, as_of: date) -> pd.DataFrame:
    target_date = as_of - timedelta(days=2)
    month_start = pd.Timestamp(target_date).replace(day=1)
    target_rows = data.loc[data.date == pd.Timestamp(target_date)].copy()
    if len(target_rows) != 2 or set(target_rows.platform) != {"Android", "iOS"}:
        raise ValueError("Need precisely one target-date cohort on each platform")
    # Crucial: do not ever require actual D7/D28/D90 for the forecast cohort.
    if not np.isfinite(target_rows[FEATURES].to_numpy(dtype=float)).all():
        missing = target_rows[FEATURES].isna().sum()
        raise ValueError(f"Target feature set incomplete: {missing[missing.gt(0)].to_dict()}")
    if (target_rows[["factor_7_1", "factor_28_7", "factor_90_28", "baseline_factor"]] <= 0).any().any():
        raise ValueError("Missing/invalid historical baseline factors for target")

    outputs = target_rows[["date", "platform", "installs", "rev_d1",
                           "factor_7_1", "factor_28_7", "factor_90_28", "baseline_factor"]].copy()
    outputs.insert(0, "run_date", as_of.isoformat())
    outputs.insert(3, "model_version", MODEL_VERSION)
    outputs["baseline_predicted_d90"] = outputs.rev_d1 * outputs.baseline_factor
    outputs["ridge_train_month"] = month_start.strftime("%Y-%m")

    correction_total = np.zeros(len(target_rows))
    for label, maturity, lag, factor_col in COMPONENTS:
        cutoff = month_start - pd.Timedelta(days=lag)
        train = data.loc[data.date <= cutoff].copy()
        if train.empty or not {"Android", "iOS"}.issubset(set(train.platform)):
            raise ValueError(f"No training data for {label}")
        if min(train.groupby("platform").size()) < 60:
            raise ValueError(f"Insufficient mature training cohorts for {label}")
        if train.date.max() >= month_start - pd.Timedelta(days=maturity):
            raise AssertionError(f"Training cutoff violated for {label}")
        if not np.isfinite(train[FEATURES].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite training predictors for {label}")

        numerator, denominator = {
            "part_7_1": ("rev_d7", "rev_d1"),
            "part_28_7": ("rev_d28", "rev_d7"),
            "part_90_28": ("rev_d90", "rev_d28"),
        }[label]
        required = [numerator, denominator, factor_col]
        if not np.isfinite(train[required].to_numpy(dtype=float)).all() or (train[required] <= 0).any().any():
            raise ValueError(f"Missing/invalid matured labels for {label} through {cutoff.date()}")
        y = np.log((train[numerator] / train[denominator]) / train[factor_col])
        train[f"y_{label}"] = y
        if not np.isfinite(y).all():
            raise ValueError(f"Invalid log correction label for {label}")
        correction = model_fit_predict(train, target_rows, f"y_{label}")
        correction_total += correction
        outputs[f"log_correction_{label}"] = correction
        outputs[f"predicted_{label}"] = target_rows[factor_col].to_numpy() * np.exp(correction)
        outputs[f"train_last_{label}"] = train.date.max().date().isoformat()
        outputs[f"train_rows_{label}"] = len(train)

    outputs["predicted_factor_d90_d1"] = outputs.baseline_factor * np.exp(correction_total)
    outputs["predicted_d90"] = outputs.rev_d1 * outputs.predicted_factor_d90_d1
    outputs["predicted_d90_per_install"] = outputs.predicted_d90 / outputs.installs
    if not np.isfinite(outputs[["predicted_d90", "predicted_factor_d90_d1",
                                "predicted_d90_per_install"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite final prediction")
    if (outputs["predicted_d90"] <= 0).any():
        raise ValueError("Nonpositive final predicted revenue")
    return outputs.reset_index(drop=True)


def write_local(predictions: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(predictions.date.iloc[0]).strftime("%Y-%m-%d")
    target_path = output_dir / f"d90_forecast_{day}.csv"
    latest_path = output_dir / "latest_d90_forecast.csv"
    for path in [target_path, latest_path]:
        temp = path.with_suffix(".csv.tmp")
        predictions.to_csv(temp, index=False)
        os.replace(temp, path)
    return target_path, latest_path


def write_bigquery(predictions: pd.DataFrame) -> None:
    """Idempotent opt-in upsert; source table is never written."""
    if not PROJECT or not OUTPUT_TABLE:
        raise ValueError("Set GOOGLE_CLOUD_PROJECT and BQ_OUTPUT_TABLE before --write-bigquery")
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    ddl = f"""CREATE TABLE IF NOT EXISTS `{OUTPUT_TABLE}` (
      cohort_date DATE NOT NULL,
      platform STRING NOT NULL,
      model_version STRING NOT NULL,
      run_date DATE,
      installs INT64,
      rev_d1 FLOAT64,
      baseline_factor FLOAT64,
      baseline_predicted_d90 FLOAT64,
      predicted_factor_d90_d1 FLOAT64,
      predicted_d90 FLOAT64,
      predicted_d90_per_install FLOAT64,
      train_month STRING,
      updated_at TIMESTAMP
    ) PARTITION BY cohort_date CLUSTER BY platform, model_version"""
    client.query(ddl).result()

    sql = f"""MERGE `{OUTPUT_TABLE}` AS t
    USING (SELECT
      @cohort_date AS cohort_date, @platform AS platform,
      @model_version AS model_version, @run_date AS run_date,
      @installs AS installs, @rev_d1 AS rev_d1,
      @baseline_factor AS baseline_factor,
      @baseline_predicted_d90 AS baseline_predicted_d90,
      @predicted_factor_d90_d1 AS predicted_factor_d90_d1,
      @predicted_d90 AS predicted_d90,
      @predicted_d90_per_install AS predicted_d90_per_install,
      @train_month AS train_month) AS s
    ON t.cohort_date = s.cohort_date AND t.platform = s.platform
       AND t.model_version = s.model_version
    WHEN MATCHED THEN UPDATE SET
      run_date=s.run_date, installs=s.installs, rev_d1=s.rev_d1,
      baseline_factor=s.baseline_factor,
      baseline_predicted_d90=s.baseline_predicted_d90,
      predicted_factor_d90_d1=s.predicted_factor_d90_d1,
      predicted_d90=s.predicted_d90,
      predicted_d90_per_install=s.predicted_d90_per_install,
      train_month=s.train_month, updated_at=CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT (
      cohort_date, platform, model_version, run_date, installs, rev_d1,
      baseline_factor, baseline_predicted_d90, predicted_factor_d90_d1,
      predicted_d90, predicted_d90_per_install, train_month, updated_at)
    VALUES (s.cohort_date,s.platform,s.model_version,s.run_date,s.installs,
      s.rev_d1,s.baseline_factor,s.baseline_predicted_d90,
      s.predicted_factor_d90_d1,s.predicted_d90,s.predicted_d90_per_install,
      s.train_month,CURRENT_TIMESTAMP())"""
    for row in predictions.to_dict("records"):
        query_params = [
            bigquery.ScalarQueryParameter("cohort_date", "DATE", pd.Timestamp(row["date"]).date()),
            bigquery.ScalarQueryParameter("platform", "STRING", row["platform"]),
            bigquery.ScalarQueryParameter("model_version", "STRING", row["model_version"]),
            bigquery.ScalarQueryParameter("run_date", "DATE", date.fromisoformat(row["run_date"])),
            bigquery.ScalarQueryParameter("installs", "INT64", int(row["installs"])),
            bigquery.ScalarQueryParameter("rev_d1", "FLOAT64", float(row["rev_d1"])),
            bigquery.ScalarQueryParameter("baseline_factor", "FLOAT64", float(row["baseline_factor"])),
            bigquery.ScalarQueryParameter("baseline_predicted_d90", "FLOAT64", float(row["baseline_predicted_d90"])),
            bigquery.ScalarQueryParameter("predicted_factor_d90_d1", "FLOAT64", float(row["predicted_factor_d90_d1"])),
            bigquery.ScalarQueryParameter("predicted_d90", "FLOAT64", float(row["predicted_d90"])),
            bigquery.ScalarQueryParameter("predicted_d90_per_install", "FLOAT64", float(row["predicted_d90_per_install"])),
            bigquery.ScalarQueryParameter("train_month", "STRING", row["ridge_train_month"]),
        ]
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=query_params)).result()
    print(f"BigQuery upserted {len(predictions)} prediction rows into {OUTPUT_TABLE}")


def main() -> None:
    args = parse_args()
    today = args.as_of or datetime.now(ZoneInfo(TIMEZONE)).date()
    target = today - timedelta(days=2)
    print(f"Run date ({TIMEZONE}): {today}; target installation date: {target}")
    print(f"Model: {MODEL_VERSION}; alpha={ALPHA:g}; training as-of first day of forecast month")
    raw = load_raw(target, args.csv)
    data = build_features(raw)
    result = forecast_for_day(data, today)
    file_path, latest = write_local(result, args.output_dir)
    print("\nD90 PREDICTION (cumulative revenue per install-date/platform cohort):")
    print(result[["date", "platform", "installs", "rev_d1", "baseline_predicted_d90",
                  "predicted_d90", "predicted_d90_per_install"]].round(4).to_string(index=False))
    print(f"Saved: {file_path}")
    print(f"Latest: {latest}")
    if args.write_bigquery:
        if args.csv:
            raise ValueError("Refusing to write local/fixture CSV predictions to BigQuery")
        write_bigquery(result)
    else:
        print("BigQuery is unchanged; add --write-bigquery to upsert forecasts into a separate table.")


if __name__ == "__main__":
    main()
