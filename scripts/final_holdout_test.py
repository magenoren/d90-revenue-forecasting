"""FINAL, ONE-TIME, untouched March-June 2026 holdout for approaches A-E.

Prerequisites: place this file beside BOTH of the following previously created files:
    ridge_model_search_calendar.py
    compare_component_hybrids.py

Usage:
    python final_holdout_test.py
    python final_holdout_test.py --csv cohort_data.csv

Do not change any feature selection, window, model family or alpha after seeing
these results. The models are retrained once at the start of each test month,
using only labels that would already have matured at that time.

The final test spans 2026-03-01 through 2026-06-30 (122 dates x 2 platforms).
Historical dates and a full 2026 test period are needed to compute features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the SAME baseline, feature engineering, fitting and prediction logic
# as the existing development experiment. Its load_data function is deliberately
# NOT reused: it hard-cuts source rows at 2025-11-30 to protect the holdout.
from ridge_model_search_calendar import (
    PROJECT, TABLE, DATA_COLUMNS, make_dataset, score,
)
from compare_component_hybrids import (
    ALPHA, FEATURES, COMPONENT_MATURITY, MODEL_NAMES,
    create_targets, fit_component_model, append_predictions,
)

TEST_START = pd.Timestamp("2026-03-01")
TEST_END = pd.Timestamp("2026-06-30")
EXPECTED_DATES = pd.date_range(TEST_START, TEST_END, freq="D")
OUT = Path("outputs")

# Assert model configuration is frozen before loading final-test outcomes.
assert ALPHA == 500.0, "This test requires the previously selected alpha=500."
assert len(FEATURES) == 13, "Unexpected R4 features. Check source scripts."
assert set(MODEL_NAMES) == {"A", "B", "C", "D", "E"}


def load_full_data(csv_path: str | None) -> pd.DataFrame:
    """Read historic + holdout cohorts; never train on future labels."""
    if csv_path:
        print(f"Reading local CSV: {csv_path}")
        raw = pd.read_csv(csv_path)
    else:
        from google.cloud import bigquery
        print(f"Reading BigQuery table (read-only): {TABLE}")
        client = bigquery.Client(project=PROJECT)
        cols = ", ".join(DATA_COLUMNS)
        sql = (f"SELECT {cols} FROM `{TABLE}` "
               f"WHERE date <= DATE '{TEST_END:%Y-%m-%d}'")
        raw = client.query(sql).to_dataframe()

    missing = set(DATA_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    raw = raw[DATA_COLUMNS].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.loc[raw["date"] <= TEST_END].copy()
    for col in DATA_COLUMNS[2:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw[DATA_COLUMNS].isna().any().any():
        raise ValueError("Missing or invalid source values")
    if raw.duplicated(["date", "platform"]).any():
        raise ValueError("Duplicate cohort keys")
    if set(raw.platform.unique()) != {"Android", "iOS"}:
        raise ValueError("Expected Android and iOS")
    if (raw[["installs", "rev_d1", "rev_d7", "rev_d28", "rev_d90"]] <= 0).any().any():
        raise ValueError("Revenue denominators and installs must be positive")
    if not raw["retention_d1"].between(0, 1).all():
        raise ValueError("Retention not on 0..1 scale")

    for platform in ["Android", "iOS"]:
        platform_data = raw.loc[raw.platform == platform, "date"]
        observed = pd.DatetimeIndex(platform_data)
        if not EXPECTED_DATES.isin(observed).all():
            raise ValueError(f"Missing test dates for {platform}")
        if len(platform_data) != len(pd.date_range(raw.date.min(), TEST_END)):
            raise ValueError(f"Non-continuous historical date coverage for {platform}")
    return raw


def evaluate() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Optional local cohort CSV instead of BigQuery")
    args = parser.parse_args()

    raw = load_full_data(args.csv)
    data = create_targets(make_dataset(raw))
    # Predictions are generated for test cohorts ONLY. Full raw data is used to
    # calculate maturity-safe historical features and eligible training labels.
    test_dates = data.loc[data.date.between(TEST_START, TEST_END), "date"].unique()
    if len(test_dates) != len(EXPECTED_DATES):
        raise ValueError("Test dataset is incomplete after feature engineering")
    print(f"Ready: {len(test_dates)} test dates across 2 platforms.")

    records: list[pd.DataFrame] = []
    folds: list[dict] = []
    for month in pd.date_range(TEST_START, TEST_END, freq="MS"):
        month_end = min(month + pd.offsets.MonthEnd(0), TEST_END)
        forecast = data.loc[data.date.between(month, month_end)].copy()
        expected_rows = 2 * len(pd.date_range(month, month_end))
        if len(forecast) != expected_rows:
            raise ValueError(f"Unexpected forecast row count in {month:%Y-%m}")

        corrections: dict[str, np.ndarray] = {}
        details = {"forecast_month": month.strftime("%Y-%m"),
                   "forecast_cohorts": len(forecast)}
        for component, maturity in COMPONENT_MATURITY.items():
            # Same conservative maturity cutoffs as development script:
            # D7 -> t-8; D28 -> t-29; D90 -> t-91.
            train_last = month - pd.Timedelta(days=maturity + 1)
            train = data.loc[data.date <= train_last].copy()
            if train.empty or min(train.groupby("platform").size()) < 60:
                raise ValueError(f"Insufficient mature training data for {component}")
            if train.date.max() >= month - pd.Timedelta(days=maturity):
                raise AssertionError("Training on immature label!")
            corrections[component] = fit_component_model(
                train, forecast, f"y_{component}"
            )
            details[f"{component}_train_last"] = train.date.max().date().isoformat()
            details[f"{component}_train_rows"] = len(train)

        zero = np.zeros(len(forecast), dtype=float)
        log_corrections = {
            "A": zero,
            "B": corrections["total_90_1"],
            "C": corrections["part_7_1"] + corrections["part_28_7"]
                 + corrections["part_90_28"],
            "D": corrections["part_7_1"],
            "E": corrections["part_28_1"],
        }
        for key, correction in log_corrections.items():
            append_predictions(records, forecast, month, MODEL_NAMES[key], correction)
        folds.append(details)
        print(f"{month:%Y-%m}: predicted {len(forecast)} cohorts | latest D7/D28/D90 "
              f"training cohorts: {details['part_7_1_train_last']} / "
              f"{details['part_28_1_train_last']} / {details['total_90_1_train_last']}")

    preds = pd.concat(records, ignore_index=True)
    if len(preds) != 5 * len(EXPECTED_DATES) * 2:
        raise AssertionError("Wrong total number of predictions")
    if preds.duplicated(["date", "platform", "config"]).any():
        raise AssertionError("Duplicate test predictions")
    if preds.date.min() != TEST_START or preds.date.max() != TEST_END:
        raise AssertionError("Test period mismatch")
    if preds.groupby("config").size().nunique() != 1:
        raise AssertionError("Inconsistent cohort counts across models")
    if preds.groupby(["date", "platform"])["actual_d90"].nunique().max() != 1:
        raise AssertionError("Actual revenue differs by model")

    overall = score(preds, ["config"]).sort_values("WAPE_%")
    platform = score(preds, ["config", "platform"]).sort_values(["platform", "WAPE_%"])
    monthly = score(preds, ["month", "config"]).sort_values(["month", "config"])

    print("\nFINAL HOLDOUT — POOLED (MARCH–JUNE 2026)")
    print(overall[["config", "cohorts", "WAPE_%", "Bias_%"]].round(3).to_string(index=False))
    print("\nFINAL HOLDOUT — BY PLATFORM")
    print(platform[["config", "platform", "cohorts", "WAPE_%", "Bias_%"]]
          .round(3).to_string(index=False))
    print("\nFINAL HOLDOUT — MONTHLY")
    formatted = monthly[["month", "config", "cohorts", "WAPE_%", "Bias_%"]].copy()
    formatted["month"] = formatted["month"].dt.strftime("%Y-%m")
    print(formatted.round(3).to_string(index=False))

    # A paired uncertainty interval for improvement in *absolute* WAPE
    # percentage points. Resampling preserves the Android/iOS date pairs and
    # samples contiguous 7-day blocks to acknowledge serial dependence.
    # It measures uncertainty conditional on these trained forecasts and does
    # NOT account for model-selection uncertainty on the development period.
    pivot = preds.assign(
        abs_error=(preds["predicted_d90"] - preds["actual_d90"]).abs()
    ).groupby(["date", "config"], as_index=False).agg(
        absolute_error=("abs_error", "sum"),
        actual_revenue=("actual_d90", "sum"),
    )
    errors = pivot.pivot(index="date", columns="config", values="absolute_error")
    actual = pivot.loc[pivot.config == MODEL_NAMES["A"]].set_index("date")
    actual = actual["actual_revenue"].reindex(errors.index)
    rng = np.random.default_rng(42)
    block_len, replications = 7, 2000
    n = len(errors)
    bootstrap_indices = np.empty((replications, n), dtype=int)
    for b in range(replications):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_len + 1))
            indices.extend(range(start, start + block_len))
        bootstrap_indices[b, :] = indices[:n]

    print("\nPAIRED 7-DAY BLOCK-BOOTSTRAP: WAPE IMPROVEMENT VS BASELINE")
    base_errors = errors[MODEL_NAMES["A"]].to_numpy()
    denominator = actual.to_numpy()[bootstrap_indices].sum(axis=1)
    uncertainty = []
    for label in ["B", "C", "D", "E"]:
        candidate_errors = errors[MODEL_NAMES[label]].to_numpy()
        improvement = 100 * (base_errors.sum() - candidate_errors.sum()) / actual.sum()
        bootstrap = 100 * (
            base_errors[bootstrap_indices].sum(axis=1)
            - candidate_errors[bootstrap_indices].sum(axis=1)
        ) / denominator
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        uncertainty.append({"config": MODEL_NAMES[label],
                            "WAPE_improvement_pp": improvement,
                            "CI95_lower_pp": lower,
                            "CI95_upper_pp": upper})
    boot_table = pd.DataFrame(uncertainty)
    print(boot_table.round(3).to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / "FINAL_HOLDOUT_2026"
    preds.to_csv(f"{prefix}_daily_predictions.csv", index=False)
    overall.to_csv(f"{prefix}_overall.csv", index=False)
    platform.to_csv(f"{prefix}_platform.csv", index=False)
    monthly.to_csv(f"{prefix}_monthly.csv", index=False)
    pd.DataFrame(folds).to_csv(f"{prefix}_training_cutoffs.csv", index=False)
    boot_table.to_csv(f"{prefix}_bootstrap.csv", index=False)
    print("\nSaved outputs/FINAL_HOLDOUT_2026_*.csv")
    print("IMPORTANT: This is the one-time final holdout evaluation. Do not "
          "retune based on these results.")


if __name__ == "__main__":
    evaluate()
