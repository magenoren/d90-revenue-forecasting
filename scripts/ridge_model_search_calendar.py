"""Walk-forward Ridge model selection for D90 revenue forecasting.

Compares 8 feature groups x 7 Ridge alphas x 2 model structures
(pooled across Android/iOS vs separate models), plus two non-ML controls.

CRITICAL: Every monthly fold trains only on cohorts whose D90 outcomes
were fully mature before the forecast month. Does not read or evaluate
March-June 2026, which remains reserved for the final test.

Run in the existing VS Code project:
    pip install google-cloud-bigquery db-dtypes pandas numpy scikit-learn
    python ridge_model_search_calendar.py

For an offline test, you may supply a CSV with the same BigQuery columns:
    python ridge_model_search_calendar.py --csv path/to/your_cohort_data.csv

The script reads data only. It never modifies BigQuery.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
TABLE = os.environ.get("BQ_SOURCE_TABLE", "")
OUTPUT = Path("outputs")

# Holdout (Mar-Jun 2026) is *not queried*. All model selection is before it.
DEVELOPMENT_END = "2025-11-30"
FIRST_TRAIN_DATE = pd.Timestamp("2024-08-28")
FIRST_CV_MONTH = "2025-04-01"
LAST_CV_MONTH = "2025-11-01"

# D90 values for training cohorts must be mature when the forecast begins.
# Cohorts from t-91 and earlier are used; this is slightly conservative.
LABEL_MATURITY_GAP_DAYS = 91

ALPHAS = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
STRUCTURES = ["Pooled", "Separate"]

# Keep the rule-based baseline fixed across all candidate ML models.
# Numerator, denominator, maturity in days, historical window in days.
FACTOR_WINDOWS = {
    "factor_7_1": ("rev_d7", "rev_d1", 7, 30),
    "factor_28_7": ("rev_d28", "rev_d7", 28, 60),
    "factor_90_28": ("rev_d90", "rev_d28", 90, 120),
}

# Five unique and nested/controlled feature sets. A platform indicator is
# included automatically only in pooled models. Calibration is a separate
# benchmark, so R0 can test whether the factor levels themselves help.
FACTOR_FEATURES = [
    "baseline_factor", "factor_7_1", "factor_28_7", "factor_90_28"
]
REVENUE_FEATURES = ["rpi_d1", "d0_share_of_d1", "installs"]
ENGAGEMENT_FEATURES = [
    "retention_d1", "playtime_d0",
    "retention_d1_vs_prev30", "playtime_d0_vs_prev30",
]
MONETIZATION_FEATURES = ["ecpm_d0", "ecpm_d0_vs_prev30"]

# New features to test independently and in combination.
# Weekend days = Saturday or Sunday; weekdays = Monday through Friday.
CALENDAR_FEATURES = ["is_weekend"]
SAME_TYPE_PLAYTIME_FEATURES = ["playtime_d0_vs_prev7_same_day_type"]

FEATURE_SETS = {
    "R0_Factors": FACTOR_FEATURES,
    "R1_Revenue": FACTOR_FEATURES + REVENUE_FEATURES,
    "R2_Engagement": FACTOR_FEATURES + REVENUE_FEATURES + ENGAGEMENT_FEATURES,
    "R3_Monetization": FACTOR_FEATURES + REVENUE_FEATURES + MONETIZATION_FEATURES,
    "R4_Full": FACTOR_FEATURES + REVENUE_FEATURES
               + ENGAGEMENT_FEATURES + MONETIZATION_FEATURES,
    # Additions to R4 are isolated so we can measure their value.
    "R5_Full_Weekend": FACTOR_FEATURES + REVENUE_FEATURES
               + ENGAGEMENT_FEATURES + MONETIZATION_FEATURES
               + CALENDAR_FEATURES,
    "R6_Full_SameTypePlaytime": FACTOR_FEATURES + REVENUE_FEATURES
               + ENGAGEMENT_FEATURES + MONETIZATION_FEATURES
               + SAME_TYPE_PLAYTIME_FEATURES,
    "R7_Full_Both": FACTOR_FEATURES + REVENUE_FEATURES
               + ENGAGEMENT_FEATURES + MONETIZATION_FEATURES
               + CALENDAR_FEATURES + SAME_TYPE_PLAYTIME_FEATURES,
}

DATA_COLUMNS = [
    "date", "platform", "installs", "rev_d0", "rev_d1", "rev_d7",
    "rev_d28", "rev_d90", "retention_d1", "playtime_d0", "ecpm_d0",
]


def mature_factor(data: pd.DataFrame, numerator: str,
                  denominator: str, maturity: int, window: int) -> pd.Series:
    """Ratio of summed eligible cohort revenues. Never uses immature labels."""
    numer = data[numerator].rolling(window, min_periods=window).sum().shift(maturity)
    denom = data[denominator].rolling(window, min_periods=window).sum().shift(maturity)
    return numer.div(denom.where(denom != 0))


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        print(f"Reading local CSV: {csv_path}")
        raw = pd.read_csv(csv_path)
    else:
        if not PROJECT or not TABLE:
            raise ValueError("Set GOOGLE_CLOUD_PROJECT and BQ_SOURCE_TABLE to use BigQuery; or pass --csv")
        from google.cloud import bigquery
        print(f"Reading BigQuery table (read only): {TABLE}")
        client = bigquery.Client(project=PROJECT)
        cols = ", ".join(DATA_COLUMNS)
        # Explicit cutoff prevents accidental inspection of the final holdout.
        sql = f"SELECT {cols} FROM `{TABLE}` WHERE date <= '{DEVELOPMENT_END}'"
        raw = client.query(sql).to_dataframe()

    missing = sorted(set(DATA_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    raw = raw[DATA_COLUMNS].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.loc[raw["date"] <= pd.Timestamp(DEVELOPMENT_END)]
    if raw.duplicated(["date", "platform"]).any():
        raise ValueError("Duplicate date/platform cohort keys")
    if set(raw["platform"].unique()) != {"Android", "iOS"}:
        raise ValueError("Expected exactly two platforms: Android and iOS")
    for name in DATA_COLUMNS[2:]:
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    if raw[DATA_COLUMNS].isna().any().any():
        raise ValueError("Source data contains missing or invalid values")
    if (raw[["installs", "rev_d1", "rev_d90"]] <= 0).any().any():
        raise ValueError("Installs, D1 revenue and D90 revenue must be positive")
    if ((raw["retention_d1"] < 0) | (raw["retention_d1"] > 1)).any():
        raise ValueError("Retention must be a fraction between 0 and 1")
    return raw


def make_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for platform, group in raw.groupby("platform"):
        data = group.set_index("date").sort_index().asfreq("D")
        if data[DATA_COLUMNS[2:]].isna().any().any():
            raise ValueError(f"Missing dates or features for {platform}")

        for name, (num, den, age, window) in FACTOR_WINDOWS.items():
            data[name] = mature_factor(data, num, den, age, window)
        data["baseline_factor"] = (
            data["factor_7_1"] * data["factor_28_7"] * data["factor_90_28"]
        )
        data["rpi_d1"] = data["rev_d1"] / data["installs"]
        data["d0_share_of_d1"] = data["rev_d0"] / data["rev_d1"]
        data["actual_factor"] = data["rev_d90"] / data["rev_d1"]

        # Calendar feature: 1 for Saturday/Sunday, 0 for Monday-Friday.
        # No target information or future cohorts are used.
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

        # Compare current D0 playtime to the last 7 PREVIOUS cohort days
        # of the same type. For a Sunday, this means the most recent 7
        # Saturdays/Sundays (excluding today); for Monday, the most recent
        # 7 Mon-Fri dates. Groupby preserves chronological order because
        # 'data' was sorted by date above.
        prior7 = data.groupby("is_weekend")["playtime_d0"].transform(
            lambda values: values.shift(1).rolling(7, min_periods=7).mean()
        )
        data["playtime_d0_vs_prev7_same_day_type"] = (
            data["playtime_d0"] / prior7.replace(0, np.nan) - 1
        )

        # For each KPI compare the target cohort with the 30 prior cohorts.
        # This statistic is available at D1; the target cohort is excluded.
        for kpi in ["retention_d1", "playtime_d0", "ecpm_d0"]:
            ref = data[kpi].rolling(30, min_periods=30).mean().shift(1)
            data[f"{kpi}_vs_prev30"] = data[kpi] / ref.replace(0, np.nan) - 1

        data["platform"] = platform
        chunks.append(data.reset_index())

    result = pd.concat(chunks, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    result["target_log_adjustment"] = np.log(
        result["actual_factor"] / result["baseline_factor"]
    )
    # The pre-forecast rows have no long-term baseline and should not be fitted.
    result = result.loc[result["date"] >= FIRST_TRAIN_DATE].copy()
    required = list(set(sum(FEATURE_SETS.values(), []))) + [
        "date", "platform", "rev_d1", "rev_d90", "target_log_adjustment"
    ]
    if result[required].isna().any().any():
        bad = result.loc[result[required].isna().any(axis=1), ["date", "platform"]]
        raise ValueError(f"Incomplete ML rows after warmup: {bad.head().to_dict('records')}")
    result = result.sort_values(["date", "platform"]).reset_index(drop=True)
    return result


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame,
                      features: list[str], alpha: float,
                      structure: str) -> np.ndarray:
    """Fit on training rows and return log corrections for test rows."""
    output = np.empty(len(test), dtype=float)

    if structure == "Pooled":
        # One binary indicator; scaler and Ridge both fit on training only.
        x_tr = train[features].copy()
        x_te = test[features].copy()
        x_tr["platform_iOS"] = (train["platform"] == "iOS").astype(float)
        x_te["platform_iOS"] = (test["platform"] == "iOS").astype(float)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(x_tr, train["target_log_adjustment"])
        output[:] = pipe.predict(x_te)
    elif structure == "Separate":
        for platform in ["Android", "iOS"]:
            tr = train[train["platform"] == platform]
            mask = (test["platform"] == platform).to_numpy()
            if not mask.any():
                continue
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            pipe.fit(tr[features], tr["target_log_adjustment"])
            output[mask] = pipe.predict(test.loc[mask, features])
    else:
        raise ValueError(f"Unknown structure: {structure}")
    return output


def to_predictions(test: pd.DataFrame, log_corrections: np.ndarray) -> np.ndarray:
    """Convert log corrections back into positive predicted cohort revenue."""
    return (
        test["rev_d1"].to_numpy()
        * test["baseline_factor"].to_numpy()
        * np.exp(log_corrections)
    )


def record_predictions(records: list[pd.DataFrame], test: pd.DataFrame,
                       month: pd.Timestamp, config: str, family: str,
                       features: str, structure: str,
                       alpha: float | None, log_adjustment: np.ndarray) -> None:
    pred = to_predictions(test, log_adjustment)
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError(f"Nonfinite/negative forecasts: {config}, {month}")
    records.append(pd.DataFrame({
        "date": test["date"].to_numpy(),
        "month": month,
        "platform": test["platform"].to_numpy(),
        "config": config,
        "family": family,
        "features": features,
        "structure": structure,
        "alpha": f"{alpha:g}" if alpha is not None else "N/A",
        "rev_d90_actual": test["rev_d90"].to_numpy(),
        "rev_d90_prediction": pred,
    }))


def score(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Revenue-weighted WAPE and signed bias (pooled over cohort rows)."""
    temp = data.assign(
        absolute_error=np.abs(data["rev_d90_prediction"] - data["rev_d90_actual"]),
        signed_error=data["rev_d90_prediction"] - data["rev_d90_actual"],
    )
    result = temp.groupby(group_cols, dropna=False, as_index=False).agg(
        cohorts=("date", "size"),
        actual_revenue=("rev_d90_actual", "sum"),
        absolute_error=("absolute_error", "sum"),
        signed_error=("signed_error", "sum"),
    )
    result["WAPE_%"] = 100 * result["absolute_error"] / result["actual_revenue"]
    result["Bias_%"] = 100 * result["signed_error"] / result["actual_revenue"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None,
                        help="Local cohort CSV instead of BigQuery")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = make_dataset(load_data(args.csv))
    print(f"Available ML rows: {len(df)}; "
          f"dates {df.date.min().date()} - {df.date.max().date()}")
    print("Target is log(actual_D90/D1 / baseline_D90/D1).")

    predictions: list[pd.DataFrame] = []
    fold_metadata = []
    fold_starts = pd.date_range(FIRST_CV_MONTH, LAST_CV_MONTH, freq="MS")

    for month in fold_starts:
        # Train only on labels that would already have matured as of month start.
        train_last = month - pd.Timedelta(days=LABEL_MATURITY_GAP_DAYS)
        month_last = month + pd.offsets.MonthEnd(0)
        train = df.loc[df["date"] <= train_last].copy()
        test = df.loc[df["date"].between(month, month_last)].copy()
        if test.empty:
            raise ValueError(f"Missing test cohorts for month {month.date()}")
        if min(train.groupby("platform").size()) < 60:
            raise ValueError(f"Too few mature training cohorts for month {month.date()}")
        if train["date"].max() >= month - pd.Timedelta(days=90):
            raise AssertionError("D90 label leakage into training")

        fold_metadata.append({
            "forecast_month": str(month.date()),
            "training_from": str(train.date.min().date()),
            "training_to": str(train.date.max().date()),
            "train_cohorts": len(train),
            "forecast_cohorts": len(test),
        })
        print(f"{month:%Y-%m}: train through {train.date.max():%Y-%m-%d} "
              f"({len(train)} rows); forecast {len(test)} rows")

        # Two non-ML controls, retrained/recalibrated each fold.
        record_predictions(predictions, test, month, "Baseline", "Baseline",
                           "None", "None", None, np.zeros(len(test)))
        avg_logs = train.groupby("platform")["target_log_adjustment"].mean()
        calibrated_logs = test["platform"].map(avg_logs).to_numpy()
        record_predictions(predictions, test, month, "Calibrated", "Calibrated",
                           "PlatformMean", "Separate", None, calibrated_logs)

        # Grid-search 8 feature groups x 7 alphas x pooled/separate structures.
        for group_name, features in FEATURE_SETS.items():
            for alpha in ALPHAS:
                for structure in STRUCTURES:
                    config = f"Ridge|{group_name}|{structure}|alpha={alpha:g}"
                    correction = fit_predict_ridge(train, test, features, alpha, structure)
                    record_predictions(predictions, test, month, config, "Ridge",
                                       group_name, structure, alpha, correction)

    preds = pd.concat(predictions, ignore_index=True)
    expected = len(fold_starts) * (2 + len(FEATURE_SETS) * len(ALPHAS) * len(STRUCTURES))
    if preds.groupby(["month", "config"]).ngroups != expected:
        raise AssertionError("Unexpected number of fold/configuration combinations")
    if preds["date"].max() > pd.Timestamp(DEVELOPMENT_END):
        raise AssertionError("Final test period must remain untouched")

    # Select model by pooled WAPE across *all* walk-forward forecasts.
    keys = ["config", "family", "features", "structure", "alpha"]
    overall = score(preds, keys).sort_values(["WAPE_%", "config"])
    by_platform = score(preds, keys + ["platform"])
    monthly = score(preds, keys + ["month", "platform"])
    ridge = overall[overall["family"] == "Ridge"].copy()
    winners_by_group = (ridge.sort_values("WAPE_%")
                        .drop_duplicates(["features", "structure"]))
    best = ridge.iloc[0]["config"]

    print("\nBENCHMARKS AND BEST RIDGE (pooled, walk-forward predictions)")
    overview = pd.concat([
        overall[overall["family"] != "Ridge"],
        overall[overall["config"] == best],
    ]).sort_values("WAPE_%")
    print(overview[["config", "cohorts", "WAPE_%", "Bias_%"]]
          .round(3).to_string(index=False))

    print("\nBEST RIDGE FOR EACH FEATURE SET AND MODEL STRUCTURE")
    print(winners_by_group[["features", "structure", "alpha", "WAPE_%", "Bias_%"]]
          .sort_values(["features", "structure"])
          .round(3).to_string(index=False))

    print("\nTOP 10 RIDGE CONFIGURATIONS")
    print(ridge[["features", "structure", "alpha", "WAPE_%", "Bias_%"]]
          .head(10).round(3).to_string(index=False))

    print("\nPER-PLATFORM PERFORMANCE: BASELINE / CALIBRATED / BEST RIDGE")
    finalists = ["Baseline", "Calibrated", best]
    platform_table = by_platform[by_platform["config"].isin(finalists)].copy()
    print(platform_table[["config", "platform", "cohorts", "WAPE_%", "Bias_%"]]
          .sort_values(["platform", "WAPE_%"])
          .round(3).to_string(index=False))

    print("\nMONTHLY PERFORMANCE OF THE CHOSEN RIDGE MODEL")
    chosen_monthly = score(preds[preds["config"].isin(finalists)],
                           ["config", "month"])
    print(chosen_monthly[["month", "config", "cohorts", "WAPE_%", "Bias_%"]]
          .sort_values(["month", "config"])
          .round(3).to_string(index=False))

    # Save complete reproducible evaluation artifacts.
    overall.to_csv(OUTPUT / "ridge_calendar_cv_rankings.csv", index=False)
    by_platform.to_csv(OUTPUT / "ridge_calendar_cv_platform_scores.csv", index=False)
    monthly.to_csv(OUTPUT / "ridge_calendar_cv_monthly_scores.csv", index=False)
    pd.DataFrame(fold_metadata).to_csv(OUTPUT / "ridge_calendar_cv_folds.csv", index=False)
    chosen_monthly.to_csv(OUTPUT / "ridge_calendar_cv_selected_monthly.csv", index=False)
    # Keep only finalist forecasts; the full grid is available in scores.
    preds[preds["config"].isin(finalists)].to_csv(
        OUTPUT / "ridge_calendar_cv_finalist_predictions.csv", index=False)

    print(f"\nBest Ridge configuration: {best}")
    print("Saved results to outputs/ridge_calendar_cv_*.csv")
    print("IMPORTANT: These WAPE scores are used for model SELECTION, not "
          "an independent final test. March-June 2026 remains untouched.")


if __name__ == "__main__":
    main()
