"""Focused Ridge feature ablation with organic-install share.

Place this file next to ridge_model_search_calendar.py in the VS Code project.
Reads BigQuery (read-only), trains pooled Ridge on mature cohorts only,
and evaluates April-November 2025 month-by-month. Never queries the final
March-June 2026 holdout.

Run:
    python ridge_orgshare_ablation.py
Or test against a locally exported table with the same column names:
    python ridge_orgshare_ablation.py --csv path/to/cohort_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from ridge_model_search_calendar import (
    PROJECT, TABLE, DEVELOPMENT_END, FIRST_CV_MONTH, LAST_CV_MONTH,
    LABEL_MATURITY_GAP_DAYS, FACTOR_FEATURES, REVENUE_FEATURES,
    ENGAGEMENT_FEATURES, MONETIZATION_FEATURES, load_data,
    make_dataset, fit_predict_ridge, to_predictions, score,
)

OUTPUT = Path("outputs")
ALPHA = 500.0  # Winner from previous unweighted Ridge experiment; held fixed.

FEATURES = {
    "R0_Factors": FACTOR_FEATURES,
    "R1_Revenue": FACTOR_FEATURES + REVENUE_FEATURES,
    "R2_Engagement": FACTOR_FEATURES + REVENUE_FEATURES + ENGAGEMENT_FEATURES,
    "R3_Monetization": FACTOR_FEATURES + REVENUE_FEATURES + MONETIZATION_FEATURES,
    "R4_Full": FACTOR_FEATURES + REVENUE_FEATURES + ENGAGEMENT_FEATURES + MONETIZATION_FEATURES,
    "R4_Full_plus_OrgShare": FACTOR_FEATURES + REVENUE_FEATURES
        + ENGAGEMENT_FEATURES + MONETIZATION_FEATURES + ["org_share"],
    "R1_Revenue_plus_OrgShare": FACTOR_FEATURES + REVENUE_FEATURES + ["org_share"],
}


def read_with_org_share(csv_path: str | None) -> pd.DataFrame:
    """Use unchanged original loader, then attach org_share by date/platform.

    The original loader intentionally limits all data to DEVELOPMENT_END.
    We obtain the new column using exactly the same cutoff. Assert exact
    key alignment and no duplicate or missing organic shares.
    """
    original = load_data(csv_path)
    if csv_path:
        extra = pd.read_csv(csv_path, usecols=["date", "platform", "org_share"])
    else:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT)
        sql = (f"SELECT date, platform, org_share FROM `{TABLE}` "
               f"WHERE date <= '{DEVELOPMENT_END}'")
        extra = client.query(sql).to_dataframe()
    extra["date"] = pd.to_datetime(extra["date"])
    extra["org_share"] = pd.to_numeric(extra["org_share"], errors="coerce")
    if extra.duplicated(["date", "platform"]).any():
        raise ValueError("Duplicate (date, platform) in org_share input")
    merged = original.merge(extra, on=["date", "platform"], how="left",
                            validate="one_to_one")
    if merged["org_share"].isna().any():
        raise ValueError("Missing org_share for at least one cohort")
    if (~merged["org_share"].between(0, 1)).any():
        raise ValueError("org_share must be a fraction in [0, 1]")
    return merged


def make_ml_dataset(csv_path: str | None) -> pd.DataFrame:
    """Reuse maturity-safe baseline/feature construction, preserving org_share.

    make_dataset returns the engineered columns but drops no extra columns;
    org_share therefore stays available without changing the old script.
    """
    df = make_dataset(read_with_org_share(csv_path))
    if "org_share" not in df.columns or df["org_share"].isna().any():
        raise ValueError("org_share was lost during feature engineering")
    return df


def save_predictions(target: list[pd.DataFrame], valid: pd.DataFrame,
                     month: pd.Timestamp, name: str,
                     correction: np.ndarray) -> None:
    pred = to_predictions(valid, correction)
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError(f"Invalid predictions for {name} in {month:%Y-%m}")
    target.append(pd.DataFrame({
        "date": valid["date"].to_numpy(), "month": month,
        "platform": valid["platform"].to_numpy(), "config": name,
        "rev_d90_actual": valid["rev_d90"].to_numpy(dtype=float),
        "rev_d90_prediction": pred,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Offline CSV containing all base columns and org_share")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    data = make_ml_dataset(args.csv)
    print(f"ML cohorts: {len(data)}; cutoff: {DEVELOPMENT_END}")
    print(f"Testing {len(FEATURES)} fixed feature sets, pooled Ridge alpha={ALPHA:g}.")
    print("No weighted Ridge, no platform-specific regression, and no final holdout.\n")

    preds: list[pd.DataFrame] = []
    folds = []
    for month in pd.date_range(FIRST_CV_MONTH, LAST_CV_MONTH, freq="MS"):
        train_end = month - pd.Timedelta(days=LABEL_MATURITY_GAP_DAYS)
        end = month + pd.offsets.MonthEnd(0)
        train = data.loc[data["date"] <= train_end].copy()
        valid = data.loc[data["date"].between(month, end)].copy()
        if len(valid) == 0 or set(valid["platform"]) != {"Android", "iOS"}:
            raise ValueError(f"Missing validation cohorts for {month:%Y-%m}")
        if min(train.groupby("platform").size()) < 60:
            raise ValueError(f"Insufficient mature training rows in {month:%Y-%m}")
        if train["date"].max() >= month - pd.Timedelta(days=90):
            raise AssertionError("D90 outcome leakage")

        folds.append({"month": month, "train_last": train.date.max(),
                      "train_rows": len(train), "validation_rows": len(valid)})
        print(f"{month:%Y-%m}: train to {train.date.max():%Y-%m-%d}; "
              f"forecast {len(valid)} cohorts")
        save_predictions(preds, valid, month, "Baseline", np.zeros(len(valid)))
        for name, features in FEATURES.items():
            corr = fit_predict_ridge(train, valid, features, ALPHA, "Pooled")
            save_predictions(preds, valid, month, name, corr)

    all_preds = pd.concat(preds, ignore_index=True)
    assert all_preds.date.max() <= pd.Timestamp(DEVELOPMENT_END)
    overall = score(all_preds, ["config"]).sort_values("WAPE_%")
    platform = score(all_preds, ["config", "platform"]).sort_values(["platform", "WAPE_%"])
    monthly = score(all_preds, ["config", "month"]).sort_values(["month", "config"])
    print("\nFEATURE ABLATION — POOLED WAPE / BIAS")
    print(overall[["config", "cohorts", "WAPE_%", "Bias_%"]].round(3).to_string(index=False))
    print("\nPER-PLATFORM PERFORMANCE")
    print(platform[["config", "platform", "WAPE_%", "Bias_%"]].round(3).to_string(index=False))
    print("\nMONTHLY PERFORMANCE: BASELINE / R4 / R4 + ORGANIC SHARE")
    keep = ["Baseline", "R4_Full", "R4_Full_plus_OrgShare"]
    shown = monthly[monthly.config.isin(keep)].copy()
    shown["month"] = shown["month"].dt.strftime("%Y-%m")
    print(shown[["month", "config", "cohorts", "WAPE_%", "Bias_%"]].round(3).to_string(index=False))

    overall.to_csv(OUTPUT / "ridge_orgshare_ablation_overall.csv", index=False)
    platform.to_csv(OUTPUT / "ridge_orgshare_ablation_platform.csv", index=False)
    monthly.to_csv(OUTPUT / "ridge_orgshare_ablation_monthly.csv", index=False)
    pd.DataFrame(folds).to_csv(OUTPUT / "ridge_orgshare_ablation_folds.csv", index=False)
    all_preds.to_csv(OUTPUT / "ridge_orgshare_ablation_predictions.csv", index=False)
    print("\nSaved outputs/ridge_orgshare_ablation_*.csv")
    print("Results are development/model-selection results, not final test performance.")


if __name__ == "__main__":
    main()
