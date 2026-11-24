"""Compare unweighted vs revenue-weighted Ridge for D90 cohort forecasting.

Uses the *same* data preparation, R4 features, baseline and walk-forward folds
as ridge_model_search_calendar.py. Changing only how strongly each historical
cohort contributes to Ridge's training loss and Ridge alpha.

Run from your VS Code project folder (both .py files in the same directory):
    python ridge_weighted_experiment.py

Offline test with an appropriately structured CSV:
    python ridge_weighted_experiment.py --csv cohort_data.csv

The March-June 2026 final test period is never queried or evaluated.
This script reads BigQuery but never modifies it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import the exact data construction and metric logic from our previous
# experiment. Keeping it identical makes the comparison apples-to-apples.
from ridge_model_search_calendar import (
    PROJECT, TABLE, DEVELOPMENT_END, FIRST_CV_MONTH, LAST_CV_MONTH,
    LABEL_MATURITY_GAP_DAYS, FEATURE_SETS,
    load_data, make_dataset, to_predictions, score,
)

OUTPUT = Path("outputs")
FEATURES = FEATURE_SETS["R4_Full"]
ALPHAS = [100.0, 200.0, 500.0, 1000.0]
WEIGHTING = ["Unweighted", "PredictedD90Weighted"]


def fit_ridge(train: pd.DataFrame, test: pd.DataFrame,
              alpha: float, weighting: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit a POOLED Ridge model and predict log adjustments for the next month.

    Pooled training gets a platform indicator, while the baseline multiplier
    components are already computed separately by platform upstream.

    In weighted mode, sample_weight = predicted D90 / mean(predicted D90).
    Crucially, predicted D90 uses only D1 revenue and the mature baseline
    multiplier, both of which would have been available at D1. We never use
    actual D90 revenue to construct weights.

    Normalizing weights to mean 1 keeps the effective scale of alpha roughly
    comparable between weighted and unweighted experiments.
    """
    x_train = train[FEATURES].copy()
    x_test = test[FEATURES].copy()
    x_train["platform_iOS"] = (train["platform"] == "iOS").astype(float)
    x_test["platform_iOS"] = (test["platform"] == "iOS").astype(float)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    fit_kwargs = {}
    if weighting == "PredictedD90Weighted":
        expected_d90 = (
            train["rev_d1"].to_numpy(dtype=float)
            * train["baseline_factor"].to_numpy(dtype=float)
        )
        if not np.isfinite(expected_d90).all() or (expected_d90 <= 0).any():
            raise ValueError("Invalid D1-based predicted D90 training weights")
        weights = expected_d90 / expected_d90.mean()
        # StandardScaler fits on training rows only, in both variants.
        # The sample weights are applied to Ridge's squared-error loss.
        fit_kwargs["ridge__sample_weight"] = weights
    elif weighting != "Unweighted":
        raise ValueError(f"Unrecognized weighting method: {weighting}")

    pipeline.fit(x_train, train["target_log_adjustment"], **fit_kwargs)
    corrections = pipeline.predict(x_test)
    if not np.isfinite(corrections).all():
        raise ValueError("Non-finite model outputs")

    reg = pipeline.named_steps["ridge"]
    coefficients = pd.DataFrame({
        "feature": x_train.columns,
        "standardized_coefficient": reg.coef_,
    })
    coefficients["intercept"] = float(reg.intercept_)
    return corrections, coefficients


def append_predictions(records: list[pd.DataFrame], test: pd.DataFrame,
                       month: pd.Timestamp, config: str,
                       log_adjustment: np.ndarray) -> None:
    prediction = to_predictions(test, log_adjustment)
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError(f"Invalid D90 predictions for {config}, {month}")
    records.append(pd.DataFrame({
        "date": test["date"].to_numpy(),
        "month": month,
        "platform": test["platform"].to_numpy(),
        "config": config,
        "rev_d90_actual": test["rev_d90"].to_numpy(dtype=float),
        "rev_d90_prediction": prediction,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None,
                        help="Read a local cohort CSV instead of BigQuery")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = make_dataset(load_data(args.csv))
    print(f"R4 has {len(FEATURES)} numeric features + 1 platform indicator")
    print(f"Candidate alphas: {ALPHAS}")
    print("Testing unweighted Ridge and Ridge weighted by predicted D90 revenue")
    print("March-June 2026 remains a protected final holdout.\n")

    predictions: list[pd.DataFrame] = []
    fold_info = []
    coefficients_all = []
    months = pd.date_range(FIRST_CV_MONTH, LAST_CV_MONTH, freq="MS")

    for month in months:
        train_end = month - pd.Timedelta(days=LABEL_MATURITY_GAP_DAYS)
        month_end = month + pd.offsets.MonthEnd(0)
        train = data.loc[data["date"] <= train_end].copy()
        valid = data.loc[data["date"].between(month, month_end)].copy()

        if valid.empty or set(valid["platform"]) != {"Android", "iOS"}:
            raise ValueError(f"Missing forecast cohorts in {month:%Y-%m}")
        if min(train.groupby("platform").size()) < 60:
            raise ValueError(f"Too few mature training cohorts in {month:%Y-%m}")
        if train["date"].max() >= month - pd.Timedelta(days=90):
            raise AssertionError("Training cohorts' D90 revenue is not mature")
        fold_info.append({
            "month": month.strftime("%Y-%m"),
            "train_first": train.date.min().date().isoformat(),
            "train_last": train.date.max().date().isoformat(),
            "n_train": len(train),
            "n_forecast": len(valid),
        })
        print(f"{month:%Y-%m}: train <= {train.date.max():%Y-%m-%d} "
              f"({len(train)} cohorts), forecast {len(valid)} cohorts")

        # Non-ML benchmarks: identical to the previous experiment.
        append_predictions(predictions, valid, month,
                           "Baseline", np.zeros(len(valid)))
        avg_log = train.groupby("platform")["target_log_adjustment"].mean()
        append_predictions(predictions, valid, month,
                           "Calibrated", valid["platform"].map(avg_log).to_numpy())

        for weighting in WEIGHTING:
            for alpha in ALPHAS:
                name = f"Ridge|R4_Full|Pooled|{weighting}|alpha={alpha:g}"
                corrections, coef = fit_ridge(train, valid, alpha, weighting)
                append_predictions(predictions, valid, month, name, corrections)
                coef.insert(0, "month", month)
                coef.insert(1, "config", name)
                coefficients_all.append(coef)

    all_preds = pd.concat(predictions, ignore_index=True)
    if all_preds["date"].max() > pd.Timestamp(DEVELOPMENT_END):
        raise AssertionError("Holdout period was included accidentally")
    expected_configs = 2 + len(ALPHAS) * len(WEIGHTING)
    if all_preds.groupby(["month", "config"]).ngroups != len(months) * expected_configs:
        raise AssertionError("Incorrect number of monthly comparisons")

    overall = score(all_preds, ["config"]).sort_values(["WAPE_%", "config"])
    platform_scores = score(all_preds, ["config", "platform"])
    monthly_scores = score(all_preds, ["config", "month"])

    ridge_scores = overall[overall["config"].str.startswith("Ridge|")]
    best_weighted = ridge_scores[
        ridge_scores["config"].str.contains("|PredictedD90Weighted|", regex=False)
    ].iloc[0]["config"]
    best_unweighted = ridge_scores[
        ridge_scores["config"].str.contains("|Unweighted|", regex=False)
    ].iloc[0]["config"]
    reference = "Ridge|R4_Full|Pooled|Unweighted|alpha=100"
    highlighted = list(dict.fromkeys([
        "Baseline", "Calibrated", reference, best_unweighted, best_weighted,
    ]))

    print("\nBENCHMARKS AND BEST RIDGE VARIANTS")
    print(overall[overall["config"].isin(highlighted)][
        ["config", "cohorts", "WAPE_%", "Bias_%"]
    ].round(3).to_string(index=False))

    print("\nALL RIDGE CONFIGURATIONS (LOWEST WAPE FIRST)")
    print(ridge_scores[["config", "WAPE_%", "Bias_%"]]
          .round(3).to_string(index=False))

    print("\nPER-PLATFORM RESULTS FOR KEY MODELS")
    print(platform_scores[platform_scores["config"].isin(highlighted)][
        ["config", "platform", "cohorts", "WAPE_%", "Bias_%"]
    ].sort_values(["platform", "WAPE_%"]).round(3).to_string(index=False))

    print("\nMONTHLY RESULTS FOR BASELINE / ORIGINAL RIDGE / BEST WEIGHTED RIDGE")
    shown_monthly = ["Baseline", reference, best_weighted]
    print(monthly_scores[monthly_scores["config"].isin(shown_monthly)][
        ["month", "config", "cohorts", "WAPE_%", "Bias_%"]
    ].sort_values(["month", "config"]).round(3).to_string(index=False))

    # New filenames ensure previous model-selection results are preserved.
    overall.to_csv(OUTPUT / "ridge_weighted_cv_scores.csv", index=False)
    platform_scores.to_csv(OUTPUT / "ridge_weighted_cv_platform_scores.csv", index=False)
    monthly_scores.to_csv(OUTPUT / "ridge_weighted_cv_monthly_scores.csv", index=False)
    pd.DataFrame(fold_info).to_csv(OUTPUT / "ridge_weighted_cv_folds.csv", index=False)
    pd.concat(coefficients_all, ignore_index=True).to_csv(
        OUTPUT / "ridge_weighted_cv_coefficients.csv", index=False)
    all_preds[all_preds["config"].isin(highlighted)].to_csv(
        OUTPUT / "ridge_weighted_cv_key_predictions.csv", index=False)

    print(f"\nBest unweighted Ridge: {best_unweighted}")
    print(f"Best weighted Ridge:   {best_weighted}")
    print("Saved results to outputs/ridge_weighted_cv_*.csv")
    print("These scores are for development/model selection; the final test remains untouched.")


if __name__ == "__main__":
    main()
