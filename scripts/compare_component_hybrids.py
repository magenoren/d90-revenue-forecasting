"""Compare five D1->D90 cohort revenue forecasting architectures (A-E).

A: Historical three-step baseline.
B: End-to-end pooled Ridge correction to D90/D1 (existing R4 model).
C: Three independent pooled Ridge corrections to D7/D1, D28/D7, D90/D28.
D: Ridge correction to D7/D1 only; historical D28/D7 and D90/D28.
E: Ridge correction to D28/D1 only; historical D90/D28.

All Ridge models use exactly the same 13 R4 D1-available numeric features and
one iOS indicator; alpha=500. The factor features are constructed with mature
historical cohorts and are recomputed daily, separately by platform.

Training endpoints vary with the target's maturity: D7, D28 and D90 require
8, 29 and 91 days' age at the *start* of each forecast month, respectively.
These are conservative maturity lags, one day beyond Dx.

Monthly walk-forward evaluation: April-November 2025 ONLY.
March-June 2026 final holdout is excluded from the BigQuery query and fitting.

Set up: place beside ridge_model_search_calendar.py and run
  python compare_component_hybrids.py
For a standalone CSV with the same base columns:
  python compare_component_hybrids.py --csv cohort_data.csv

Outputs are created under outputs/component_hybrids_*.csv. This experiment
uses development data; do not treat its winning model as holdout-validated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ridge_model_search_calendar import (
    DEVELOPMENT_END, FIRST_CV_MONTH, LAST_CV_MONTH,
    FEATURE_SETS, load_data, make_dataset, score,
)

OUTPUT = Path("outputs")
ALPHA = 500.0
FEATURES = FEATURE_SETS["R4_Full"]

# Training labels for a component become available once that Dx matures.
# Use an extra conservative one-day gap, consistent with prior D90 scripts.
COMPONENT_MATURITY = {
    "total_90_1": 90,
    "part_7_1": 7,
    "part_28_7": 28,
    "part_90_28": 90,
    "part_28_1": 28,
}

MODEL_NAMES = {
    "A": "A_Baseline",
    "B": "B_EndToEnd_Ridge",
    "C": "C_ThreeComponent_Ridge",
    "D": "D_Ridge_D7_only",
    "E": "E_Ridge_to_D28",
}


def create_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Build historic-only ML labels; labels are never used as predictors."""
    data = data.copy()
    # All Dx revenue values here are LABELS for historical model fitting.
    # For a fresh target cohort, only rev_d1 is used to make the prediction.
    data["y_total_90_1"] = data["target_log_adjustment"]
    data["y_part_7_1"] = np.log(
        (data["rev_d7"] / data["rev_d1"]) / data["factor_7_1"]
    )
    data["y_part_28_7"] = np.log(
        (data["rev_d28"] / data["rev_d7"]) / data["factor_28_7"]
    )
    data["y_part_90_28"] = np.log(
        (data["rev_d90"] / data["rev_d28"]) / data["factor_90_28"]
    )
    # E has one independent model for *cumulative* D28/D1, not a product
    # of D7 and D28 Ridge models. Its baseline is the product of the two
    # corresponding historical factors to match the original 3-step path.
    historic_28_1 = data["factor_7_1"] * data["factor_28_7"]
    data["y_part_28_1"] = np.log(
        (data["rev_d28"] / data["rev_d1"]) / historic_28_1
    )

    required = [f"y_{component}" for component in COMPONENT_MATURITY]
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite component targets")
    return data


def fit_component_model(train: pd.DataFrame, forecast: pd.DataFrame,
                        label: str) -> np.ndarray:
    """Train a pooled Ridge model, independently for the given component."""
    x_train = train[FEATURES].copy()
    x_forecast = forecast[FEATURES].copy()
    # Only one dummy is needed. The underlying historical factors already
    # have platform-specific estimates.
    x_train["platform_iOS"] = (train["platform"] == "iOS").astype(float)
    x_forecast["platform_iOS"] = (forecast["platform"] == "iOS").astype(float)

    model = Pipeline([
        ("standardize", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])
    model.fit(x_train, train[label])
    correction = model.predict(x_forecast)
    if not np.isfinite(correction).all():
        raise ValueError(f"Invalid correction for {label}")
    return correction


def append_predictions(records: list[pd.DataFrame], forecast: pd.DataFrame,
                       month: pd.Timestamp, name: str,
                       correction: np.ndarray) -> None:
    """Convert a summed log correction into a D90 forecast per daily cohort."""
    prediction = (
        forecast["rev_d1"].to_numpy(dtype=float)
        * forecast["baseline_factor"].to_numpy(dtype=float)
        * np.exp(correction)
    )
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError(f"Invalid D90 predictions for {name}")

    records.append(pd.DataFrame({
        "date": forecast["date"].to_numpy(),
        "platform": forecast["platform"].to_numpy(),
        "month": month,
        "config": name,
        "actual_d90": forecast["rev_d90"].to_numpy(dtype=float),
        "predicted_d90": prediction,
        "rev_d90_actual": forecast["rev_d90"].to_numpy(dtype=float),
        "rev_d90_prediction": prediction,
        "baseline_factor": forecast["baseline_factor"].to_numpy(dtype=float),
        "predicted_factor": prediction / forecast["rev_d1"].to_numpy(dtype=float),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Offline source CSV instead of BigQuery")
    args = parser.parse_args()

    data = create_targets(make_dataset(load_data(args.csv)))
    assert data.date.max() <= pd.Timestamp(DEVELOPMENT_END), (
        "Final holdout was accidentally included"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    folds = []
    components = []

    for month in pd.date_range(FIRST_CV_MONTH, LAST_CV_MONTH, freq="MS"):
        month_end = month + pd.offsets.MonthEnd(0)
        forecast = data.loc[data.date.between(month, month_end)].copy()
        if set(forecast.platform.unique()) != {"Android", "iOS"}:
            raise ValueError(f"Forecast month lacks both platforms: {month:%Y-%m}")
        # Unlike a D90-only model, each component can train from the newest
        # mature labels appropriate to its *own* target horizon.
        corrections: dict[str, np.ndarray] = {}
        details = {"forecast_month": month.strftime("%Y-%m"),
                   "forecast_cohorts": len(forecast)}
        for component, maturity in COMPONENT_MATURITY.items():
            end_of_training = month - pd.Timedelta(days=maturity + 1)
            mature = data.loc[data.date <= end_of_training].copy()
            if mature.empty or min(mature.groupby("platform").size()) < 60:
                raise ValueError(f"Insufficient training rows for {component} in {month:%Y-%m}")
            if mature.date.max() >= month - pd.Timedelta(days=maturity):
                raise AssertionError(f"Immature training labels used for {component}")
            corrections[component] = fit_component_model(
                mature, forecast, f"y_{component}"
            )
            details[f"{component}_train_last"] = mature.date.max().date().isoformat()
            details[f"{component}_train_rows"] = len(mature)

            # Helpful optional diagnostics: actual versus predicted component
            # ratios. These are evaluation ONLY and do not enter predictions.
            component_to_revenues = {
                "part_7_1": ("rev_d7", "rev_d1", "factor_7_1"),
                "part_28_7": ("rev_d28", "rev_d7", "factor_28_7"),
                "part_90_28": ("rev_d90", "rev_d28", "factor_90_28"),
                "part_28_1": ("rev_d28", "rev_d1", None),
            }
            if component in component_to_revenues:
                num, den, hist_col = component_to_revenues[component]
                hist = (forecast[hist_col].to_numpy(dtype=float)
                        if hist_col is not None else
                        (forecast["factor_7_1"]*forecast["factor_28_7"]).to_numpy(dtype=float))
                actual = (forecast[num] / forecast[den]).to_numpy(dtype=float)
                predicted = hist * np.exp(corrections[component])
                components.append(pd.DataFrame({
                    "month": month,
                    "date": forecast.date.to_numpy(),
                    "platform": forecast.platform.to_numpy(),
                    "component": component,
                    "actual_factor": actual,
                    "predicted_factor": predicted,
                    "historical_factor": hist,
                }))

        zero = np.zeros(len(forecast), dtype=float)
        formulas = {
            "A": zero,
            "B": corrections["total_90_1"],
            "C": (corrections["part_7_1"] + corrections["part_28_7"]
                  + corrections["part_90_28"]),
            "D": corrections["part_7_1"],
            "E": corrections["part_28_1"],
        }
        for key, adjustment in formulas.items():
            append_predictions(predictions, forecast, month, MODEL_NAMES[key], adjustment)
        folds.append(details)
        print(f"{month:%Y-%m}: forecast {len(forecast)} cohorts; "
              f"last training cohort D7={details['part_7_1_train_last']}, "
              f"D28={details['part_28_1_train_last']}, "
              f"D90={details['total_90_1_train_last']}")

    all_predictions = pd.concat(predictions, ignore_index=True)
    if all_predictions.date.max() > pd.Timestamp(DEVELOPMENT_END):
        raise AssertionError("A final-holdout cohort was evaluated")
    if all_predictions.groupby(["month", "config"]).ngroups != 8 * len(MODEL_NAMES):
        raise AssertionError("Incorrect number of configurations per month")

    overall = score(all_predictions, ["config"]).sort_values("WAPE_%")
    by_platform = score(all_predictions, ["config", "platform"]).sort_values(
        ["platform", "WAPE_%"]
    )
    monthly = score(all_predictions, ["config", "month"]).sort_values(
        ["month", "config"]
    )
    print("\nFIVE-APPROACH COMPARISON — POOLED")
    print(overall[["config", "cohorts", "WAPE_%", "Bias_%"]].round(3).to_string(index=False))
    print("\nPER-PLATFORM PERFORMANCE")
    print(by_platform[["config", "platform", "WAPE_%", "Bias_%"]].round(3).to_string(index=False))
    print("\nMONTHLY PERFORMANCE")
    monthly_display = monthly[["month", "config", "cohorts", "WAPE_%", "Bias_%"]].copy()
    monthly_display["month"] = monthly_display["month"].dt.strftime("%Y-%m")
    print(monthly_display.round(3).to_string(index=False))

    prefix = OUTPUT / "component_hybrids"
    all_predictions.to_csv(f"{prefix}_daily_predictions.csv", index=False)
    overall.to_csv(f"{prefix}_overall.csv", index=False)
    by_platform.to_csv(f"{prefix}_platform.csv", index=False)
    monthly.to_csv(f"{prefix}_monthly.csv", index=False)
    pd.DataFrame(folds).to_csv(f"{prefix}_folds.csv", index=False)
    pd.concat(components, ignore_index=True).to_csv(
        f"{prefix}_component_diagnostics.csv", index=False
    )
    print("\nSaved outputs/component_hybrids_*.csv")
    print("Development results only; the final March-June 2026 test remains untouched.")


if __name__ == "__main__":
    main()
