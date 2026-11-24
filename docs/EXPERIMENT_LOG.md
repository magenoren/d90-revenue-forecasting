# Experiment Log — How the Project Was Built

This is the chronology of the project rather than a suggested sequence of repeated hyperparameter searches. Do not use the final holdout to tune more models.

## Stage 1. Historical baseline

- Goal: forecast total cumulative D90 revenue from cumulative D1 revenue per install-date/platform cohort.
- Explored a direct D90/D1 multiplier and decompositions through D7 and/or D28. Historical windows of 30, 60, 90, 120 days were studied, with maturity-aware lookbacks.
- Selected the three-step method with windows **30 days for D7/D1; 60 for D28/D7; 120 for D90/D28**. Ratios are weighted implicitly by revenue because each ratio is `sum(numerator revenue)/sum(denominator revenue)`.
- The initial historical baseline configuration-search script is not retained in this curated repo; the selected baseline calculation is fully implemented in `ridge_model_search_calendar.py` and downstream scripts.

## Stage 2. Introduce ML and early-cohort features

- Added D1 retention, D0 playtime, D0 eCPM and relative changes versus the previous 30 cohorts.
- Chose a **logarithmic correction** to the statistical baseline: `log(actual multiplier / historical multiplier)`, with prediction reconstructed by exponentiation.
- Initially compared constant calibration, Ridge regression, and LightGBM. The initial LightGBM comparison is discussed as historical context but its older exploratory script is not part of the curated final repo.
- `ridge_model_search_calendar.py` runs month-by-month Ridge feature-set and regularization experiments, including weekend features, pooled/separate models and baseline/calibration benchmarks. It only uses data through November 2025.

## Stage 3. Stronger regularization and alternative weighting

- `ridge_weighted_experiment.py` tests pooled Ridge with alpha **100, 200, 500, 1000** and either uniform sample weighting or sample weights proportional to predicted D90 revenue.
- The strongest unweighted Ridge configuration used **R4 features, alpha=500** with development WAPE **8.970%**, compared with **9.040%** for the historical baseline. Predicted-revenue weighting did not improve on this choice.
- The comparison optimizes development WAPE, not Ridge's own squared-error training loss.

## Stage 4. Feature ablation and organic share

- `ridge_orgshare_ablation.py` compares nested feature sets using fixed pooled Ridge `alpha=500`.
- Adding organic install share to the full feature set yielded development WAPE **8.980%**, versus **8.970%** without it. It was excluded from the production feature set.
- Historical factors, early revenue, engagement and ad monetization remained in R4. Additional features are not necessarily useful because much of their signal may already be captured in other predictors.

## Stage 5. Reframe the ML target — five architectures

`compare_component_hybrids.py` evaluates five frozen formulas in the development period (April–November 2025, 488 date/platform cohorts):

- A: three historical factors, no ML.
- B: Ridge corrects the entire D90/D1 multiplier.
- C: separate Ridge corrections to D7/D1, D28/D7 and D90/D28.
- D: Ridge corrects D7/D1; later multipliers remain historical.
- E: Ridge corrects D28/D1 directly; remaining D90/D28 multiplier remains historical.

Each component model is trained using its own label maturity cutoff. D and E performed best during development (both approximately **8.59% WAPE**), illustrating the advantage of specialized shorter-horizon targets. These were *development*, not independent, results.

## Stage 6. Final independent holdout

`final_holdout_test.py` evaluates all five fixed models on March–June 2026 and saves platform, monthly, daily, fold-cutoff and bootstrap summaries. The **best pooled holdout** was C at **5.720% WAPE**, versus A at **6.447%**. B was 5.775%, E 5.950%, D 6.274%.

The ranking shifted: D/E won development, while C/B won holdout. This is important evidence against overinterpreting development choices or declaring any one architecture universally superior. See `RESULTS.md` for detailed tables and uncertainty limitations.

## Stage 7. Operationalize the final forecasting method

`daily_d90_forecast.py` uses the chosen **C three-component Ridge** approach with fixed R4 predictors and `alpha=500`. It uses target `run date - 2` and generates Android and iOS predictions. For C, the three Ridge models are retrained with monthly maturity cutoffs (D7: 8, D28: 29, D90: 91 days before the forecast month's first day). Each target's baseline factors and lagged D1 features are computed from data available at the target date.

The script can run from the bundled synthetic CSV; BigQuery output is **explicitly opt-in** and never alters the source cohort table. Windows scheduling is optional.

## Engineering safeguards

- Unique `date × platform` key, contiguous daily cohort history and input validation.
- No D7/D28/D90 target-cohort labels required at D1 forecast time.
- Historical windows shift by outcome maturity; training rows have mature labels at the forecast month's start.
- Training-only scaling and fixed hyperparameters in the final test.
- Real data and generated prediction CSVs excluded from public Git tracking.

## Interpretation

- Accuracy gains on a development period do not guarantee the same architecture wins on an independent time period.
- Platform-level behavior matters: on the reported holdout E led Android and B led iOS; C led pooled WAPE.
- Negative Bias denotes underestimation; near-zero pooled Bias can hide offsetting platform biases.
- The final holdout was already examined; new performance decisions require genuinely new later cohorts or another designated independent evaluation period.
