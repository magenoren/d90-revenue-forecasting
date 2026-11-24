# Reported Benchmark Results (Private Data, Aggregates Only)

> **Disclosure:** The following tables reproduce *aggregate scores supplied during the project* for a private, US-only Android/iOS cohort dataset. The underlying real cohort data are **not included**. The public `examples/synthetic_cohorts.csv` generates **different** results and is intended only to demonstrate runnable methods. Obtain approval before posting even these aggregate performance figures publicly.

## Development period — April through November 2025

488 date/platform cohorts, daily prediction grain, monthly walk-forward training.

| Method | WAPE (%) | Bias (%) |
|---|---:|---:|
| A: historical baseline | 9.040 | -4.389 |
| B: end-to-end Ridge | 8.970 | -5.570 |
| C: three-component Ridge | 8.894 | -6.615 |
| D: Ridge to D7 | 8.590 | -4.620 |
| E: Ridge to D28 | 8.589 | -4.964 |

**Feature ablation:** On pooled Ridge (alpha=500), R4 full 8.970% WAPE; R4 plus `org_share` 8.980%. The 0.010 percentage-point deterioration did not support including organic share. On the initial Ridge alpha comparison, unweighted alpha=500 performed best among alpha 100/200/500/1000; predicted-revenue weighting offered no improvement.

## Reserved final holdout — March through June 2026

244 date/platform cohorts: 122 Android, 122 iOS. The model configurations were frozen before this evaluation.

| Model | Pooled WAPE (%) | Pooled Bias (%) | Relative WAPE reduction vs A |
|---|---:|---:|---:|
| **C — three-component Ridge** | **5.720** | **-0.375** | **11.3%** |
| B — end-to-end Ridge | 5.775 | -1.374 | 10.4% |
| E — Ridge to D28 | 5.950 | +0.641 | 7.7% |
| D — Ridge to D7 | 6.274 | +1.326 | 2.7% |
| A — historical baseline | 6.447 | +0.912 | — |

### Holdout results by platform

| Model | Android WAPE (%) | Android Bias (%) | iOS WAPE (%) | iOS Bias (%) |
|---|---:|---:|---:|---:|
| A | 7.737 | -0.829 | 5.618 | +2.030 |
| B | 8.176 | -5.094 | **4.233** | +1.015 |
| C | 7.628 | -4.192 | 4.494 | +2.076 |
| D | 7.402 | -0.760 | 5.550 | +2.666 |
| E | **7.286** | -2.174 | 5.091 | +2.448 |

C wins pooled WAPE, but B is better on iOS and E is better on Android. The favorable pooled Bias for C masks differences in signed platform errors.

### Paired 7-day block-bootstrap intervals

WAPE **improvement over A**, measured in **percentage points** on the holdout:

| Model | WAPE improvement (pp) | Reported 95% bootstrap interval (pp) |
|---|---:|---:|
| B | +0.672 | [+0.002, +1.619] |
| C | +0.727 | [+0.199, +1.445] |
| D | +0.173 | [+0.035, +0.338] |
| E | +0.497 | [+0.239, +0.790] |

The resampling retains both platform observations for each sampled date and uses contiguous seven-day date blocks. These intervals are conditional on the already trained predictions and do **not** account for all model-search uncertainty, data changes, or future deployment regimes. They do **not** establish that C significantly outperforms B.

## Why final and development results differ

- Holdout cohorts may exhibit less structural volatility than the development period.
- Model ranking can shift across time; D/E led development, but C/B led holdout.
- D1 signals may have different predictive strength on Android versus iOS.
- The February 2024–June 2026 historical timeframe, two-platform sample, and fixed feature set limit generalization.
- **Do not re-tune on March–June 2026.** Additional evaluation should use later, genuinely unobserved cohorts.

## How to reproduce the *code* without the private data

```bash
python scripts/compare_component_hybrids.py --csv examples/synthetic_cohorts.csv
python scripts/final_holdout_test.py --csv examples/synthetic_cohorts.csv
python scripts/daily_d90_forecast.py --as-of 2026-06-30 --csv examples/synthetic_cohorts.csv
```

These commands demonstrate the full pipeline, evaluation structure, and daily scoring, but will **not** recreate the scores shown in the tables above because they use independently generated, fictitious data.
