# Data Dictionary and Feature Availability

## Grain and source

One observation per `date × platform` (two platforms: Android and iOS) from a US-only cohort study. The included CSV is completely synthetic; the original experiment used a private BigQuery table with 1,762 daily-platform rows dated 2024-02-01 through 2026-06-30.

| Input | Type | Availability / meaning |
|---|---|---|
| `date` | date | Installation cohort date, not revenue report date |
| `platform` | string | Android or iOS |
| `installs` | integer | Users installed in that cohort; known by D1 |
| `rev_d0` | float | Cumulative revenue by D0; known by D1 |
| `rev_d1` | float | Cumulative revenue by D1; known by D1 |
| `rev_d7` | float / nullable for recent cohorts | Cumulative revenue by D7; training label or mature historical factor only |
| `rev_d28` | float / nullable | Cumulative revenue by D28; training label or mature historical factor only |
| `rev_d90` | float / nullable | Cumulative revenue by D90; training label, evaluation truth, or mature historical factor only |
| `retention_d1` | float | D1 retention as a **fraction** from 0 to 1; available after D1 reporting |
| `playtime_d0` | float | Average D0 playtime in source-defined units (unit not verified in original data); available by D1 |
| `ecpm_d0` | float | Advertising eCPM measured for D0; available by D1 |
| `org_share` | float | Organic installs divided by total installs, 0–1; optional experiment, excluded from final R4 |

An earlier source also contained `eCPM`, which was not used by the final ML feature set and is not required by these scripts. For real production, the organic/paid attribution snapshot must be available by D1 if used as a predictor.

## Engineered R4 predictors

`R4_Full` has 13 numeric features (plus a binary `platform_iOS` indicator in pooled models):

**Historical factors (4)**
- `factor_7_1`: rolling revenue-weighted `SUM(rev_d7)/SUM(rev_d1)` from 30 fully D7-mature cohorts, shifted 7 days.
- `factor_28_7`: `SUM(rev_d28)/SUM(rev_d7)` from 60 fully D28-mature cohorts, shifted 28 days.
- `factor_90_28`: `SUM(rev_d90)/SUM(rev_d28)` from 120 fully D90-mature cohorts, shifted 90 days.
- `baseline_factor`: product of those three factors.

**Revenue / size (3)**
- `rpi_d1 = rev_d1 / installs`.
- `d0_share_of_d1 = rev_d0 / rev_d1`.
- `installs`: cohort size.

**Engagement (4)**
- `retention_d1`.
- `playtime_d0`.
- `retention_d1_vs_prev30 = current_retention_d1 / mean(retention_d1 of prior 30 dates on same platform) - 1`.
- `playtime_d0_vs_prev30`, same construction using `playtime_d0`.

**Monetization (2)**
- `ecpm_d0`.
- `ecpm_d0_vs_prev30`, same construction using `ecpm_d0`.

The previous-30 baseline uses the preceding cohorts only: the current cohort is excluded by a one-day `shift(1)`.

## Research-only features that were not selected

- `org_share`: organic install share.
- `is_weekend`: 1 for Saturday or Sunday in the original implementation.
- `playtime_d0_vs_prev7_same_day_type`: comparison with the seven most recent *weekend* or *weekday* cohort dates, not the seven matching weekdays.

## Training targets and maturity

Each Ridge model predicts a log-correction relative to the corresponding historical factor. To train `D7/D1`, `D28/D7` or `D90/D28`, the corresponding numerator/denominator must be mature **for that historical training cohort**. Training cutoff is the first of each forecast month minus 8, 29, and 91 days for D7, D28, and D90, respectively.

These dates and the older historical-factor windows are separate safeguards. Neither a forecast row's actual `rev_d90` nor the future `rev_d7`/`rev_d28` enters any model's *predictors* for that row.

## Daily pipeline contract

Two rows for target `as_of - 2`, one per platform. Required target data: `installs`, `rev_d0`, `rev_d1`, `retention_d1`, `playtime_d0`, `ecpm_d0`. D7/D28/D90 may be `NULL` for recent cohorts; the daily program validates the most recently mature historical training rows and factor windows independently. Output contains total D90 dollars and D90 revenue per install, but is **not** a forecast of revenue booked on D90 alone.
