"""Generate a reproducible, fictional two-platform cohort dataset.

No organization, account, game, customer, or real revenue information is used.
Produces exactly the feature and label columns expected by the research and
production scripts, from 2024-02-01 through 2026-06-30 inclusive.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --output examples/synthetic_cohorts.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(20261124)
    days = pd.date_range('2024-02-01', '2026-06-30', freq='D')
    records = []
    for platform in ['Android', 'iOS']:
        for i, d in enumerate(days):
            period = 2 * np.pi * i / 365.25
            is_ios = platform == 'iOS'
            installs = int(max(100, rng.normal(680 if is_ios else 920, 100)))
            retention = float(np.clip(0.30 + 0.03 * np.sin(period) + (0.025 if is_ios else 0) + rng.normal(0, 0.012), 0.1, 0.8))
            playtime = float(max(2, 12 + (1.0 if is_ios else 0) + 2 * np.sin(period - 0.6) + rng.normal(0, 0.9)))
            ecpm = float(max(1, 15 + (3 if is_ios else 0) + 2 * np.sin(period + 0.3) + rng.normal(0, 0.8)))
            org_share = float(np.clip(0.32 + (0.10 if is_ios else 0) + 0.05 * np.cos(period) + rng.normal(0, 0.035), 0.05, 0.95))
            rpi = float(max(0.08, 0.17 + (0.05 if is_ios else 0) + 0.025 * np.sin(period) + rng.normal(0, 0.012)))
            d1 = installs * rpi
            d0 = d1 * float(np.clip(0.79 + rng.normal(0, 0.018), 0.60, 0.96))
            f71 = float(1.35 + 0.8*(retention-.30) + 0.006*(playtime-12) + rng.normal(0,0.035))
            f287 = float(1.28 + 0.6*(retention-.30) + 0.007*(ecpm-15) + rng.normal(0,0.035))
            f9028 = float(1.40 + 0.3*(retention-.30) + 0.008*(playtime-12) + rng.normal(0,0.055))
            d7 = d1 * max(1.05, f71)
            d28 = d7 * max(1.05, f287)
            d90 = d28 * max(1.05, f9028)
            records.append({
                'date':d.strftime('%Y-%m-%d'), 'platform':platform,
                'installs':installs,'rev_d0':round(d0,4),'rev_d1':round(d1,4),
                'rev_d7':round(d7,4),'rev_d28':round(d28,4),'rev_d90':round(d90,4),
                'retention_d1':round(retention,6),'playtime_d0':round(playtime,5),
                'ecpm_d0':round(ecpm,5),'org_share':round(org_share,6),
            })
    return pd.DataFrame(records).sort_values(['date','platform']).reset_index(drop=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('examples/synthetic_cohorts.csv'))
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sample=generate()
    sample.to_csv(args.output,index=False)
    print(f'Saved {len(sample)} synthetic date/platform cohorts to {args.output}')


if __name__=='__main__':
    main()
