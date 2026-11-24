"""Offline smoke tests of forecasting invariants; run via unittest discover."""
from __future__ import annotations
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import daily_d90_forecast as daily
from ridge_model_search_calendar import load_data as load_development_data
from ridge_model_search_calendar import make_dataset
from compare_component_hybrids import create_targets, fit_component_model
from final_holdout_test import load_full_data

CSV=ROOT/'examples'/'synthetic_cohorts.csv'


class ForecastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=pd.read_csv(CSV)

    def test_sample_grain_and_nonnegative_revenue(self):
        x=self.source
        self.assertEqual(len(x),1762)
        self.assertFalse(x.duplicated(['date','platform']).any())
        self.assertEqual(set(x.platform),{'Android','iOS'})
        self.assertTrue((x.rev_d90>=x.rev_d28).all())
        self.assertTrue((x.rev_d28>=x.rev_d7).all())
        self.assertTrue((x.rev_d7>=x.rev_d1).all())

    def test_development_loader_cannot_see_holdout_rows(self):
        raw=load_development_data(str(CSV))
        self.assertLessEqual(raw.date.max(),pd.Timestamp('2025-11-30'))
        self.assertEqual(raw.platform.nunique(),2)

    def test_daily_three_component_matches_holdout_formula(self):
        # Test date March 1, 2026 is an exact forecast-month boundary.
        as_of=date(2026,3,3)
        target=as_of - pd.Timedelta(days=2)
        daily_raw=daily.load_raw(target,CSV)
        live_features=daily.build_features(daily_raw)
        actual_daily=daily.forecast_for_day(live_features,as_of)

        research_raw=load_full_data(str(CSV))
        research=create_targets(make_dataset(research_raw))
        march1=pd.Timestamp('2026-03-01')
        forecast=research.loc[research.date==march1].copy()
        corrections=np.zeros(len(forecast))
        for component,maturity in [('part_7_1',7),('part_28_7',28),('part_90_28',90)]:
            cutoff=march1-pd.Timedelta(days=maturity+1)
            training=research.loc[research.date<=cutoff].copy()
            corrections+=fit_component_model(training,forecast,f'y_{component}')
        reconstructed=(forecast.rev_d1.to_numpy()*forecast.baseline_factor.to_numpy()*np.exp(corrections))
        expected=pd.DataFrame({'platform':forecast.platform.to_numpy(),'pred':reconstructed}).set_index('platform')
        observed=actual_daily.set_index('platform')
        np.testing.assert_allclose(observed.loc[expected.index,'predicted_d90'].to_numpy(),expected.pred.to_numpy(),rtol=1e-10,atol=1e-8)

    def test_recent_immature_outcomes_not_needed(self):
        # Compare data with all future outcomes filled vs what production
        # actually receives (recent cohort outcomes are unavailable).
        as_of=date(2026,6,30)
        target=date(2026,6,28)
        whole=daily.load_raw(target,CSV)
        full=daily.forecast_for_day(daily.build_features(whole),as_of)
        censored=whole.copy()
        censored.loc[censored.date>=pd.Timestamp('2026-06-22'),'rev_d7']=np.nan
        censored.loc[censored.date>=pd.Timestamp('2026-06-01'),'rev_d28']=np.nan
        censored.loc[censored.date>=pd.Timestamp('2026-04-01'),'rev_d90']=np.nan
        masked=daily.forecast_for_day(daily.build_features(censored),as_of)
        for col in ['predicted_d90','baseline_predicted_d90','predicted_factor_d90_d1']:
            np.testing.assert_allclose(full[col].to_numpy(),masked[col].to_numpy(),rtol=1e-10,atol=1e-8)
        self.assertEqual(len(masked),2)

    def test_duplicate_cohorts_rejected(self):
        as_of=date(2026,6,30)
        raw=self.source.copy()
        raw=pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
        path=ROOT/'outputs'/'test_duplicate.csv'
        path.parent.mkdir(exist_ok=True)
        try:
            raw.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate install date/platform'):
                daily.load_raw(date(2026,6,28),path)
        finally:
            path.unlink(missing_ok=True)


if __name__=='__main__':
    unittest.main()
