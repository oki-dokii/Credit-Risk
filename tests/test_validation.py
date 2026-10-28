"""Numerical regressions and alarm tests. SAS fixtures are Python test doubles."""
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from src.data_utils import FrozenPreprocessor, split_data
from src.woe_encoder import WOEBinEncoder
from src.scorecard_model import LogisticScorecard
from src.validation import (synthetic_data, metrics, population_stability,
                            monitoring_experiments, stress_tests)
from src.reconciliation import export_bridge, compare_bridge


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_data(4000, 42)
        a, b, c, d, e, f = split_data(cls.data, .2, .2, 42)
        cls.prep = FrozenPreprocessor().fit(a)
        cls.raw_proxy, cls.yproxy = c, f
        cls.splits = {name: (cls.prep.transform(X), y) for name, X, y in
                      [('train', a, d), ('validation', b, e), ('simulated_oot_proxy', c, f)]}
        cls.model = LogisticScorecard(WOEBinEncoder(), lr_params={'max_iter': 3000, 'tol': 1e-8})
        cls.model.fit(*cls.splits['train'])

    def test_known_metrics_and_reversed_ranking(self):
        perfect = metrics([0, 0, 1, 1], [.1, .2, .8, .9])
        reverse = metrics([0, 0, 1, 1], [.9, .8, .2, .1])
        self.assertEqual(perfect['auc'], 1)
        self.assertEqual(perfect['gini'], 1)
        self.assertEqual(reverse['gini'], -1)
        self.assertEqual(reverse['ks'], 1)
        self.assertAlmostEqual(perfect['brier'], .025)

    def test_probability_validation_and_single_class(self):
        for p in [[np.nan, .2], [-.1, .2], [.1, 1.1]]:
            with self.assertRaises(ValueError):
                metrics([0, 1], p)
        self.assertTrue(np.isnan(metrics([0, 0], [.1, .2])['auc']))

    def test_joint_calibration_regression(self):
        rng = np.random.default_rng(3)
        p = expit(rng.normal(size=20000))
        result = metrics(rng.binomial(1, p), p)
        self.assertTrue(result['calibration_fit_ok'])
        self.assertAlmostEqual(result['calibration_intercept'], 0, delta=.07)
        self.assertAlmostEqual(result['calibration_slope'], 1, delta=.07)

    def test_no_sample_overlap(self):
        sets = [set(X.index) for X, y in self.splits.values()]
        self.assertEqual(sum(map(len, sets)), len(self.data))
        for i in range(3):
            for j in range(i+1, 3):
                self.assertFalse(sets[i] & sets[j])

    def test_imputation_frozen_to_train(self):
        training = self.data.iloc[:20].copy()
        training['MonthlyIncome'] = np.arange(20.)
        prep = FrozenPreprocessor().fit(training)
        learned = prep.medians_.copy()
        holdout = self.data.iloc[:2].copy()
        holdout['MonthlyIncome'] = [np.nan, 50000.]
        result = prep.transform(holdout)
        self.assertEqual(result.MonthlyIncome.iloc[0], 9.5)
        pd.testing.assert_series_equal(prep.medians_, learned)

    def test_woe_counts_and_exact_boundaries(self):
        X = pd.DataFrame({'discrete': [0, 1, 2, 0, 1, 2]*20,
                          'constant': np.ones(120), 'continuous': np.linspace(0, 1, 120)})
        y = pd.Series([0, 1, 0, 1, 0, 1]*20)
        enc = WOEBinEncoder(enforce_monotone=False, min_bin_pct=0).fit(X, y)
        for feature in X:
            table = enc.get_bin_table(feature)
            self.assertEqual(table['count'].sum(), len(X))
            edges = table.bin_lower.to_numpy()
            encoded = enc.transform(X)
            for row in table.itertuples():
                mask = X[feature].ge(row.bin_lower) & X[feature].lt(row.bin_upper)
                self.assertTrue(np.allclose(encoded.loc[mask, feature], row.woe))
            finite = edges[np.isfinite(edges)]
            if len(finite):
                probe = X.iloc[:len(finite)].copy()
                probe[feature] = finite
                transformed = enc.transform(probe)[feature]
                for boundary, value in zip(finite, transformed):
                    expected = table.loc[table.bin_lower.eq(boundary), 'woe'].iloc[0]
                    self.assertAlmostEqual(value, expected)

    def test_scorecard_points_intercept_sign(self):
        table = self.model.get_scorecard_table()
        X = self.splits['validation'][0].iloc[:10]
        woe = self.model.woe_encoder.transform(X)
        scaler = self.model.scorecard_scaler
        for index in X.index:
            total = 0.
            for feature in self.model.selected_features_:
                row = table.loc[table.variable.eq(feature) &
                    np.isclose(table.woe, round(woe.loc[index, feature], 4))].iloc[0]
                total += row.points
            expected = scaler.proba_to_score(self.model.predict_proba(X.loc[[index]]))[0]
            self.assertAlmostEqual(total, expected, delta=1.5)

    def test_psi_constant_duplicate_and_identity(self):
        self.assertEqual(population_stability(np.ones(20), np.ones(20)), 0)
        self.assertGreater(population_stability(np.ones(20), np.ones(20)*2), .1)
        self.assertEqual(population_stability([0, 0, 1, 1], [0, 0, 1, 1]), 0)
        with self.assertRaises(ValueError):
            population_stability([], [1])

    def test_monitoring_alert_paths(self):
        thresholds = dict(psi_alert_threshold=.1, gini_alert_threshold=.05, calibration_gap_threshold=.03)
        train_pd = self.model.predict_proba(self.splits['train'][0])
        result, _ = monitoring_experiments(self.model, self.prep, self.raw_proxy, self.yproxy, train_pd, thresholds)
        result = result.set_index('scenario')
        self.assertTrue(result.loc['covariate_shift_fixed_labels', 'psi_alert'])
        self.assertTrue(result.loc['synthetic_outcome_inversion', 'performance_alert'])
        self.assertFalse(result.loc['unshifted_control', 'performance_alert'])

    def test_stress_zero_approvals_is_nan(self):
        stress = stress_tests(self.model, self.prep, self.raw_proxy, self.yproxy, approval_score=1000)
        self.assertTrue(stress.approved_expected_default_rate.isna().all())
        self.assertTrue(stress.approval_rate.eq(0).all())

    def test_bridge_matching_and_rejection(self):
        # These fixtures test comparator mechanics; they are NOT SAS execution.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                export_bridge(self.model, self.splits, root)
            pred = pd.read_csv(root/'python_predictions.csv').rename(columns={'python_pd': 'sas_pd'})
            coeff = pd.read_csv(root/'python_coefficients.csv')
            pfile, cfile = root/'fixture_predictions.csv', root/'fixture_coefficients.csv'
            coeff.to_csv(cfile, index=False)
            pred.sample(frac=1, random_state=4).to_csv(pfile, index=False)
            self.assertEqual(compare_bridge(root, pfile, cfile)['status'], 'PASS_NUMERICAL_CHECKS')
            for mutation in ['missing', 'duplicate', 'target', 'split', 'design', 'nan']:
                altered = pred.copy()
                if mutation == 'missing':
                    altered = altered.iloc[1:]
                elif mutation == 'duplicate':
                    altered.loc[0, 'row_id'] = altered.loc[1, 'row_id']
                elif mutation == 'target':
                    altered.loc[0, 'target'] = 1-altered.loc[0, 'target']
                elif mutation == 'split':
                    altered.loc[0, 'split'] = 'wrong'
                elif mutation == 'design':
                    altered.loc[0, 'w001'] += .001
                else:
                    altered.loc[0, 'sas_pd'] = np.nan
                altered.to_csv(pfile, index=False)
                with self.assertRaises(ValueError, msg=mutation):
                    compare_bridge(root, pfile, cfile)
            pred.to_csv(pfile, index=False)
            coeff.loc[0, 'estimate'] += .1
            coeff.to_csv(cfile, index=False)
            self.assertEqual(compare_bridge(root, pfile, cfile)['status'], 'FAIL')
            coeff = pd.read_csv(root/'python_coefficients.csv')
            coeff.to_csv(cfile, index=False)
            pred['sas_pd'] = .5
            pred.to_csv(pfile, index=False)
            self.assertEqual(compare_bridge(root, pfile, cfile)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
