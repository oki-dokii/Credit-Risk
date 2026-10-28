"""Independent metrics and controlled experiments; never fit on evaluation rows."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from src.data_utils import FEATURE_COLS, TARGET_COL


def synthetic_data(n=12000, seed=42):
    """Artificial applicants, not a reconstruction of Kaggle or lending evidence."""
    if n < 1000:
        raise ValueError('Use at least 1000 synthetic rows for the three-way demo')
    rng = np.random.default_rng(seed)
    income = rng.lognormal(8.6, .65, n)
    late90 = rng.poisson(.18, n)
    late30 = rng.poisson(.35, n)
    late60 = rng.poisson(.12, n)
    util = rng.beta(1.4, 3.5, n)
    age = rng.integers(21, 85, n)
    debt = rng.beta(1.5, 4, n) * 2
    df = pd.DataFrame(dict(zip(FEATURE_COLS, [
        util, age, late30, debt, income, rng.poisson(7, n), late90,
        rng.poisson(1, n), late60, rng.poisson(1.1, n)])))
    latent = (-4.4 + 3.1*util + .65*late30 + .9*late90 + .75*late60
              + .5*debt - .65*np.log(income/5000) - .015*(age-45))
    df[TARGET_COL] = rng.binomial(1, expit(latent))
    for col in ['MonthlyIncome', 'NumberOfDependents']:
        df.loc[rng.random(n) < .06, col] = np.nan
    return df


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned, nonempty binary labels and predictions required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Predictions must be finite probabilities')
    two_classes = np.unique(y).size == 2
    auc = roc_auc_score(y, p) if two_classes else np.nan
    if two_classes:
        fpr, tpr, _ = roc_curve(y, p)
        ks = float(np.max(np.abs(tpr - fpr)))
    else:
        ks = np.nan
    z = logit(np.clip(p, 1e-10, 1-1e-10))
    # Joint calibration regression logit(Y) = intercept + slope*logit(PD).
    intercept = slope = np.nan
    converged = False
    if two_classes and np.std(z) > 1e-10:
        design = np.column_stack([np.ones(len(z)), z])
        def objective(b):
            eta = design @ b
            return np.mean(np.logaddexp(0, eta) - y*eta)
        def gradient(b):
            return design.T @ (expit(design @ b)-y) / len(y)
        fit = minimize(objective, [0., 1.], jac=gradient, method='BFGS',
                       options={'gtol': 1e-8, 'maxiter': 2000})
        converged = bool(fit.success or np.max(np.abs(gradient(fit.x))) < 1e-6)
        if converged:
            intercept, slope = map(float, fit.x)
    return dict(n=len(y), event_rate=float(y.mean()), mean_pd=float(p.mean()),
                auc=float(auc), gini=float(2*auc-1), ks=ks,
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                calibration_gap=float(p.mean()-y.mean()),
                calibration_intercept=intercept, calibration_slope=slope,
                calibration_fit_ok=converged)


def calibration_table(y, p, n_bins=10):
    frame = pd.DataFrame({'observed': np.asarray(y), 'pd': np.asarray(p)})
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins+1)))
    if len(edges) < 2:
        frame['band'] = 0
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        frame['band'] = pd.cut(frame.pd, edges, labels=False, include_lowest=True)
    result = frame.groupby('band').agg(n=('observed', 'size'),
        observed_rate=('observed', 'mean'), mean_pd=('pd', 'mean')).reset_index()
    result['gap'] = result.mean_pd-result.observed_rate
    return result


def population_stability(reference, actual, n_bins=10, eps=1e-6):
    reference, actual = np.asarray(reference, float), np.asarray(actual, float)
    if not len(reference) or not len(actual) or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('PSI requires nonempty finite arrays')
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins+1)[1:-1]))
    if np.ptp(reference) == 0:
        inner = np.array([reference[0], np.nextafter(reference[0], np.inf)])
    edges = np.r_[-np.inf, inner, np.inf]
    a = np.histogram(reference, bins=edges)[0].astype(float) + eps
    b = np.histogram(actual, bins=edges)[0].astype(float) + eps
    a, b = a/a.sum(), b/b.sum()
    return float(np.sum((b-a)*np.log(b/a)))


def monitoring_experiments(model, prep, X, y, train_pd, thresholds):
    """Feature shift plus an explicitly artificial outcome-inversion alarm test."""
    baseline = model.predict_proba(prep.transform(X))
    base_metrics = metrics(y, baseline)
    shifted = X.copy()
    shifted['MonthlyIncome'] = shifted.MonthlyIncome.fillna(prep.medians_['MonthlyIncome']) * .35
    shifted['RevolvingUtilizationOfUnsecuredLines'] = 1.5
    shifted['NumberOfTimes90DaysLate'] = shifted.NumberOfTimes90DaysLate + 4
    batches = [('unshifted_control', X, y),
               ('covariate_shift_fixed_labels', shifted, y),
               ('artificial_outcome_inversion', X, 1-np.asarray(y))]
    rows = []
    for scenario, frame, labels in batches:
        p = model.predict_proba(prep.transform(frame))
        m = metrics(labels, p)
        value = population_stability(train_pd, p)
        drop = base_metrics['gini'] - m['gini']
        gap = abs(m['calibration_gap'])
        rows.append(dict(scenario=scenario, **m, psi=value, gini_drop=drop,
            psi_alert=value > thresholds['psi_alert_threshold'],
            performance_alert=drop > thresholds['gini_alert_threshold'],
            calibration_alert=gap > thresholds['calibration_gap_threshold']))
    return pd.DataFrame(rows), shifted


def stress_tests(model, prep, X, y, approval_score=620):
    """Model sensitivities and fixed-label selection rates, not causal outcomes."""
    rows = []
    scenarios = [('baseline', None, None), ('income_x0.5', 'income', .5),
                 ('income_x1.5', 'income', 1.5), ('late90_plus1', 'late90', 1),
                 ('late90_plus3', 'late90', 3), ('income_down_late90_up', 'combined', 1)]
    baseline_p = model.predict_proba(prep.transform(X))
    for name, kind, value in scenarios:
        frame = X.copy()
        if kind in ('income', 'combined'):
            scale = value if kind == 'income' else .5
            frame['MonthlyIncome'] = frame.MonthlyIncome.fillna(prep.medians_['MonthlyIncome']) * scale
        if kind in ('late90', 'combined'):
            delta = value if kind == 'late90' else 1
            frame['NumberOfTimes90DaysLate'] = frame.NumberOfTimes90DaysLate + delta
        transformed = prep.transform(frame)
        p = model.predict_proba(transformed)
        scores = model.predict_score(transformed)
        approved = scores > approval_score
        rows.append(dict(scenario=name, n=len(p), approved_count=int(approved.sum()),
            approval_rate=float(approved.mean()), portfolio_expected_default_rate=float(p.mean()),
            approved_expected_default_rate=float(p[approved].mean()) if approved.any() else np.nan,
            approved_observed_rate_fixed_labels=float(np.asarray(y)[approved].mean()) if approved.any() else np.nan,
            mean_pd_change=float((p-baseline_p).mean()), approval_score=approval_score,
            minimum_pd=float(p.min()), maximum_score=int(scores.max()),
            minimum_score=int(scores.min())))
    return pd.DataFrame(rows)


def sign_checks(model, prep, X):
    base = prep.transform(X)
    p = model.predict_proba(base)
    rows = []
    for feature, delta, expected in [('MonthlyIncome', 1000, -1),
                                    ('NumberOfTimes90DaysLate', 1, 1),
                                    ('NumberOfTime30-59DaysPastDueNotWorse', 1, 1),
                                    ('NumberOfTime60-89DaysPastDueNotWorse', 1, 1)]:
        changed = base.copy()
        changed[feature] += delta
        effect = model.predict_proba(prep.transform(changed)) - p
        rows.append(dict(feature=feature, perturbation=delta, expected_pd_direction=expected,
            mean_pd_change=float(effect.mean()),
            violation_rate=float((expected*effect < -1e-10).mean()),
            unchanged_rate=float((np.abs(effect) <= 1e-10).mean()),
            selected=feature in model.selected_features_))
    return pd.DataFrame(rows)
