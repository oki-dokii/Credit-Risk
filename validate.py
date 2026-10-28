"""Reproducible validation demo or assessment of an explicitly supplied dataset."""
import argparse
import hashlib
import json
import logging
import platform
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from src.data_utils import FrozenPreprocessor, load_data, split_data
from src.scorecard_model import LogisticScorecard, XGBoostChallenger
from src.woe_encoder import WOEBinEncoder
from src.validation import (synthetic_data, metrics, calibration_table, population_stability,
                            monitoring_experiments, stress_tests, sign_checks)
from src.reconciliation import export_bridge


def markdown_table(frame, digits=4):
    formatted = frame.copy()
    for col in formatted.select_dtypes(include='number'):
        formatted[col] = formatted[col].map(lambda v: f'{v:.{digits}f}' if pd.notna(v) else 'N/A')
    lines = ['| '+' | '.join(map(str, formatted.columns))+' |',
             '| '+' | '.join(['---']*len(formatted.columns))+' |']
    lines += ['| '+' | '.join(map(str, row))+' |' for row in formatted.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    df = load_data(args.data) if args.data else synthetic_data(args.rows, args.seed)
    source = 'SUPPLIED DATA' if args.data else 'SYNTHETIC DEMONSTRATION — NOT REAL CREDIT DATA'
    Xtrain, Xval, Xproxy, ytrain, yval, yproxy = split_data(df, **config['split'])
    prep = FrozenPreprocessor().fit(Xtrain)
    raw = dict(train=(Xtrain, ytrain), validation=(Xval, yval), simulated_oot_proxy=(Xproxy, yproxy))
    splits = {name: (prep.transform(X), y) for name, (X, y) in raw.items()}
    encoder = WOEBinEncoder(**config['woe_encoder'])
    champion = LogisticScorecard(encoder, config['scorecard_scaler'], config['logistic_regression'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        champion.fit(*splits['train'], iv_threshold=config['iv_threshold'])
    models = {'logistic_champion': champion}
    if not args.skip_challenger:
        challenger = XGBoostChallenger(config['xgboost'])
        challenger.fit(*splits['train'], *splits['validation'])
        models['xgboost_challenger'] = challenger
    results, reliability, predictions = [], [], []
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False)
    for index, (name, model) in enumerate(models.items()):
        train_pd = model.predict_proba(splits['train'][0])
        axis = axes[0, index]
        for split, (X, y) in splits.items():
            p = model.predict_proba(X)
            results.append(dict(model=name, split=split, **metrics(y, p), psi_vs_train=population_stability(train_pd, p)))
            bins = calibration_table(y, p)
            reliability.append(bins.assign(model=name, split=split))
            axis.plot(bins.mean_pd, bins.observed_rate, marker='o', label=split)
            predictions.append(pd.DataFrame(dict(row_id=X.index, model=name, split=split,
                                                target=np.asarray(y), predicted_pd=p)))
        bands = pd.concat([frame for frame in reliability if frame['model'].iloc[0] == name])
        limit = min(1., max(float(bands.mean_pd.max()), float(bands.observed_rate.max())) * 1.15 + .01)
        axis.plot([0, limit], [0, limit], '--', color='gray')
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set(xlabel='Mean predicted PD', ylabel='Observed event rate', title=name)
        axis.legend(fontsize=8)
    fig.suptitle('Synthetic demonstration' if not args.data else 'Supplied dataset — random split validation')
    fig.tight_layout()
    fig.savefig(out/'calibration.png', dpi=140)
    plt.close(fig)
    assessment = pd.DataFrame(results)
    training_gini = assessment.loc[assessment['split'].eq('train')].set_index('model').gini
    assessment['gini_drop_vs_train'] = assessment['model'].map(training_gini) - assessment.gini
    assessment['gini_drop_alert_vs_train'] = assessment.gini_drop_vs_train > config['monitoring']['gini_alert_threshold']
    assessment.to_csv(out/'metrics.csv', index=False)
    pd.concat(reliability).to_csv(out/'calibration_bands.csv', index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv', index=False)
    monitor, shifted = monitoring_experiments(champion, prep, Xproxy, yproxy,
        champion.predict_proba(splits['train'][0]), config['monitoring'])
    monitor.to_csv(out/'monitoring.csv', index=False)
    shifted.to_csv(out/'shifted_batch.csv', index_label='row_id')
    stress = stress_tests(champion, prep, Xproxy, yproxy, config['approval_score'])
    stress.to_csv(out/'decision_stress.csv', index=False)
    signs = sign_checks(champion, prep, Xproxy)
    signs.to_csv(out/'sign_checks.csv', index=False)
    coefficients = pd.DataFrame({'feature': champion.selected_features_, 'coefficient': champion.model.coef_[0]})
    coefficients['woe_convention'] = 'log(event_distribution / non_event_distribution)'
    coefficients['negative_coefficient_review'] = coefficients.coefficient < 0
    coefficients.to_csv(out/'champion_coefficients.csv', index=False)
    tables = pd.concat([encoder.get_bin_table(feature).assign(feature=feature)
                        for feature in encoder.feature_names_], ignore_index=True)
    tables.to_csv(out/'woe_bins.csv', index=False)
    champion.get_scorecard_table().to_csv(out/'scorecard_points.csv', index=False)
    prep.medians_.rename('training_median').to_csv(out/'preprocessing_medians.csv')
    membership = pd.concat([pd.DataFrame({'row_id': X.index, 'split': split})
                            for split, (X, y) in raw.items()])
    membership.to_csv(out/'split_membership.csv', index=False)
    export_bridge(champion, splits, out/'reconciliation')
    joblib.dump(dict(preprocessor=prep, models=models, config=config), out/'validation_bundle.pkl')
    expected_alarms = dict(
        psi_shift=bool(monitor.loc[monitor.scenario.eq('covariate_shift_fixed_labels'), 'psi_alert'].iloc[0]),
        gini_inversion=bool(monitor.loc[monitor.scenario.eq('synthetic_outcome_inversion'), 'performance_alert'].iloc[0]))
    run_manifest = dict(data_source=source, rows=len(df), seed=args.seed if not args.data else None,
        split_config=config['split'], models=list(models), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__,
        data_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
        config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        sas_status='PENDING_SAS_EXECUTION', demonstration_alerts=expected_alarms,
        temporal_validation=False)
    (out/'run_manifest.json').write_text(json.dumps(run_manifest, indent=2)+'\n')
    # Store the actual source too, so the synthetic run is reproducible and inspectable.
    if not args.data:
        df.to_csv(out/'synthetic_applicants.csv', index_label='row_id')
    subset = assessment[['model', 'split', 'auc', 'gini', 'ks', 'brier', 'calibration_intercept', 'calibration_slope']]
    champion_rows = assessment.loc[assessment.model.eq('logistic_champion')].set_index('split')
    proxy_metrics = champion_rows.loc['simulated_oot_proxy']
    baseline_stress = stress.iloc[0]
    income_stress = stress.loc[stress.scenario.eq('income_x0.5')].iloc[0]
    memo = f'''# Validation memo

Data status: **{source}**. Seed: {args.seed}. Rows: {len(df):,}.
This is an executable validation demonstration, not evidence of production readiness.

## Scope and independence

Metrics are recomputed from row-level predictions using a separate validation module.
Both champion and challenger are evaluated when enabled. This is implementation-level
independent checking, not an organizationally independent validation review.
Caps are fixed business rules. Medians, WOE bins, feature selection, and model parameters
are learned only from training observations, then frozen for validation and inference.
Split membership, medians, WOE tables, predictions, configuration and data hashes are exported.

## Discrimination and calibration

{markdown_table(subset)}

AUC measures ranking; Gini = 2*AUC-1. KS is the maximum absolute separation of
event and non-event score CDFs. Brier and log loss assess probability accuracy.
Joint calibration regression has ideal intercept 0 and slope 1. Fit failures are
flagged in metrics.csv; a failed calibration fit must not be interpreted as a pass.
Reliability bands and mean-PD minus observed-rate gaps are exported separately.
These point estimates have no uncertainty intervals and should not be overinterpreted.
Training WOE uses training outcomes, so in-sample performance is optimistic.

Observed champion findings: validation AUC is **{champion_rows.loc['validation', 'auc']:.4f}**;
proxy AUC is **{proxy_metrics['auc']:.4f}** and proxy Gini is **{proxy_metrics['gini']:.4f}**.
The training-to-proxy Gini drop is **{proxy_metrics['gini_drop_vs_train']:.4f}**
(configured review flag: **{bool(proxy_metrics['gini_drop_alert_vs_train'])}**).
Proxy calibration intercept/slope are **{proxy_metrics['calibration_intercept']:.4f} /
{proxy_metrics['calibration_slope']:.4f}**; review their departure from 0/1 rather than
inferring stability from PSI alone. No statistical significance is asserted.

![Calibration by split](calibration.png)

## Simulated OOT proxy limitation

The configured split is {config['split']['val_size']:.0%} validation and
{config['split']['oot_size']:.0%} simulated OOT proxy, with the remainder for training.
Both held-out partitions are stratified random samples of the same population.
There is no date-based separation, seasoning test, macroeconomic change, vintage analysis,
or genuine future observation. Low proxy PSI does **not** prove temporal stability.
Synthetic data further limits all results to software and methodological demonstration.
A genuine OOT test requires dated applications and fully matured default outcomes,
with training cutoffs preceding evaluation vintages.

## Monitoring simulation

{markdown_table(monitor[['scenario', 'psi', 'gini', 'gini_drop', 'psi_alert', 'performance_alert', 'calibration_alert']])}

Monitoring experiments use the frozen logistic champion.
The shifted batch sets utilization to 1.5, multiplies imputed income by 0.35,
and adds four 90-day delinquencies before applying frozen preprocessing.
Labels stay fixed to isolate a controlled covariate stress; they are not new realized outcomes.
The outcome-inversion batch deliberately flips synthetic labels (1-y), keeping features
fixed, solely to exercise the performance and calibration alert paths. It is not a forecast.
PSI uses fixed training-score quantile edges with smoothing. Gini drops are absolute
changes from the unshifted proxy cohort. Calibration alerts use absolute mean-PD gaps.
Thresholds are illustrative project settings, not regulatory standards.
Expected PSI alarm triggered: **{expected_alarms['psi_shift']}**.
Expected Gini alarm triggered: **{expected_alarms['gini_inversion']}**.

## Decision stress tests

{markdown_table(stress[['scenario', 'approval_rate', 'portfolio_expected_default_rate', 'approved_expected_default_rate', 'approved_observed_rate_fixed_labels']])}

Approval requires rounded, clipped score > {config['approval_score']}; review cases are not counted as approvals.
All changes hold other fields fixed; income and DebtRatio are not economically reconciled.
Expected default rates are averages of model PDs, not observed stressed default rates.
The fixed-label approved rate changes only with selection into the approved portfolio.
No causal income or delinquency effect is identified, and there is no loss-given-default model.
Zero approvals produce N/A approved-portfolio rates rather than a misleading zero.
In this run, halving income changes approval from **{baseline_stress.approval_rate:.2%}**
to **{income_stress.approval_rate:.2%}**, and whole-portfolio expected default rate from
**{baseline_stress.portfolio_expected_default_rate:.2%}** to **{income_stress.portfolio_expected_default_rate:.2%}**.
Additional delinquency increments can map to the same learned WOE bin, yielding identical
stress results. This is a coarse-binning limitation, not evidence that extra delinquencies
have no economic effect.

## Sign checks and implementation findings

{markdown_table(signs)}

WOE is log(event share / non-event share), so positive WOE coefficients are the
usual risk-consistent direction. Negative coefficients require multicollinearity,
selection and economic interpretation review, not automatic reversal.
Negative selected coefficient count: **{int(coefficients.negative_coefficient_review.sum())}**.
Perturbation violations must be reviewed alongside the unchanged fraction: frozen bins
make local sensitivities stepwise, and an unselected variable can have no direct effect.
The baseline was corrected for whole-sample median leakage, inconsistent WOE interval
boundaries, minimum-value exclusion for discrete features, constant-score PSI handling,
and the sign of the intercept contribution in the scorecard points table.

## Python–SAS reconciliation

**PENDING — SAS is unavailable; no cross-platform agreement is claimed.**
The exported SAS program fits an unpenalized companion logistic model on the identical
training rows and precomputed WOE matrix. It models target=1, includes an intercept,
and uses no sampling weights, class weights, variable selection or standardization.
This avoids confusing SAS default maximum likelihood with the champion's L2 fit.
Run reconciliation/reconcile.sas, inspect its convergence/separation log, then use
reconcile.py to compare returned coefficients, row-level probabilities and Gini on all splits.
The comparator rejects duplicate/missing IDs, mismatched targets, split membership,
WOE columns, nonfinite probabilities or missing coefficient terms before testing tolerances.
Tolerance settings are in reconciliation/manifest.json. Numerical agreement cannot
replace SAS log review or establish economic/model validity.

## Recommended actions and disposition

- Investigate input drift and data quality when score PSI exceeds {config['monitoring']['psi_alert_threshold']}; review feature-level drift and sample composition.
- Escalate an absolute Gini drop over {config['monitoring']['gini_alert_threshold']} after outcomes mature; assess uncertainty and cohort mix before changing policy.
- Investigate a mean-PD calibration gap over {config['monitoring']['calibration_gap_threshold']}; recalibrate only on approved, representative development data.
- Review sign violations, negative WOE coefficients and approval/expected-risk changes with credit policy owners.
- Obtain real development data, a genuine dated OOT cohort, and actual SAS execution before seeking model approval.
- Add sample-size-dependent limits, confidence intervals, fairness/segment validation and realized-loss backtesting before deployment.

Disposition: **demonstration only; production approval and SAS reconciliation pending**.
'''
    (out/'validation_memo.md').write_text(memo)
    print(f'Artifacts: {out.resolve()}')
    print(f'Data: {source}; SAS: PENDING; temporal validation: not performed')
    print(f'Demonstration alerts: {expected_alarms}')
    if not args.data and not all(expected_alarms.values()):
        raise RuntimeError('Synthetic monitoring demonstration failed to trigger the expected alarms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Optional CSV; omit for explicitly synthetic demonstration')
    parser.add_argument('--rows', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='validation_config.yaml')
    parser.add_argument('--output', default='outputs/validation')
    parser.add_argument('--skip-challenger', action='store_true')
    logging.basicConfig(level=logging.WARNING)
    run(parser.parse_args())
