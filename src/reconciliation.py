"""Export a matched unpenalized logistic bridge; compare genuine SAS outputs."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from src.validation import metrics


def export_bridge(champion, splits, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    features = champion.selected_features_
    mapping = {feature: f'w{i:03d}' for i, feature in enumerate(features, 1)}
    frames = []
    for split, (X, y) in splits.items():
        design = champion.woe_encoder.transform(X)[features].rename(columns=mapping)
        design.insert(0, 'target', np.asarray(y))
        design.insert(0, 'split', split)
        frames.append(design.reset_index(drop=True))
    design = pd.concat(frames, ignore_index=True)
    design.insert(0, 'row_id', np.arange(1, len(design)+1))
    columns = list(mapping.values())
    train = design['split'].eq('train')
    # PROC LOGISTIC default MLE is unpenalized. Keep this companion distinct
    # from the production champion, whose L2 regularization remains unchanged.
    bridge = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        bridge.fit(design.loc[train, columns], design.loc[train, 'target'])
    design.to_csv(out/'design.csv', index=False, float_format='%.17g')
    predictions = design.copy()
    predictions['python_pd'] = bridge.predict_proba(design[columns])[:, 1]
    predictions.to_csv(out/'python_predictions.csv', index=False, float_format='%.17g')
    coefficients = pd.DataFrame({'term': ['Intercept']+columns,
                                'estimate': np.r_[bridge.intercept_, bridge.coef_[0]]})
    coefficients.to_csv(out/'python_coefficients.csv', index=False, float_format='%.17g')
    summary = pd.DataFrame([dict(split=split, **metrics(part.target, part.python_pd))
                           for split, part in predictions.groupby('split')])
    summary.to_csv(out/'python_metrics.csv', index=False)
    pd.DataFrame({'feature': features, 'sas_term': columns}).to_csv(out/'feature_mapping.csv', index=False)
    manifest = dict(status='PENDING_SAS_EXECUTION', event=1, intercept=True,
                    penalty='none', weights='none', selection='none',
                    training_split='train', feature_order=columns,
                    design_sha256=hashlib.sha256((out/'design.csv').read_bytes()).hexdigest(),
                    note='Unpenalized companion; not the regularized champion. No SAS result is implied.',
                    tolerances=dict(coefficient_atol=1e-4, probability_atol=1e-5, gini_atol=1e-6))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    names = ' '.join(columns)
    inputs = ' '.join(f'{name} :best32.' for name in columns)
    sas = f'''/* Set bridge_dir to the directory containing design.csv. */
%let bridge_dir=/absolute/path/to/reconciliation;
data design;
  infile "&bridge_dir./design.csv" dsd dlm=',' firstobs=2 lrecl=32767 truncover;
  length split $24;
  input row_id :best32. split :$24. target :best32. {inputs};
run;
ods output ParameterEstimates=parameters;
proc logistic data=design(where=(split='train'));
  model target(event='1') = {names} / maxiter=1000 gconv=1e-10;
  score data=design out=scored;
run;
ods output close;
data sas_predictions;
  set scored;
  sas_pd=P_1;
  keep row_id split target {names} sas_pd;
  format _numeric_ best32.;
run;
data sas_coefficients;
  set parameters;
  length term $32;
  term=Variable;
  keep term estimate;
  format estimate best32.;
run;
proc export data=sas_predictions outfile="&bridge_dir./sas_predictions.csv" dbms=csv replace; run;
proc export data=sas_coefficients outfile="&bridge_dir./sas_coefficients.csv" dbms=csv replace; run;
/* Check the SAS log for convergence, separation, and discarded observations.
   Do not report a pass when SAS has issued fitting warnings. */
'''
    (out/'reconcile.sas').write_text(sas)
    return manifest


def compare_bridge(directory, sas_predictions, sas_coefficients):
    """Strict sample/design matching before numerical coefficient/PD/Gini checks."""
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    if hashlib.sha256((root/'design.csv').read_bytes()).hexdigest() != manifest['design_sha256']:
        raise ValueError('Bridge design changed after export')
    expected = pd.read_csv(root/'python_predictions.csv')
    actual = pd.read_csv(sas_predictions)
    expected_coef = pd.read_csv(root/'python_coefficients.csv')
    actual_coef = pd.read_csv(sas_coefficients)
    actual.columns = actual.columns.str.lower()
    actual_coef.columns = actual_coef.columns.str.lower()
    required = ['row_id', 'split', 'target', 'sas_pd'] + manifest['feature_order']
    if not set(required).issubset(actual):
        raise ValueError(f'SAS predictions must include {required}')
    if actual.row_id.duplicated().any() or actual.row_id.isna().any() or set(actual.row_id) != set(expected.row_id):
        raise ValueError('SAS sample IDs do not match one-to-one')
    actual = actual.set_index('row_id').loc[expected.row_id].reset_index()
    if not np.array_equal(actual['split'], expected['split']) or not np.array_equal(actual.target, expected.target):
        raise ValueError('Split membership or target event differs')
    cols = manifest['feature_order']
    if not np.isfinite(actual[cols].to_numpy()).all() or not np.allclose(actual[cols], expected[cols], rtol=0, atol=1e-12):
        raise ValueError('SAS design differs from frozen Python preprocessing/WOE')
    if not np.isfinite(actual.sas_pd).all() or not actual.sas_pd.between(0, 1).all():
        raise ValueError('SAS probabilities must be finite and within [0,1]')
    if not {'term', 'estimate'}.issubset(actual_coef):
        raise ValueError('SAS coefficient CSV requires term and estimate')
    for frame in [expected_coef, actual_coef]:
        frame['term'] = frame.term.str.strip().str.lower()
    if actual_coef.term.duplicated().any() or set(actual_coef.term) != set(expected_coef.term):
        raise ValueError('SAS coefficient terms differ')
    coefs = expected_coef.merge(actual_coef, on='term', suffixes=('_python', '_sas'), validate='one_to_one')
    if not np.isfinite(coefs[['estimate_python', 'estimate_sas']].to_numpy()).all():
        raise ValueError('SAS coefficients must be finite')
    coefs['absolute_difference'] = abs(coefs.estimate_python-coefs.estimate_sas)
    tolerance = manifest['tolerances']
    coefs['pass'] = coefs.absolute_difference <= tolerance['coefficient_atol']
    coefs.to_csv(root/'coefficient_comparison.csv', index=False)
    checks = []
    for split in expected['split'].unique():
        mask = expected['split'].eq(split)
        py = metrics(expected.loc[mask, 'target'], expected.loc[mask, 'python_pd'])
        sas = metrics(actual.loc[mask, 'target'], actual.loc[mask, 'sas_pd'])
        difference = float(abs(expected.loc[mask, 'python_pd']-actual.loc[mask, 'sas_pd']).max())
        gini_delta = abs(py['gini']-sas['gini'])
        checks.append(dict(split=split, max_probability_difference=difference,
                           python_gini=py['gini'], sas_gini=sas['gini'], gini_difference=gini_delta,
                           pass_probability=difference <= tolerance['probability_atol'],
                           pass_gini=gini_delta <= tolerance['gini_atol']))
    comparison = pd.DataFrame(checks)
    comparison.to_csv(root/'prediction_comparison.csv', index=False)
    passed = bool(coefs['pass'].all() and comparison.pass_probability.all() and comparison.pass_gini.all())
    result = dict(status='PASS_NUMERICAL_CHECKS' if passed else 'FAIL',
                  sas_log_review_required=True, checks=checks)
    (root/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
