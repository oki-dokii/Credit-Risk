# Give Me Some Credit: measured validation findings

Run date: 2026-10-28. Dataset: 150,000 observations; 10,026 defaults (6.684%).
The official Kaggle download required authentication, so a public mirror was used.
The exact URL, source-file SHA-256 and missingness counts are recorded in
[data provenance](../data/raw/cs-training.provenance.json). The mirror's bytes have not
been authenticated against an official Kaggle checksum. Raw rows and applicant-level
predictions are kept local.

## Reproduction

```bash
python validate.py --data data/raw/cs-training.csv --output outputs/real_validation
```

The 60/20/20 split is stratified and random, with seed 42: 90,000 training, 30,000 validation,
and 30,000 **random holdout** observations. No partition is out-of-time. There are no dated
application cohorts here, so these results cannot establish temporal stability.
Preprocessing, WOE bins, feature selection and all model parameters are fitted on training only.
This is a single-split assessment with no confidence intervals or significance claims.

## Results

| Model | Partition | Gini | KS | Score PSI vs train |
| --- | --- | --- | --- | --- |
| Logistic champion | Train | 0.7022 | 0.5459 | 0.000000 |
| Logistic champion | Validation | 0.7062 | 0.5469 | 0.000604 |
| Logistic champion | Random holdout | 0.7153 | 0.5553 | 0.000412 |
| XGBoost challenger | Train | 0.7364 | 0.5806 | 0.000000 |
| XGBoost challenger | Validation | 0.7277 | 0.5692 | 0.001214 |
| XGBoost challenger | Random holdout | 0.7350 | 0.5803 | 0.000573 |

Champion random-holdout AUC is 0.8576, Brier score 0.05049, calibration intercept 0.0611
and slope 1.0238. Challenger AUC is 0.8675 and Brier score 0.04867.
Low random-split PSI is consistent with similar score distributions across random samples;
it does not establish future-vintage stability or suitability for a new borrower population.
Full-precision results are in [metrics.csv](../outputs/real_validation/metrics.csv).

## Delinquency stress: why zero approvals?

The experiment adds one 90-day delinquency to **every** applicant in the random holdout.
It is not a shock to one individual or a statement that one newly late borrower should
cause all other applicants to be declined.

The fitted training bins for NumberOfTimes90DaysLate are:

| Count | Training rows | Training default rate | WOE |
| --- | --- | --- | --- |
| Zero | 84,964 | 4.61% | -0.3943 |
| Any positive count | 5,036 | 41.74% | 2.3027 |

These are marginal bin statistics, not individual PDs. With the fitted multivariate model,
the minimum PD after the all-applicant shock is 2.54%, and the maximum score is 607.
The illustrative policy requires score >620, approximately PD <1.61% after accounting
for rounding. Consequently, no stressed score clears the cutoff.

The model/policy combination is highly sensitive to this variable. The positive-count bin
also gives identical responses for one and three additional delinquencies. Credit-policy review
should assess finer expert bins, sample-size constraints and alternative cutoffs without
selecting them for a preferred holdout result. No cutoff changes were made to avoid this finding.

A sparse-count binning defect was corrected before this reported run: when the first nine
quantiles were all zero, the baseline could produce a single bin and discard the predictor.
Cut points now retain a training-derived zero-versus-positive boundary, and a regression test
protects it. The published metrics and stress results are from the corrected implementation.

See [bin statistics](../outputs/real_validation/delinquency_bins.csv),
[score/PD diagnostics](../outputs/real_validation/delinquency_stress_diagnostics.json), and
[all stress scenarios](../outputs/real_validation/decision_stress.csv).

## Monitoring

The deliberately shifted batch triggers PSI, performance and calibration-gap alerts.
Its score PSI is 21.7116, and Gini drops by 0.1544 relative to the unshifted random holdout.
The separate artificial label-inversion scenario exercises the outcome-based alarm path.
Both scenarios are controlled tests, not measured future drift or new realized borrower outcomes.

## SAS result

**Pending execution in SAS OnDemand.** The real-data matched-design bridge has been exported;
no SAS-versus-Python numerical comparison has been measured yet. The bridge uses a separate
unpenalized companion logistic fit so its specification matches SAS default maximum likelihood.
It must not be confused with the L2-regularized champion reported above.

Once actual SAS exports are available:

```bash
python reconcile.py --directory outputs/real_validation/reconciliation \
  --sas-predictions /absolute/path/sas_predictions.csv \
  --sas-coefficients /absolute/path/sas_coefficients.csv
```

Check the SAS log for convergence and separation before interpreting numerical agreement.
