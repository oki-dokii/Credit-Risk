# Validation memo

Data status: **SYNTHETIC DEMONSTRATION — NOT REAL CREDIT DATA**. Seed: 42. Rows: 12,000.
This is an executable validation demonstration, not evidence of production readiness.

## Scope and independence

Metrics are recomputed from row-level predictions using a separate validation module.
Both champion and challenger are evaluated when enabled. This is implementation-level
independent checking, not an organizationally independent validation review.
Caps are fixed business rules. Medians, WOE bins, feature selection, and model parameters
are learned only from training observations, then frozen for validation and inference.
Split membership, medians, WOE tables, predictions, configuration and data hashes are exported.

## Discrimination and calibration

| model | split | auc | gini | ks | brier | calibration_intercept | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_champion | train | 0.7570 | 0.5141 | 0.3870 | 0.0644 | 0.0347 | 1.0162 |
| logistic_champion | validation | 0.7548 | 0.5097 | 0.4014 | 0.0634 | 0.0010 | 1.0055 |
| logistic_champion | simulated_oot_proxy | 0.7207 | 0.4414 | 0.3229 | 0.0661 | -0.3813 | 0.8429 |
| xgboost_challenger | train | 0.8228 | 0.6455 | 0.4876 | 0.0588 | 0.8136 | 1.3986 |
| xgboost_challenger | validation | 0.7465 | 0.4930 | 0.3849 | 0.0643 | -0.0601 | 0.9737 |
| xgboost_challenger | simulated_oot_proxy | 0.7060 | 0.4120 | 0.3437 | 0.0668 | -0.5061 | 0.7878 |

AUC measures ranking; Gini = 2*AUC-1. KS is the maximum absolute separation of
event and non-event score CDFs. Brier and log loss assess probability accuracy.
Joint calibration regression has ideal intercept 0 and slope 1. Fit failures are
flagged in metrics.csv; a failed calibration fit must not be interpreted as a pass.
Reliability bands and mean-PD minus observed-rate gaps are exported separately.
These point estimates have no uncertainty intervals and should not be overinterpreted.
Training WOE uses training outcomes, so in-sample performance is optimistic.

Observed champion findings: validation AUC is **0.7548**;
proxy AUC is **0.7207** and proxy Gini is **0.4414**.
The training-to-proxy Gini drop is **0.0726**
(configured review flag: **True**).
Proxy calibration intercept/slope are **-0.3813 /
0.8429**; review their departure from 0/1 rather than
inferring stability from PSI alone. No statistical significance is asserted.

![Calibration by split](calibration.png)

## Simulated OOT proxy limitation

The configured split is 20% validation and
20% simulated OOT proxy, with the remainder for training.
Both held-out partitions are stratified random samples of the same population.
There is no date-based separation, seasoning test, macroeconomic change, vintage analysis,
or genuine future observation. Low proxy PSI does **not** prove temporal stability.
Synthetic data further limits all results to software and methodological demonstration.
A genuine OOT test requires dated applications and fully matured default outcomes,
with training cutoffs preceding evaluation vintages.

## Monitoring simulation

| scenario | psi | gini | gini_drop | psi_alert | performance_alert | calibration_alert |
| --- | --- | --- | --- | --- | --- | --- |
| unshifted_control | 0.0046 | 0.4414 | 0.0000 | False | False | False |
| covariate_shift_fixed_labels | 16.2690 | 0.2973 | 0.1441 | True | True | True |
| synthetic_outcome_inversion | 0.0046 | -0.4414 | 0.8829 | False | True | True |

Monitoring experiments use the frozen logistic champion.
The shifted batch sets utilization to 1.5, multiplies imputed income by 0.35,
and adds four 90-day delinquencies before applying frozen preprocessing.
Labels stay fixed to isolate a controlled covariate stress; they are not new realized outcomes.
The outcome-inversion batch deliberately flips synthetic labels (1-y), keeping features
fixed, solely to exercise the performance and calibration alert paths. It is not a forecast.
PSI uses fixed training-score quantile edges with smoothing. Gini drops are absolute
changes from the unshifted proxy cohort. Calibration alerts use absolute mean-PD gaps.
Thresholds are illustrative project settings, not regulatory standards.
Expected PSI alarm triggered: **True**.
Expected Gini alarm triggered: **True**.

## Decision stress tests

| scenario | approval_rate | portfolio_expected_default_rate | approved_expected_default_rate | approved_observed_rate_fixed_labels |
| --- | --- | --- | --- | --- |
| baseline | 0.0712 | 0.0791 | 0.0124 | 0.0234 |
| income_x0.5 | 0.0225 | 0.1001 | 0.0135 | 0.0370 |
| income_x1.5 | 0.1342 | 0.0664 | 0.0121 | 0.0124 |
| late90_plus1 | 0.0000 | 0.1498 | N/A | N/A |
| late90_plus3 | 0.0000 | 0.1498 | N/A | N/A |
| income_down_late90_up | 0.0000 | 0.1852 | N/A | N/A |

Approval requires rounded, clipped score > 620; review cases are not counted as approvals.
All changes hold other fields fixed; income and DebtRatio are not economically reconciled.
Expected default rates are averages of model PDs, not observed stressed default rates.
The fixed-label approved rate changes only with selection into the approved portfolio.
No causal income or delinquency effect is identified, and there is no loss-given-default model.
Zero approvals produce N/A approved-portfolio rates rather than a misleading zero.
In this run, halving income changes approval from **7.12%**
to **2.25%**, and whole-portfolio expected default rate from
**7.91%** to **10.01%**.
Additional delinquency increments can map to the same learned WOE bin, yielding identical
stress results. This is a coarse-binning limitation, not evidence that extra delinquencies
have no economic effect.

## Sign checks and implementation findings

| feature | perturbation | expected_pd_direction | mean_pd_change | violation_rate | unchanged_rate | selected |
| --- | --- | --- | --- | --- | --- | --- |
| MonthlyIncome | 1000.0000 | -1.0000 | -0.0068 | 0.0000 | 0.4725 | True |
| NumberOfTimes90DaysLate | 1.0000 | 1.0000 | 0.0707 | 0.0000 | 0.1771 | True |
| NumberOfTime30-59DaysPastDueNotWorse | 1.0000 | 1.0000 | 0.0512 | 0.0000 | 0.0488 | True |
| NumberOfTime60-89DaysPastDueNotWorse | 1.0000 | 1.0000 | 0.0591 | 0.0000 | 0.1217 | True |

WOE is log(event share / non-event share), so positive WOE coefficients are the
usual risk-consistent direction. Negative coefficients require multicollinearity,
selection and economic interpretation review, not automatic reversal.
Negative selected coefficient count: **0**.
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

- Investigate input drift and data quality when score PSI exceeds 0.1; review feature-level drift and sample composition.
- Escalate an absolute Gini drop over 0.05 after outcomes mature; assess uncertainty and cohort mix before changing policy.
- Investigate a mean-PD calibration gap over 0.03; recalibrate only on approved, representative development data.
- Review sign violations, negative WOE coefficients and approval/expected-risk changes with credit policy owners.
- Obtain real development data, a genuine dated OOT cohort, and actual SAS execution before seeking model approval.
- Add sample-size-dependent limits, confidence intervals, fairness/segment validation and realized-loss backtesting before deployment.

Disposition: **demonstration only; production approval and SAS reconciliation pending**.
