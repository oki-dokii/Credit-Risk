# Model card — Credit Risk validation baseline

Updated: 2026-10-28. Status: synthetic demonstration; not approved for lending.
The baseline model documentation was authored by Anil Jhanwar; see THIRD_PARTY_NOTICES.md.

## Models and target

The champion is an L2-regularized logistic regression on training-fitted WOE features.
The challenger is XGBoost on capped, training-median-imputed features.
The target convention is 1 = default and 0 = non-default. Higher scorecard points mean lower PD.
WOE = log(event distribution / non-event distribution); increasing WOE typically increases PD.
Feature selection uses training Information Value only. No evaluation labels are used in fitting.

## Data and evaluation

No real training dataset or SAS installation was supplied. The default validation run generates
12,000 artificial applicants with a fixed seed and a documented logistic outcome mechanism in
src/validation.py. Its results demonstrate software behavior, not performance on real borrowers.
The real-data path accepts the Give Me Some Credit feature schema and target SeriousDlqin2yrs;
synthetic outcomes are artificial and have no real 24-month follow-up period.

validation_config.yaml uses 60% training, 20% validation and 20% **simulated OOT proxy**.
All splits are stratified random samples, not contiguous rows or dated vintages.
The legacy training configuration uses its own declared fractions in config.yaml.
The proxy is not a genuine out-of-time test and cannot establish temporal stability.
Use dated application cohorts with matured outcomes to assess future-vintage performance.

Run `python validate.py` to regenerate discrimination, calibration, monitoring experiments,
decision stresses, sign checks and a validation memo. Results are in outputs/validation/metrics.csv
and outputs/validation/validation_memo.md. Historical CSVs directly under outputs/ are imported
reference artifacts; they were not reproduced and must not be presented as results of this work.

## Decisions and sensitivities

Default illustrative score cutoffs: <=580 decline, >580 through 620 review, >620 approve.
Score conversion uses good:bad odds, PDO 20, base score 600 and base odds 30:1;
scores are rounded and clipped to 300–850 before applying cutoffs.
Stress experiments vary income and 90-day delinquency while freezing the model and preprocessing.
Average predicted PDs are expected default rates, not observed causal stressed outcomes.
Observed default rates among approved applicants use unchanged labels and describe selection only.
The cutoffs are demonstration settings, not validated lending policy.

## Monitoring and limitations

Illustrative alerts: score PSI >0.10, absolute Gini drop >0.05 from the unshifted reference cohort,
and absolute mean-PD calibration gap >0.03. These are configurable project thresholds,
not universal regulatory requirements. Performance alerts require matured outcome labels.
The simulations include artificial label inversion solely to prove the alarm path can fire.
PSI depends on the frozen reference bins, smoothing and sample size; low PSI does not establish stability.

Missing work before production: genuine temporal validation, uncertainty intervals, data provenance,
segment and fairness assessment, feature-economic review, stress assumptions backed by evidence,
realized-loss validation, operational controls and independent model approval.
Income shocks hold DebtRatio fixed and therefore do not model a complete household balance sheet.

## Python–SAS status

Pending actual SAS execution. The exported bridge compares the same frozen WOE design and samples
under unpenalized logistic maximum likelihood, explicitly distinct from the regularized champion.
No SAS coefficients, probabilities or Gini have been observed. Python-generated comparator fixtures
in unit tests are test doubles, not evidence of SAS agreement. Review the SAS convergence log and
run reconcile.py on actual SAS exports before reporting numerical reconciliation.
