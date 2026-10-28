# Model card — Credit Risk validation baseline

Updated: 2026-10-28. Status: historical public-data validation; not approved for lending.
The baseline model documentation was authored by Anil Jhanwar; see THIRD_PARTY_NOTICES.md.

## Models and target

The champion is an L2-regularized logistic regression on training-fitted WOE features.
The challenger is XGBoost on capped, training-median-imputed features.
The target convention is 1 = default and 0 = non-default. Higher scorecard points mean lower PD.
WOE = log(event distribution / non-event distribution); increasing WOE typically increases PD.
Feature selection uses training Information Value only. No evaluation labels are used in fitting.

## Data and evaluation

The real-data assessment uses 150,000 Give Me Some Credit training rows with 10,026 defaults
(6.684%), obtained from a documented public mirror because Kaggle required authentication.
The source file hash, URL and missingness profile are recorded in data/raw/cs-training.provenance.json.
The mirror was not authenticated against an official Kaggle checksum. No applicant-level data are committed.
The target is SeriousDlqin2yrs, representing serious delinquency within two years in the public dataset.
There are no application dates available for constructing a temporal test in this workflow.
The default no-data command still generates 12,000 artificial applicants for software demonstrations;
those outputs are kept separate and have no real follow-up period.

validation_config.yaml uses 60% training, 20% validation and 20% **random holdout**.
All splits are stratified random samples, not contiguous rows or dated vintages.
The legacy training configuration uses its own declared fractions in config.yaml.
The random holdout is not a genuine out-of-time test and cannot establish temporal stability.
Use dated application cohorts with matured outcomes to assess future-vintage performance.

Run `python validate.py --data data/raw/cs-training.csv --output outputs/real_validation`
to reproduce the real-data metrics, calibration, monitoring, stresses, signs and memo.
On the 30,000-row random holdout, champion AUC/Gini/KS are 0.8576/0.7153/0.5553,
with PSI 0.000412 versus training; challenger values are 0.8675/0.7350/0.5803 and PSI 0.000573.
These are single-split historical estimates without uncertainty intervals or temporal conclusions.
See outputs/real_validation/validation_memo.md and metrics.csv. The separate synthetic run is
under outputs/validation/. Historical CSVs directly under outputs/ are imported reference artifacts.

## Decisions and sensitivities

Default illustrative score cutoffs: <=580 decline, >580 through 620 review, >620 approve.
Score conversion uses good:bad odds, PDO 20, base score 600 and base odds 30:1;
scores are rounded and clipped to 300–850 before applying cutoffs.
Stress experiments vary income and 90-day delinquency while freezing the model and preprocessing.
Average predicted PDs are expected default rates, not observed causal stressed outcomes.
Observed default rates among approved applicants use unchanged labels and describe selection only.
The cutoffs are demonstration settings, not validated lending policy.
Adding one 90-day delinquency to every holdout applicant results in a maximum score of 607
and minimum PD of 2.54%, below the >620 approval rule (approximately PD <1.61%).
The zero-approval outcome is a fitted model/policy sensitivity, not a recommended universal rule.
Training bins separate zero from any positive 90-day count; additional positive counts share a bin.
See delinquency_bins.csv and delinquency_stress_diagnostics.json in the real-validation outputs.

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
