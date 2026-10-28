# Verification results

Executed on 2026-10-28 with Python 3.12 and the packages in requirements-validation-lock.txt.

- `python validate.py`: completed on 12,000 explicitly synthetic applicants, both models,
  all three splits; required PSI and Gini alarm demonstrations both triggered.
- `python -m unittest discover -s tests -v`: 11 tests passed. Covers known metric values,
  reversed ranking, calibration regression, input validity, split disjointness, frozen medians,
  exact WOE boundaries and count conservation, scorecard point arithmetic, PSI degeneracy,
  monitoring alerts, zero-approval handling, and strict reconciliation mismatch rejection.
- CSV-input path: completed using the generated synthetic CSV as an explicit input and a
  separate temporary output directory. This tests ingestion and does not add real-data evidence.
- Inference integration: 50 applicants scored through predict.py using saved training preprocessing
  and champion artifacts. Rounded probabilities agreed with direct model predictions, and all
  decisions were assigned.
- Source compilation and `git diff --check`: passed.
- Calibration chart rendered and visually inspected.

No SAS runtime was available. Reconciliation comparator tests use Python-produced fixtures,
not SAS evidence. Actual cross-platform validation and real-data/temporal validation remain pending.
