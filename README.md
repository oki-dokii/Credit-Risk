# Credit Risk

A credit risk scorecard baseline for experimentation and future extensions. The pipeline uses Weight of Evidence (WOE) encoding, a logistic regression scorecard, and an XGBoost challenger to estimate applicant default risk.

## Setup

Use Python 3.9 or later:

```bash
git clone https://github.com/oki-dokii/Credit-Risk.git
cd Credit-Risk
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `cs-training.csv` from the [Give Me Some Credit dataset](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in `data/raw/`. The dataset and trained model files are not included.

## Train and score

```bash
python train.py --config config.yaml --data data/raw/cs-training.csv --output outputs
python predict.py --input data/new_applicants.csv --output outputs/scores.csv
```

Training saves models under `outputs/models/`. Run training before inference. Applicant input must contain the features listed in `src/data_utils.py`.

## Project layout

- `src/data_utils.py`: loading, preprocessing, and data splitting.
- `src/woe_encoder.py`: WOE binning and Information Value calculations.
- `src/scorecard_model.py`: logistic scorecard and XGBoost challenger.
- `src/evaluation.py`: discrimination, stability, decile tables, and reporting.
- `notebooks/`: EDA and WOE exploration in Python and notebook formats.
- `config.yaml`: model, binning, scaling, and monitoring parameters.
- `outputs/`: existing reference results; regenerate these for each experiment.
- `model_card.md`: baseline model documentation to update after validation.

The imported results and model card have not been independently reproduced for this repository. This baseline requires validation before use in lending decisions.

## Attribution and licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for baseline authorship and the licensing status recorded at import.
