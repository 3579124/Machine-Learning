# AIDAMS ML Competition Pipeline

This project trains a non-deep-learning model and generates a Kaggle submission for the AIDAMS ML competition.

## Files
- `train.py` — trains a CatBoost model and saves `model.cbm` + `features.json`
- `predict.py` — generates `submission.csv` from `test.csv`
- `requirements.txt` — Python dependencies

## Data placement
Place the Kaggle files in this folder:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

If you only have `data.csv` (original dataset with `Target`), you can still train locally by running:

```
python train.py --train data.csv
```

## Install dependencies
```
pip install -r requirements.txt
```

## Train
```
python train.py --train train.csv
```

## Predict / Create submission
```
python predict.py --test test.csv --sample sample_submission.csv --submission submission.csv
```

## Notes
- The pipeline automatically detects the CSV delimiter (comma or semicolon).
- Column names are cleaned to remove tabs and leading/trailing spaces.
- The model uses gradient boosting (CatBoost), which is allowed by the competition rules.
