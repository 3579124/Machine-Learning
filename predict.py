import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )
    return df


def load_metadata(feature_path: Path) -> dict:
    with feature_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_to_submission(
    test_path: Path,
    model_path: Path,
    feature_path: Path,
    submission_path: Path,
    sample_submission: Path | None = None,
) -> None:
    test_df = clean_columns(read_table(test_path))

    model = CatBoostClassifier()
    model.load_model(model_path)

    meta = load_metadata(feature_path)
    features = meta["features"]

    if "id" in test_df.columns:
        ids = test_df["id"].copy()
        X = test_df.drop(columns=["id"])
    else:
        ids = pd.Series(np.arange(len(test_df)), name="id")
        X = test_df

    missing_cols = [c for c in features if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test data: {missing_cols}")

    X = X[features]

    preds = model.predict(X)
    preds = np.array(preds).ravel()

    submission = pd.DataFrame({"id": ids, "Target": preds})

    if sample_submission and sample_submission.exists():
        sample_df = read_table(sample_submission)
        if "id" in sample_df.columns:
            submission = sample_df[["id"]].merge(submission, on="id", how="left")

    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="test.csv")
    parser.add_argument("--model", type=str, default="model.cbm")
    parser.add_argument("--features", type=str, default="features.json")
    parser.add_argument("--submission", type=str, default="submission.csv")
    parser.add_argument("--sample", type=str, default="sample_submission.csv")
    args = parser.parse_args()

    sample_path = Path(args.sample)
    if not sample_path.exists():
        sample_path = None

    predict_to_submission(
        test_path=Path(args.test),
        model_path=Path(args.model),
        feature_path=Path(args.features),
        submission_path=Path(args.submission),
        sample_submission=sample_path,
    )
