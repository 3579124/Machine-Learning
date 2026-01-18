import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


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


def detect_categorical(X: pd.DataFrame) -> list[int]:
    cat_cols = []
    n_rows = len(X)
    for i, col in enumerate(X.columns):
        series = X[col]
        if series.dtype == "object":
            cat_cols.append(i)
            continue
        # treat low-cardinality integers as categorical codes
        if pd.api.types.is_integer_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique <= 50 or nunique <= max(20, int(0.05 * n_rows)):
                cat_cols.append(i)
    return cat_cols


def train_model(
    train_path: Path,
    model_path: Path,
    label_path: Path,
    feature_path: Path,
    seed: int = 42,
) -> None:
    df = clean_columns(read_table(train_path))
    if "Target" not in df.columns:
        raise ValueError("Target column not found in training data.")

    y = df["Target"].astype(str)
    X = df.drop(columns=["Target"]).copy()

    if "id" in X.columns:
        X = X.drop(columns=["id"])

    cat_features = detect_categorical(X)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=seed,
        depth=6,
        learning_rate=0.1,
        iterations=800,
        l2_leaf_reg=3.0,
        early_stopping_rounds=50,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
    )

    valid_pred = model.predict(X_valid)
    valid_pred = np.array(valid_pred).ravel()
    acc = accuracy_score(y_valid, valid_pred)
    print(f"Validation accuracy: {acc:.4f}")

    model.fit(X, y, cat_features=cat_features, verbose=False)

    model.save_model(model_path)

    labels = sorted(y.unique().tolist())
    label_map = {label: idx for idx, label in enumerate(labels)}
    with label_path.open("w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label_map": label_map}, f, indent=2)

    with feature_path.open("w", encoding="utf-8") as f:
        json.dump({"features": X.columns.tolist(), "cat_features": cat_features}, f, indent=2)

    joblib.dump({"seed": seed}, model_path.with_suffix(".meta.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv")
    parser.add_argument("--model", type=str, default="model.cbm")
    parser.add_argument("--labels", type=str, default="labels.json")
    parser.add_argument("--features", type=str, default="features.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        train_path=Path(args.train),
        model_path=Path(args.model),
        label_path=Path(args.labels),
        feature_path=Path(args.features),
        seed=args.seed,
    )
