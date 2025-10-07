
"""
preprocessing.py
----------------
"Human-written" utilities for the Kaggle House Prices dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 7

@dataclass
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame

def load_data(data_dir: str = "data") -> DataBundle:
    train_fp = os.path.join(data_dir, "train.csv")
    test_fp = os.path.join(data_dir, "test.csv")
    if os.path.exists(train_fp) and os.path.exists(test_fp):
        train = pd.read_csv(train_fp)
        test = pd.read_csv(test_fp)
        return DataBundle(train=train, test=test)

    rng = np.random.default_rng(RANDOM_STATE)
    def synthesize(n: int) -> pd.DataFrame:
        df = pd.DataFrame({
            "Id": np.arange(1, n+1),
            "OverallQual": rng.integers(1, 10, size=n),
            "GrLivArea": rng.normal(1500, 500, size=n).clip(400, 4500).round().astype(int),
            "GarageCars": rng.integers(0, 4, size=n),
            "YearBuilt": rng.integers(1900, 2010, size=n),
            "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"], size=n),
            "LotArea": rng.normal(10000, 5000, size=n).clip(800, 60000).round().astype(int),
            "1stFlrSF": rng.normal(1100, 300, size=n).clip(200, 3000).round().astype(int),
            "2ndFlrSF": rng.normal(300, 300, size=n).clip(0, 2000).round().astype(int),
            "TotalBsmtSF": rng.normal(800, 300, size=n).clip(0, 3000).round().astype(int),
        })
        return df

    train = synthesize(300)
    noise = rng.normal(0, 25000, size=len(train))
    train["SalePrice"] = (
        25000
        + 12000 * train["OverallQual"]
        + 55 * train["GrLivArea"]
        + 9000 * train["GarageCars"]
        + 12 * train["LotArea"]
        + 6 * train["TotalBsmtSF"]
        + np.where(train["Neighborhood"].eq("Somerst"), 25000, 0)
        + np.where(train["YearBuilt"] >= 2000, 20000, 0)
        + noise
    ).clip(40000, 900000).round().astype(int)

    test = synthesize(150)
    return DataBundle(train=train, test=test)

def basic_clean(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "Neighborhood" in out.columns:
            out["Neighborhood"] = out["Neighborhood"].astype(str)
        parts = []
        for col in ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF"):
            parts.append(out[col] if col in out.columns else 0)
        out["TotalSF"] = sum(parts)
        return out

    return _apply(train), _apply(test)

def split_features_target(train: pd.DataFrame, target: str = "SalePrice"):
    if target not in train.columns:
        raise KeyError(f"'{target}' not found in training data.")
    X = train.drop(columns=[target])
    y = train[target]
    return X, y

def build_preprocessor(X: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = SimpleImputer(strategy="median")
    cat = make_categorical_pipeline()

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", cat, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre, numeric_cols, cat_cols

def make_categorical_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

def fit_transform(pre, X: pd.DataFrame, X_test: pd.DataFrame):
    X_proc = pre.fit_transform(X)
    X_test_proc = pre.transform(X_test)

    feature_names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            feature_names.extend(cols)
        if name == "cat":
            ohe = trans.named_steps.get("ohe")
            if hasattr(ohe, "get_feature_names_out"):
                feature_names.extend(ohe.get_feature_names_out(cols).tolist())
    return X_proc, X_test_proc, feature_names

def train_valid_split(X, y, test_size: float = 0.2, random_state: int = 7):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

def save_submission(test_ids, preds, out_path="results/submission.csv") -> str:
    sub = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    return save_csv(sub, out_path)
