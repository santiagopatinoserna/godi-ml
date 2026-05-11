"""Preprocessor: split, impute, encode, scale (fit on train only)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    """Splits, imputes, encodes, and scales tabular data.

    Stratified split by default. fit_transform on TRAIN only — transform
    applied to val/test using TRAIN-fitted statistics. This is the canonical
    pattern that prevents leakage.
    """

    def __init__(
        self,
        target: str,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_seed: int = 42,
        max_categorical_cardinality: int = 50,
    ) -> None:
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.max_categorical_cardinality = max_categorical_cardinality
        self._fitted = False
        self.feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.transformer_: ColumnTransformer | None = None

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified split into (train, val, test)."""
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=df[self.target],
        )
        rel_val = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=rel_val,
            random_state=self.random_seed,
            stratify=train_val[self.target],
        )
        return train, val, test

    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Decide numeric vs categorical, exclude high-cardinality and target."""
        n = len(df)
        numeric: list[str] = []
        categorical: list[str] = []
        for col in df.columns:
            if col == self.target:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric.append(col)
            elif df[col].dtype == "object" or df[col].dtype.name == "category":
                if df[col].nunique() == n:
                    continue  # high-cardinality non-numeric ID
                if df[col].nunique() <= self.max_categorical_cardinality:
                    categorical.append(col)
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical
        self.feature_columns_ = numeric + categorical

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        """Fit transformer on TRAIN, return transformed feature matrix."""
        self._identify_columns(train)

        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.transformer_ = ColumnTransformer(
            [
                ("num", numeric_pipe, self.numeric_columns_),
                ("cat", categorical_pipe, self.categorical_columns_),
            ],
            remainder="drop",
        )

        X = self.transformer_.fit_transform(train[self.feature_columns_])
        self._fitted = True
        return np.asarray(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform val/test data using TRAIN-fitted stats."""
        if not self._fitted or self.transformer_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")
        return np.asarray(self.transformer_.transform(df[self.feature_columns_]))

    def y(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target column as numpy array."""
        return df[self.target].to_numpy()
