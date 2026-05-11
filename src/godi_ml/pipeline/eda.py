"""EDA: compute summary statistics and mutual information with target."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def compute_eda(
    df: pd.DataFrame,
    target: str,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute summary EDA statistics on a tabular DataFrame.

    Returns dict with: n_rows, n_cols, dtypes, missing_pct, target_distribution,
    mutual_info, suspicious_id_columns.

    `suspicious_id_columns` lists columns with cardinality == n_rows
    (typical IDs that should be excluded). DataDoctor (M2) reads this.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in DataFrame")

    n_rows, n_cols = df.shape

    dtypes = {col: str(df[col].dtype) for col in df.columns}

    missing_pct = {col: float((df[col].isna().sum() / n_rows) * 100) for col in df.columns}

    target_dist = df[target].value_counts().sort_index().to_dict()

    suspicious_id_columns = [
        col for col in df.columns if col != target and df[col].nunique() == n_rows
    ]

    feature_cols = [c for c in df.columns if c != target]
    mi_features = df[feature_cols].copy()
    for col in mi_features.columns:
        if pd.api.types.is_numeric_dtype(mi_features[col]):
            mi_features[col] = mi_features[col].fillna(mi_features[col].median())
        else:
            mi_features[col] = LabelEncoder().fit_transform(
                mi_features[col].astype(str).fillna("__missing__")
            )

    if len(feature_cols) > 0:
        mi_array = mutual_info_classif(
            mi_features.values,
            df[target].values,
            random_state=random_seed,
        )
        mutual_info = {col: float(mi) for col, mi in zip(feature_cols, mi_array, strict=True)}
    else:
        mutual_info = {}

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": dtypes,
        "missing_pct": missing_pct,
        "target_distribution": {
            int(k) if isinstance(k, np.integer) else k: int(v) for k, v in target_dist.items()
        },
        "mutual_info": mutual_info,
        "suspicious_id_columns": suspicious_id_columns,
    }
