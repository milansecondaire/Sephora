"""Cluster stability validation module (Epic 4 — US-4.6)."""

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.config import RANDOM_STATE


def bootstrap_stability(
    X: pd.DataFrame,
    full_labels: pd.Series,
    algorithm_fn,
    n_bootstraps: int = 5,
    subsample_frac: float = 0.80,
) -> pd.DataFrame:
    """Run algorithm on N subsamples and compute ARI vs. full-data labels.

    Args:
        X: Feature matrix (n_samples, n_features).
        full_labels: Cluster labels for the full dataset.
        algorithm_fn: Callable that takes a DataFrame X and returns cluster labels array.
        n_bootstraps: Number of bootstrap iterations.
        subsample_frac: Fraction of data to sample each iteration.

    Returns:
        DataFrame with columns: bootstrap, n_samples, ari, n_clusters.
    """
    results = []
    for i in range(n_bootstraps):
        seed = RANDOM_STATE + i
        idx = X.sample(frac=subsample_frac, random_state=seed).index
        X_sub = X.loc[idx]
        sub_labels = algorithm_fn(X_sub)
        ari = adjusted_rand_score(full_labels.loc[idx], sub_labels)
        results.append({
            "bootstrap": i + 1,
            "n_samples": len(idx),
            "ari": round(ari, 4),
            "n_clusters": len(set(sub_labels)),
        })
    return pd.DataFrame(results)
