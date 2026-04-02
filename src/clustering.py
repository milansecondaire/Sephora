"""Clustering module for Sephora customer segmentation (Epic 4)."""

import pandas as pd
from typing import Iterable
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.config import RANDOM_STATE, K_RANGE


def evaluate_kmeans_k_range(
    X: pd.DataFrame,
    k_range: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Evaluate KMeans for all k in k_range. Returns metrics DataFrame.

    Args:
        X: Feature matrix (n_samples, n_features).
        k_range: Range of k values to evaluate. Defaults to K_RANGE from config.

    Returns:
        DataFrame with columns: k, inertia, silhouette, davies_bouldin, calinski_harabasz.
    """
    if k_range is None:
        k_range = K_RANGE

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        sil_kwargs = {"random_state": RANDOM_STATE}
        if len(X) > 10_000:
            sil_kwargs["sample_size"] = 10_000
        results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels, **sil_kwargs),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(results)


def get_top_k_candidates(metrics_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Rank k candidates by silhouette score (descending) and return top_n.

    Args:
        metrics_df: Output of evaluate_kmeans_k_range().
        top_n: Number of candidates to return.

    Returns:
        DataFrame with top_n rows sorted by silhouette descending.
    """
    return metrics_df.nlargest(top_n, "silhouette").reset_index(drop=True)


def run_kmeans_final(X: pd.DataFrame, k: int, random_state: int = RANDOM_STATE) -> tuple:
    """Run final KMeans with given k. Returns (labels, fitted_model).

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Number of clusters (k_optimal from US-4.1).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (labels ndarray, fitted KMeans model).
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    return labels, km


def run_hierarchical(X: pd.DataFrame, k: int) -> "np.ndarray":
    """Fit AgglomerativeClustering(ward) on X and return cluster labels.

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Number of clusters (same k_optimal as used in KMeans).

    Returns:
        labels ndarray of shape (n_samples,).
    """
    import numpy as np  # noqa: F401 — kept local to avoid module-level dep
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    return labels
