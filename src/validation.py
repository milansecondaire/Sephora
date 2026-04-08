"""Cluster validation module (Epic 4 — US-4.6, Epic 5 — US-5.5)."""

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu
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


def run_kruskal_wallis(df: pd.DataFrame, kpis: list) -> pd.DataFrame:
    """Run Kruskal-Wallis test for each KPI across cluster groups.

    Args:
        df: DataFrame with 'cluster_id' column and KPI columns.
        kpis: List of KPI column names to test.

    Returns:
        DataFrame with columns: kpi, H_statistic, p_value, significant.
    """
    groups_by_cluster = {cid: grp for cid, grp in df.groupby("cluster_id")}
    results = []
    for kpi in kpis:
        groups = [grp[kpi].dropna().values for grp in groups_by_cluster.values()]
        stat, pval = kruskal(*groups)
        results.append({
            "kpi": kpi,
            "H_statistic": round(stat, 3),
            "p_value": pval,
            "significant": pval < 0.05,
        })
    return pd.DataFrame(results)


def run_posthoc_mannwhitney(df: pd.DataFrame, kpis: list) -> pd.DataFrame:
    """Run pairwise Mann-Whitney U for each KPI across all cluster pairs.

    Only tests KPIs that are significant in Kruskal-Wallis (pass kpis
    already filtered, or pass all — caller decides).

    Args:
        df: DataFrame with 'cluster_id' column and KPI columns.
        kpis: List of KPI column names to test pairwise.

    Returns:
        DataFrame with columns: kpi, cluster_a, cluster_b, U_statistic, p_value, significant.
    """
    cluster_ids = sorted(df["cluster_id"].unique())
    results = []
    for kpi in kpis:
        for a, b in combinations(cluster_ids, 2):
            vals_a = df.loc[df["cluster_id"] == a, kpi].dropna().values
            vals_b = df.loc[df["cluster_id"] == b, kpi].dropna().values
            stat, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            results.append({
                "kpi": kpi,
                "cluster_a": a,
                "cluster_b": b,
                "U_statistic": round(stat, 3),
                "p_value": pval,
                "significant": pval < 0.05,
            })
    return pd.DataFrame(results)


def print_kruskal_summary(kw_results: pd.DataFrame, total_kpis: int) -> str:
    """Print and return summary of Kruskal-Wallis results.

    Args:
        kw_results: Output of run_kruskal_wallis().
        total_kpis: Total number of KPIs tested.

    Returns:
        Summary string.
    """
    n_significant = int(kw_results["significant"].sum())
    summary = (
        f"Statistical validation: {n_significant}/{total_kpis} KPIs "
        f"significantly differ across segments at p < 0.05"
    )
    print(summary)
    return summary
