"""Clustering module for Sephora customer segmentation (Epic 4)."""

import mlflow
import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
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


def run_hierarchical(X: pd.DataFrame, k: int, n_samples: int = None) -> "np.ndarray":
    """Fit AgglomerativeClustering(ward) on X and return cluster labels.

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Number of clusters (same k_optimal as used in KMeans).
        n_samples: If provided, fit only on a random subsample of this size,
                   then assign full dataset via nearest-centroid propagation.

    Returns:
        labels ndarray of shape (n_samples,).
    """
    import numpy as np
    from sklearn.metrics import pairwise_distances_argmin_min

    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_sub = X.iloc[idx]
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        sub_labels = model.fit_predict(X_sub)
        # Compute centroid of each cluster on the subsample
        centroids = np.vstack([
            X_sub.values[sub_labels == c].mean(axis=0) for c in range(k)
        ])
        # Assign all points to nearest centroid
        full_labels, _ = pairwise_distances_argmin_min(X.values, centroids)
        return full_labels
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        return labels


def plot_dendrogram(
    X: pd.DataFrame,
    k: int,
    n_samples: int = 2_000,
    save_path: str | None = None,
):
    """Compute Ward linkage on a subsample and plot the dendrogram.

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Number of clusters — used to draw the cut-line.
        n_samples: Number of rows to subsample before computing linkage.
                   Keep ≤ 5 000 for reasonable speed.
        save_path: If provided, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram

    rng = np.random.default_rng(42)
    n = len(X)
    n_sub = min(n_samples, n)
    idx = rng.choice(n, size=n_sub, replace=False)
    X_sub = X.iloc[idx].values

    Z = linkage(X_sub, method="ward")

    # Cut threshold: midpoint between the merge that creates k clusters and
    # the one that creates k-1 clusters.  Z[-k] is the distance at which we
    # go from k+1 → k clusters; Z[-k+1] from k → k-1 clusters.
    if k >= 2:
        cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2
    else:
        cut = Z[-1, 2] * 1.05

    p_show = min(n_sub - 1, max(k * 4, 40))

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=p_show,
        ax=ax,
        color_threshold=cut,
        above_threshold_color="grey",
        leaf_rotation=90,
        leaf_font_size=8,
    )
    ax.axhline(cut, color="red", linestyle="--", linewidth=1.4,
               label=f"Coupure k={k}  (seuil={cut:.1f})")
    ax.set_title(
        f"Dendrogramme Ward — {n_sub:,} points échantillonnés sur {n:,}  |  k={k} clusters",
        fontsize=12,
    )
    ax.set_xlabel("Clients (groupes de feuilles)")
    ax.set_ylabel("Distance Ward")
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def run_gmm(X: pd.DataFrame, k: int) -> tuple:
    """Fit GMM with diagonal covariance. Returns (hard labels, fitted model).

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Number of components (clusters).

    Returns:
        Tuple of (labels ndarray, fitted GaussianMixture model).
    """
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=RANDOM_STATE,
        max_iter=200,
    )
    labels = gmm.fit_predict(X)
    return labels, gmm


def run_hdbscan(
    X: pd.DataFrame,
    min_cluster_size: int = 500,
    min_samples: int = 5,
) -> tuple:
    """Fit HDBSCAN. Returns (labels ndarray, fitted model).
    Noise points are labelled -1.
    """
    import numpy as np
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
    )
    labels = model.fit_predict(X)
    return labels.astype(np.intp), model


def plot_kdistance(
    X: pd.DataFrame,
    k: int = 5,
    n_samples: int = 10_000,
    save_path: str | None = None,
):
    """Plot sorted k-th nearest-neighbour distances (k-distance graph).
    The elbow of the curve is the candidate eps for DBSCAN.

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Neighbour count — should equal DBSCAN min_samples.
        n_samples: Subsample size for speed.
        save_path: Optional figure save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    X_sub = X.values[idx] if hasattr(X, "values") else np.asarray(X)[idx]

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_sub)
    distances, _ = nn.kneighbors(X_sub)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_distances, linewidth=1.5)
    ax.set_xlabel("Points triés par distance croissante")
    ax.set_ylabel(f"Distance au {k}e plus proche voisin (eps candidat)")
    ax.set_title(f"k-distance graph (k={k}) — Identifier le coude pour choisir eps")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def run_dbscan(
    X: pd.DataFrame,
    eps: float = 1.5,
    min_samples: int = 5,
) -> tuple:
    """Fit DBSCAN. Returns (labels ndarray, fitted model).
    Noise points are labelled -1.

    Args:
        X: Feature matrix (n_samples, n_features).
        eps: Neighbourhood radius — tune via k-distance graph.
        min_samples: Minimum points to form a core point.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)
    return labels, model


def evaluate_gmm_bic_aic(X: pd.DataFrame, k_range=None) -> pd.DataFrame:
    """Evaluate BIC and AIC for a range of k.

    Args:
        X: Feature matrix (n_samples, n_features).
        k_range: Range of k values. Defaults to range(2, 21).

    Returns:
        DataFrame with columns: k, bic, aic.
    """
    k_range = k_range or range(2, 21)
    results = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=RANDOM_STATE,
            max_iter=200,
        )
        gmm.fit(X)
        results.append({"k": k, "bic": gmm.bic(X), "aic": gmm.aic(X)})
    return pd.DataFrame(results)


def build_comparison_table(
    comparison_results: list[dict],
    df_customers: pd.DataFrame,
) -> pd.DataFrame:
    """Build algorithm comparison DataFrame with min cluster size %.

    Args:
        comparison_results: List of dicts with keys: algorithm, silhouette,
            davies_bouldin, calinski_harabasz.
        df_customers: Customer DataFrame with '<algo>_label' columns.

    Returns:
        DataFrame with columns: algorithm, k, silhouette, davies_bouldin,
        calinski_harabasz, min_cluster_pct.
    """
    comp_df = pd.DataFrame(comparison_results)
    # Compute min cluster size % for each algorithm
    min_pcts = []
    for _, row in comp_df.iterrows():
        algo = row["algorithm"]
        # Use explicit label_col if provided, otherwise use heuristic
        if "label_col" in row and pd.notna(row.get("label_col")):
            candidates = [row["label_col"]]
        else:
            label_col = f"{algo.lower().replace(' ', '_').replace('(', '').replace(')', '')}_label"
            # Try common naming patterns
            candidates = [label_col]
            if "hierarchical" in algo.lower() or "ward" in algo.lower():
                candidates = ["hclust_label", "hierarchical_label"]
            elif "gmm" in algo.lower():
                candidates = ["gmm_label", "gmm_diag_label"]
            elif "kmeans" in algo.lower():
                candidates = ["kmeans_label"]
        found = False
        for col in candidates:
            if col in df_customers.columns:
                labels = df_customers[col]
                valid_labels = labels[labels != -1]
                if len(valid_labels) > 0:
                    min_pcts.append(valid_labels.value_counts(normalize=True).min() * 100)
                else:
                    min_pcts.append(float("nan"))
                found = True
                break
        if not found:
            min_pcts.append(float("nan"))
    comp_df["min_cluster_pct"] = min_pcts
    if "notes" not in comp_df.columns:
        comp_df["notes"] = ""
    return comp_df


def select_best_algorithm(comp_df: pd.DataFrame) -> pd.Series:
    """Select best algorithm based on combined score.

    Score = silhouette - davies_bouldin / max(davies_bouldin).
    Rows with min_cluster_pct < 1% are penalised.

    Args:
        comp_df: Output of build_comparison_table().

    Returns:
        Series for the best algorithm row.
    """
    df = comp_df.copy()
    db_max = df["davies_bouldin"].max()
    # Score incorporates silhouette, normalized negative DB, and normalized CH
    ch_max = df["calinski_harabasz"].max()
    db_norm = df["davies_bouldin"] / db_max if db_max > 0 else 0
    ch_norm = df["calinski_harabasz"] / ch_max if ch_max > 0 else 0
    df["score"] = df["silhouette"] - db_norm + 0.1 * ch_norm

    # Penalise trivially small clusters
    if "min_cluster_pct" in df.columns:
        df.loc[df["min_cluster_pct"] < 1.0, "score"] -= 0.5
    return df.loc[df["score"].idxmax()]


def log_clustering_run(
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: dict | None = None,
    parent_run_id: str | None = None,
) -> str:
    """Log a clustering run to MLflow. Returns the run_id."""
    with mlflow.start_run(run_name=run_name, nested=parent_run_id is not None) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
        return run.info.run_id
