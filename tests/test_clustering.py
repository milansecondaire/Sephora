"""Tests for src/clustering.py — Stories 4.1 & 4.2 & 4.3 & 4.4."""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from src.clustering import evaluate_kmeans_k_range, get_top_k_candidates, run_kmeans_final


@pytest.fixture
def synthetic_X():
    """Create a small synthetic dataset with 3 clear clusters."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(50, 4) + np.array([0, 0, 0, 0])
    c2 = rng.randn(50, 4) + np.array([5, 5, 5, 5])
    c3 = rng.randn(50, 4) + np.array([10, 10, 10, 10])
    X = np.vstack([c1, c2, c3])
    return pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])


@pytest.fixture
def sample_metrics():
    """Sample metrics DataFrame for get_top_k_candidates tests."""
    return pd.DataFrame({
        "k": [2, 3, 4, 5, 6],
        "inertia": [1000, 600, 400, 350, 320],
        "silhouette": [0.35, 0.55, 0.42, 0.38, 0.30],
        "davies_bouldin": [1.5, 0.8, 1.0, 1.2, 1.4],
        "calinski_harabasz": [100, 200, 180, 160, 140],
    })


class TestEvaluateKmeansKRange:
    def test_returns_dataframe(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 5))
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 5))
        expected_cols = {"k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"}
        assert set(result.columns) == expected_cols

    def test_correct_k_values(self, synthetic_X):
        k_range = range(2, 6)
        result = evaluate_kmeans_k_range(synthetic_X, k_range=k_range)
        assert list(result["k"]) == list(k_range)

    def test_inertia_decreases(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 8))
        inertias = result["inertia"].values
        # Inertia should generally decrease as k increases
        assert inertias[0] > inertias[-1]

    def test_silhouette_in_range(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 5))
        assert (result["silhouette"] >= -1).all()
        assert (result["silhouette"] <= 1).all()

    def test_davies_bouldin_positive(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 5))
        assert (result["davies_bouldin"] > 0).all()

    def test_calinski_harabasz_positive(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 5))
        assert (result["calinski_harabasz"] > 0).all()

    def test_best_k_is_3_for_clear_clusters(self, synthetic_X):
        result = evaluate_kmeans_k_range(synthetic_X, k_range=range(2, 8))
        best_k = result.loc[result["silhouette"].idxmax(), "k"]
        assert best_k == 3

    def test_defaults_to_config_k_range(self, synthetic_X):
        """Check that default k_range is used (from config) when not specified."""
        from src.config import K_RANGE
        result = evaluate_kmeans_k_range(synthetic_X)
        assert list(result["k"]) == list(K_RANGE)


class TestGetTopKCandidates:
    def test_returns_top_n(self, sample_metrics):
        top = get_top_k_candidates(sample_metrics, top_n=3)
        assert len(top) == 3

    def test_sorted_by_silhouette_desc(self, sample_metrics):
        top = get_top_k_candidates(sample_metrics, top_n=3)
        sil_values = top["silhouette"].values
        assert all(sil_values[i] >= sil_values[i + 1] for i in range(len(sil_values) - 1))

    def test_best_k_first(self, sample_metrics):
        top = get_top_k_candidates(sample_metrics, top_n=1)
        assert top.iloc[0]["k"] == 3  # k=3 has highest silhouette (0.55)

    def test_reset_index(self, sample_metrics):
        top = get_top_k_candidates(sample_metrics, top_n=3)
        assert list(top.index) == [0, 1, 2]

    def test_all_columns_preserved(self, sample_metrics):
        top = get_top_k_candidates(sample_metrics, top_n=2)
        assert set(top.columns) == set(sample_metrics.columns)


class TestPlotElbowCurves:
    def test_returns_figure(self, sample_metrics):
        from src.visualization import plot_elbow_curves
        fig = plot_elbow_curves(sample_metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_three_panels(self, sample_metrics):
        from src.visualization import plot_elbow_curves
        fig = plot_elbow_curves(sample_metrics)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_saves_to_file(self, sample_metrics, tmp_path):
        from src.visualization import plot_elbow_curves
        path = str(tmp_path / "elbow_test.png")
        fig = plot_elbow_curves(sample_metrics, save_path=path)
        import os
        assert os.path.exists(path)
        plt.close(fig)


# ── Story 4.2: K-Means Final Run & Assignment ──────────────────────

class TestRunKmeansFinal:
    """Tests for run_kmeans_final() — AC 1-5."""

    def test_returns_tuple_of_labels_and_model(self, synthetic_X):
        """AC-1: function returns (labels, fitted KMeans model)."""
        labels, model = run_kmeans_final(synthetic_X, k=3)
        assert isinstance(labels, np.ndarray)
        from sklearn.cluster import KMeans
        assert isinstance(model, KMeans)

    def test_labels_length_matches_input(self, synthetic_X):
        labels, _ = run_kmeans_final(synthetic_X, k=3)
        assert len(labels) == len(synthetic_X)

    def test_labels_contain_k_unique_values(self, synthetic_X):
        """AC-1: final run with k clusters produces k distinct labels."""
        k = 3
        labels, _ = run_kmeans_final(synthetic_X, k=k)
        assert len(set(labels)) == k

    def test_deterministic_with_random_state(self, synthetic_X):
        """Fixed seed → same labels every run."""
        labels1, _ = run_kmeans_final(synthetic_X, k=3, random_state=42)
        labels2, _ = run_kmeans_final(synthetic_X, k=3, random_state=42)
        np.testing.assert_array_equal(labels1, labels2)
        
        # Verify changing random_state gives different initialization trajectory
        # (Though they might converge to same labels on clear data, the model itself is different)
        labels3, model3 = run_kmeans_final(synthetic_X, k=3, random_state=999)
        _, model1 = run_kmeans_final(synthetic_X, k=3, random_state=42)
        assert not np.array_equal(model1.cluster_centers_, model3.cluster_centers_)

    def test_kmeans_initialization_params(self, monkeypatch, synthetic_X):
        """Verify KMeans is instantiated with n_init=10 and max_iter=300."""
        from sklearn.cluster import KMeans
        import src.clustering
        
        # Mock KMeans entirely to spy on initialization parameters
        called_args = {}
        original_init = KMeans.__init__
        
        def mock_init(self, n_clusters, *, random_state, n_init, max_iter, **kwargs):
            called_args['n_clusters'] = n_clusters
            called_args['random_state'] = random_state
            called_args['n_init'] = n_init
            called_args['max_iter'] = max_iter
            original_init(self, n_clusters=n_clusters, random_state=random_state, 
                          n_init=n_init, max_iter=max_iter, **kwargs)
            
        monkeypatch.setattr(KMeans, "__init__", mock_init)
        
        run_kmeans_final(synthetic_X, k=5, random_state=123)
        assert called_args['n_clusters'] == 5
        assert called_args['n_init'] == 10
        assert called_args['max_iter'] == 300
        assert called_args['random_state'] == 123

    def test_model_is_fitted(self, synthetic_X):
        """Model must have cluster_centers_ attribute after fitting."""
        _, model = run_kmeans_final(synthetic_X, k=3)
        assert hasattr(model, "cluster_centers_")
        assert model.cluster_centers_.shape == (3, synthetic_X.shape[1])

    def test_labels_are_integers(self, synthetic_X):
        labels, _ = run_kmeans_final(synthetic_X, k=3)
        assert labels.dtype in (np.int32, np.int64, np.intp)

    def test_different_k_produces_different_clusters(self, synthetic_X):
        labels_3, _ = run_kmeans_final(synthetic_X, k=3)
        labels_4, _ = run_kmeans_final(synthetic_X, k=4)
        assert len(set(labels_3)) == 3
        assert len(set(labels_4)) == 4


# ── Story 4.3: Agglomerative Hierarchical Clustering ─────────────────────

class TestRunHierarchical:
    """Tests for run_hierarchical() — AC 2–4."""

    def test_returns_ndarray(self, synthetic_X):
        """AC-2: AgglomerativeClustering returns an ndarray."""
        from src.clustering import run_hierarchical
        labels = run_hierarchical(synthetic_X, k=3)
        assert isinstance(labels, np.ndarray)

    def test_labels_length_matches_input(self, synthetic_X):
        """Labels length equals number of samples."""
        from src.clustering import run_hierarchical
        labels = run_hierarchical(synthetic_X, k=3)
        assert len(labels) == len(synthetic_X)

    def test_labels_contain_k_unique_values(self, synthetic_X):
        """AC-2: k clusters → k distinct label values."""
        from src.clustering import run_hierarchical
        k = 3
        labels = run_hierarchical(synthetic_X, k=k)
        assert len(set(labels)) == k

    def test_different_k_produces_different_n_clusters(self, synthetic_X):
        """Varying k produces expected cluster count."""
        from src.clustering import run_hierarchical
        for k in [2, 3, 4]:
            labels = run_hierarchical(synthetic_X, k=k)
            assert len(set(labels)) == k

    def test_labels_are_integers(self, synthetic_X):
        """Labels must be integer dtype."""
        from src.clustering import run_hierarchical
        labels = run_hierarchical(synthetic_X, k=3)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_metrics_computable_on_labels(self, synthetic_X):
        """AC-3/4: silhouette, DB, CH scores computable without error."""
        from src.clustering import run_hierarchical
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
        )
        labels = run_hierarchical(synthetic_X, k=3)
        sil = silhouette_score(synthetic_X, labels)
        db = davies_bouldin_score(synthetic_X, labels)
        ch = calinski_harabasz_score(synthetic_X, labels)
        assert -1.0 <= sil <= 1.0
        assert db > 0
        assert ch > 0


class TestPlotDendrogram:
    """Tests for plot_dendrogram() — AC 1."""

    def test_returns_figure(self, synthetic_X):
        from src.visualization import plot_dendrogram
        fig = plot_dendrogram(synthetic_X)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, synthetic_X, tmp_path):
        from src.visualization import plot_dendrogram
        path = str(tmp_path / "dendrogram_test.png")
        fig = plot_dendrogram(synthetic_X, save_path=path)
        import os
        assert os.path.exists(path)
        plt.close(fig)

    def test_has_axes(self, synthetic_X):
        from src.visualization import plot_dendrogram
        fig = plot_dendrogram(synthetic_X)
        assert len(fig.axes) == 1
        plt.close(fig)


class TestPlotUmapKmeansVsHclust:
    """Tests for plot_umap_kmeans_vs_hclust() — AC 5."""

    @pytest.fixture
    def umap_and_customers(self):
        """Minimal UMAP embedding + customer DataFrame with both label columns."""
        n = 30
        rng = np.random.RandomState(0)
        umap_df = pd.DataFrame(
            {"umap_1": rng.randn(n), "umap_2": rng.randn(n)},
            index=range(n),
        )
        df_customers = pd.DataFrame(
            {
                "kmeans_label": rng.randint(0, 3, n),
                "hclust_label": rng.randint(0, 3, n),
            },
            index=range(n),
        )
        return umap_df, df_customers

    def test_returns_figure(self, umap_and_customers, tmp_path):
        from src.visualization import plot_umap_kmeans_vs_hclust
        umap_df, df_customers = umap_and_customers
        save_path = str(tmp_path / "umap_cmp.png")
        fig = plot_umap_kmeans_vs_hclust(umap_df, df_customers, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_panels(self, umap_and_customers, tmp_path):
        from src.visualization import plot_umap_kmeans_vs_hclust
        umap_df, df_customers = umap_and_customers
        save_path = str(tmp_path / "umap_cmp2.png")
        fig = plot_umap_kmeans_vs_hclust(umap_df, df_customers, save_path=save_path)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_saves_to_file(self, umap_and_customers, tmp_path):
        from src.visualization import plot_umap_kmeans_vs_hclust
        import os
        umap_df, df_customers = umap_and_customers
        path = str(tmp_path / "umap_side.png")
        fig = plot_umap_kmeans_vs_hclust(umap_df, df_customers, save_path=path)
        assert os.path.exists(path)
        plt.close(fig)


# ── Story R1.5: MLflow Integration ──────────────────────────────────

class TestLogClusteringRun:
    """Tests for log_clustering_run() — AC 2."""

    def test_returns_run_id_string(self, tmp_path):
        import mlflow
        from src.clustering import log_clustering_run
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-exp")
        run_id = log_clustering_run(
            run_name="test-run",
            params={"k": 3, "algorithm": "kmeans"},
            metrics={"silhouette": 0.5, "davies_bouldin": 1.0},
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 32

    def test_params_logged(self, tmp_path):
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.clustering import log_clustering_run
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-exp")
        run_id = log_clustering_run(
            run_name="test-params",
            params={"k": 5, "algorithm": "kmeans"},
            metrics={"silhouette": 0.3},
        )
        client = MlflowClient()
        run = client.get_run(run_id)
        assert run.data.params["k"] == "5"
        assert run.data.params["algorithm"] == "kmeans"

    def test_metrics_logged(self, tmp_path):
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.clustering import log_clustering_run
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-exp")
        run_id = log_clustering_run(
            run_name="test-metrics",
            params={"k": 3},
            metrics={"silhouette": 0.42, "davies_bouldin": 1.5},
        )
        client = MlflowClient()
        run = client.get_run(run_id)
        assert abs(run.data.metrics["silhouette"] - 0.42) < 1e-6
        assert abs(run.data.metrics["davies_bouldin"] - 1.5) < 1e-6

    def test_artifacts_logged(self, tmp_path):
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.clustering import log_clustering_run
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-exp")
        # Create a temp CSV artifact
        csv_path = str(tmp_path / "labels.csv")
        pd.DataFrame({"label": [0, 1, 2]}).to_csv(csv_path, index=False)
        run_id = log_clustering_run(
            run_name="test-artifact",
            params={"k": 3},
            metrics={"silhouette": 0.3},
            artifacts={"labels": csv_path},
        )
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id, path="labels")
        assert len(artifacts) > 0

    def test_no_artifacts_when_none(self, tmp_path):
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.clustering import log_clustering_run
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-exp")
        run_id = log_clustering_run(
            run_name="test-no-artifact",
            params={"k": 3},
            metrics={"silhouette": 0.3},
            artifacts=None,
        )
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert len(artifacts) == 0

    def test_nested_run(self, tmp_path):
        import mlflow
        from src.clustering import log_clustering_run
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-exp")
        with mlflow.start_run(run_name="parent") as parent:
            child_id = log_clustering_run(
                run_name="child",
                params={"k": 2},
                metrics={"silhouette": 0.1},
                parent_run_id=parent.info.run_id,
            )
        assert isinstance(child_id, str)
        assert child_id != parent.info.run_id


# ── Story 4.4: Gaussian Mixture Models ──────────────────────────────

class TestRunGmm:
    """Tests for run_gmm() — AC 1-2."""

    def test_returns_tuple_of_labels_and_model(self, synthetic_X):
        """AC-1: function returns (labels, fitted GaussianMixture model)."""
        from src.clustering import run_gmm
        labels, model = run_gmm(synthetic_X, k=3)
        assert isinstance(labels, np.ndarray)
        assert isinstance(model, GaussianMixture)

    def test_labels_length_matches_input(self, synthetic_X):
        from src.clustering import run_gmm
        labels, _ = run_gmm(synthetic_X, k=3)
        assert len(labels) == len(synthetic_X)

    def test_labels_contain_k_unique_values(self, synthetic_X):
        """AC-1: n_components=k produces k distinct labels on clear clusters."""
        from src.clustering import run_gmm
        k = 3
        labels, _ = run_gmm(synthetic_X, k=k)
        assert len(set(labels)) == k

    def test_covariance_type_is_diag(self, synthetic_X):
        """AC-1: covariance_type must be 'diag' (not 'full')."""
        from src.clustering import run_gmm
        _, model = run_gmm(synthetic_X, k=3)
        assert model.covariance_type == "diag"

    def test_random_state_is_42(self, synthetic_X):
        """AC-1: random_state=RANDOM_STATE (42)."""
        from src.clustering import run_gmm
        from src.config import RANDOM_STATE
        _, model = run_gmm(synthetic_X, k=3)
        assert model.random_state == RANDOM_STATE

    def test_deterministic(self, synthetic_X):
        """Same input → same labels."""
        from src.clustering import run_gmm
        labels1, _ = run_gmm(synthetic_X, k=3)
        labels2, _ = run_gmm(synthetic_X, k=3)
        np.testing.assert_array_equal(labels1, labels2)

    def test_labels_are_integers(self, synthetic_X):
        from src.clustering import run_gmm
        labels, _ = run_gmm(synthetic_X, k=3)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_model_is_fitted(self, synthetic_X):
        """Model must have converged_ attribute after fitting."""
        from src.clustering import run_gmm
        _, model = run_gmm(synthetic_X, k=3)
        assert hasattr(model, "converged_")

    def test_metrics_computable_on_labels(self, synthetic_X):
        """AC-4: silhouette, DB, CH scores computable on gmm labels."""
        from src.clustering import run_gmm
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
        )
        labels, _ = run_gmm(synthetic_X, k=3)
        sil = silhouette_score(synthetic_X, labels)
        db = davies_bouldin_score(synthetic_X, labels)
        ch = calinski_harabasz_score(synthetic_X, labels)
        assert -1.0 <= sil <= 1.0
        assert db > 0
        assert ch > 0


class TestEvaluateGmmBicAic:
    """Tests for evaluate_gmm_bic_aic() — AC 3."""

    def test_returns_dataframe(self, synthetic_X):
        from src.clustering import evaluate_gmm_bic_aic
        result = evaluate_gmm_bic_aic(synthetic_X, k_range=range(2, 5))
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, synthetic_X):
        from src.clustering import evaluate_gmm_bic_aic
        result = evaluate_gmm_bic_aic(synthetic_X, k_range=range(2, 5))
        assert set(result.columns) == {"k", "bic", "aic"}

    def test_correct_k_values(self, synthetic_X):
        from src.clustering import evaluate_gmm_bic_aic
        k_range = range(2, 6)
        result = evaluate_gmm_bic_aic(synthetic_X, k_range=k_range)
        assert list(result["k"]) == list(k_range)

    def test_defaults_to_2_through_20(self, synthetic_X):
        from src.clustering import evaluate_gmm_bic_aic
        result = evaluate_gmm_bic_aic(synthetic_X)
        assert list(result["k"]) == list(range(2, 21))

    def test_bic_aic_are_finite(self, synthetic_X):
        from src.clustering import evaluate_gmm_bic_aic
        result = evaluate_gmm_bic_aic(synthetic_X, k_range=range(2, 5))
        assert np.isfinite(result["bic"]).all()
        assert np.isfinite(result["aic"]).all()

    def test_best_bic_at_k3_for_clear_clusters(self, synthetic_X):
        """With 3 clear clusters, best BIC (lowest) should be at k=3."""
        from src.clustering import evaluate_gmm_bic_aic
        result = evaluate_gmm_bic_aic(synthetic_X, k_range=range(2, 7))
        best_k = result.loc[result["bic"].idxmin(), "k"]
        assert best_k == 3


class TestPlotGmmBicAic:
    """Tests for plot_gmm_bic_aic() in visualization.py."""

    @pytest.fixture
    def bic_aic_df(self):
        return pd.DataFrame({
            "k": [2, 3, 4, 5],
            "bic": [1000, 800, 850, 900],
            "aic": [950, 750, 800, 850],
        })

    def test_returns_figure(self, bic_aic_df):
        from src.visualization import plot_gmm_bic_aic
        fig = plot_gmm_bic_aic(bic_aic_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, bic_aic_df, tmp_path):
        from src.visualization import plot_gmm_bic_aic
        path = str(tmp_path / "gmm_bic_aic_test.png")
        fig = plot_gmm_bic_aic(bic_aic_df, save_path=path)
        import os
        assert os.path.exists(path)
        plt.close(fig)

    def test_has_axes(self, bic_aic_df):
        from src.visualization import plot_gmm_bic_aic
        fig = plot_gmm_bic_aic(bic_aic_df)
        assert len(fig.axes) >= 1
        plt.close(fig)


# ── Story 4.5: Algorithm Comparison & Final Selection ────────────────

class TestBuildComparisonTable:
    """Tests for build_comparison_table() — AC 1."""

    @pytest.fixture
    def comparison_results_fixture(self):
        return [
            {"algorithm": "KMeans", "k": 5, "silhouette": 0.35, "davies_bouldin": 1.2, "calinski_harabasz": 180},
            {"algorithm": "Hierarchical (Ward)", "k": 5, "silhouette": 0.32, "davies_bouldin": 1.3, "calinski_harabasz": 170},
            {"algorithm": "GMM (diag)", "k": 5, "silhouette": 0.30, "davies_bouldin": 1.5, "calinski_harabasz": 160},
        ]

    @pytest.fixture
    def df_customers_fixture(self):
        n = 100
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "kmeans_label": rng.randint(0, 5, n),
            "hclust_label": rng.randint(0, 5, n),
            "gmm_label": rng.randint(0, 5, n),
        })

    def test_returns_dataframe(self, comparison_results_fixture, df_customers_fixture):
        from src.clustering import build_comparison_table
        comp_df = build_comparison_table(comparison_results_fixture, df_customers_fixture)
        assert isinstance(comp_df, pd.DataFrame)

    def test_has_min_cluster_pct_column(self, comparison_results_fixture, df_customers_fixture):
        from src.clustering import build_comparison_table
        comp_df = build_comparison_table(comparison_results_fixture, df_customers_fixture)
        assert "min_cluster_pct" in comp_df.columns

    def test_min_cluster_pct_is_positive(self, comparison_results_fixture, df_customers_fixture):
        from src.clustering import build_comparison_table
        comp_df = build_comparison_table(comparison_results_fixture, df_customers_fixture)
        assert (comp_df["min_cluster_pct"] > 0).all()

    def test_min_cluster_pct_max_100(self, comparison_results_fixture, df_customers_fixture):
        from src.clustering import build_comparison_table
        comp_df = build_comparison_table(comparison_results_fixture, df_customers_fixture)
        assert (comp_df["min_cluster_pct"] <= 100).all()

    def test_preserves_original_columns(self, comparison_results_fixture, df_customers_fixture):
        from src.clustering import build_comparison_table
        comp_df = build_comparison_table(comparison_results_fixture, df_customers_fixture)
        for col in ["algorithm", "k", "silhouette", "davies_bouldin", "calinski_harabasz"]:
            assert col in comp_df.columns

    def test_row_count_matches_input(self, comparison_results_fixture, df_customers_fixture):
        from src.clustering import build_comparison_table
        comp_df = build_comparison_table(comparison_results_fixture, df_customers_fixture)
        assert len(comp_df) == len(comparison_results_fixture)


class TestSelectBestAlgorithm:
    """Tests for select_best_algorithm() — AC 2."""

    def test_returns_series(self):
        from src.clustering import select_best_algorithm
        comp_df = pd.DataFrame({
            "algorithm": ["KMeans", "Hierarchical", "GMM"],
            "k": [5, 5, 5],
            "silhouette": [0.40, 0.35, 0.30],
            "davies_bouldin": [1.0, 1.2, 1.5],
            "calinski_harabasz": [200, 180, 160],
            "min_cluster_pct": [10.0, 12.0, 8.0],
        })
        result = select_best_algorithm(comp_df)
        assert isinstance(result, pd.Series)

    def test_picks_highest_score(self):
        from src.clustering import select_best_algorithm
        comp_df = pd.DataFrame({
            "algorithm": ["KMeans", "Hierarchical", "GMM"],
            "k": [5, 5, 5],
            "silhouette": [0.40, 0.35, 0.30],
            "davies_bouldin": [1.0, 1.2, 1.5],
            "calinski_harabasz": [200, 180, 160],
            "min_cluster_pct": [10.0, 12.0, 8.0],
        })
        result = select_best_algorithm(comp_df)
        # KMeans has highest silhouette and lowest DB → best score
        assert result["algorithm"] == "KMeans"

    def test_penalises_tiny_clusters(self):
        from src.clustering import select_best_algorithm
        comp_df = pd.DataFrame({
            "algorithm": ["A", "B"],
            "k": [5, 5],
            "silhouette": [0.50, 0.45],
            "davies_bouldin": [1.0, 1.0],
            "calinski_harabasz": [200, 200],
            "min_cluster_pct": [0.5, 15.0],  # A has tiny cluster
        })
        result = select_best_algorithm(comp_df)
        assert result["algorithm"] == "B"

    def test_result_has_algorithm_key(self):
        from src.clustering import select_best_algorithm
        comp_df = pd.DataFrame({
            "algorithm": ["KMeans"],
            "k": [3],
            "silhouette": [0.5],
            "davies_bouldin": [1.0],
            "calinski_harabasz": [200],
            "min_cluster_pct": [20.0],
        })
        result = select_best_algorithm(comp_df)
        assert "algorithm" in result.index


# ── Story 4.7: HDBSCAN Clustering ───────────────────────────────────

class TestRunHdbscan:
    """Tests for run_hdbscan() — AC 1, 3."""

    def test_returns_tuple_of_labels_and_model(self, synthetic_X):
        """AC-1: function returns (labels, fitted HDBSCAN model)."""
        from src.clustering import run_hdbscan
        from sklearn.cluster import HDBSCAN
        labels, model = run_hdbscan(synthetic_X, min_cluster_size=10)
        assert isinstance(labels, np.ndarray)
        assert isinstance(model, HDBSCAN)

    def test_labels_length_matches_input(self, synthetic_X):
        from src.clustering import run_hdbscan
        labels, _ = run_hdbscan(synthetic_X, min_cluster_size=10)
        assert len(labels) == len(synthetic_X)

    def test_noise_points_labelled_minus_one(self, synthetic_X):
        """AC-3: noise points have label -1."""
        from src.clustering import run_hdbscan
        labels, _ = run_hdbscan(synthetic_X, min_cluster_size=10)
        # Labels should contain -1 or non-negative integers
        for lbl in labels:
            assert lbl >= -1

    def test_finds_clusters_on_clear_data(self, synthetic_X):
        """On well-separated data, HDBSCAN should find at least 2 clusters."""
        from src.clustering import run_hdbscan
        labels, _ = run_hdbscan(synthetic_X, min_cluster_size=10)
        non_noise = labels[labels != -1]
        assert len(set(non_noise)) >= 2

    def test_default_params(self, synthetic_X):
        """Default min_cluster_size=500, min_samples=5."""
        from src.clustering import run_hdbscan
        _, model = run_hdbscan(synthetic_X)
        assert model.min_cluster_size == 500
        assert model.min_samples == 5

    def test_custom_params(self, synthetic_X):
        """Custom min_cluster_size and min_samples are respected."""
        from src.clustering import run_hdbscan
        _, model = run_hdbscan(synthetic_X, min_cluster_size=20, min_samples=3)
        assert model.min_cluster_size == 20
        assert model.min_samples == 3

    def test_labels_are_integers(self, synthetic_X):
        from src.clustering import run_hdbscan
        labels, _ = run_hdbscan(synthetic_X, min_cluster_size=10)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_model_is_fitted(self, synthetic_X):
        """Model must have labels_ attribute after fitting."""
        from src.clustering import run_hdbscan
        _, model = run_hdbscan(synthetic_X, min_cluster_size=10)
        assert hasattr(model, "labels_")

    def test_accepts_dataframe(self, synthetic_X):
        """Must accept pandas DataFrame."""
        from src.clustering import run_hdbscan
        labels, _ = run_hdbscan(synthetic_X, min_cluster_size=10)
        assert len(labels) == len(synthetic_X)

    def test_cluster_selection_method_eom(self, synthetic_X):
        """cluster_selection_method should be 'eom'."""
        from src.clustering import run_hdbscan
        _, model = run_hdbscan(synthetic_X, min_cluster_size=10)
        assert model.cluster_selection_method == "eom"

