"""Tests for src/clustering.py — Stories 4.1 & 4.2 & 4.3."""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

