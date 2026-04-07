"""Tests for src/validation.py — Story 4.6: Cluster Stability Validation."""
import pytest
import pandas as pd
import numpy as np

from src.validation import bootstrap_stability


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
def full_labels(synthetic_X):
    """KMeans labels on the full synthetic dataset."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(synthetic_X)
    return pd.Series(labels, index=synthetic_X.index)


def _kmeans_3(X):
    """Simple algorithm_fn for testing: KMeans k=3."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    return km.fit_predict(X)


class TestBootstrapStability:
    """Tests for bootstrap_stability() — AC 1-5."""

    def test_returns_dataframe(self, synthetic_X, full_labels):
        """AC-1: returns a DataFrame."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, synthetic_X, full_labels):
        """AC-4: table has bootstrap, n_samples, ari, n_clusters."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3)
        expected = {"bootstrap", "n_samples", "ari", "n_clusters"}
        assert set(result.columns) == expected

    def test_correct_number_of_rows(self, synthetic_X, full_labels):
        """AC-1: 5 bootstraps → 5 rows."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=5)
        assert len(result) == 5

    def test_custom_n_bootstraps(self, synthetic_X, full_labels):
        """Supports non-default n_bootstraps."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=3)
        assert len(result) == 3

    def test_bootstrap_numbers_sequential(self, synthetic_X, full_labels):
        """Bootstrap column is 1-indexed sequential."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=5)
        assert list(result["bootstrap"]) == [1, 2, 3, 4, 5]

    def test_subsample_size(self, synthetic_X, full_labels):
        """AC-1: 80% subsample → n_samples ≈ 0.8 * 150 = 120."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, subsample_frac=0.80)
        for _, row in result.iterrows():
            assert row["n_samples"] == int(0.80 * len(synthetic_X))

    def test_ari_in_valid_range(self, synthetic_X, full_labels):
        """AC-2: ARI is between -1 and 1."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3)
        assert (result["ari"] >= -1.0).all()
        assert (result["ari"] <= 1.0).all()

    def test_high_ari_on_clear_clusters(self, synthetic_X, full_labels):
        """AC-3: well-separated clusters → mean ARI ≥ 0.70."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3)
        mean_ari = result["ari"].mean()
        assert mean_ari >= 0.70, f"Mean ARI {mean_ari:.3f} < 0.70 on clear clusters"

    def test_n_clusters_column(self, synthetic_X, full_labels):
        """AC-4: n_clusters reported correctly."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3)
        assert (result["n_clusters"] == 3).all()

    def test_deterministic_results(self, synthetic_X, full_labels):
        """Same inputs → same results (seeded sampling)."""
        r1 = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=3)
        r2 = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=3)
        pd.testing.assert_frame_equal(r1, r2)

    def test_custom_subsample_frac(self, synthetic_X, full_labels):
        """Different subsample_frac changes n_samples."""
        r50 = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=1, subsample_frac=0.50)
        r90 = bootstrap_stability(synthetic_X, full_labels, _kmeans_3, n_bootstraps=1, subsample_frac=0.90)
        assert r50.iloc[0]["n_samples"] < r90.iloc[0]["n_samples"]

    def test_ari_rounded_to_4_decimals(self, synthetic_X, full_labels):
        """ARI values are rounded to 4 decimal places."""
        result = bootstrap_stability(synthetic_X, full_labels, _kmeans_3)
        for ari in result["ari"]:
            # Check string representation has at most 4 decimal digits
            s = f"{ari:.10f}"
            decimal_part = s.split(".")[1]
            assert decimal_part[4:] == "000000", f"ARI {ari} not rounded to 4 decimals"
