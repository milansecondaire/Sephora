"""Tests for src/validation.py — Story 4.6 + Story 5.5."""
import pytest
import pandas as pd
import numpy as np

from src.validation import bootstrap_stability, run_kruskal_wallis, run_posthoc_mannwhitney, print_kruskal_summary


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


# ---------- US-5.5 fixtures ----------

@pytest.fixture
def clustered_df():
    """DataFrame with 3 clusters and 4 KPIs, clearly separated."""
    rng = np.random.RandomState(42)
    n = 60
    data = {
        "cluster_id": [0] * 20 + [1] * 20 + [2] * 20,
        "kpi_a": np.concatenate([rng.normal(10, 1, 20), rng.normal(50, 1, 20), rng.normal(90, 1, 20)]),
        "kpi_b": np.concatenate([rng.normal(5, 0.5, 20), rng.normal(25, 0.5, 20), rng.normal(45, 0.5, 20)]),
        "kpi_c": np.concatenate([rng.normal(100, 2, 20), rng.normal(100, 2, 20), rng.normal(100, 2, 20)]),
        "kpi_d": np.concatenate([rng.normal(0, 1, 20), rng.normal(20, 1, 20), rng.normal(40, 1, 20)]),
    }
    return pd.DataFrame(data)


@pytest.fixture
def kpi_list():
    return ["kpi_a", "kpi_b", "kpi_c", "kpi_d"]


class TestRunKruskalWallis:
    """Tests for run_kruskal_wallis() — AC 1-3."""

    def test_returns_dataframe(self, clustered_df, kpi_list):
        """AC-1: Returns a DataFrame."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, clustered_df, kpi_list):
        """AC-1: Has kpi, H_statistic, p_value, significant columns."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        assert set(result.columns) == {"kpi", "H_statistic", "p_value", "significant"}

    def test_one_row_per_kpi(self, clustered_df, kpi_list):
        """AC-1: One row per KPI."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        assert len(result) == len(kpi_list)
        assert list(result["kpi"]) == kpi_list

    def test_p_value_range(self, clustered_df, kpi_list):
        """AC-2: p-values between 0 and 1."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_significant_flag_matches_threshold(self, clustered_df, kpi_list):
        """AC-2: significant == (p_value < 0.05)."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        for _, row in result.iterrows():
            assert row["significant"] == (row["p_value"] < 0.05)

    def test_well_separated_clusters_significant(self, clustered_df, kpi_list):
        """AC-3: kpi_a, kpi_b, kpi_d are well separated → significant."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        sig = result.set_index("kpi")["significant"]
        assert sig["kpi_a"] is True or sig["kpi_a"] == True
        assert sig["kpi_b"] is True or sig["kpi_b"] == True
        assert sig["kpi_d"] is True or sig["kpi_d"] == True

    def test_non_separated_cluster_not_significant(self, clustered_df, kpi_list):
        """AC-3: kpi_c has same distribution → not significant."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        sig = result.set_index("kpi")["significant"]
        assert sig["kpi_c"] == False

    def test_at_least_n_significant(self, clustered_df, kpi_list):
        """AC-3: At least 3 of 4 KPIs significant (a, b, d)."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        assert result["significant"].sum() >= 3

    def test_h_statistic_positive(self, clustered_df, kpi_list):
        """H-statistic is always non-negative."""
        result = run_kruskal_wallis(clustered_df, kpi_list)
        assert (result["H_statistic"] >= 0).all()


class TestRunPosthocMannWhitney:
    """Tests for run_posthoc_mannwhitney() — AC 4."""

    def test_returns_dataframe(self, clustered_df, kpi_list):
        """AC-4: Returns a DataFrame."""
        result = run_posthoc_mannwhitney(clustered_df, kpi_list)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, clustered_df, kpi_list):
        """AC-4: Has expected columns."""
        result = run_posthoc_mannwhitney(clustered_df, kpi_list)
        assert set(result.columns) == {"kpi", "cluster_a", "cluster_b", "U_statistic", "p_value", "significant"}

    def test_correct_number_of_rows(self, clustered_df, kpi_list):
        """AC-4: 4 KPIs × C(3,2)=3 pairs = 12 rows."""
        result = run_posthoc_mannwhitney(clustered_df, kpi_list)
        assert len(result) == 4 * 3  # 4 kpis * 3 pairs

    def test_significant_pairs_for_separated_kpis(self, clustered_df):
        """AC-4: kpi_a pairs should all be significant."""
        result = run_posthoc_mannwhitney(clustered_df, ["kpi_a"])
        assert result["significant"].all()

    def test_nonsignificant_pairs_for_same_kpi(self, clustered_df):
        """AC-4: kpi_c pairs should NOT be significant."""
        result = run_posthoc_mannwhitney(clustered_df, ["kpi_c"])
        assert not result["significant"].any()

    def test_p_values_in_range(self, clustered_df, kpi_list):
        """p-values between 0 and 1."""
        result = run_posthoc_mannwhitney(clustered_df, kpi_list)
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_cluster_pairs_sorted(self, clustered_df, kpi_list):
        """cluster_a < cluster_b for all rows."""
        result = run_posthoc_mannwhitney(clustered_df, kpi_list)
        assert (result["cluster_a"] < result["cluster_b"]).all()


class TestPrintKruskalSummary:
    """Tests for print_kruskal_summary() — AC 5."""

    def test_returns_string(self, clustered_df, kpi_list):
        """AC-5: Returns a summary string."""
        kw = run_kruskal_wallis(clustered_df, kpi_list)
        summary = print_kruskal_summary(kw, len(kpi_list))
        assert isinstance(summary, str)

    def test_summary_contains_count(self, clustered_df, kpi_list):
        """AC-5: Summary contains 'X/4' format."""
        kw = run_kruskal_wallis(clustered_df, kpi_list)
        summary = print_kruskal_summary(kw, len(kpi_list))
        n_sig = int(kw["significant"].sum())
        assert f"{n_sig}/{len(kpi_list)}" in summary

    def test_summary_format(self, clustered_df, kpi_list):
        """AC-5: Summary matches expected format."""
        kw = run_kruskal_wallis(clustered_df, kpi_list)
        summary = print_kruskal_summary(kw, len(kpi_list))
        assert "significantly differ" in summary
        assert "p < 0.05" in summary
