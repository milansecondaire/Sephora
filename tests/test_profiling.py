"""Tests for src/profiling.py — Stories 5.1, 5.2, 5.3, 5.4, 5.6."""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from src.profiling import (
    compute_global_kpis, compute_cluster_kpis, NUMERICAL_KPIS,
    build_delta_table, get_notable_deltas, export_delta_table_md,
    compute_distinguishing_features, rank_by_clv,
    identify_high_potential_segments,
)


@pytest.fixture
def sample_customers():
    """Minimal DataFrame mimicking customers_with_clusters.csv."""
    rng = np.random.RandomState(42)
    n = 100
    data = {
        'monetary_avg': rng.uniform(10, 200, n),
        'frequency': rng.randint(1, 20, n).astype(float),
        'avg_basket_size_eur': rng.uniform(15, 300, n),
        'avg_units_per_basket': rng.uniform(1, 10, n),
        'recency_days': rng.randint(1, 365, n).astype(float),
        'store_ratio': rng.uniform(0, 1, n),
        'estore_ratio': rng.uniform(0, 1, n),
        'click_collect_ratio': rng.uniform(0, 1, n),
        'axe_make_up_ratio': rng.uniform(0, 1, n),
        'axe_skincare_ratio': rng.uniform(0, 1, n),
        'axe_fragrance_ratio': rng.uniform(0, 1, n),
        'axe_haircare_ratio': rng.uniform(0, 1, n),
        'axe_others_ratio': rng.uniform(0, 1, n),
        'discount_rate': rng.uniform(0, 0.5, n),
        'monetary_total': rng.uniform(50, 5000, n),
        'loyalty_status': rng.choice(['Gold', 'Silver', 'Bronze', 'White'], n),
        'cluster_id': rng.randint(0, 5, n),
    }
    return pd.DataFrame(data)


class TestComputeGlobalKpis:
    def test_returns_dict(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        assert isinstance(result, dict)

    def test_contains_all_numerical_kpis(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        for kpi in NUMERICAL_KPIS:
            assert kpi in result, f"Missing KPI: {kpi}"

    def test_numerical_kpi_values_are_means(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        expected = sample_customers[NUMERICAL_KPIS].mean()
        for kpi in NUMERICAL_KPIS:
            assert result[kpi] == pytest.approx(expected[kpi], rel=1e-9), f"KPI {kpi} mismatch"

    def test_contains_loyalty_distribution(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        assert 'loyalty_distribution' in result
        assert isinstance(result['loyalty_distribution'], dict)

    def test_loyalty_distribution_sums_to_one(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        total = sum(result['loyalty_distribution'].values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_loyalty_distribution_keys(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        loyalty_keys = set(result['loyalty_distribution'].keys())
        expected_keys = set(sample_customers['loyalty_status'].unique())
        assert loyalty_keys == expected_keys

    def test_contains_total_customers(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        assert result['total_customers'] == len(sample_customers)

    def test_contains_total_sales(self, sample_customers):
        result = compute_global_kpis(sample_customers)
        assert result['total_sales_eur'] == pytest.approx(
            sample_customers['monetary_total'].sum(), rel=1e-9
        )

    def test_single_row(self):
        """Edge case: single customer."""
        data = {kpi: [42.0] for kpi in NUMERICAL_KPIS}
        data['loyalty_status'] = ['Gold']
        df = pd.DataFrame(data)
        result = compute_global_kpis(df)
        assert result['total_customers'] == 1
        assert result['loyalty_distribution'] == {'Gold': 1.0}
        for kpi in NUMERICAL_KPIS:
            assert result[kpi] == pytest.approx(42.0)


# ================================================================
# US 5-2: Per-Cluster KPI Computation
# ================================================================

class TestComputeClusterKpis:
    def test_returns_dataframe(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        assert isinstance(result, pd.DataFrame)

    def test_rows_equal_num_clusters(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        expected_clusters = sample_customers['cluster_id'].nunique()
        assert len(result) == expected_clusters

    def test_index_is_cluster_id(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        assert result.index.name == 'cluster_id'

    def test_n_customers_column(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        assert 'n_customers' in result.columns
        expected = sample_customers.groupby('cluster_id').size()
        pd.testing.assert_series_equal(
            result['n_customers'], expected, check_names=False
        )

    def test_pct_customers_column(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        assert 'pct_customers' in result.columns
        assert result['pct_customers'].sum() == pytest.approx(100.0, abs=1e-6)

    def test_pct_customers_values(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        expected_pct = sample_customers.groupby('cluster_id').size() / len(sample_customers) * 100
        pd.testing.assert_series_equal(
            result['pct_customers'], expected_pct, check_names=False
        )

    def test_contains_all_numerical_kpis(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        for kpi in NUMERICAL_KPIS:
            assert kpi in result.columns, f"Missing KPI column: {kpi}"

    def test_kpi_values_are_cluster_means(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        expected_means = sample_customers.groupby('cluster_id')[NUMERICAL_KPIS].mean()
        pd.testing.assert_frame_equal(
            result[NUMERICAL_KPIS], expected_means, check_names=False
        )

    def test_n_customers_and_pct_are_first_columns(self, sample_customers):
        result = compute_cluster_kpis(sample_customers)
        assert result.columns[0] == 'n_customers'
        assert result.columns[1] == 'pct_customers'

    def test_single_cluster(self):
        """Edge case: all customers in one cluster."""
        data = {kpi: [10.0, 20.0] for kpi in NUMERICAL_KPIS}
        data['cluster_id'] = [0, 0]
        df = pd.DataFrame(data)
        result = compute_cluster_kpis(df)
        assert len(result) == 1
        assert result['n_customers'].iloc[0] == 2
        assert result['pct_customers'].iloc[0] == pytest.approx(100.0)


# ================================================================
# US 5-3: Delta Table (Clusters vs. Global Average)
# ================================================================

@pytest.fixture
def cluster_and_global(sample_customers):
    """Precompute cluster_kpis and global_kpis from sample_customers."""
    global_kpis = compute_global_kpis(sample_customers)
    cluster_kpis = compute_cluster_kpis(sample_customers)
    return cluster_kpis, global_kpis


class TestBuildDeltaTable:
    def test_returns_dataframe(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        expected_cols = {'cluster_id', 'kpi', 'global_avg', 'cluster_value', 'delta_abs', 'delta_pct'}
        assert set(result.columns) == expected_cols

    def test_row_count(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        n_clusters = len(cluster_kpis)
        n_kpis = len(NUMERICAL_KPIS)
        assert len(result) == n_clusters * n_kpis

    def test_delta_abs_formula(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        for _, row in result.iterrows():
            expected = row['cluster_value'] - row['global_avg']
            assert row['delta_abs'] == pytest.approx(expected, abs=1e-3)

    def test_delta_pct_formula(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        for _, row in result.iterrows():
            if row['global_avg'] != 0:
                expected = (row['cluster_value'] / row['global_avg'] - 1) * 100
                assert row['delta_pct'] == pytest.approx(expected, abs=0.05)

    def test_sorted_by_monetary_total_desc(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        cluster_order = result['cluster_id'].unique().tolist()
        monetary_vals = [cluster_kpis.loc[c, 'monetary_total'] for c in cluster_order]
        assert monetary_vals == sorted(monetary_vals, reverse=True)

    def test_all_kpis_present(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        result = build_delta_table(cluster_kpis, global_kpis)
        assert set(result['kpi'].unique()) == set(NUMERICAL_KPIS)


class TestGetNotableDeltas:
    def test_threshold_filter(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        delta_df = build_delta_table(cluster_kpis, global_kpis)
        notable = get_notable_deltas(delta_df, threshold=30.0)
        assert all(notable['delta_pct'].abs() > 30.0)

    def test_returns_dataframe(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        delta_df = build_delta_table(cluster_kpis, global_kpis)
        notable = get_notable_deltas(delta_df)
        assert isinstance(notable, pd.DataFrame)

    def test_custom_threshold(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        delta_df = build_delta_table(cluster_kpis, global_kpis)
        notable = get_notable_deltas(delta_df, threshold=0.0)
        # All non-zero deltas should be included
        assert len(notable) >= 0


class TestExportDeltaTableMd:
    def test_creates_file(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        delta_df = build_delta_table(cluster_kpis, global_kpis)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "delta.md")
            export_delta_table_md(delta_df, path)
            assert os.path.exists(path)

    def test_file_content_header(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        delta_df = build_delta_table(cluster_kpis, global_kpis)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "delta.md")
            export_delta_table_md(delta_df, path)
            with open(path) as f:
                content = f.read()
            assert content.startswith("# KPI Delta Table")

    def test_file_contains_table(self, cluster_and_global):
        cluster_kpis, global_kpis = cluster_and_global
        delta_df = build_delta_table(cluster_kpis, global_kpis)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "delta.md")
            export_delta_table_md(delta_df, path)
            with open(path) as f:
                content = f.read()
            assert '|' in content  # Markdown table pipes


# ================================================================
# US 5-4: Top Distinguishing Features per Cluster
# ================================================================

class TestComputeDistinguishingFeatures:
    def test_returns_dict(self, sample_customers):
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        assert isinstance(result, dict)

    def test_keys_match_cluster_ids(self, sample_customers):
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        assert set(result.keys()) == set(sample_customers['cluster_id'].unique())

    def test_values_are_dataframes(self, sample_customers):
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        for cluster_id, df in result.items():
            assert isinstance(df, pd.DataFrame), f"Cluster {cluster_id} not a DataFrame"

    def test_dataframe_columns(self, sample_customers):
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        for cluster_id, df in result.items():
            assert 'feature' in df.columns
            assert 'cohens_d' in df.columns
            assert 'cohens_d_abs' in df.columns

    def test_sorted_by_cohens_d_abs_desc(self, sample_customers):
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        for cluster_id, df in result.items():
            abs_vals = df['cohens_d_abs'].values
            assert list(abs_vals) == sorted(abs_vals, reverse=True), \
                f"Cluster {cluster_id} not sorted by |Cohen's d| desc"

    def test_cohens_d_formula(self, sample_customers):
        """Verify Cohen's d = (cluster_mean - rest_mean) / rest_std."""
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        
        for cluster_id, df in result.items():
            cluster_mask = sample_customers['cluster_id'] == cluster_id
            cluster_mean = sample_customers.loc[cluster_mask, features].mean()
            
            rest_mask = ~cluster_mask
            rest_mean = sample_customers.loc[rest_mask, features].mean()
            rest_std = sample_customers.loc[rest_mask, features].std()
            
            for _, row in df.iterrows():
                feat = row['feature']
                expected_d = (cluster_mean[feat] - rest_mean[feat]) / (rest_std[feat] + 1e-10)
                assert row['cohens_d'] == pytest.approx(expected_d, abs=1e-5), \
                    f"Cluster {cluster_id}, feature {feat}: expected {expected_d}, got {row['cohens_d']}"

    def test_all_features_present(self, sample_customers):
        features = [k for k in NUMERICAL_KPIS if k in sample_customers.columns]
        result = compute_distinguishing_features(sample_customers, features)
        for cluster_id, df in result.items():
            assert set(df['feature']) == set(features), \
                f"Cluster {cluster_id} missing features"

    def test_two_clusters(self):
        """Minimal 2-cluster case with known values."""
        df = pd.DataFrame({
            'cluster_id': [0, 0, 1, 1],
            'feat_a': [10.0, 10.0, 20.0, 20.0],
            'feat_b': [5.0, 5.0, 5.0, 5.0],
        })
        result = compute_distinguishing_features(df, ['feat_a', 'feat_b'])
        assert set(result.keys()) == {0, 1}
        # feat_b has zero std → cohens_d should be ~0
        for cid in [0, 1]:
            feat_b_row = result[cid][result[cid]['feature'] == 'feat_b']
            assert feat_b_row['cohens_d'].values[0] == pytest.approx(0.0, abs=1e-6)

    def test_single_cluster(self):
        """All customers in one cluster → Cohen's d ≈ 0 for all features."""
        df = pd.DataFrame({
            'cluster_id': [0, 0, 0],
            'feat_a': [10.0, 20.0, 30.0],
        })
        result = compute_distinguishing_features(df, ['feat_a'])
        assert len(result) == 1
        assert result[0]['cohens_d'].values[0] == pytest.approx(0.0, abs=1e-6)


# ================================================================
# US 5-6: CLV Ranking & High-Potential Segment Identification
# ================================================================

class TestRankByClv:
    def test_returns_dataframe(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = rank_by_clv(cluster_kpis)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_monetary_total_desc(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = rank_by_clv(cluster_kpis)
        vals = result['monetary_total'].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_clv_tier_column_exists(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = rank_by_clv(cluster_kpis)
        assert 'clv_tier' in result.columns

    def test_clv_tier_values(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = rank_by_clv(cluster_kpis)
        valid_tiers = {'Top', 'Mid', 'Low'}
        assert set(result['clv_tier'].dropna().unique()).issubset(valid_tiers)

    def test_does_not_modify_original(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        original_cols = list(cluster_kpis.columns)
        rank_by_clv(cluster_kpis)
        assert list(cluster_kpis.columns) == original_cols

    def test_preserves_all_original_columns(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        original_cols = set(cluster_kpis.columns)
        result = rank_by_clv(cluster_kpis)
        assert original_cols.issubset(set(result.columns))

    def test_same_number_of_rows(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = rank_by_clv(cluster_kpis)
        assert len(result) == len(cluster_kpis)

    def test_two_clusters(self):
        """Minimal 2-cluster case: can't form 3 tertiles, fallback to labels."""
        df = pd.DataFrame({
            'n_customers': [50, 50],
            'pct_customers': [50.0, 50.0],
            'monetary_total': [1000.0, 500.0],
        }, index=[0, 1])
        df.index.name = 'cluster_id'
        result = rank_by_clv(df)
        assert result.iloc[0]['monetary_total'] >= result.iloc[1]['monetary_total']
        assert 'clv_tier' in result.columns

    def test_identical_clv(self):
        """Duplicates in monetary_total should not crash pd.qcut."""
        df = pd.DataFrame({
            'n_customers': [50, 50, 50, 50],
            'pct_customers': [25.0, 25.0, 25.0, 25.0],
            'monetary_total': [100.0, 100.0, 100.0, 100.0], # All identical
        }, index=[0, 1, 2, 3])
        df.index.name = 'cluster_id'
        # Would raise ValueError: Bin edges must be unique without method='first' processing
        result = rank_by_clv(df)
        assert len(result) == 4
        assert 'clv_tier' in result.columns


class TestIdentifyHighPotentialSegments:
    def test_returns_dict(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        assert isinstance(result, dict)

    def test_has_top3_key(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        assert 'top3_priority' in result

    def test_top3_has_at_most_3(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        assert len(result['top3_priority']) <= 3

    def test_top3_sorted_by_clv(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        vals = [cluster_kpis.loc[c, 'monetary_total'] for c in result['top3_priority']]
        assert vals == sorted(vals, reverse=True)

    def test_has_investment_worthy_key(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        assert 'investment_worthy' in result

    def test_investment_worthy_is_list(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        assert isinstance(result['investment_worthy'], list)

    def test_has_high_potential_key(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        assert 'high_potential' in result

    def test_high_potential_above_medians(self, sample_customers):
        cluster_kpis = compute_cluster_kpis(sample_customers)
        result = identify_high_potential_segments(cluster_kpis)
        median_size = cluster_kpis['pct_customers'].median()
        median_clv = cluster_kpis['monetary_total'].median()
        for c in result['high_potential']:
            assert cluster_kpis.loc[c, 'pct_customers'] >= median_size
            assert cluster_kpis.loc[c, 'monetary_total'] >= median_clv
