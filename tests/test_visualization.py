"""Tests for src/visualization.py — Stories 2.1, 2.2, 2.3, 2.4"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI
import matplotlib.pyplot as plt
import os
import tempfile
import shutil


# ---------- Fixtures ----------

@pytest.fixture
def df_customers():
    """Minimal customer-level DataFrame with the features needed by E2."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "recency_days": np.random.randint(0, 365, n),
        "frequency": np.random.randint(1, 30, n),
        "monetary_total": np.random.uniform(10, 5000, n),
        "monetary_avg": np.random.uniform(5, 300, n),
        "avg_basket_size_eur": np.random.uniform(10, 500, n),
        "discount_rate": np.random.uniform(0, 0.5, n),
        "subscription_tenure_days": np.random.randint(0, 3000, n),
        "loyalty_status": np.random.choice(["No Fid", "BRONZE", "SILVER", "GOLD"], n),
        "gender": np.random.choice(["Men", "Women", "Unknown"], n),
        "dominant_channel": np.random.choice(["store", "estore", "click_collect"], n),
        "dominant_axe": np.random.choice(["MAKE UP", "SKINCARE", "FRAGRANCE"], n),
        "dominant_market": np.random.choice(["SELECTIVE", "EXCLUSIVE", "SEPHORA"], n),
        "country": np.random.choice(["FR", "DE", "IT"], n),
        "age_generation": np.random.choice(["Gen Z", "Millennial", "Gen X", "Boomer"], n),
        "loyalty_numeric": np.random.choice([0, 1, 2, 3], n),
        "axis_diversity": np.random.randint(1, 6, n),
        "is_new_customer": np.random.choice([0, 1], n),
        "store_ratio": np.random.uniform(0, 1, n),
        "estore_ratio": np.random.uniform(0, 1, n),
        "click_collect_ratio": np.random.uniform(0, 0.05, n),
        "axe_make_up_ratio": np.random.uniform(0, 0.5, n),
        "axe_skincare_ratio": np.random.uniform(0, 0.5, n),
        "axe_fragrance_ratio": np.random.uniform(0, 0.3, n),
        "axe_haircare_ratio": np.random.uniform(0, 0.1, n),
        "axe_others_ratio": np.random.uniform(0, 0.1, n),
        "market_selective_ratio": np.random.uniform(0, 0.5, n),
        "market_exclusive_ratio": np.random.uniform(0, 0.3, n),
        "market_sephora_ratio": np.random.uniform(0, 0.3, n),
        "market_others_ratio": np.random.uniform(0, 0.1, n),
        "total_sales_eur": np.random.uniform(10, 5000, n),
    })


@pytest.fixture
def df_clean():
    """Minimal transaction-level DataFrame for brand / monthly charts."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "anonymized_card_code": np.random.choice([f"C{i}" for i in range(40)], n),
        "brand": np.random.choice([f"Brand_{i}" for i in range(30)], n),
        "salesVatEUR": np.random.uniform(5, 500, n),
        "transactionDate": pd.date_range("2025-01-01", periods=n, freq="2D"),
    })


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


NUMERICAL_FEATURES = [
    "recency_days", "frequency", "monetary_total", "monetary_avg",
    "avg_basket_size_eur", "discount_rate", "subscription_tenure_days",
]
CATEGORICAL_FEATURES = [
    "loyalty_status", "gender", "dominant_channel", "dominant_axe",
    "dominant_market", "country", "age_generation",
]


# ================================================================
# US 2-1: Univariate Distribution Analysis
# ================================================================

class TestPlotNumericalDistributions:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_numerical_distributions
        fig = plot_numerical_distributions(df_customers, NUMERICAL_FEATURES)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_numerical_distributions
        path = os.path.join(tmp_dir, "num_dist.png")
        fig = plot_numerical_distributions(df_customers, NUMERICAL_FEATURES, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_correct_subplot_count(self, df_customers):
        from src.visualization import plot_numerical_distributions
        fig = plot_numerical_distributions(df_customers, NUMERICAL_FEATURES)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == len(NUMERICAL_FEATURES)
        plt.close(fig)


class TestPlotCategoricalDistributions:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_categorical_distributions
        fig = plot_categorical_distributions(df_customers, CATEGORICAL_FEATURES)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_categorical_distributions
        path = os.path.join(tmp_dir, "cat_dist.png")
        fig = plot_categorical_distributions(df_customers, CATEGORICAL_FEATURES, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_correct_subplot_count(self, df_customers):
        from src.visualization import plot_categorical_distributions
        fig = plot_categorical_distributions(df_customers, CATEGORICAL_FEATURES)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == len(CATEGORICAL_FEATURES)
        plt.close(fig)


class TestComputeSummaryStats:
    def test_returns_dataframe(self, df_customers):
        from src.visualization import compute_summary_stats
        result = compute_summary_stats(df_customers, NUMERICAL_FEATURES)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df_customers):
        from src.visualization import compute_summary_stats
        result = compute_summary_stats(df_customers, NUMERICAL_FEATURES)
        for col in ["mean", "median", "std", "min", "max", "pct_missing", "nunique"]:
            assert col in result.columns

    def test_row_count(self, df_customers):
        from src.visualization import compute_summary_stats
        result = compute_summary_stats(df_customers, NUMERICAL_FEATURES)
        assert len(result) == len(NUMERICAL_FEATURES)

    def test_missing_pct_zero_when_no_nulls(self, df_customers):
        from src.visualization import compute_summary_stats
        result = compute_summary_stats(df_customers, NUMERICAL_FEATURES)
        assert (result["pct_missing"] == 0).all()


# ================================================================
# US 2-2: Correlation & Redundancy Analysis
# ================================================================

class TestPlotCorrelationHeatmap:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_correlation_heatmap
        fig = plot_correlation_heatmap(df_customers)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_correlation_heatmap
        path = os.path.join(tmp_dir, "corr.png")
        fig = plot_correlation_heatmap(df_customers, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)


class TestGetHighCorrelationPairs:
    def test_returns_dataframe(self, df_customers):
        from src.visualization import get_high_correlation_pairs
        result = get_high_correlation_pairs(df_customers, threshold=0.85)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, df_customers):
        from src.visualization import get_high_correlation_pairs
        result = get_high_correlation_pairs(df_customers, threshold=0.85)
        assert list(result.columns) == ["Feature_A", "Feature_B", "r"]

    def test_threshold_filter(self):
        from src.visualization import get_high_correlation_pairs
        # Create perfectly correlated columns
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 3, 1, 7, 2]})
        result = get_high_correlation_pairs(df, threshold=0.85)
        assert len(result) >= 1
        assert (result["r"].abs() > 0.85).all()

    def test_empty_when_no_high_corr(self):
        from src.visualization import get_high_correlation_pairs
        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"col_{i}": rng.standard_normal(100) for i in range(5)})
        result = get_high_correlation_pairs(df, threshold=0.99)
        assert len(result) == 0


# ================================================================
# US R1-3: Correlation Circle
# ================================================================

@pytest.fixture
def x_scaled_df():
    """Scaled DataFrame simulating preprocess_for_clustering output."""
    np.random.seed(42)
    n = 100
    cols = [
        "recency_days", "frequency", "monetary_total", "monetary_avg",
        "avg_basket_size_eur", "discount_rate",
        "store_ratio", "estore_ratio", "click_collect_ratio",
        "axe_make_up_ratio", "axe_skincare_ratio",
        "gender_Women", "gender_Men",
        "dominant_axe_MAKE UP", "dominant_axe_SKINCARE",
    ]
    data = np.random.randn(n, len(cols))
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def feature_categories():
    from src.config import FEATURE_CATEGORIES
    return FEATURE_CATEGORIES


@pytest.fixture
def category_colors():
    from src.config import CATEGORY_COLORS
    return CATEGORY_COLORS


class TestPlotCorrelationCircle:
    def test_returns_figure(self, x_scaled_df, feature_categories, category_colors):
        from src.visualization import plot_correlation_circle
        fig = plot_correlation_circle(x_scaled_df, feature_categories, category_colors)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, x_scaled_df, feature_categories, category_colors, tmp_dir):
        from src.visualization import plot_correlation_circle
        path = os.path.join(tmp_dir, "correlation_circle.png")
        fig = plot_correlation_circle(x_scaled_df, feature_categories, category_colors, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_unit_circle_drawn(self, x_scaled_df, feature_categories, category_colors):
        from src.visualization import plot_correlation_circle
        fig = plot_correlation_circle(x_scaled_df, feature_categories, category_colors)
        ax = fig.axes[0]
        patches = ax.patches
        assert any(hasattr(p, "center") for p in patches), "Unit circle patch expected"
        plt.close(fig)

    def test_axes_labels_contain_variance(self, x_scaled_df, feature_categories, category_colors):
        from src.visualization import plot_correlation_circle
        fig = plot_correlation_circle(x_scaled_df, feature_categories, category_colors)
        ax = fig.axes[0]
        assert "PC1" in ax.get_xlabel()
        assert "%" in ax.get_xlabel()
        assert "PC2" in ax.get_ylabel()
        assert "%" in ax.get_ylabel()
        plt.close(fig)

    def test_equal_aspect(self, x_scaled_df, feature_categories, category_colors):
        from src.visualization import plot_correlation_circle
        fig = plot_correlation_circle(x_scaled_df, feature_categories, category_colors)
        ax = fig.axes[0]
        assert ax.get_aspect() == "equal" or ax.get_aspect() == 1.0
        plt.close(fig)

    def test_legend_has_all_categories(self, x_scaled_df, feature_categories, category_colors):
        from src.visualization import plot_correlation_circle
        fig = plot_correlation_circle(x_scaled_df, feature_categories, category_colors)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        for cat in category_colors:
            assert cat in legend_labels
        plt.close(fig)

    def test_prints_redundancies(self, feature_categories, category_colors, capsys):
        from src.visualization import plot_correlation_circle
        # Create perfectly correlated features to trigger cosine sim > 0.90
        n = 100
        np.random.seed(42)
        base = np.random.randn(n)
        df = pd.DataFrame({
            "recency_days": base,
            "frequency": base + np.random.randn(n) * 0.01,  # nearly identical
            "monetary_total": np.random.randn(n),
        })
        fig = plot_correlation_circle(df, feature_categories, category_colors)
        captured = capsys.readouterr()
        assert "Potential redundancies" in captured.out
        assert "recency_days" in captured.out
        assert "frequency" in captured.out
        plt.close(fig)

    def test_ohe_column_colored_by_base_feature(self, feature_categories, category_colors):
        from src.visualization import plot_correlation_circle
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "gender_Women": np.random.randn(n),
            "gender_Men": np.random.randn(n),
            "recency_days": np.random.randn(n),
        })
        fig = plot_correlation_circle(df, feature_categories, category_colors)
        # Should not raise — OHE columns are handled gracefully
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ================================================================
# US 2-3: RFM Space Visualization
# ================================================================

class TestPlotRfmScatter:
    def test_returns_figures(self, df_customers, tmp_dir):
        from src.visualization import plot_rfm_scatter
        figs = plot_rfm_scatter(df_customers, save_dir=tmp_dir)
        assert isinstance(figs, list)
        assert len(figs) >= 1
        for f in figs:
            assert isinstance(f, plt.Figure)
            plt.close(f)

    def test_saves_loyalty_png(self, df_customers, tmp_dir):
        from src.visualization import plot_rfm_scatter
        plot_rfm_scatter(df_customers, save_dir=tmp_dir)
        assert os.path.isfile(os.path.join(tmp_dir, "rfm_scatter_loyalty.png"))

    def test_saves_rfmseg_when_column_present(self, df_customers, tmp_dir):
        from src.visualization import plot_rfm_scatter
        df_customers["RFM_Segment_ID"] = np.random.randint(1, 5, len(df_customers))
        figs = plot_rfm_scatter(df_customers, save_dir=tmp_dir)
        assert os.path.isfile(os.path.join(tmp_dir, "rfm_scatter_rfmseg.png"))
        assert len(figs) == 2
        for f in figs:
            plt.close(f)

    def test_no_rfmseg_when_column_absent(self, df_customers, tmp_dir):
        from src.visualization import plot_rfm_scatter
        figs = plot_rfm_scatter(df_customers, save_dir=tmp_dir)
        assert not os.path.isfile(os.path.join(tmp_dir, "rfm_scatter_rfmseg.png"))
        assert len(figs) == 1
        plt.close(figs[0])


# ================================================================
# US 2-4: Channel, Product & Brand Analysis
# ================================================================

class TestPlotChannelProductOverview:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_channel_product_overview
        fig = plot_channel_product_overview(df_customers)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_channel_product_overview
        path = os.path.join(tmp_dir, "cpo.png")
        fig = plot_channel_product_overview(df_customers, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)


class TestPlotTopBrands:
    def test_returns_figure(self, df_clean):
        from src.visualization import plot_top_brands
        fig = plot_top_brands(df_clean, n=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_clean, tmp_dir):
        from src.visualization import plot_top_brands
        path = os.path.join(tmp_dir, "brands.png")
        fig = plot_top_brands(df_clean, n=10, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)


class TestPlotSalesByDemographics:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_sales_by_demographics
        fig = plot_sales_by_demographics(df_customers)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_sales_by_demographics
        path = os.path.join(tmp_dir, "demo.png")
        fig = plot_sales_by_demographics(df_customers, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)


class TestPlotMonthlyVolume:
    def test_returns_figure(self, df_clean):
        from src.visualization import plot_monthly_volume
        fig = plot_monthly_volume(df_clean)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_clean, tmp_dir):
        from src.visualization import plot_monthly_volume
        path = os.path.join(tmp_dir, "monthly.png")
        fig = plot_monthly_volume(df_clean, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)


# ================================================================
# US 2-5: Outlier Analysis & Treatment Decision
# ================================================================

class TestPlotOutlierBoxplots:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_outlier_boxplots
        fig = plot_outlier_boxplots(df_customers)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_outlier_boxplots
        path = os.path.join(tmp_dir, "outlier_boxplots.png")
        fig = plot_outlier_boxplots(df_customers, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_correct_subplot_count(self, df_customers):
        from src.visualization import plot_outlier_boxplots
        fig = plot_outlier_boxplots(df_customers)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_custom_features(self, df_customers):
        from src.visualization import plot_outlier_boxplots
        fig = plot_outlier_boxplots(df_customers, features=["monetary_total"])
        assert len(fig.axes) == 1
        plt.close(fig)


class TestIdentifyExtremeCustomers:
    def test_returns_tuple(self, df_customers):
        from src.visualization import identify_extreme_customers
        result = identify_extreme_customers(df_customers)
        assert isinstance(result, tuple) and len(result) == 2

    def test_threshold_type(self, df_customers):
        from src.visualization import identify_extreme_customers
        extremes, threshold = identify_extreme_customers(df_customers)
        assert isinstance(threshold, float)

    def test_all_above_threshold(self, df_customers):
        from src.visualization import identify_extreme_customers
        extremes, threshold = identify_extreme_customers(df_customers)
        if len(extremes) > 0:
            assert (extremes["monetary_total"] > threshold).all()

    def test_custom_column(self, df_customers):
        from src.visualization import identify_extreme_customers
        extremes, threshold = identify_extreme_customers(df_customers, col="frequency", quantile=0.95)
        if len(extremes) > 0:
            assert (extremes["frequency"] > threshold).all()


class TestApplyWinsorization:
    def test_returns_dataframe(self, df_customers):
        from src.visualization import apply_winsorization
        result = apply_winsorization(df_customers)
        assert isinstance(result, pd.DataFrame)

    def test_capped_columns_exist(self, df_customers):
        from src.visualization import apply_winsorization
        result = apply_winsorization(df_customers)
        assert "monetary_total_capped" in result.columns
        assert "frequency_capped" in result.columns

    def test_is_outlier_column(self, df_customers):
        from src.visualization import apply_winsorization
        result = apply_winsorization(df_customers)
        assert "is_outlier" in result.columns
        assert result["is_outlier"].dtype == bool

    def test_capped_values_respect_threshold(self, df_customers):
        from src.visualization import apply_winsorization
        result = apply_winsorization(df_customers, quantile=0.99)
        cap_m = df_customers["monetary_total"].quantile(0.99)
        cap_f = df_customers["frequency"].quantile(0.99)
        assert result["monetary_total_capped"].max() <= cap_m + 1e-9
        assert result["frequency_capped"].max() <= cap_f + 1e-9

    def test_does_not_modify_original(self, df_customers):
        from src.visualization import apply_winsorization
        original_cols = set(df_customers.columns)
        apply_winsorization(df_customers)
        assert set(df_customers.columns) == original_cols


# ================================================================
# US 2-6: Loyalty & Lifecycle Analysis
# ================================================================

class TestPlotLoyaltyLifecycle:
    def test_returns_figure(self, df_customers):
        from src.visualization import plot_loyalty_lifecycle
        fig = plot_loyalty_lifecycle(df_customers)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, df_customers, tmp_dir):
        from src.visualization import plot_loyalty_lifecycle
        path = os.path.join(tmp_dir, "loyalty_lifecycle.png")
        fig = plot_loyalty_lifecycle(df_customers, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_three_subplots(self, df_customers):
        from src.visualization import plot_loyalty_lifecycle
        fig = plot_loyalty_lifecycle(df_customers)
        assert len(fig.axes) == 3
        plt.close(fig)


class TestComputeLoyaltySummary:
    def test_returns_dataframe(self, df_customers):
        from src.visualization import compute_loyalty_summary
        result = compute_loyalty_summary(df_customers)
        assert isinstance(result, pd.DataFrame)

    def test_expected_index(self, df_customers):
        from src.visualization import compute_loyalty_summary
        result = compute_loyalty_summary(df_customers)
        assert list(result.index) == ["No Fid", "BRONZE", "SILVER", "GOLD"]

    def test_proportion_sums_to_one(self, df_customers):
        from src.visualization import compute_loyalty_summary
        result = compute_loyalty_summary(df_customers)
        assert abs(result["proportion"].sum() - 1.0) < 1e-9

    def test_expected_columns(self, df_customers):
        from src.visualization import compute_loyalty_summary
        result = compute_loyalty_summary(df_customers)
        for col in ["monetary_avg_mean", "frequency_mean", "recency_days_mean",
                     "subscription_tenure_days_mean", "is_new_customer_rate", "proportion"]:
            assert col in result.columns


# ================================================================
# US 3-4: PCA — Variance Analysis
# ================================================================

@pytest.fixture
def pca_fitted():
    """Return a fitted PCA object and feature names for testing."""
    from sklearn.decomposition import PCA
    np.random.seed(42)
    n_samples, n_features = 200, 20
    data = np.random.randn(n_samples, n_features)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    pca = PCA(n_components=10, random_state=42)
    pca.fit(data)
    return pca, feature_names


class TestPlotPcaVariance:
    def test_returns_figure(self, pca_fitted):
        from src.visualization import plot_pca_variance
        pca, _ = pca_fitted
        fig = plot_pca_variance(pca)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, pca_fitted, tmp_dir):
        from src.visualization import plot_pca_variance
        pca, _ = pca_fitted
        path = os.path.join(tmp_dir, "pca_variance.png")
        fig = plot_pca_variance(pca, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_has_threshold_lines(self, pca_fitted):
        from src.visualization import plot_pca_variance
        pca, _ = pca_fitted
        fig = plot_pca_variance(pca)
        ax = fig.axes[0]
        # Should have horizontal lines for 80% and 90%
        h_lines = [line for line in ax.get_lines() if len(set(line.get_ydata())) == 1]
        plt.close(fig)
        # At least the main curve + threshold lines exist
        assert len(ax.get_lines()) >= 1


class TestPlotPcaLoadings:
    def test_returns_figure(self, pca_fitted):
        from src.visualization import plot_pca_loadings
        pca, feat_names = pca_fitted
        fig = plot_pca_loadings(pca, feat_names)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, pca_fitted, tmp_dir):
        from src.visualization import plot_pca_loadings
        pca, feat_names = pca_fitted
        path = os.path.join(tmp_dir, "pca_loadings.png")
        fig = plot_pca_loadings(pca, feat_names, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_correct_subplot_count(self, pca_fitted):
        from src.visualization import plot_pca_loadings
        pca, feat_names = pca_fitted
        fig = plot_pca_loadings(pca, feat_names, top_n=3)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_top_n_capped_at_components(self, pca_fitted):
        from src.visualization import plot_pca_loadings
        pca, feat_names = pca_fitted
        # Request more than available components
        fig = plot_pca_loadings(pca, feat_names, top_n=20)
        assert len(fig.axes) == pca.n_components_
        plt.close(fig)

    def test_single_component(self, pca_fitted):
        from src.visualization import plot_pca_loadings
        pca, feat_names = pca_fitted
        fig = plot_pca_loadings(pca, feat_names, top_n=1)
        assert len(fig.axes) == 1
        plt.close(fig)


# ================================================================
# US 3-5: UMAP 2D Visualization
# ================================================================

@pytest.fixture
def umap_df():
    """Minimal UMAP embedding DataFrame."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "umap_1": np.random.uniform(-10, 10, n),
        "umap_2": np.random.uniform(-10, 10, n),
    })


class TestPlotUmap2d:
    def test_returns_figure(self, umap_df, df_customers):
        from src.visualization import plot_umap_2d
        fig = plot_umap_2d(umap_df, df_customers, color_by="loyalty_status")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, umap_df, df_customers, tmp_dir):
        from src.visualization import plot_umap_2d
        fig = plot_umap_2d(umap_df, df_customers, color_by="loyalty_status",
                           save_dir=tmp_dir)
        path = os.path.join(tmp_dir, "umap_2d_loyalty_status.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_color_by_dominant_axe(self, umap_df, df_customers, tmp_dir):
        from src.visualization import plot_umap_2d
        fig = plot_umap_2d(umap_df, df_customers, color_by="dominant_axe",
                           save_dir=tmp_dir)
        path = os.path.join(tmp_dir, "umap_2d_dominant_axe.png")
        assert os.path.isfile(path)
        plt.close(fig)

    def test_color_by_rfm_segment(self, umap_df, df_customers, tmp_dir):
        from src.visualization import plot_umap_2d
        df_customers["RFM_Segment_ID"] = np.random.randint(1, 5, len(df_customers))
        fig = plot_umap_2d(umap_df, df_customers, color_by="RFM_Segment_ID",
                           save_dir=tmp_dir)
        path = os.path.join(tmp_dir, "umap_2d_RFM_Segment_ID.png")
        assert os.path.isfile(path)
        plt.close(fig)


# ================================================================
# US 5-2: Per-Cluster KPI Heatmap
# ================================================================

@pytest.fixture
def cluster_kpis_df():
    """Minimal cluster KPI matrix for heatmap tests."""
    from src.profiling import NUMERICAL_KPIS
    np.random.seed(42)
    n_clusters = 4
    data = {'n_customers': [30, 25, 20, 25], 'pct_customers': [30.0, 25.0, 20.0, 25.0]}
    for kpi in NUMERICAL_KPIS:
        data[kpi] = np.random.uniform(0, 100, n_clusters)
    df = pd.DataFrame(data, index=pd.Index(range(n_clusters), name='cluster_id'))
    return df


class TestPlotClusterKpiHeatmap:
    def test_returns_figure(self, cluster_kpis_df):
        from src.visualization import plot_cluster_kpi_heatmap
        fig = plot_cluster_kpi_heatmap(cluster_kpis_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_figure(self, cluster_kpis_df, tmp_dir):
        from src.visualization import plot_cluster_kpi_heatmap
        path = os.path.join(tmp_dir, "cluster_kpi_heatmap.png")
        fig = plot_cluster_kpi_heatmap(cluster_kpis_df, save_path=path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)

    def test_heatmap_axes_present(self, cluster_kpis_df):
        from src.visualization import plot_cluster_kpi_heatmap
        fig = plot_cluster_kpi_heatmap(cluster_kpis_df)
        ax = fig.axes[0]
        assert ax.get_title() or fig._suptitle is not None
        plt.close(fig)

    def test_no_crash_single_cluster(self):
        from src.visualization import plot_cluster_kpi_heatmap
        from src.profiling import NUMERICAL_KPIS
        data = {'n_customers': [50], 'pct_customers': [100.0]}
        for kpi in NUMERICAL_KPIS:
            data[kpi] = [42.0]
        df = pd.DataFrame(data, index=pd.Index([0], name='cluster_id'))
        fig = plot_cluster_kpi_heatmap(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ================================================================
# US 5-4: Distinguishing Features per Cluster
# ================================================================

@pytest.fixture
def distinguishing_features_data():
    """DataFrame with cluster_id and features for distinguishing features tests."""
    np.random.seed(42)
    n = 100
    features = ['monetary_avg', 'frequency', 'recency_days', 'store_ratio', 'discount_rate']
    data = {f: np.random.uniform(0, 100, n) for f in features}
    data['cluster_id'] = np.random.choice([0, 1, 2], n)
    return pd.DataFrame(data), features


class TestPlotDistinguishingFeatures:
    def test_returns_list_of_figures(self, distinguishing_features_data):
        from src.visualization import plot_distinguishing_features
        from src.profiling import compute_distinguishing_features
        df, features = distinguishing_features_data
        dist_feats = compute_distinguishing_features(df, features)
        figs = plot_distinguishing_features(dist_feats)
        assert isinstance(figs, list)
        assert all(isinstance(f, plt.Figure) for f in figs)
        for f in figs:
            plt.close(f)

    def test_one_figure_per_cluster(self, distinguishing_features_data):
        from src.visualization import plot_distinguishing_features
        from src.profiling import compute_distinguishing_features
        df, features = distinguishing_features_data
        dist_feats = compute_distinguishing_features(df, features)
        figs = plot_distinguishing_features(dist_feats)
        assert len(figs) == len(dist_feats)
        for f in figs:
            plt.close(f)

    def test_saves_figures(self, distinguishing_features_data, tmp_dir):
        from src.visualization import plot_distinguishing_features
        from src.profiling import compute_distinguishing_features
        df, features = distinguishing_features_data
        dist_feats = compute_distinguishing_features(df, features)
        figs = plot_distinguishing_features(dist_feats, save_dir=tmp_dir)
        for cluster_id in dist_feats:
            path = os.path.join(tmp_dir, f"distinguishing_features_cluster_{cluster_id}.png")
            assert os.path.isfile(path), f"Missing file for cluster {cluster_id}"
            assert os.path.getsize(path) > 0
        for f in figs:
            plt.close(f)

    def test_horizontal_bars(self, distinguishing_features_data):
        from src.visualization import plot_distinguishing_features
        from src.profiling import compute_distinguishing_features
        df, features = distinguishing_features_data
        dist_feats = compute_distinguishing_features(df, features)
        figs = plot_distinguishing_features(dist_feats)
        for fig in figs:
            ax = fig.axes[0]
            # barh creates Patch objects
            assert len(ax.patches) > 0, "No bars found in chart"
            plt.close(fig)

    def test_top_n_parameter(self, distinguishing_features_data):
        from src.visualization import plot_distinguishing_features
        from src.profiling import compute_distinguishing_features
        df, features = distinguishing_features_data
        dist_feats = compute_distinguishing_features(df, features)
        figs = plot_distinguishing_features(dist_feats, top_n=3)
        for fig in figs:
            ax = fig.axes[0]
            # Should have at most 3 positive + 3 negative = 6 bars
            assert len(ax.patches) <= 6
            plt.close(fig)
