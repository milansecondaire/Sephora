"""Tests for src/preprocessing.py — Story 3.2, 3.3, 3.4, 3.5"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.preprocessing import impute_features, scale_features, preprocess_for_clustering, apply_pca, apply_umap


@pytest.fixture
def feature_df_no_missing():
    """Feature matrix with zero missing values."""
    return pd.DataFrame({
        "recency_days": [10.0, 20.0, 30.0, 40.0],
        "frequency": [5, 10, 15, 20],
        "monetary_total": [100.0, 200.0, 300.0, 400.0],
        "subscription_tenure_days": [365.0, 730.0, 100.0, 50.0],
        "is_new_customer": [0, 0, 1, 1],
    })


@pytest.fixture
def feature_df_with_missing():
    """Feature matrix with some missing values (<30%)."""
    return pd.DataFrame({
        "recency_days": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "frequency": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "monetary_total": [100.0, 200.0, np.nan, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "subscription_tenure_days": [365.0, np.nan, 100.0, 50.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
        "is_new_customer": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def feature_df_high_missing():
    """Feature matrix with one feature having >30% missing."""
    n = 10
    # recency_days has 40% missing (4 out of 10)
    recency = [np.nan, np.nan, np.nan, np.nan, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    return pd.DataFrame({
        "recency_days": recency,
        "frequency": list(range(1, n + 1)),
        "monetary_total": [100.0] * n,
        "subscription_tenure_days": [365.0] * n,
        "is_new_customer": [0] * n,
    })


@pytest.fixture
def feature_df_with_categorical():
    """Feature matrix containing a categorical column with <30% missing."""
    df = pd.DataFrame({
        "recency_days": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "frequency": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "category_col": ["A", np.nan, "B", np.nan, "C", "A", "B", "C", "A", "B"],
    })
    df["category_col"] = df["category_col"].astype("category")
    return df

@pytest.fixture
def feature_df_high_missing_categorical():
    """Feature matrix containing a categorical column with >30% missing to test AC-2 vs AC-3 conflict."""
    df = pd.DataFrame({
        "recency_days": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "category_col": [np.nan, np.nan, np.nan, np.nan, "C", "A", "B", "C", "A", "B"],
    })
    df["category_col"] = df["category_col"].astype("category")
    return df


class TestImputeFeaturesReturnsDataFrame:
    """impute_features() must return a DataFrame."""

    def test_returns_dataframe(self, feature_df_no_missing):
        result = impute_features(feature_df_no_missing)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_index(self, feature_df_with_missing):
        original_index = feature_df_with_missing.index.tolist()
        result = impute_features(feature_df_with_missing)
        assert result.index.tolist() == original_index


class TestZeroNaNGuarantee:
    """AC-5: Zero NaN values confirmed in final feature matrix."""

    def test_no_nan_after_imputation_clean(self, feature_df_no_missing):
        result = impute_features(feature_df_no_missing)
        assert result.isnull().sum().sum() == 0

    def test_no_nan_after_imputation_with_missing(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_no_nan_after_high_missing_drop(self, feature_df_high_missing):
        result = impute_features(feature_df_high_missing)
        assert result.isnull().sum().sum() == 0


class TestMissingRatePrint:
    """AC-1: Missing rate per selected feature printed."""

    def test_missing_rate_printed(self, feature_df_with_missing, capsys):
        impute_features(feature_df_with_missing)
        captured = capsys.readouterr()
        assert "missing" in captured.out.lower() or "Missing" in captured.out


class TestMedianImputation:
    """AC-2: Numerical missing values with <30% missing get median imputation."""

    def test_median_imputation_applied(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        # recency_days had 1 NaN out of 10 => median imputed
        # The NaN was at index 1 — should now equal the median of non-NaN values
        non_nan_values = [10.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        expected_median = np.median(non_nan_values)
        assert result.loc[1, "recency_days"] == expected_median


class TestHighMissingExclusion:
    """AC-2: Features with >30% missing are flagged and excluded."""

    def test_high_missing_feature_dropped(self, feature_df_high_missing, capsys):
        result = impute_features(feature_df_high_missing)
        # recency_days has 40% missing => should be dropped
        assert "recency_days" not in result.columns

    def test_high_missing_warning_printed(self, feature_df_high_missing, capsys):
        impute_features(feature_df_high_missing)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "warning" in captured.out.lower()
        assert "recency_days" in captured.out

    def test_low_missing_feature_kept(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        # recency_days has 10% missing => should be kept
        assert "recency_days" in result.columns


class TestCategoricalImputation:
    """AC-3: Categorical missing values filled with 'Unknown'."""

    def test_categorical_filled_with_unknown(self, feature_df_with_categorical):
        result = impute_features(feature_df_with_categorical)
        assert (result["category_col"] == "Unknown").sum() == 2
        assert result["category_col"].isnull().sum() == 0
        assert "Unknown" in result["category_col"].cat.categories

    def test_high_missing_categorical_not_dropped(self, feature_df_high_missing_categorical):
        # 40% missing in 'category_col', which is categorical, so it shouldn't be dropped.
        result = impute_features(feature_df_high_missing_categorical)
        assert "category_col" in result.columns
        assert (result["category_col"] == "Unknown").sum() == 4
        assert result["category_col"].isnull().sum() == 0


class TestSubscriptionTenureDays:
    """AC-4: subscription_tenure_days NaN imputed (median preferred)."""

    def test_subscription_tenure_imputed(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        # subscription_tenure_days had 1 NaN => should be imputed
        assert not np.isnan(result.loc[1, "subscription_tenure_days"])

    def test_subscription_tenure_median_value(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        non_nan_values = [365.0, 100.0, 50.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
        expected_median = np.median(non_nan_values)
        assert result.loc[1, "subscription_tenure_days"] == expected_median


class TestCustomThreshold:
    """missing_threshold parameter is respected."""

    def test_threshold_20_percent(self, feature_df_with_missing):
        # recency_days has 10% missing => kept at 20% threshold
        result = impute_features(feature_df_with_missing, missing_threshold=0.20)
        assert "recency_days" in result.columns

    def test_threshold_05_percent(self, feature_df_with_missing):
        # recency_days has 10% missing => dropped at 5% threshold
        result = impute_features(feature_df_with_missing, missing_threshold=0.05)
        assert "recency_days" not in result.columns


class TestShapePreservation:
    """Row count must remain the same; only high-missing columns dropped."""

    def test_row_count_preserved(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        assert len(result) == len(feature_df_with_missing)

    def test_no_columns_dropped_when_all_below_threshold(self, feature_df_with_missing):
        result = impute_features(feature_df_with_missing)
        # All features have <30% missing => all columns kept
        assert set(result.columns) == set(feature_df_with_missing.columns)

    def test_high_missing_column_dropped(self, feature_df_high_missing):
        result = impute_features(feature_df_high_missing)
        assert result.shape[1] == feature_df_high_missing.shape[1] - 1


# ========== Story 3.3: Feature Scaling ==========

@pytest.fixture
def numeric_df():
    """Simple numeric DataFrame for scaling tests."""
    return pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]},
        index=pd.Index([100, 101, 102, 103, 104], name="anonymized_card_code"),
    )


class TestScaleFeaturesReturnsDataFrame:
    """scale_features() must return a DataFrame preserving structure."""

    def test_returns_dataframe(self, numeric_df):
        result = scale_features(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_columns(self, numeric_df):
        result = scale_features(numeric_df)
        assert list(result.columns) == list(numeric_df.columns)

    def test_preserves_index(self, numeric_df):
        result = scale_features(numeric_df)
        assert list(result.index) == list(numeric_df.index)

    def test_preserves_shape(self, numeric_df):
        result = scale_features(numeric_df)
        assert result.shape == numeric_df.shape


class TestStandardScalerProperties:
    """AC-1: StandardScaler applied — zero mean, unit variance."""

    def test_zero_mean(self, numeric_df):
        result = scale_features(numeric_df)
        means = result.mean()
        for col in result.columns:
            assert means[col] == pytest.approx(0.0, abs=1e-9), f"{col} mean not ~0: {means[col]}"

    def test_unit_variance(self, numeric_df):
        result = scale_features(numeric_df)
        stds = result.std(ddof=0)
        for col in result.columns:
            assert stds[col] == pytest.approx(1.0, abs=1e-9), f"{col} std not ~1: {stds[col]}"


class TestScaleFeaturesShapePrint:
    """AC-4: Shape printed."""

    def test_shape_printed(self, numeric_df, capsys):
        scale_features(numeric_df)
        captured = capsys.readouterr()
        assert "X_scaled shape" in captured.out
        assert str(numeric_df.shape) in captured.out


# ========== Story 3.3 (revised): preprocess_for_clustering() ==========

@pytest.fixture
def minimal_customers_df():
    """Minimal df_customers-like DataFrame satisfying config feature lists.

    Uses patched config constants so the test is decoupled from the real dataset.
    Includes hardcoded columns required by preprocess_for_clustering() (age,
    channel_recruitment) alongside test continuous/OHE/frequency features.
    """
    n = 10
    return pd.DataFrame({
        # continuous → median-imputed + scaled (includes 'age' required for has_age_info)
        "recency_days": [float(i * 10) for i in range(1, n + 1)],
        "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        # one-hot categorical (includes 'channel_recruitment' required for has_first_purchase_info)
        "gender": ["Women", "Men", "Women", "Men", "Women", "Men", "Women", "Men", "Women", "Men"],
        "channel_recruitment": ["store", "estore", "store", "estore", "store",
                                 "estore", "store", "estore", "store", "estore"],
        # frequency categorical
        "customer_city": ["Paris", "Lyon", "Paris", "Lille", "Paris", "Lyon", "Lille", "Paris", "Lyon", "Paris"],
    })


_MOCK_CONTINUOUS = ["recency_days", "age"]
_MOCK_ONEHOT = ["gender", "channel_recruitment"]
_MOCK_FREQUENCY = ["customer_city"]
_MOCK_DROP = []


@pytest.fixture
def patched_config():
    """Patch config constants used by preprocess_for_clustering()."""
    with patch("src.preprocessing.FEATURES_DROP", _MOCK_DROP), \
         patch("src.preprocessing.FEATURES_CONTINUOUS", _MOCK_CONTINUOUS), \
         patch("src.preprocessing.FEATURES_ONEHOT", _MOCK_ONEHOT), \
         patch("src.preprocessing.FEATURES_FREQUENCY", _MOCK_FREQUENCY):
        yield


class TestPreprocessForClusteringOutputShape:
    """AC-5, 6: function returns a DataFrame of the expected shape."""

    def test_returns_dataframe(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        assert len(result) == len(minimal_customers_df)

    def test_original_categorical_dropped(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        # OHE and frequency-encoded originals must be replaced
        assert "gender" not in result.columns
        assert "customer_city" not in result.columns

    def test_ohe_dummies_created(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        ohe_cols = [c for c in result.columns if c.startswith("gender_")]
        assert len(ohe_cols) >= 2

    def test_frequency_column_created(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        assert "customer_city_freq" in result.columns


class TestPreprocessForClusteringScaling:
    """AC-3, 7: StandardScaler applied and output passes quality checks."""

    def test_zero_mean(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        means = result.mean()
        # Ensure all means are practically 0
        np.testing.assert_allclose(means.values, 0.0, atol=1e-9)

    def test_unit_std(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        stds = result.std(ddof=0)
        # Constant columns (e.g. a dummy with all 0s) are excluded from std≈1 check
        non_constant = stds[stds > 0]
        np.testing.assert_allclose(non_constant.values, 1.0, atol=1e-9)

    def test_no_nan(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        assert result.isnull().sum().sum() == 0, "NaN values remain in X_scaled"

    def test_no_inf(self, minimal_customers_df, patched_config):
        result = preprocess_for_clustering(minimal_customers_df)
        assert not np.isinf(result.values).any(), "Inf values remain in X_scaled"


# ========== Story 3.4: PCA — Variance Analysis ==========

@pytest.fixture
def scaled_df_20():
    """Scaled DataFrame with 20 features for PCA tests."""
    np.random.seed(42)
    n_samples, n_features = 100, 20
    data = np.random.randn(n_samples, n_features)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def scaled_df_50():
    """Scaled DataFrame with 50 features (>30) for PCA default capping test."""
    np.random.seed(42)
    n_samples, n_features = 100, 50
    data = np.random.randn(n_samples, n_features)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


class TestApplyPcaReturnsCorrectTypes:
    """apply_pca() returns (DataFrame, PCA)."""

    def test_returns_tuple(self, scaled_df_20):
        result = apply_pca(scaled_df_20)
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_dataframe(self, scaled_df_20):
        X_pca, _ = apply_pca(scaled_df_20)
        assert isinstance(X_pca, pd.DataFrame)

    def test_returns_pca_object(self, scaled_df_20):
        _, pca = apply_pca(scaled_df_20)
        from sklearn.decomposition import PCA
        assert isinstance(pca, PCA)


class TestApplyPcaDefaultComponents:
    """AC-1: n_components = min(n_features, 30)."""

    def test_default_20_features(self, scaled_df_20):
        X_pca, pca = apply_pca(scaled_df_20)
        # min(20, 30) = 20
        assert X_pca.shape[1] == 20
        assert pca.n_components == 20

    def test_default_50_features_caps_at_30(self, scaled_df_50):
        X_pca, pca = apply_pca(scaled_df_50)
        # min(50, 30) = 30
        assert X_pca.shape[1] == 30
        assert pca.n_components == 30


class TestApplyPcaExplicitComponents:
    """apply_pca() respects explicit n_components."""

    def test_explicit_5_components(self, scaled_df_20):
        X_pca, pca = apply_pca(scaled_df_20, n_components=5)
        assert X_pca.shape[1] == 5

    def test_explicit_10_components(self, scaled_df_50):
        X_pca, pca = apply_pca(scaled_df_50, n_components=10)
        assert X_pca.shape[1] == 10


class TestApplyPcaPreservesIndex:
    """Index preserved from input DataFrame."""

    def test_preserves_index(self, scaled_df_20):
        scaled_df_20.index = [f"C{i}" for i in range(len(scaled_df_20))]
        X_pca, _ = apply_pca(scaled_df_20)
        assert list(X_pca.index) == list(scaled_df_20.index)

    def test_preserves_row_count(self, scaled_df_20):
        X_pca, _ = apply_pca(scaled_df_20)
        assert len(X_pca) == len(scaled_df_20)


class TestApplyPcaColumnNames:
    """Columns named PC1, PC2, ..."""

    def test_column_names(self, scaled_df_20):
        X_pca, _ = apply_pca(scaled_df_20, n_components=5)
        assert list(X_pca.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5"]


class TestApplyPcaReproducibility:
    """random_state=RANDOM_STATE ensures reproducible results."""

    def test_reproducible(self, scaled_df_20):
        X1, _ = apply_pca(scaled_df_20, n_components=5)
        X2, _ = apply_pca(scaled_df_20, n_components=5)
        pd.testing.assert_frame_equal(X1, X2)


class TestApplyPcaVariance:
    """PCA object captures explained variance."""

    def test_explained_variance_sums_le_1(self, scaled_df_20):
        _, pca = apply_pca(scaled_df_20)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

    def test_explained_variance_length(self, scaled_df_20):
        _, pca = apply_pca(scaled_df_20, n_components=5)
        assert len(pca.explained_variance_ratio_) == 5


# ========== Story 3.5: UMAP 2D Visualization ==========

@pytest.fixture
def scaled_df_umap():
    """Scaled DataFrame with 10 features for UMAP tests."""
    np.random.seed(42)
    n_samples, n_features = 50, 10
    data = np.random.randn(n_samples, n_features)
    cols = [f"feat_{i}" for i in range(n_features)]
    idx = pd.Index([f"C{i}" for i in range(n_samples)], name="anonymized_card_code")
    return pd.DataFrame(data, columns=cols, index=idx)


class TestApplyUmapReturnsDataFrame:
    """apply_umap() returns a 2D DataFrame with columns ['umap_1', 'umap_2']."""

    def test_returns_dataframe(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap)
        assert isinstance(result, pd.DataFrame)

    def test_two_columns(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap)
        assert result.shape[1] == 2

    def test_column_names(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap)
        assert list(result.columns) == ["umap_1", "umap_2"]


class TestApplyUmapPreservesIndex:
    """Index is preserved from input DataFrame."""

    def test_preserves_index(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap)
        assert list(result.index) == list(scaled_df_umap.index)

    def test_preserves_row_count(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap)
        assert len(result) == len(scaled_df_umap)


class TestApplyUmapDefaultParams:
    """AC-1: Default n_neighbors=15, min_dist=0.1, random_state=42."""

    def test_reproducible(self, scaled_df_umap):
        r1 = apply_umap(scaled_df_umap)
        r2 = apply_umap(scaled_df_umap)
        pd.testing.assert_frame_equal(r1, r2)

    def test_custom_params(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap, n_neighbors=5, min_dist=0.05)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(scaled_df_umap), 2)


class TestApplyUmapNoNaN:
    """Output has no NaN values."""

    def test_no_nan(self, scaled_df_umap):
        result = apply_umap(scaled_df_umap)
        assert result.isnull().sum().sum() == 0
