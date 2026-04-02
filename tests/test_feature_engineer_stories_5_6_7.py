"""Tests for src/feature_engineer.py — Stories 1.5, 1.6, 1.7"""
import pytest
import pandas as pd
import numpy as np


# ── Story 1.5 — compute_behavioral_features ─────────────────────────────────


@pytest.fixture
def behavioral_clean_df():
    """Transaction-level DataFrame for behavioral feature tests.

    Customer A: 3 rows, 2 tickets (T1×2 lines, T2×1 line)
      - T1: store, T2: estore + click_collect
    Customer B: 1 row, 1 ticket (T3: estore, not C&C)
    """
    return pd.DataFrame({
        "anonymized_card_code": ["A", "A", "A", "B"],
        "anonymized_Ticket_ID": ["T1", "T1", "T2", "T3"],
        "channel": ["store", "store", "estore", "estore"],
        "is_click_collect": [False, False, True, False],
        "salesVatEUR": [10.0, 20.0, 30.0, 50.0],
        "brand": ["BrandX", "BrandY", "BrandX", "BrandZ"],
        "store_code_name": ["S1", "S1", "S2", "S3"],
    })


@pytest.fixture
def behavioral_customer_df():
    """Customer-level DataFrame matching behavioral_clean_df aggregation."""
    return pd.DataFrame(
        {
            "total_transactions": [2, 1],
            "total_sales_eur": [60.0, 50.0],
            "total_quantity": [4, 3],
            "total_discount_eur": [6.0, 5.0],
        },
        index=pd.Index(["A", "B"], name="anonymized_card_code"),
    )


class TestComputeBehavioralFeatures:
    """Tests for compute_behavioral_features() — Story 1.5"""

    def test_returns_dataframe(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        original_cols = list(behavioral_customer_df.columns)
        _ = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        assert list(behavioral_customer_df.columns) == original_cols

    def test_avg_basket_size_eur(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: 60/2=30, B: 50/1=50
        assert result.loc["A", "avg_basket_size_eur"] == pytest.approx(30.0)
        assert result.loc["B", "avg_basket_size_eur"] == pytest.approx(50.0)

    def test_avg_units_per_basket(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: 4/2=2, B: 3/1=3
        assert result.loc["A", "avg_units_per_basket"] == pytest.approx(2.0)
        assert result.loc["B", "avg_units_per_basket"] == pytest.approx(3.0)

    def test_discount_rate(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: 6/60=0.1, B: 5/50=0.1
        assert result.loc["A", "discount_rate"] == pytest.approx(0.1)
        assert result.loc["B", "discount_rate"] == pytest.approx(0.1)

    def test_discount_rate_capped(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        df = behavioral_customer_df.copy()
        df.loc["A", "total_discount_eur"] = 999.0
        result = compute_behavioral_features(df, behavioral_clean_df)
        assert result.loc["A", "discount_rate"] <= 1.0

    def test_discount_rate_zero_sales(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        df = behavioral_customer_df.copy()
        df.loc["A", "total_sales_eur"] = 0.0
        df.loc["A", "total_discount_eur"] = 0.0
        result = compute_behavioral_features(df, behavioral_clean_df)
        assert result.loc["A", "discount_rate"] == 0.0

    def test_store_ratio(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: 1 store ticket (T1) / 2 total = 0.5; B: 0/1 = 0
        assert result.loc["A", "store_ratio"] == pytest.approx(0.5)
        assert result.loc["B", "store_ratio"] == pytest.approx(0.0)

    def test_estore_ratio(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: T2 is estore but is_click_collect=True → 0 estore / 2 = 0
        # B: T3 is estore, not C&C → 1/1 = 1.0
        assert result.loc["A", "estore_ratio"] == pytest.approx(0.0)
        assert result.loc["B", "estore_ratio"] == pytest.approx(1.0)

    def test_click_collect_ratio(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: T2 is C&C → 1/2 = 0.5; B: 0/1=0
        assert result.loc["A", "click_collect_ratio"] == pytest.approx(0.5)
        assert result.loc["B", "click_collect_ratio"] == pytest.approx(0.0)

    def test_dominant_channel(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: store=0.5, estore=0, cc=0.5 → idxmax picks first → store
        assert result.loc["A", "dominant_channel"] == "store"
        assert result.loc["B", "dominant_channel"] == "estore"

    def test_nb_unique_brands(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: BrandX, BrandY → 2; B: BrandZ → 1
        assert result.loc["A", "nb_unique_brands"] == 2
        assert result.loc["B", "nb_unique_brands"] == 1

    def test_nb_unique_stores(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        # A: S1, S2 → 2; B: S3 → 1
        assert result.loc["A", "nb_unique_stores"] == 2
        assert result.loc["B", "nb_unique_stores"] == 1

    def test_all_new_columns_present(self, behavioral_customer_df, behavioral_clean_df):
        from src.feature_engineer import compute_behavioral_features
        result = compute_behavioral_features(behavioral_customer_df, behavioral_clean_df)
        expected = {
            "avg_basket_size_eur", "avg_units_per_basket", "discount_rate",
            "store_ratio", "estore_ratio", "click_collect_ratio",
            "dominant_channel", "nb_unique_brands", "nb_unique_stores",
        }
        assert expected.issubset(set(result.columns))


# ── Story 1.6 — compute_product_affinity_features ───────────────────────────


@pytest.fixture
def affinity_clean_df():
    """Transaction DataFrame with Axe_Desc, Market_Desc, salesVatEUR."""
    return pd.DataFrame({
        "anonymized_card_code": ["A", "A", "A", "B", "B"],
        "Axe_Desc": ["MAKE UP", "SKINCARE", "MAKE UP", "FRAGRANCE", "FRAGRANCE"],
        "Market_Desc": ["SELECTIVE", "EXCLUSIVE", "SELECTIVE", "SEPHORA", "SEPHORA"],
        "salesVatEUR": [30.0, 20.0, 10.0, 40.0, 10.0],
    })


@pytest.fixture
def affinity_customer_df():
    """Customer-level DataFrame matching affinity_clean_df."""
    return pd.DataFrame(
        {
            "total_sales_eur": [60.0, 50.0],
        },
        index=pd.Index(["A", "B"], name="anonymized_card_code"),
    )


class TestComputeProductAffinityFeatures:
    """Tests for compute_product_affinity_features() — Story 1.6"""

    def test_returns_dataframe(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        original_cols = list(affinity_customer_df.columns)
        _ = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        assert list(affinity_customer_df.columns) == original_cols

    def test_axis_ratios_sum_to_one(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features, AXIS_COL_MAP
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        axe_cols = list(AXIS_COL_MAP.values())
        row_sums = result[axe_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.001)

    def test_axis_ratio_values(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        # A: MAKE UP=(30+10)/60=0.667, SKINCARE=20/60=0.333
        assert result.loc["A", "axe_make_up_ratio"] == pytest.approx(40 / 60, abs=0.001)
        assert result.loc["A", "axe_skincare_ratio"] == pytest.approx(20 / 60, abs=0.001)
        assert result.loc["A", "axe_fragrance_ratio"] == pytest.approx(0.0)
        # B: FRAGRANCE=50/50=1.0
        assert result.loc["B", "axe_fragrance_ratio"] == pytest.approx(1.0)

    def test_market_ratio_values(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        # A: SELECTIVE=(30+10)/60=0.667, EXCLUSIVE=20/60=0.333
        assert result.loc["A", "market_selective_ratio"] == pytest.approx(40 / 60, abs=0.001)
        assert result.loc["A", "market_exclusive_ratio"] == pytest.approx(20 / 60, abs=0.001)
        # B: SEPHORA=50/50=1.0
        assert result.loc["B", "market_sephora_ratio"] == pytest.approx(1.0)

    def test_market_ratios_sum_to_one(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features, MARKET_COL_MAP
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        mkt_cols = list(MARKET_COL_MAP.values())
        row_sums = result[mkt_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.001)

    def test_dominant_axe(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        assert result.loc["A", "dominant_axe"] == "MAKE UP"
        assert result.loc["B", "dominant_axe"] == "FRAGRANCE"

    def test_dominant_market(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        assert result.loc["A", "dominant_market"] == "SELECTIVE"
        assert result.loc["B", "dominant_market"] == "SEPHORA"

    def test_axis_diversity(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        # A: MAKE UP, SKINCARE → 2; B: FRAGRANCE → 1
        assert result.loc["A", "axis_diversity"] == 2
        assert result.loc["B", "axis_diversity"] == 1

    def test_all_new_columns_present(self, affinity_customer_df, affinity_clean_df):
        from src.feature_engineer import compute_product_affinity_features
        result = compute_product_affinity_features(affinity_customer_df, affinity_clean_df)
        expected = {
            "axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio",
            "axe_haircare_ratio", "axe_others_ratio",
            "market_selective_ratio", "market_exclusive_ratio",
            "market_sephora_ratio", "market_others_ratio",
            "dominant_axe", "dominant_market", "axis_diversity",
        }
        assert expected.issubset(set(result.columns))


# ── Story 1.7 — compute_lifecycle_features ───────────────────────────────────


@pytest.fixture
def lifecycle_customer_df():
    """Customer-level DataFrame for lifecycle feature tests."""
    return pd.DataFrame(
        {
            "subscription_date": pd.to_datetime(["2020-01-01", "2023-06-15", pd.NaT]),
            "loyalty_status": ["GOLD", "BRONZE", "No Fid"],
            "first_purchase_date": pd.to_datetime(["2024-06-01", "2025-03-15", "2025-11-20"]),
            "Axe_Desc_first_purchase": ["SKINCARE", "['MAKE UP']", np.nan],
            "channel_recruitment": ["store", "estore", "store"],
            "salesVatEUR_first_purchase": [100.0, 50.0, 25.0],
        },
        index=pd.Index(["C1", "C2", "C3"], name="anonymized_card_code"),
    )


class TestComputeLifecycleFeatures:
    """Tests for compute_lifecycle_features() — Story 1.7"""

    def test_returns_dataframe(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        original_cols = list(lifecycle_customer_df.columns)
        _ = compute_lifecycle_features(lifecycle_customer_df)
        assert list(lifecycle_customer_df.columns) == original_cols

    def test_subscription_tenure_days(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C1", "subscription_tenure_days"] == (pd.Timestamp("2025-12-31") - pd.Timestamp("2020-01-01")).days
        assert result.loc["C2", "subscription_tenure_days"] == (pd.Timestamp("2025-12-31") - pd.Timestamp("2023-06-15")).days

    def test_subscription_tenure_nan_when_missing(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert pd.isna(result.loc["C3", "subscription_tenure_days"])

    def test_loyalty_numeric(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C1", "loyalty_numeric"] == 3  # GOLD
        assert result.loc["C2", "loyalty_numeric"] == 1  # BRONZE
        assert result.loc["C3", "loyalty_numeric"] == 0  # No Fid

    def test_is_new_customer(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C1", "is_new_customer"] == 0
        assert result.loc["C2", "is_new_customer"] == 1
        assert result.loc["C3", "is_new_customer"] == 1

    def test_first_purchase_axe_scalar(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C1", "first_purchase_axe"] == "SKINCARE"

    def test_first_purchase_axe_list_string(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C2", "first_purchase_axe"] == "MAKE UP"

    def test_first_purchase_axe_nan(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert pd.isna(result.loc["C3", "first_purchase_axe"])

    def test_first_purchase_channel(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C1", "first_purchase_channel"] == "store"
        assert result.loc["C2", "first_purchase_channel"] == "estore"

    def test_first_purchase_amount(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        assert result.loc["C1", "first_purchase_amount"] == pytest.approx(100.0)
        assert result.loc["C2", "first_purchase_amount"] == pytest.approx(50.0)

    def test_all_new_columns_present(self, lifecycle_customer_df):
        from src.feature_engineer import compute_lifecycle_features
        result = compute_lifecycle_features(lifecycle_customer_df)
        expected = {
            "subscription_tenure_days", "loyalty_numeric", "is_new_customer",
            "first_purchase_axe", "first_purchase_channel", "first_purchase_amount",
        }
        assert expected.issubset(set(result.columns))


# ── Story 1.7 — save_customers_features ──────────────────────────────────────


class TestSaveCustomersFeatures:
    """Tests for save_customers_features() — Story 1.7"""

    def test_saves_csv_file(self, tmp_path, monkeypatch):
        from src.feature_engineer import save_customers_features
        import src.feature_engineer as fe_mod
        monkeypatch.setattr(fe_mod, "DATA_PROCESSED_PATH", str(tmp_path) + "/")
        df = pd.DataFrame({"col": [1, 2]}, index=pd.Index(["A", "B"], name="anonymized_card_code"))
        path = save_customers_features(df)
        assert path.endswith("customers_features.csv")
        import os
        assert os.path.exists(path)

    def test_csv_content(self, tmp_path, monkeypatch):
        from src.feature_engineer import save_customers_features
        import src.feature_engineer as fe_mod
        monkeypatch.setattr(fe_mod, "DATA_PROCESSED_PATH", str(tmp_path) + "/")
        df = pd.DataFrame({"col": [1, 2]}, index=pd.Index(["A", "B"], name="anonymized_card_code"))
        path = save_customers_features(df)
        loaded = pd.read_csv(path, index_col=0)
        assert len(loaded) == 2
        assert "col" in loaded.columns
