"""Tests for src/feature_engineer.py — Stories 1.2 & 1.3"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking raw Sephora transaction data."""
    return pd.DataFrame({
        "anonymized_card_code": ["A", "B", "C", "D", "E", "F", "G"],
        "age": [25, 0, 14, 30, 45, 10, 50],
        "gender": [1, 2, 99999, 1, 2, 2, 1],
        "status": [2, 3, 4, 2, 3, 4, 2],
        "Axe_Desc": ["SKINCARE", "MAEK UP", "MAKE UP", "HAIRCARE", "MAEK UP", "FRAGRANCE", "OTHERS"],
        "channel": ["store", "estore", "estore", "estore", "store", "estore", "estore"],
        "store_type_app": ["STORE", "STORE", "ESTORE", "WEB", "STORE", "MOBILE", "STORE"],
        "salesVatEUR": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    })


class TestAssessDataQuality:
    """Tests for assess_data_quality()."""

    def test_empty_dataframe(self):
        from src.feature_engineer import assess_data_quality
        df = pd.DataFrame(columns=["age", "gender"])
        result = assess_data_quality(df)
        assert result == {"missing_count": {}, "missing_rate": {}}

    def test_returns_dict(self, sample_df):
        from src.feature_engineer import assess_data_quality
        result = assess_data_quality(sample_df)
        assert isinstance(result, dict)

    def test_missing_count_key(self, sample_df):
        from src.feature_engineer import assess_data_quality
        result = assess_data_quality(sample_df)
        assert "missing_count" in result
        assert "missing_rate" in result

    def test_missing_count_per_column(self, sample_df):
        from src.feature_engineer import assess_data_quality
        # Add a NaN to check detection
        df = sample_df.copy()
        df.loc[0, "salesVatEUR"] = np.nan
        result = assess_data_quality(df)
        assert result["missing_count"]["salesVatEUR"] == 1

    def test_missing_rate_is_percentage(self, sample_df):
        from src.feature_engineer import assess_data_quality
        df = sample_df.copy()
        df.loc[0, "salesVatEUR"] = np.nan
        result = assess_data_quality(df)
        expected_rate = 1 / len(df) * 100
        assert abs(result["missing_rate"]["salesVatEUR"] - expected_rate) < 0.01


class TestCleanRawData:
    """Tests for clean_raw_data()."""

    def test_returns_dataframe(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_original(self, sample_df):
        from src.feature_engineer import clean_raw_data
        original_values = sample_df["Axe_Desc"].tolist()
        _ = clean_raw_data(sample_df)
        assert sample_df["Axe_Desc"].tolist() == original_values

    def test_axe_desc_typo_fixed(self, sample_df):
        from src.feature_engineer import clean_raw_data
        sample_df["Axe_Desc_first_purchase"] = ["MAEK UP"] * len(sample_df)
        result = clean_raw_data(sample_df)
        assert "MAEK UP" not in result["Axe_Desc"].values
        assert "MAKE UP" in result["Axe_Desc"].values
        assert "MAEK UP" not in result["Axe_Desc_first_purchase"].values
        assert "MAKE UP" in result["Axe_Desc_first_purchase"].values

    def test_age_zero_flagged_nan(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        # age == 0 (row B) should be NaN
        assert pd.isna(result.loc[1, "age"])

    def test_age_under_15_flagged_nan(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        # age == 14 (row C) and age == 10 (row F) should be NaN
        assert pd.isna(result.loc[2, "age"])
        assert pd.isna(result.loc[5, "age"])

    def test_age_valid_not_changed(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        # age == 25 (row A) should remain
        assert result.loc[0, "age"] == 25

    def test_gender_99999_replaced(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        assert 99999 not in result["gender"].values
        assert "Unknown" in result["gender"].values

    def test_gender_recoded_men_women(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        assert "Men" in result["gender"].values
        assert "Women" in result["gender"].values
        assert 1 not in result["gender"].values
        assert 2 not in result["gender"].values

    def test_status_recoded(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        expected_labels = {"BRONZE", "SILVER", "GOLD"}
        actual_labels = set(result["status"].unique())
        assert actual_labels == expected_labels

    def test_status_mapping_correct(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        # Row A: status=2 → BRONZE
        assert result.loc[0, "status"] == "BRONZE"
        # Row B: status=3 → SILVER
        assert result.loc[1, "status"] == "SILVER"
        # Row C: status=4 → GOLD
        assert result.loc[2, "status"] == "GOLD"
        
    def test_string_columns_are_categoricals(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        assert isinstance(result["gender"].dtype, pd.CategoricalDtype)
        assert isinstance(result["status"].dtype, pd.CategoricalDtype)

    def test_is_click_collect_column_exists(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        assert "is_click_collect" in result.columns

    def test_click_collect_true_cases(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        # Row B: channel=estore, store_type_app=STORE → click & collect
        assert result.loc[1, "is_click_collect"] is True or result.loc[1, "is_click_collect"] == True
        # Row G: channel=estore, store_type_app=STORE → click & collect
        assert result.loc[6, "is_click_collect"] is True or result.loc[6, "is_click_collect"] == True

    def test_click_collect_false_cases(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        # Row A: channel=store → NOT click & collect
        assert result.loc[0, "is_click_collect"] == False
        # Row C: channel=estore, store_type_app=ESTORE → NOT click & collect
        assert result.loc[2, "is_click_collect"] == False
        # Row D: channel=estore, store_type_app=WEB → NOT click & collect
        assert result.loc[3, "is_click_collect"] == False

    def test_row_count_preserved(self, sample_df):
        from src.feature_engineer import clean_raw_data
        result = clean_raw_data(sample_df)
        assert len(result) == len(sample_df)


# ── Story 1.3 — aggregate_to_customer_level ─────────────────────────────────


@pytest.fixture
def clean_df():
    """Multi-row cleaned transaction DataFrame for aggregation tests.

    Customer A: 2 transactions (ticket T1 with 2 lines, ticket T2 with 1 line) → 3 rows
    Customer B: 1 transaction (ticket T3, 1 line) → 1 row
    """
    return pd.DataFrame({
        "anonymized_card_code": ["A", "A", "A", "B"],
        "anonymized_Ticket_ID": ["T1", "T1", "T2", "T3"],
        "salesVatEUR": [10.0, 20.0, 30.0, 50.0],
        "discountEUR": [1.0, 2.0, 3.0, 5.0],
        "quantity": [1, 2, 1, 3],
        "transactionDate": pd.to_datetime(["2025-01-10", "2025-01-10", "2025-06-15", "2025-03-20"]),
        "status": pd.Categorical(["GOLD", "GOLD", "GOLD", "SILVER"]),
        "age": [30.0, 30.0, 30.0, 45.0],
        "age_category": ["30-39", "30-39", "30-39", "40-49"],
        "age_generation": ["Millennials", "Millennials", "Millennials", "Gen X"],
        "gender": pd.Categorical(["Women", "Women", "Women", "Men"]),
        "countryIsoCode": ["FR", "FR", "FR", "DE"],
        "customer_city": ["Paris", "Paris", "Paris", "Berlin"],
        "subscription_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01", "2021-06-01"]),
        "channel_recruitment": ["store", "store", "store", "estore"],
        "salesVatEUR_first_purchase": [15.0, 15.0, 15.0, 50.0],
        "Axe_Desc_first_purchase": ["SKINCARE", "SKINCARE", "SKINCARE", "FRAGRANCE"],
        "is_click_collect": [False, False, True, False],
    })


class TestAggregateToCustomerLevel:
    """Tests for aggregate_to_customer_level() — Story 1.3"""

    def test_returns_dataframe(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_customer(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert len(result) == 2  # A, B
        assert result.index.nunique() == len(result)

    def test_index_is_anonymized_card_code(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.index.name == "anonymized_card_code"

    def test_total_transactions_nunique(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "total_transactions"] == 2  # T1, T2
        assert result.loc["B", "total_transactions"] == 1  # T3

    def test_total_lines_count(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "total_lines"] == 3
        assert result.loc["B", "total_lines"] == 1

    def test_total_sales_eur_sum(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "total_sales_eur"] == pytest.approx(60.0)
        assert result.loc["B", "total_sales_eur"] == pytest.approx(50.0)

    def test_avg_sales_eur_mean(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "avg_sales_eur"] == pytest.approx(20.0)
        assert result.loc["B", "avg_sales_eur"] == pytest.approx(50.0)

    def test_total_discount_eur_sum(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "total_discount_eur"] == pytest.approx(6.0)
        assert result.loc["B", "total_discount_eur"] == pytest.approx(5.0)

    def test_total_quantity_sum(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "total_quantity"] == 4
        assert result.loc["B", "total_quantity"] == 3

    def test_last_purchase_date_max(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "last_purchase_date"] == pd.Timestamp("2025-06-15")
        assert result.loc["B", "last_purchase_date"] == pd.Timestamp("2025-03-20")

    def test_first_purchase_date_min(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "first_purchase_date"] == pd.Timestamp("2025-01-10")
        assert result.loc["B", "first_purchase_date"] == pd.Timestamp("2025-03-20")

    def test_loyalty_status_last(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "loyalty_status"] == "GOLD"
        assert result.loc["B", "loyalty_status"] == "SILVER"

    def test_demographic_first_values(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "age"] == 30.0
        assert result.loc["A", "age_category"] == "30-39"
        assert result.loc["A", "age_generation"] == "Millennials"
        assert result.loc["A", "gender"] == "Women"
        assert result.loc["A", "country"] == "FR"
        assert result.loc["A", "customer_city"] == "Paris"
        assert result.loc["B", "age"] == 45.0
        assert result.loc["B", "country"] == "DE"

    def test_subscription_and_recruitment(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "subscription_date"] == pd.Timestamp("2020-01-01")
        assert result.loc["A", "channel_recruitment"] == "store"
        assert result.loc["B", "channel_recruitment"] == "estore"

    def test_first_purchase_columns(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "salesVatEUR_first_purchase"] == 15.0
        assert result.loc["A", "Axe_Desc_first_purchase"] == "SKINCARE"
        assert result.loc["B", "Axe_Desc_first_purchase"] == "FRAGRANCE"

    def test_cc_transactions_column(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert result.loc["A", "cc_transactions"] == 1  # one True in 3 rows
        assert result.loc["B", "cc_transactions"] == 0

    def test_all_expected_columns_present(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        expected = {
            "total_transactions", "total_lines", "total_sales_eur", "avg_sales_eur",
            "total_discount_eur", "total_quantity", "last_purchase_date",
            "first_purchase_date", "loyalty_status", "age", "age_category",
            "age_generation", "gender", "country", "customer_city",
            "subscription_date", "channel_recruitment", "salesVatEUR_first_purchase",
            "Axe_Desc_first_purchase", "cc_transactions",
        }
        assert expected.issubset(set(result.columns))

    def test_no_duplicate_index(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        result = aggregate_to_customer_level(clean_df)
        assert not result.index.duplicated().any()

    def test_nan_handling_in_aggregations(self, clean_df):
        from src.feature_engineer import aggregate_to_customer_level
        # Introduce NaNs into numeric columns
        df_with_nans = clean_df.copy()
        df_with_nans.loc[0, "salesVatEUR"] = np.nan
        df_with_nans.loc[1, "is_click_collect"] = np.nan
        
        result = aggregate_to_customer_level(df_with_nans)
        
        # A has three rows originally: 10, 20, 30. Now NaN, 20, 30. sum=50, mean=25
        assert result.loc["A", "total_sales_eur"] == pytest.approx(50.0)
        assert result.loc["A", "avg_sales_eur"] == pytest.approx(25.0)
        
        # is_click_collect: rows 0(False->NaN), 1(False->NaN), 2(True). sum=1
        assert pd.notna(result.loc["A", "cc_transactions"])
        assert result.loc["A", "cc_transactions"] == 1


# ── Story 1.4 — compute_rfm_features ────────────────────────────────────────


@pytest.fixture
def customer_df():
    """Customer-level DataFrame as produced by aggregate_to_customer_level()."""
    return pd.DataFrame(
        {
            "total_transactions": [5, 1, 12],
            "total_sales_eur": [250.0, 80.0, 1500.0],
            "avg_sales_eur": [50.0, 80.0, 125.0],
            "last_purchase_date": pd.to_datetime(
                ["2025-11-01", "2025-06-15", "2025-12-30"]
            ),
        },
        index=pd.Index(["C1", "C2", "C3"], name="anonymized_card_code"),
    )


class TestComputeRfmFeatures:
    """Tests for compute_rfm_features() — Story 1.4"""

    def test_returns_dataframe(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert isinstance(result, pd.DataFrame)

    def test_recency_days_column_exists(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert "recency_days" in result.columns

    def test_recency_days_values(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        # 2025-12-31 − 2025-11-01 = 60 days
        assert result.loc["C1", "recency_days"] == 60
        # 2025-12-31 − 2025-06-15 = 199 days
        assert result.loc["C2", "recency_days"] == 199
        # 2025-12-31 − 2025-12-30 = 1 day
        assert result.loc["C3", "recency_days"] == 1

    def test_recency_days_is_int(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert result["recency_days"].dtype in (np.int64, np.int32, int)

    def test_recency_days_non_negative(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert (result["recency_days"] >= 0).all()

    def test_frequency_equals_total_transactions(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert "frequency" in result.columns
        pd.testing.assert_series_equal(
            result["frequency"], customer_df["total_transactions"], check_names=False
        )

    def test_monetary_total_equals_total_sales_eur(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert "monetary_total" in result.columns
        pd.testing.assert_series_equal(
            result["monetary_total"], customer_df["total_sales_eur"], check_names=False
        )

    def test_monetary_avg_equals_avg_sales_eur(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert "monetary_avg" in result.columns
        pd.testing.assert_series_equal(
            result["monetary_avg"], customer_df["avg_sales_eur"], check_names=False
        )

    def test_no_nulls_in_rfm_columns(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        rfm_cols = ["recency_days", "frequency", "monetary_total", "monetary_avg"]
        assert result[rfm_cols].isnull().sum().sum() == 0

    def test_preserves_existing_columns(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        for col in customer_df.columns:
            assert col in result.columns

    def test_preserves_index(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        result = compute_rfm_features(customer_df)
        assert result.index.name == "anonymized_card_code"
        assert list(result.index) == ["C1", "C2", "C3"]

    def test_does_not_mutate_original(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        original_cols = list(customer_df.columns)
        _ = compute_rfm_features(customer_df)
        assert list(customer_df.columns) == original_cols
        assert "recency_days" not in customer_df.columns

    def test_recency_days_clipped_to_zero(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        # Add a date strictly after 2025-12-31 to test clipping
        df = customer_df.copy()
        df.loc["C1", "last_purchase_date"] = pd.Timestamp("2026-01-10")
        result = compute_rfm_features(df)
        assert result.loc["C1", "recency_days"] == 0

    def test_raises_error_on_missing_last_purchase_date(self, customer_df):
        from src.feature_engineer import compute_rfm_features
        df = customer_df.copy()
        df.loc["C2", "last_purchase_date"] = pd.NaT
        with pytest.raises(AssertionError, match="last_purchase_date contains nulls"):
            compute_rfm_features(df)

