"""Tests for configuration and feature groups (Story 3.1 REVISED)."""
import pytest
from src.config import (
    CLUSTERING_FEATURES,
    FEATURES_DROP,
    FEATURES_CONTINUOUS,
    FEATURES_ONEHOT,
    FEATURES_FREQUENCY,
)


class TestClusteringFeatures:
    """Validate legacy CLUSTERING_FEATURES constant (backward compatibility)."""
    
    def test_legacy_is_list(self):
        assert isinstance(CLUSTERING_FEATURES, list)
        assert len(CLUSTERING_FEATURES) == 21
        assert len(CLUSTERING_FEATURES) == len(set(CLUSTERING_FEATURES))


class TestFeatureGroups:
    """Validate new comprehensive Feature Groups meet ACs."""

    def test_groups_are_lists(self):
        """All feature groups must be lists."""
        assert isinstance(FEATURES_DROP, list)
        assert isinstance(FEATURES_CONTINUOUS, list)
        assert isinstance(FEATURES_ONEHOT, list)
        assert isinstance(FEATURES_FREQUENCY, list)

    def test_mutually_exclusive(self):
        """AC-1: Features should not appear in multiple groups."""
        all_features = FEATURES_DROP + FEATURES_CONTINUOUS + FEATURES_ONEHOT + FEATURES_FREQUENCY
        assert len(all_features) == len(set(all_features)), "Overlap found between feature groups"

    def test_correct_column_count(self):
        """Ensure all 54 columns from the dataset are classified."""
        total_count = len(FEATURES_DROP) + len(FEATURES_CONTINUOUS) + len(FEATURES_ONEHOT) + len(FEATURES_FREQUENCY)
        assert total_count == 54, f"Expected 54 features, found {total_count}."

    def test_dropped_features_criteria(self):
        """AC-2: Validate specific known drops are in FEATURES_DROP."""
        expected_drops = [
            "is_new_customer", "total_sales_eur", "total_transactions", 
            "first_purchase_date", "last_purchase_date", "is_outlier"
        ]
        for col in expected_drops:
            assert col in FEATURES_DROP, f"{col} should be marked for dropping"

    def test_family_coverage(self):
        """AC-4: At least one feature from each of the 6 families in retained features."""
        retained = set(FEATURES_CONTINUOUS + FEATURES_ONEHOT + FEATURES_FREQUENCY)
        
        # Checking presence of known features for each family in retained features
        family_checks = {
            "RFM": ["recency_days", "frequency", "monetary_total", "monetary_avg"],
            "Behavior": ["avg_basket_size_eur", "avg_units_per_basket", "discount_rate"],
            "Product": ["axe_make_up_ratio", "market_selective_ratio"],
            "Channel": ["store_ratio", "estore_ratio", "dominant_channel"],
            "Sociodemographic": ["loyalty_numeric", "age", "gender"],
            "Lifecycle": ["subscription_tenure_days", "first_purchase_amount"]
        }
        
        for family, cols in family_checks.items():
            assert any(c in retained for c in cols), f"No {family} feature found in retained groups"
