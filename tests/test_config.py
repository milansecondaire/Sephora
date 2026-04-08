"""Tests for configuration and feature groups (Story 3.1 REVISED + R1.1)."""
import pytest
import re
from src.config import (
    CLUSTERING_FEATURES,
    FEATURES_DROP,
    FEATURES_CONTINUOUS,
    FEATURES_ONEHOT,
    FEATURES_FREQUENCY,
    FEATURE_CATEGORIES,
    CATEGORY_COLORS,
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
            "Lifecycle": ["subscription_tenure_days"]
        }
        
        for family, cols in family_checks.items():
            assert any(c in retained for c in cols), f"No {family} feature found in retained groups"


class TestFeatureCategories:
    """Validate FEATURE_CATEGORIES and CATEGORY_COLORS (Story R1.1)."""

    EXPECTED_KEYS = {"profil", "valeur", "affinite_produit", "comportement", "canal", "dates"}

    def test_is_dict_with_six_keys(self):
        """AC-1: FEATURE_CATEGORIES must be a dict with exactly 6 keys."""
        assert isinstance(FEATURE_CATEGORIES, dict)
        assert set(FEATURE_CATEGORIES.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_lists(self):
        """AC-1: Every category value must be a non-empty list."""
        for key, val in FEATURE_CATEGORIES.items():
            assert isinstance(val, list), f"{key} must be a list"
            assert len(val) > 0, f"{key} must not be empty"

    def test_no_duplicates_across_categories(self):
        """AC-2: Each feature appears in exactly one category."""
        all_features: list = []
        for features in FEATURE_CATEGORIES.values():
            all_features.extend(features)
        assert len(all_features) == len(set(all_features)), "Duplicate feature found across categories"

    def test_no_dropped_feature_in_categories(self):
        """AC-3: No feature in FEATURE_CATEGORIES should be in FEATURES_DROP."""
        all_categorized = {f for features in FEATURE_CATEGORIES.values() for f in features}
        dropped = set(FEATURES_DROP)
        overlap = all_categorized & dropped
        assert not overlap, f"Features in both FEATURE_CATEGORIES and FEATURES_DROP: {overlap}"

    def test_required_features_present(self):
        """AC-2: Check all features from each category are exhaustively present."""
        expected = {
            "profil": ["age", "gender", "country", "loyalty_numeric"],
            "valeur": ["recency_days", "frequency", "monetary_total", "monetary_avg",
                       "avg_basket_size_eur", "discount_rate"],
            "affinite_produit": ["axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio",
                                 "axe_haircare_ratio", "axe_others_ratio",
                                 "market_selective_ratio", "market_exclusive_ratio",
                                 "market_sephora_ratio",
                                 "axis_diversity"],
            "comportement": ["avg_units_per_basket", "nb_unique_stores"],
            "canal": ["store_ratio", "click_collect_ratio", "dominant_channel"],
            "dates": ["subscription_tenure_days"],
        }
        total_expected = 0
        for cat, features in expected.items():
            total_expected += len(features)
            assert set(FEATURE_CATEGORIES[cat]) == set(features), (
                f"Features mismatch in FEATURE_CATEGORIES['{cat}']"
            )
        assert total_expected == 25, "Total features must be exactly 25"

        actual_total = sum(len(v) for v in FEATURE_CATEGORIES.values())
        assert actual_total == 25, f"Expected 25 features across categories, got {actual_total}"

    def test_category_colors_matches_categories(self):
        """AC-5: CATEGORY_COLORS must have the same 6 keys as FEATURE_CATEGORIES."""
        assert isinstance(CATEGORY_COLORS, dict)
        assert set(CATEGORY_COLORS.keys()) == self.EXPECTED_KEYS

    def test_category_colors_are_hex(self):
        """AC-5: CATEGORY_COLORS values must be valid hex color strings."""
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        for cat, color in CATEGORY_COLORS.items():
            assert hex_pattern.match(color), (
                f"CATEGORY_COLORS['{cat}'] = '{color}' is not a valid hex color"
            )
