"""Tests for src/preprocessing.py — Story 3.2 & 3.3 New features"""
import pytest
import pandas as pd
import numpy as np

from src.preprocessing import _clean_primary_axe, preprocess_for_clustering


class TestCleanPrimaryAxe:
    def test_nan(self):
        assert pd.isna(_clean_primary_axe(np.nan))
    
    def test_valid_category(self):
        assert _clean_primary_axe("SKINCARE") == "SKINCARE"
        
    def test_multi_label(self):
        assert _clean_primary_axe("FRAGRANCE|SKINCARE") == "FRAGRANCE"
        
    def test_typo_fix(self):
        assert _clean_primary_axe("MAEK UP|OTHERS") == "MAKE UP"
        
    def test_others_fallback(self):
        assert _clean_primary_axe("WEIRD CATEGORY") == "OTHERS"

class TestPreprocessForClustering:
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "age": [25, np.nan, 35],
            "first_purchase_axe": ["SKINCARE", "MAEK UP", np.nan],
            "some_ratio": [0.5, np.inf, -np.inf],
            "is_new_customer": [1, 1, 1],
        })
        res = preprocess_for_clustering(df)
        assert isinstance(res, pd.DataFrame)
        assert res.isnull().sum().sum() == 0
        if "some_ratio" in res.columns:
            assert np.isinf(res["some_ratio"]).sum() == 0
