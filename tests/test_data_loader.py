"""Tests for src/data_loader.py — Story 1.1"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_raw_data, validate_schema, EXPECTED_COLUMNS, STRING_COLUMNS, DATE_COLUMNS
from src.config import DATA_RAW_PATH

@pytest.fixture(scope="module")
def raw_data():
    """Fixture to load data once for all tests, saving significant execution time."""
    return load_raw_data()

class TestLoadRawData:
    """Tests for load_raw_data()."""

    def test_returns_dataframe(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Raw data file not found"):
            load_raw_data("nonexistent/path.csv")

    def test_encoding_no_error(self, raw_data):
        """AC-1: File loads without encoding error using utf-8-sig."""
        # If we get here, no encoding error occurred
        assert len(raw_data) > 0

    def test_column_count(self, raw_data):
        """AC-2: 34 columns expected."""
        assert raw_data.shape[1] == EXPECTED_COLUMNS

    def test_dtypes_printed(self, raw_data):
        """AC-3: dtypes can be accessed (df.dtypes)."""
        dtypes = raw_data.dtypes
        assert len(dtypes) == EXPECTED_COLUMNS

    def test_card_code_is_string(self, raw_data):
        """AC-4: anonymized_card_code parsed as string."""
        assert raw_data["anonymized_card_code"].dtype == "object"

    def test_date_columns_are_datetime(self, raw_data):
        """AC-5: transactionDate and first_purchase_dt parsed as datetime."""
        assert pd.api.types.is_datetime64_any_dtype(raw_data["transactionDate"])
        assert pd.api.types.is_datetime64_any_dtype(raw_data["first_purchase_dt"])

    def test_subscription_date_is_datetime(self, raw_data):
        """subscription_date should also be parsed as datetime."""
        assert pd.api.types.is_datetime64_any_dtype(raw_data["subscription_date"])

    def test_string_columns_all_object(self, raw_data):
        """All STRING_COLUMNS should be dtype object."""
        for col in STRING_COLUMNS:
            assert raw_data[col].dtype == "object", f"{col} should be object, got {raw_data[col].dtype}"

    def test_head_five_rows(self, raw_data):
        """AC-6: Sample of 5 rows available."""
        assert len(raw_data.head(5)) == 5

class TestValidateSchema:
    """Tests for validate_schema()."""

    def test_valid_schema_passes(self, raw_data):
        """No exception on real data."""
        validate_schema(raw_data)  # Should not raise

    def test_wrong_column_count_raises(self, raw_data):
        df_bad = raw_data.drop(columns=[raw_data.columns[0]])
        with pytest.raises(ValueError, match="Expected 34 columns"):
            validate_schema(df_bad)

    def test_card_code_wrong_dtype_raises(self, raw_data):
        df = raw_data.copy()
        df["anonymized_card_code"] = pd.to_numeric(df["anonymized_card_code"], errors="coerce")
        with pytest.raises(ValueError, match="anonymized_card_code dtype"):
            validate_schema(df)

    def test_date_column_wrong_dtype_raises(self, raw_data):
        df = raw_data.copy()
        df["transactionDate"] = df["transactionDate"].astype(str)
        with pytest.raises(ValueError, match="transactionDate dtype"):
            validate_schema(df)

    def test_prints_summary_on_success(self, raw_data, capsys):
        validate_schema(raw_data)
        captured = capsys.readouterr()
        assert "Schema validation passed" in captured.out
        assert "Rows:" in captured.out
        assert "Columns:" in captured.out
