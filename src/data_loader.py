import pandas as pd
from pathlib import Path
from src.config import DATA_RAW_PATH

EXPECTED_COLUMNS = 34
STRING_COLUMNS = ["anonymized_card_code", "anonymized_Ticket_ID", "anonymized_first_purchase_id"]
DATE_COLUMNS = ["transactionDate", "first_purchase_dt"]


def load_raw_data(path: str = DATA_RAW_PATH) -> pd.DataFrame:
    """Load the Sephora raw transaction CSV with correct dtypes and date parsing.

    Raises FileNotFoundError if path does not exist.
    Returns a pd.DataFrame — never modifies the source file.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    dtype_overrides = {col: str for col in STRING_COLUMNS}
    dtype_overrides["age_category"] = str
    dtype_overrides["age_generation"] = str

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=dtype_overrides,
        parse_dates=DATE_COLUMNS,
    )

    # subscription_date contains " UTC" suffix and mixed formats (fractional seconds) — parse manually
    df["subscription_date"] = pd.to_datetime(df["subscription_date"], utc=True, format="mixed")
    df["subscription_date"] = df["subscription_date"].dt.tz_localize(None)

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that df matches the expected schema.

    Raises ValueError with a descriptive message if:
    - Column count != EXPECTED_COLUMNS
    - anonymized_card_code is not dtype object (str)
    - transactionDate / first_purchase_dt are not datetime64

    Prints a summary on success.
    """
    actual_cols = df.shape[1]
    if actual_cols != EXPECTED_COLUMNS:
        missing = set(range(EXPECTED_COLUMNS)) - set(range(actual_cols))
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, got {actual_cols}. "
            f"Columns: {list(df.columns)}"
        )

    if df["anonymized_card_code"].dtype != "object":
        raise ValueError(
            f"anonymized_card_code dtype is {df['anonymized_card_code'].dtype}, expected object (str)"
        )

    for col in ["transactionDate", "first_purchase_dt", "subscription_date"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"{col} dtype is {df[col].dtype}, expected datetime64")

    print("✅ Schema validation passed")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {actual_cols}")
    print(f"   anonymized_card_code dtype: {df['anonymized_card_code'].dtype}")
    print(f"   transactionDate dtype: {df['transactionDate'].dtype}")
    print(f"   first_purchase_dt dtype: {df['first_purchase_dt'].dtype}")
    print(f"   subscription_date dtype: {df['subscription_date'].dtype}")
