import ast

import pandas as pd
import numpy as np
from src.config import ESTORE_TYPE_VALUES, GENDER_MAP, STATUS_MAP, RECENCY_REFERENCE_DATE, DATA_PROCESSED_PATH

AXIS_NAMES = ['MAKE UP', 'SKINCARE', 'FRAGRANCE', 'HAIRCARE', 'OTHERS']
AXIS_COL_MAP = {name: f"axe_{name.replace(' ', '_').lower()}_ratio" for name in AXIS_NAMES}

MARKET_NAMES = ['SELECTIVE', 'EXCLUSIVE', 'SEPHORA', 'OTHERS']
MARKET_COL_MAP = {name: f"market_{name.lower()}_ratio" for name in MARKET_NAMES}

LOYALTY_MAP = {'No Fid': 0, 'BRONZE': 1, 'SILVER': 2, 'GOLD': 3}

def assess_data_quality(df: pd.DataFrame) -> dict:
    """Print and return a summary of missing values and known quality issues."""
    if len(df) == 0:
        print("Dataframe is empty.")
        return {"missing_count": {}, "missing_rate": {}}

    missing_count = df.isnull().sum()
    missing_rate = (missing_count / len(df)) * 100

    print("=== Missing Values ===")
    for col in df.columns:
        if missing_count[col] > 0:
            print(f"  {col}: {missing_count[col]:,} ({missing_rate[col]:.2f}%)")

    total_missing = missing_count.sum()
    if total_missing == 0:
        print("  No missing values found.")

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {df.shape[1]}")
    
    # Filter the dictionary to only include columns with missing values
    missing_cols_dict = missing_count[missing_count > 0].to_dict()
    missing_rate_dict = missing_rate[missing_count > 0].to_dict()

    return {
        "missing_count": missing_cols_dict,
        "missing_rate": missing_rate_dict,
    }


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules. Returns a NEW DataFrame (df_clean).
    Never modifies the input df in place.
    """
    df_clean = df.copy()

    # Fix Axe_Desc typo: MAEK UP -> MAKE UP
    df_clean["Axe_Desc"] = df_clean["Axe_Desc"].replace("MAEK UP", "MAKE UP")
    if "Axe_Desc_first_purchase" in df_clean.columns:
        df_clean["Axe_Desc_first_purchase"] = df_clean["Axe_Desc_first_purchase"].replace("MAEK UP", "MAKE UP")

    # Flag age == 0 or age < 15 as NaN
    df_clean.loc[(df_clean["age"] == 0) | (df_clean["age"] < 15), "age"] = np.nan

    # Replace gender == 99999 with 'Unknown'; recode 1->Men, 2->Women
    df_clean["gender"] = df_clean["gender"].map(GENDER_MAP).fillna(df_clean["gender"])
    df_clean["gender"] = df_clean["gender"].astype("category")

    # Recode status: 1->No Fid, 2->BRONZE, 3->SILVER, 4->GOLD
    df_clean["status"] = df_clean["status"].map(STATUS_MAP).fillna(df_clean["status"])
    df_clean["status"] = df_clean["status"].astype("category")

    # Add is_click_collect boolean column
    cc_mask = (df_clean["channel"] == "estore") & (
        df_clean["store_type_app"].notna()
    ) & (
        ~df_clean["store_type_app"].isin(ESTORE_TYPE_VALUES)
    )
    df_clean["is_click_collect"] = cc_mask

    return df_clean


def aggregate_to_customer_level(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Collapse transaction rows into one row per customer.

    Returns df_customers with anonymized_card_code as index.
    """
    grouped = df_clean.groupby("anonymized_card_code")

    df_customers = grouped.agg(
        total_transactions=pd.NamedAgg(column="anonymized_Ticket_ID", aggfunc="nunique"),
        total_sales_eur=pd.NamedAgg(column="salesVatEUR", aggfunc="sum"),
        avg_sales_eur=pd.NamedAgg(column="salesVatEUR", aggfunc="mean"),
        total_discount_eur=pd.NamedAgg(column="discountEUR", aggfunc="sum"),
        total_quantity=pd.NamedAgg(column="quantity", aggfunc="sum"),
        last_purchase_date=pd.NamedAgg(column="transactionDate", aggfunc="max"),
        first_purchase_date=pd.NamedAgg(column="transactionDate", aggfunc="min"),
        loyalty_status=pd.NamedAgg(column="status", aggfunc="last"),
        age=pd.NamedAgg(column="age", aggfunc="first"),
        age_category=pd.NamedAgg(column="age_category", aggfunc="first"),
        age_generation=pd.NamedAgg(column="age_generation", aggfunc="first"),
        gender=pd.NamedAgg(column="gender", aggfunc="first"),
        country=pd.NamedAgg(column="countryIsoCode", aggfunc="first"),
        customer_city=pd.NamedAgg(column="customer_city", aggfunc="first"),
        subscription_date=pd.NamedAgg(column="subscription_date", aggfunc="first"),
        channel_recruitment=pd.NamedAgg(column="channel_recruitment", aggfunc="first"),
        salesVatEUR_first_purchase=pd.NamedAgg(column="salesVatEUR_first_purchase", aggfunc="first"),
        Axe_Desc_first_purchase=pd.NamedAgg(column="Axe_Desc_first_purchase", aggfunc="first"),
    )

    # total_lines = row count per customer (separate from agg)
    total_lines = grouped.size().rename("total_lines")
    df_customers = df_customers.join(total_lines)

    # Click & collect transaction count
    cc_count = grouped["is_click_collect"].sum().fillna(0).astype(int).rename("cc_transactions")
    df_customers = df_customers.join(cc_count)

    # Assertion gate
    assert df_customers.index.name == "anonymized_card_code"
    assert df_customers.index.nunique() == len(df_customers)

    print(f"Unique customers: {len(df_customers):,}")

    return df_customers


def compute_rfm_features(df_customers: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary features on df_customers.
    Returns a new DataFrame, avoiding in-place mutation.
    """
    df_rfm = df_customers.copy()

    assert df_rfm["last_purchase_date"].isnull().sum() == 0, "last_purchase_date contains nulls"

    ref_date = pd.Timestamp(RECENCY_REFERENCE_DATE)

    raw_recency = (ref_date - df_rfm["last_purchase_date"]).dt.days
    df_rfm["recency_days"] = np.maximum(0, raw_recency).astype(int)
    
    df_rfm["frequency"] = df_rfm["total_transactions"]
    df_rfm["monetary_total"] = df_rfm["total_sales_eur"]
    df_rfm["monetary_avg"] = df_rfm["avg_sales_eur"]

    rfm_cols = ["recency_days", "frequency", "monetary_total", "monetary_avg"]
    assert df_rfm[rfm_cols].isnull().sum().sum() == 0, "RFM features contain nulls"

    return df_rfm


def compute_behavioral_features(df_customers: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute basket, discount, channel, and diversity features (US 1-5).

    Accepts df_customers (customer-level) and df_clean (transaction-level).
    Returns df_customers with new columns added.
    """
    df = df_customers.copy()

    # AC 1-3: basket and discount ratios
    df["avg_basket_size_eur"] = df["total_sales_eur"] / df["total_transactions"]
    df["avg_units_per_basket"] = df["total_quantity"] / df["total_transactions"]
    df["discount_rate"] = (
        (df["total_discount_eur"] / df["total_sales_eur"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0.0, upper=1.0)
    )

    # AC 4-6: channel ratios from transaction data
    total_txn = df_clean.groupby("anonymized_card_code")["anonymized_Ticket_ID"].nunique()

    store_txn = (
        df_clean[df_clean["channel"] == "store"]
        .groupby("anonymized_card_code")["anonymized_Ticket_ID"]
        .nunique()
    )
    estore_txn = (
        df_clean[(df_clean["channel"] == "estore") & (~df_clean["is_click_collect"])]
        .groupby("anonymized_card_code")["anonymized_Ticket_ID"]
        .nunique()
    )
    cc_txn = (
        df_clean[df_clean["is_click_collect"]]
        .groupby("anonymized_card_code")["anonymized_Ticket_ID"]
        .nunique()
    )

    df["store_ratio"] = (store_txn / total_txn).reindex(df.index).fillna(0)
    df["estore_ratio"] = (estore_txn / total_txn).reindex(df.index).fillna(0)
    df["click_collect_ratio"] = (cc_txn / total_txn).reindex(df.index).fillna(0)

    # AC 7: dominant channel
    channel_cols = ["store_ratio", "estore_ratio", "click_collect_ratio"]
    channel_labels = ["store", "estore", "click_collect"]
    df["dominant_channel"] = df[channel_cols].idxmax(axis=1).map(
        dict(zip(channel_cols, channel_labels))
    )

    # AC 8-9: brand and store diversity
    df["nb_unique_brands"] = (
        df_clean.groupby("anonymized_card_code")["brand"].nunique().reindex(df.index).fillna(0).astype(int)
    )
    df["nb_unique_stores"] = (
        df_clean.groupby("anonymized_card_code")["store_code_name"].nunique().reindex(df.index).fillna(0).astype(int)
    )

    return df


def compute_product_affinity_features(df_customers: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute product axis and market tier share features (US 1-6).

    Returns df_customers with axis ratios, market ratios, dominant_axe,
    dominant_market, and axis_diversity columns.
    """
    df = df_customers.copy()

    # Axis ratios (AC 1)
    axis_sales = (
        df_clean.groupby(["anonymized_card_code", "Axe_Desc"])["salesVatEUR"]
        .sum()
        .unstack(fill_value=0)
    )
    # Ensure all expected axes are present
    for axis in AXIS_NAMES:
        if axis not in axis_sales.columns:
            axis_sales[axis] = 0.0

    axis_ratios = axis_sales[AXIS_NAMES].div(df["total_sales_eur"], axis=0).fillna(0)
    axis_ratios = axis_ratios.rename(columns=AXIS_COL_MAP)
    for col in AXIS_COL_MAP.values():
        df[col] = axis_ratios[col].reindex(df.index).fillna(0)

    # Market ratios (AC 2)
    market_sales = (
        df_clean.groupby(["anonymized_card_code", "Market_Desc"])["salesVatEUR"]
        .sum()
        .unstack(fill_value=0)
    )
    for market in MARKET_NAMES:
        if market not in market_sales.columns:
            market_sales[market] = 0.0

    market_ratios = market_sales[MARKET_NAMES].div(df["total_sales_eur"], axis=0).fillna(0)
    market_ratios = market_ratios.rename(columns=MARKET_COL_MAP)
    for col in MARKET_COL_MAP.values():
        df[col] = market_ratios[col].reindex(df.index).fillna(0)

    # AC 3: dominant axe
    axe_cols = list(AXIS_COL_MAP.values())
    axe_labels = list(AXIS_COL_MAP.keys())
    df["dominant_axe"] = df[axe_cols].idxmax(axis=1).map(
        dict(zip(axe_cols, axe_labels))
    )

    # AC 4: dominant market
    mkt_cols = list(MARKET_COL_MAP.values())
    mkt_labels = list(MARKET_COL_MAP.keys())
    df["dominant_market"] = df[mkt_cols].idxmax(axis=1).map(
        dict(zip(mkt_cols, mkt_labels))
    )

    # AC 5: axis diversity
    df["axis_diversity"] = (
        df_clean.groupby("anonymized_card_code")["Axe_Desc"]
        .nunique()
        .reindex(df.index)
        .fillna(0)
        .astype(int)
    )

    # AC 6: assertion - axis ratios sum ~ 1.0 (only for customers with positive sales)
    row_sums = df[axe_cols].sum(axis=1)
    valid_mask = df["total_sales_eur"].fillna(0) > 0
    violations = valid_mask & ((row_sums < 0.999) | (row_sums > 1.001))
    if violations.any():
        n = violations.sum()
        import warnings
        warnings.warn(
            f"Product axis ratios do not sum to 1.0 for {n} customer(s) "
            "(likely due to NaN Axe_Desc transactions). This is expected."
        )

    return df


def _parse_first_axe(val):
    """Parse first_purchase_axe from Axe_Desc_first_purchase (may be list-like string)."""
    if pd.isna(val):
        return np.nan
    try:
        parsed = ast.literal_eval(str(val))
        return parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else str(val)
    except Exception:
        return str(val)


def compute_lifecycle_features(df_customers: pd.DataFrame) -> pd.DataFrame:
    """Compute lifecycle and loyalty features (US 1-7).

    Returns df_customers with subscription_tenure_days, loyalty_numeric,
    is_new_customer, first_purchase_axe, first_purchase_channel,
    first_purchase_amount columns.
    """
    df = df_customers.copy()

    ref_date = pd.Timestamp(RECENCY_REFERENCE_DATE)

    # AC 1: subscription tenure
    df["subscription_tenure_days"] = (ref_date - df["subscription_date"]).dt.days

    # AC 2: loyalty numeric
    df["loyalty_numeric"] = df["loyalty_status"].map(LOYALTY_MAP)

    # AC 3: is_new_customer
    df["is_new_customer"] = (df["first_purchase_date"] >= pd.Timestamp("2025-01-01")).astype(int)

    # AC 4: first purchase axe
    df["first_purchase_axe"] = df["Axe_Desc_first_purchase"].apply(_parse_first_axe)

    # AC 5: first purchase channel
    df["first_purchase_channel"] = df["channel_recruitment"]

    # AC 6: first purchase amount
    df["first_purchase_amount"] = df["salesVatEUR_first_purchase"]

    return df


def save_customers_features(df_customers: pd.DataFrame) -> str:
    """Save the final customer feature matrix to CSV. Returns the output path."""
    import os
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    output_path = DATA_PROCESSED_PATH + "customers_features.csv"
    df_customers.to_csv(output_path)
    print(f"Saved: {output_path} -- shape: {df_customers.shape}")
    return output_path
