# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES_DROP, FEATURES_CONTINUOUS, FEATURES_ONEHOT, FEATURES_FREQUENCY,
    RANDOM_STATE,
)


def impute_features(X: pd.DataFrame, missing_threshold: float = 0.30) -> pd.DataFrame:
    """Impute missing values in the clustering feature matrix.

    - Prints missing rate per feature.
    - Numerical columns with missing rate < threshold: median imputation.
    - Numerical columns with missing rate >= threshold: flagged (printed), dropped from X.
    - Categorical columns: filled with 'Unknown'.
    - Asserts zero NaN in output.
    Returns imputed DataFrame.
    """
    X_imputed = X.copy()

    # --- AC-1: Print missing rate per feature ---
    missing_rates = X_imputed.isnull().mean()
    cols_with_missing_rates = missing_rates[missing_rates > 0]
    print("=== Missing Rate per Feature ===")
    if len(cols_with_missing_rates) == 0:
        print("  No missing values found.")
    else:
        for col, rate in cols_with_missing_rates.items():
            count = X_imputed[col].isnull().sum()
            print(f"  {col}: {count} missing ({rate * 100:.2f}%)")

    # Ensure types are identified correctly
    num_cols = X_imputed.select_dtypes(include=[np.number]).columns

    # --- AC-2: Identify high-missing numerical features ---
    num_missing_rates = X_imputed[num_cols].isnull().mean()
    high_missing = num_missing_rates[num_missing_rates >= missing_threshold]
    if len(high_missing) > 0:
        print(f"\nWARNING: Numerical features with >={missing_threshold * 100:.0f}% missing (excluded from clustering):")
        for col, rate in high_missing.items():
            print(f"  {col}: {rate * 100:.1f}%")
        X_imputed = X_imputed.drop(columns=high_missing.index)

    # --- AC-3: Categorical columns → 'Unknown' ---
    cat_cols = X_imputed.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if isinstance(X_imputed[col].dtype, pd.CategoricalDtype):
            if "Unknown" not in X_imputed[col].cat.categories:
                X_imputed[col] = X_imputed[col].cat.add_categories("Unknown")
        X_imputed[col] = X_imputed[col].fillna("Unknown")

    # --- AC-2 & AC-4: Numerical columns → median imputation ---
    num_cols_remaining = X_imputed.select_dtypes(include=[np.number]).columns
    for col in num_cols_remaining:
        if X_imputed[col].isnull().any():
            median_val = X_imputed[col].median()
            X_imputed[col] = X_imputed[col].fillna(median_val)

    # --- AC-5: Assert zero NaN ---
    assert X_imputed.isnull().sum().sum() == 0, "Imputation incomplete: NaN values remain"
    print(f"\nX_imputed shape: {X_imputed.shape} — zero NaN confirmed")

    return X_imputed


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Scale features using StandardScaler. Returns a DataFrame preserving column names and index."""
    # Replace infinite values (caused by upstream division by zero in product/channel ratios)
    # with 0.0, assuming if divisor was 0, the ratio property itself is effectively 0.
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled_arr, index=X.index, columns=X.columns)
    print(f"X_scaled shape: {X_scaled.shape}")
    return X_scaled


# === Comprehensive preprocessing pipeline ===

KNOWN_AXES = {"MAKE UP", "SKINCARE", "FRAGRANCE", "HAIRCARE", "OTHERS"}


def _clean_primary_axe(val):
    """Extract and normalize the primary axis from first_purchase_axe.

    Handles pipe-separated multi-label values (e.g. 'SKINCARE|SKINCARE')
    and the 'MAEK UP' typo.
    """
    if pd.isna(val):
        return np.nan
    primary = str(val).split("|")[0].strip()
    primary = primary.replace("MAEK UP", "MAKE UP")
    return primary if primary in KNOWN_AXES else "OTHERS"


def preprocess_for_clustering(df_customers: pd.DataFrame, return_transformers: bool = False):
    """Comprehensive preprocessing pipeline for clustering.

    1. Drops truly useless features (zero variance, exact duplicates, raw dates)
    2. Creates missing indicators for high-missing features
    3. Cleans first_purchase_axe (simplifies to primary axis)
    4. Cleans infinite values in ratio columns
    5. Imputes missing values per type (median for numeric, 'Unknown' for categorical)
    6. Frequency-encodes high-cardinality categoricals (customer_city)
    7. One-hot encodes low-cardinality categoricals
    8. Applies StandardScaler on all resulting features

    Args:
        df_customers (pd.DataFrame): Raw customer features data.
        return_transformers (bool): If True, returns a tuple (df, transformers_dict). Default False.

    Returns:
        pd.DataFrame (or tuple): A fully preprocessed DataFrame ready for clustering.
    """
    df = df_customers.copy()
    transformers = {}
    
    # --- AC-1: Print missing rate per feature ---
    missing_rates = df.isnull().mean()
    cols_with_missing_rates = missing_rates[missing_rates > 0]
    print("=== Missing Rate per Feature ===")
    if len(cols_with_missing_rates) == 0:
        print("  No missing values found.")
    else:
        for col, rate in cols_with_missing_rates.items():
            count = df[col].isnull().sum()
            print(f"  {col}: {count} missing ({rate * 100:.2f}%)")

    # --- Step 1: Clean first_purchase_axe → simplify to primary axis ---
    if "first_purchase_axe" in df.columns:
        df["first_purchase_axe"] = df["first_purchase_axe"].apply(_clean_primary_axe)

    # --- Step 2: Create missing indicators (before imputation) ---
    if "age" in df.columns:
        df["has_age_info"] = df["age"].notna().astype(int)
    if "channel_recruitment" in df.columns:
        df["has_first_purchase_info"] = df["channel_recruitment"].notna().astype(int)

    # --- Step 3: Drop useless features ---
    cols_to_drop = [c for c in FEATURES_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} useless features: {cols_to_drop}")

    # --- Step 4: Clean infinite values in ratio columns ---
    ratio_cols = [c for c in df.columns if c.endswith("_ratio")]
    for col in ratio_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            print(f"  Cleaned {n_inf} inf values in {col}")

    # --- Step 5: Impute missing values ---
    transformers["median_imputation"] = {}
    # Continuous → median
    continuous_in_df = [c for c in FEATURES_CONTINUOUS if c in df.columns]
    for col in continuous_in_df:
        n_miss = df[col].isnull().sum()
        median_val = df[col].median()
        transformers["median_imputation"][col] = median_val
        if n_miss > 0:
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {col}: {n_miss} NaN → median ({median_val:.2f})")

    # Categorical → 'Unknown'
    cat_cols = [c for c in FEATURES_ONEHOT + FEATURES_FREQUENCY if c in df.columns]
    for col in cat_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            df[col] = df[col].fillna("Unknown")
            print(f"  Imputed {col}: {n_miss} NaN → 'Unknown'")

    # --- Step 6: Frequency encoding (high-cardinality categoricals) ---
    transformers["frequency_encoding"] = {}
    for col in FEATURES_FREQUENCY:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            transformers["frequency_encoding"][col] = freq_map
            new_col = col + "_freq"
            df[new_col] = df[col].map(freq_map).fillna(0.0)
            print(f"  Frequency-encoded {col} ({df[col].nunique()} categories) → {new_col}")
            df = df.drop(columns=[col])

    # --- Step 7: One-hot encoding (low-cardinality categoricals) ---
    transformers["ohe"] = None
    ohe_cols = [c for c in FEATURES_ONEHOT if c in df.columns]
    if ohe_cols:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_arr = ohe.fit_transform(df[ohe_cols])
        ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_feature_names, index=df.index)
        n_ohe_created = df_ohe.shape[1]
        
        df = df.drop(columns=ohe_cols)
        df = pd.concat([df, df_ohe], axis=1)
        transformers["ohe"] = ohe
    else:
        n_ohe_created = 0
    print(f"  One-hot encoded {len(ohe_cols)} features → {n_ohe_created} dummy columns")

    # --- Step 8: Ensure all columns are numeric ---
    bool_cols = df.select_dtypes(include=[bool]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  WARNING: Dropping unhandled non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # --- Step 9: Assert no NaN ---
    assert df.isnull().sum().sum() == 0, "Imputation incomplete: NaN values remain"

    # --- Step 10: StandardScaler on all features ---
    scaler = StandardScaler()
    arr = scaler.fit_transform(df)
    df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    
    transformers["scaler"] = scaler

    print(f"\n=== Preprocessing Complete ===")
    print(f"  Final shape: {df.shape}")
    print(f"  Continuous features: {len(continuous_in_df)}")
    print(f"  One-hot encoded: {len(ohe_cols)} → {n_ohe_created} dummies" if 'n_ohe_created' in locals() else "")
    print(f"  Frequency encoded: {len(FEATURES_FREQUENCY)}")
    print(f"  Missing indicators: 2 (has_age_info, has_first_purchase_info)")
    means = df.mean()
    stds = df.std(ddof=0)
    print(f"  Mean range: [{means.min():.6f}, {means.max():.6f}] ≈ 0 ✓")
    print(f"  Std range:  [{stds.min():.6f}, {stds.max():.6f}] ≈ 1 ✓")

    if return_transformers:
        return df, transformers
    return df


def apply_pca(X: pd.DataFrame, n_components: int | float = None) -> tuple[pd.DataFrame, PCA]:
    """Fit PCA on scaled feature matrix.

    Args:
        X: Scaled feature DataFrame (rows = customers, cols = features).
        n_components: Number of components. If None, uses min(n_features, 30).

    Returns:
        Tuple of (X_pca DataFrame with columns PC1…PCn, fitted PCA object).
    """
    n = n_components if n_components is not None else min(X.shape[1], 50)
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    components = pca.fit_transform(X)
    col_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    return pd.DataFrame(components, index=X.index, columns=col_names), pca


def apply_umap(X: pd.DataFrame, n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
    """Fit UMAP and return 2D embedding DataFrame.

    Args:
        X: Scaled/PCA feature DataFrame (rows = customers, cols = features).
        n_neighbors: UMAP n_neighbors parameter (default 15).
        min_dist: UMAP min_dist parameter (default 0.1).

    Returns:
        DataFrame with columns ['umap_1', 'umap_2'], preserving X.index.
    """
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("umap-learn is not installed. Please install it using: pip install umap-learn")

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(X)
    return pd.DataFrame(embedding, index=X.index, columns=["umap_1", "umap_2"])
