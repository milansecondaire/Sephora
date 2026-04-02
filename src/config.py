# src/config.py
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# File paths
DATA_RAW_PATH = "data/BDD#7_Database_Albert_School_Sephora.csv"
DATA_PROCESSED_PATH = "data/processed/"
OUTPUT_PATH = "_bmad-output/implementation-artifacts/"

# Dates
RECENCY_REFERENCE_DATE = "2025-12-31"

# Clustering
K_RANGE = range(2, 31)

# --- Legacy feature list (kept for backward compatibility with old tests) ---
CLUSTERING_FEATURES = [
    "recency_days", "frequency", "monetary_total", "monetary_avg",
    "avg_basket_size_eur", "avg_units_per_basket", "discount_rate",
    "store_ratio", "estore_ratio", "click_collect_ratio",
    "axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio",
    "axe_haircare_ratio", "axe_others_ratio",
    "market_selective_ratio", "market_exclusive_ratio", "axis_diversity",
    "loyalty_numeric", "subscription_tenure_days", "is_new_customer",
]

# === Comprehensive Preprocessing Feature Groups ===

# Features to drop (zero variance, exact duplicates, raw dates, non-behavioral flags)
FEATURES_DROP = [
    "is_new_customer",              # constant=1 for all rows, zero variance
    "total_sales_eur",              # exact duplicate of monetary_total
    "total_transactions",           # exact duplicate of frequency
    "avg_sales_eur",                # exact duplicate of monetary_avg
    "salesVatEUR_first_purchase",   # exact duplicate of first_purchase_amount
    "first_purchase_channel",       # identical to channel_recruitment
    "monetary_total_capped",        # winsorized duplicate of monetary_total
    "frequency_capped",             # winsorized duplicate of frequency
    "loyalty_status",               # string version of loyalty_numeric
    "Axe_Desc_first_purchase",      # raw version — first_purchase_axe retained
    "last_purchase_date",           # raw date — recency_days retained
    "first_purchase_date",          # raw date — derivatives retained
    "subscription_date",            # raw date — subscription_tenure_days retained
    "is_outlier",                   # traceability flag, not a behavioral feature
]

# Continuous / numeric features → median imputation + StandardScaler
FEATURES_CONTINUOUS = [
    # RFM
    "recency_days", "frequency", "monetary_total", "monetary_avg",
    # Behavior
    "avg_basket_size_eur", "avg_units_per_basket", "discount_rate",
    "total_discount_eur", "total_quantity", "total_lines",
    # Channel ratios
    "store_ratio", "estore_ratio", "click_collect_ratio",
    # Product affinity ratios
    "axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio",
    "axe_haircare_ratio", "axe_others_ratio",
    # Market ratios
    "market_selective_ratio", "market_exclusive_ratio",
    "market_sephora_ratio", "market_others_ratio",
    # Diversity & counts
    "nb_unique_brands", "nb_unique_stores", "axis_diversity", "cc_transactions",
    # Demographics
    "age",
    # Lifecycle
    "subscription_tenure_days", "first_purchase_amount",
    # Ordinal (0=No Fid, 1=BRONZE, 2=SILVER, 3=GOLD)
    "loyalty_numeric",
]

# Low-cardinality categoricals → fill "Unknown" + One-Hot Encoding
FEATURES_ONEHOT = [
    "gender",              # Women, Men, Unknown (3)
    "dominant_channel",    # store, estore, click_collect (3)
    "dominant_axe",        # MAKE UP, SKINCARE, FRAGRANCE, HAIRCARE, OTHERS (5)
    "dominant_market",     # SELECTIVE, EXCLUSIVE, SEPHORA, OTHERS (4)
    "country",             # FR, LU, MC (3)
    "channel_recruitment", # store, estore + Unknown (3)
    "age_category",        # 15-25yo, 26-35yo, 46-60yo, m60yo + Unknown (5)
    "age_generation",      # genz, gena, geny, babyboomers + Unknown (5)
    "first_purchase_axe",  # cleaned to 5 primary axes + Unknown (6)
]

# High-cardinality categoricals → fill "Unknown" + Frequency Encoding
FEATURES_FREQUENCY = [
    "customer_city",       # ~12K unique cities → frequency = proxy for urbanization
]

# Visualization
SEGMENT_COLORS = plt.cm.tab20.colors  # up to 20 distinct segments
PALETTE_AXES = {
    "SKINCARE":   "#FF6B8A",
    "MAKE UP":    "#A855F7",
    "FRAGRANCE":  "#F59E0B",
    "HAIRCARE":   "#10B981",
    "OTHERS":     "#6B7280",
}

# Domain Maps
ESTORE_TYPE_VALUES = {"ESTORE", "WEB", "MOBILE", "APP", "CSC"}
GENDER_MAP = {1: "Men", 2: "Women", 99999: "Unknown"}
STATUS_MAP = {1: "No Fid", 2: "BRONZE", 3: "SILVER", 4: "GOLD"}

# Figure defaults
FIGSIZE_BAR = (12, 6)
FIGSIZE_SCATTER = (10, 8)
FIGURE_DPI = 300
