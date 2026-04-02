"""Visualization module for Sephora customer segmentation EDA (Epic 2) + PCA (US-3.4)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from src.config import (
    FIGSIZE_BAR, FIGSIZE_SCATTER, FIGURE_DPI, OUTPUT_PATH,
    SEGMENT_COLORS, PALETTE_AXES, RANDOM_STATE,
)

FIGURES_DIR = os.path.join(OUTPUT_PATH, "figures")


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# US 2-1: Univariate Distribution Analysis
# ---------------------------------------------------------------------------

def plot_numerical_distributions(
    df: pd.DataFrame,
    features: list,
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram + KDE for each numerical feature in *features*."""
    _ensure_figures_dir()
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna()
        ax.hist(data, bins=50, density=True, alpha=0.6, edgecolor="white")
        try:
            data.plot.kde(ax=ax, color="crimson", linewidth=1.5)
        except Exception as e:
            print(f"Warning: Could not plot KDE for '{feat}': {e}")
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    features: list,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart for each categorical feature in *features*."""
    _ensure_figures_dir()
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        counts = df[feat].value_counts()
        counts.plot.bar(ax=ax, color=sns.color_palette("pastel", len(counts)), edgecolor="grey")
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def compute_summary_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Return summary stats table: mean, median, std, min, max, % missing, and nunique."""
    records = []
    for feat in features:
        col = df[feat]
        is_num = pd.api.types.is_numeric_dtype(col)
        records.append({
            "feature": feat,
            "mean": col.mean() if is_num else np.nan,
            "median": col.median() if is_num else np.nan,
            "std": col.std() if is_num else np.nan,
            "min": col.min() if is_num else np.nan,
            "max": col.max() if is_num else np.nan,
            "pct_missing": col.isnull().mean() * 100,
            "nunique": col.nunique(),
        })
    return pd.DataFrame(records).set_index("feature")


# ---------------------------------------------------------------------------
# US 2-2: Correlation & Redundancy Analysis
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Pearson correlation heatmap for all numeric columns."""
    _ensure_figures_dir()
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def get_high_correlation_pairs(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Return pairs with |r| > threshold from the upper triangle."""
    corr = df.select_dtypes("number").corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    high = pairs[pairs.abs() > threshold].reset_index()
    high.columns = ["Feature_A", "Feature_B", "r"]
    high = high.sort_values("r", key=abs, ascending=False).reset_index(drop=True)
    return high


# ---------------------------------------------------------------------------
# US 2-3: RFM Space Visualization
# ---------------------------------------------------------------------------

def plot_rfm_scatter(
    df: pd.DataFrame,
    save_dir: str | None = None,
) -> list[plt.Figure]:
    """2D scatter plots in RFM space, colored by loyalty_status (and RFM_Segment_ID if present)."""
    _ensure_figures_dir()
    save_dir = save_dir or FIGURES_DIR

    pairs = [
        ("recency_days", "frequency"),
        ("recency_days", "monetary_total"),
        ("frequency", "monetary_total"),
    ]
    log_cols = {"frequency", "monetary_total"}

    # Sample for readability
    plot_df = df.sample(min(10_000, len(df)), random_state=RANDOM_STATE) if len(df) > 50_000 else df.copy()

    color_sets = [("loyalty_status", "rfm_scatter_loyalty.png")]
    if "RFM_Segment_ID" in df.columns:
        color_sets.append(("RFM_Segment_ID", "rfm_scatter_rfmseg.png"))

    figs = []
    for color_col, fname in color_sets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        n_colors = plot_df[color_col].nunique()
        pal = list(SEGMENT_COLORS[:n_colors])
        for ax, (x, y) in zip(axes, pairs):
            sns.scatterplot(
                data=plot_df, x=x, y=y, hue=plot_df[color_col].astype(str),
                palette=pal,
                alpha=0.4, s=12, ax=ax, edgecolor="none",
            )
            if x in log_cols:
                ax.set_xscale("log", nonpositive="clip")
                ax.set_xlabel(f"{x} (log)")
            if y in log_cols:
                ax.set_yscale("log", nonpositive="clip")
                ax.set_ylabel(f"{y} (log)")
            ax.legend(fontsize=7, loc="best")

        fig.suptitle(f"RFM Space — colored by {color_col}", fontsize=14, y=1.02)
        fig.tight_layout()
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# US 2-4: Channel, Product & Brand Analysis
# ---------------------------------------------------------------------------

def plot_channel_product_overview(
    df_customers: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Three-panel figure: channel share bar, axis mix by channel, market mix by loyalty."""
    _ensure_figures_dir()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1 — Global channel share
    channel_means = df_customers[["store_ratio", "estore_ratio", "click_collect_ratio"]].mean()
    channel_means.index = ["Store", "E-Store", "Click & Collect"]
    channel_means.plot.bar(ax=axes[0], color=["#3B82F6", "#F59E0B", "#10B981"], edgecolor="grey")
    axes[0].set_title("Global Channel Share (mean ratio)")
    axes[0].set_ylabel("Mean Ratio")
    axes[0].tick_params(axis="x", rotation=0)

    # Panel 2 — Axis mix by dominant channel
    axe_cols = [
        "axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio",
        "axe_haircare_ratio", "axe_others_ratio",
    ]
    ax_colors = [PALETTE_AXES[col.split("_")[1].upper()] if col.split("_")[1] != "make" else PALETTE_AXES["MAKE UP"] for col in axe_cols]
    
    ax_mix = df_customers.groupby("dominant_channel")[axe_cols].mean()
    ax_mix.columns = ["Make up", "Skincare", "Fragrance", "Haircare", "Others"]
    ax_mix.plot(kind="bar", stacked=True, ax=axes[1], color=ax_colors)
    axes[1].set_title("Product Axis Mix by Channel")
    axes[1].set_ylabel("Mean Ratio")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].tick_params(axis="x", rotation=0)

    # Panel 3 — Market mix by loyalty status
    mkt_cols = [
        "market_selective_ratio", "market_exclusive_ratio",
        "market_sephora_ratio", "market_others_ratio",
    ]
    mkt_mix = df_customers.groupby("loyalty_status")[mkt_cols].mean()
    mkt_mix.columns = ["Selective", "Exclusive", "Sephora", "Others"]
    mkt_mix.plot(kind="bar", stacked=True, ax=axes[2], color=["#6366F1", "#EC4899", "#14B8A6", "#94A3B8"])
    axes[2].set_title("Market Tier Mix by Loyalty Status")
    axes[2].set_ylabel("Mean Ratio")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].tick_params(axis="x", rotation=0)

    fig.suptitle("Channel & Product Overview", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_top_brands(
    df_clean: pd.DataFrame,
    n: int = 20,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top *n* brands by total sales EUR."""
    _ensure_figures_dir()
    
    brand_col = "brand" if "brand" in df_clean.columns else "Marque_Desc"
    
    top = (
        df_clean.groupby(brand_col)["salesVatEUR"]
        .sum()
        .nlargest(n)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    top.plot.barh(ax=ax, color="#6366F1", edgecolor="grey")
    ax.set_xlabel("Total Sales (EUR)")
    ax.set_title(f"Top {n} Brands by Total Sales EUR")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_sales_by_demographics(
    df_customers: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of mean monetary_total by age_generation × gender."""
    _ensure_figures_dir()
    ct = df_customers.pivot_table(
        values="monetary_total",
        index="age_generation",
        columns="gender",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Mean Monetary Total by Age Generation × Gender")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# US 2-5: Outlier Analysis & Treatment Decision
# ---------------------------------------------------------------------------

OUTLIER_FEATURES = ["monetary_total", "frequency", "recency_days"]


def plot_outlier_boxplots(
    df: pd.DataFrame,
    features: list | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Side-by-side box plots for outlier inspection (default: monetary_total, frequency, recency_days)."""
    _ensure_figures_dir()
    features = features or OUTLIER_FEATURES
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        sns.boxplot(y=df[feat], ax=ax, color="#6366F1", width=0.4)
        ax.set_title(feat, fontsize=12)
    fig.suptitle("Outlier Box Plots", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def identify_extreme_customers(
    df: pd.DataFrame,
    col: str = "monetary_total",
    quantile: float = 0.99,
) -> tuple[pd.DataFrame, float]:
    """Return customers above *quantile* on *col* and the threshold value."""
    threshold = df[col].quantile(quantile)
    mask = df[col] > threshold
    extremes = df.loc[mask].copy()
    return extremes, threshold


def apply_winsorization(
    df: pd.DataFrame,
    cols: list | None = None,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """Cap *cols* at *quantile*, create `<col>_capped` columns and `is_outlier` flag.

    Returns a copy of the dataframe.
    """
    cols = cols or ["monetary_total", "frequency"]
    df = df.copy()

    outlier_mask = pd.Series(False, index=df.index)
    for col in cols:
        cap = df[col].quantile(quantile)
        outlier_mask = outlier_mask | (df[col] > cap)
        df[f"{col}_capped"] = df[col].clip(upper=cap)

    df["is_outlier"] = outlier_mask
    return df


# ---------------------------------------------------------------------------
# US 2-6: Loyalty & Lifecycle Analysis
# ---------------------------------------------------------------------------

LOYALTY_ORDER = ["No Fid", "BRONZE", "SILVER", "GOLD"]


def plot_loyalty_lifecycle(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Box plots of monetary_avg, frequency, recency_days grouped by loyalty_status."""
    _ensure_figures_dir()
    metrics = ["monetary_avg", "frequency", "recency_days"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, metrics):
        sns.boxplot(
            data=df, x="loyalty_status", y=col,
            order=LOYALTY_ORDER, ax=ax, palette="pastel",
        )
        ax.set_title(f"{col} by Loyalty Status", fontsize=12)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Loyalty & Lifecycle Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def compute_loyalty_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated stats per loyalty tier: mean monetary_avg, frequency, recency, tenure, new customer rate, and proportion."""
    agg = df.groupby("loyalty_status").agg(
        monetary_avg_mean=("monetary_avg", "mean"),
        frequency_mean=("frequency", "mean"),
        recency_days_mean=("recency_days", "mean"),
        subscription_tenure_days_mean=("subscription_tenure_days", "mean"),
        is_new_customer_rate=("is_new_customer", "mean"),
        count=("monetary_avg", "size"),
    )
    agg["proportion"] = agg["count"] / agg["count"].sum()
    return agg.reindex(LOYALTY_ORDER)


def plot_monthly_volume(
    df_clean: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart of monthly transaction volume (seasonality check)."""
    _ensure_figures_dir()
    monthly = df_clean.groupby(df_clean["transactionDate"].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()
    
    # get the dominant year for title
    years = monthly.index.year.value_counts()
    year_str = str(years.index[0]) if not years.empty else ""
    title_str = f"Monthly Transaction Volume ({year_str})" if year_str else "Monthly Transaction Volume"
    
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, color="#3B82F6")
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#3B82F6")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transaction Count")
    ax.set_title(title_str)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# US 3-4: PCA — Variance Analysis
# ---------------------------------------------------------------------------

def plot_pca_variance(
    pca: PCA,
    save_path: str | None = None,
) -> plt.Figure:
    """Cumulative explained variance curve with 80% and 90% threshold markers.

    Args:
        pca: Fitted PCA object.
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    _ensure_figures_dir()
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    n_components = len(cum_var)
    x = np.arange(1, n_components + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.plot(x, cum_var, marker="o", linewidth=2, color="#3B82F6", markersize=4)
    ax.fill_between(x, cum_var, alpha=0.15, color="#3B82F6")

    # 80% threshold
    n80 = int(np.searchsorted(cum_var, 80.0) + 1) if cum_var[-1] >= 80.0 else None
    ax.axhline(80, color="#F59E0B", linestyle="--", linewidth=1.2, label="80% threshold")
    if n80 is not None and n80 <= n_components:
        ax.axvline(n80, color="#F59E0B", linestyle=":", alpha=0.6)
        ax.annotate(f"{n80} PCs → {cum_var[n80-1]:.1f}%",
                    xy=(n80, cum_var[n80-1]), xytext=(n80 + 1, cum_var[n80-1] - 5),
                    fontsize=9, color="#F59E0B",
                    arrowprops=dict(arrowstyle="->", color="#F59E0B"))

    # 90% threshold
    n90 = int(np.searchsorted(cum_var, 90.0) + 1) if cum_var[-1] >= 90.0 else None
    ax.axhline(90, color="#EF4444", linestyle="--", linewidth=1.2, label="90% threshold")
    if n90 is not None and n90 <= n_components:
        ax.axvline(n90, color="#EF4444", linestyle=":", alpha=0.6)
        ax.annotate(f"{n90} PCs → {cum_var[n90-1]:.1f}%",
                    xy=(n90, cum_var[n90-1]), xytext=(n90 + 1, cum_var[n90-1] - 5),
                    fontsize=9, color="#EF4444",
                    arrowprops=dict(arrowstyle="->", color="#EF4444"))

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA — Cumulative Explained Variance")
    ax.legend(loc="lower right")
    ax.set_xlim(1, n_components)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_pca_loadings(
    pca: PCA,
    feature_names: list[str],
    top_n: int = 3,
    n_features_per_chart: int = 15,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar charts showing top feature loadings for the first *top_n* PCA components.

    Args:
        pca: Fitted PCA object.
        feature_names: Original feature names matching pca.components_ columns.
        top_n: Number of principal components to plot (default 3).
        n_features_per_chart: Number of top features to show per component.
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    _ensure_figures_dir()
    top_n = min(top_n, pca.n_components_)
    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 6))
    if top_n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        loadings = pd.Series(pca.components_[i], index=feature_names)
        top_abs = loadings.abs().nlargest(n_features_per_chart)
        top_loadings = loadings[top_abs.index].sort_values()

        colors = ["#EF4444" if v < 0 else "#3B82F6" for v in top_loadings]
        top_loadings.plot.barh(ax=ax, color=colors, edgecolor="grey")
        ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)", fontsize=12)
        ax.set_xlabel("Loading")
        ax.axvline(0, color="black", linewidth=0.5)

    fig.suptitle("PCA — Top Feature Loadings per Component", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# US 3-5: UMAP 2D Visualization
# ---------------------------------------------------------------------------

def plot_umap_2d(
    umap_df: pd.DataFrame,
    df_customers: pd.DataFrame,
    color_by: str,
    save_dir: str | None = None,
) -> plt.Figure:
    """Scatter plot of UMAP 2D embedding colored by a categorical column.

    Args:
        umap_df: DataFrame with columns ['umap_1', 'umap_2'].
        df_customers: Customer DataFrame containing the *color_by* column.
        color_by: Column name in df_customers to use for coloring.
        save_dir: Directory to save the figure. Defaults to FIGURES_DIR.

    Returns:
        matplotlib Figure.
    """
    _ensure_figures_dir()
    save_dir = save_dir or FIGURES_DIR

    plot_df = umap_df.copy()
    plot_df[color_by] = df_customers[color_by].reindex(plot_df.index).astype(str)

    n_colors = plot_df[color_by].nunique()
    pal = list(SEGMENT_COLORS[:n_colors]) if n_colors <= len(SEGMENT_COLORS) else list(sns.color_palette("tab20", n_colors))

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    sns.scatterplot(
        data=plot_df, x="umap_1", y="umap_2",
        hue=color_by, palette=pal,
        alpha=0.5, s=10, ax=ax, edgecolor="none",
    )
    ax.set_title(f"UMAP 2D — colored by {color_by}", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=7, loc="best", markerscale=2)
    fig.tight_layout()

    path = os.path.join(save_dir, f"umap_2d_{color_by}.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# US 4-1: K-Means — Optimal k Selection (Elbow Curves)
# ---------------------------------------------------------------------------

def plot_elbow_curves(
    metrics_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """3-panel elbow plot: inertia, silhouette, Davies-Bouldin vs. k.

    Args:
        metrics_df: DataFrame with columns k, inertia, silhouette, davies_bouldin.
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    _ensure_figures_dir()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    k = metrics_df["k"]

    # Panel 1 — Inertia (elbow method)
    axes[0].plot(k, metrics_df["inertia"], marker="o", linewidth=2, color="#3B82F6", markersize=4)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method — Inertia")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2 — Silhouette score (higher is better)
    axes[1].plot(k, metrics_df["silhouette"], marker="o", linewidth=2, color="#10B981", markersize=4)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs. k")
    axes[1].grid(axis="y", alpha=0.3)

    # Highlight best silhouette
    best_idx = metrics_df["silhouette"].idxmax()
    axes[1].axvline(metrics_df.loc[best_idx, "k"], color="#F59E0B", linestyle="--", alpha=0.7,
                    label=f"best k={int(metrics_df.loc[best_idx, 'k'])}")
    axes[1].legend(fontsize=9)

    # Panel 3 — Davies-Bouldin (lower is better)
    axes[2].plot(k, metrics_df["davies_bouldin"], marker="o", linewidth=2, color="#EF4444", markersize=4)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].set_title("Davies-Bouldin Index vs. k")
    axes[2].grid(axis="y", alpha=0.3)

    # Highlight best DB
    best_db_idx = metrics_df["davies_bouldin"].idxmin()
    axes[2].axvline(metrics_df.loc[best_db_idx, "k"], color="#F59E0B", linestyle="--", alpha=0.7,
                    label=f"best k={int(metrics_df.loc[best_db_idx, 'k'])}")
    axes[2].legend(fontsize=9)

    fig.suptitle("K-Means — Optimal k Selection", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_silhouette_diagram(
    X: pd.DataFrame | np.ndarray,
    k: int,
    save_path: str | None = None,
) -> plt.Figure:
    """Per-cluster silhouette blade diagram for a given k.

    Fits KMeans(k) on X and draws horizontal bar-like silhouette coefficients
    for every sample, grouped and colored by cluster.

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Number of clusters.
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    _ensure_figures_dir()
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    sample_silhouette_values = silhouette_samples(X, labels)
    from sklearn.metrics import silhouette_score as _sil_score
    avg_score = _sil_score(X, labels)

    fig, ax = plt.subplots(figsize=(10, max(6, k * 1.2)))
    y_lower = 10
    cmap = plt.cm.get_cmap("tab20", k)

    for i in range(k):
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_values,
            facecolor=cmap(i),
            edgecolor=cmap(i),
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=9, va="center")
        y_lower = y_upper + 10

    ax.axvline(avg_score, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean silhouette = {avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Diagram — k = {k}")
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# US 4-3: Agglomerative Hierarchical Clustering
# ---------------------------------------------------------------------------

def plot_dendrogram(
    X: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Hierarchical clustering dendrogram (truncated to last 30 merges).

    Uses scipy Ward linkage on a sample (≤10 000 rows for performance).

    Args:
        X: Feature matrix (n_samples, n_features).
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    _ensure_figures_dir()

    # Sample for performance — linkage is O(n²) memory
    X_sample = X.sample(min(10_000, len(X)), random_state=RANDOM_STATE) if len(X) > 10_000 else X

    Z = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (last 30 merges)")
    ax.set_xlabel("Sample index (or cluster size)")
    ax.set_ylabel("Ward linkage distance")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_umap_kmeans_vs_hclust(
    umap_df: pd.DataFrame,
    df_customers: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Side-by-side UMAP scatter: K-Means labels vs. Hierarchical labels.

    Args:
        umap_df: DataFrame with columns ['umap_1', 'umap_2'] (index aligned to df_customers).
        df_customers: Customer DataFrame with columns 'kmeans_label' and 'hclust_label'.
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    _ensure_figures_dir()

    plot_df = umap_df.copy()
    plot_df["kmeans_label"] = df_customers["kmeans_label"].reindex(plot_df.index).astype(str)
    plot_df["hclust_label"] = df_customers["hclust_label"].reindex(plot_df.index).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, title in zip(
        axes,
        ["kmeans_label", "hclust_label"],
        ["K-Means labels", "Hierarchical (Ward) labels"],
    ):
        n_colors = plot_df[col].nunique()
        pal = (
            list(SEGMENT_COLORS[:n_colors])
            if n_colors <= len(SEGMENT_COLORS)
            else list(sns.color_palette("tab20", n_colors))
        )
        sns.scatterplot(
            data=plot_df,
            x="umap_1",
            y="umap_2",
            hue=col,
            palette=pal,
            alpha=0.5,
            s=10,
            ax=ax,
            edgecolor="none",
        )
        ax.set_title(f"UMAP 2D — {title}", fontsize=12)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=7, loc="best", markerscale=2)

    fig.suptitle("K-Means vs. Hierarchical Clustering — UMAP Projection", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    else:
        default_path = os.path.join(FIGURES_DIR, "umap_kmeans_vs_hclust.png")
        fig.savefig(default_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig
