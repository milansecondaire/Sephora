# src/profiling.py
import os
import pandas as pd
import numpy as np

NUMERICAL_KPIS = [
    'monetary_avg', 'frequency', 'avg_basket_size_eur', 'avg_units_per_basket',
    'recency_days', 'store_ratio', 'estore_ratio', 'click_collect_ratio',
    'axe_make_up_ratio', 'axe_skincare_ratio', 'axe_fragrance_ratio',
    'axe_haircare_ratio', 'axe_others_ratio',
    'discount_rate', 'monetary_total',
]

def compute_global_kpis(df: pd.DataFrame) -> dict:
    """Compute global mean KPIs across all customers.
    Returns a dict with KPI name → global mean value.
    """
    # Use only KPIs present in the dataframe to avoid KeyErrors
    valid_kpis = [kpi for kpi in NUMERICAL_KPIS if kpi in df.columns]
    global_kpis = df[valid_kpis].mean().to_dict()
    
    # Loyalty distribution
    if 'loyalty_status' in df.columns:
        global_kpis['loyalty_distribution'] = df['loyalty_status'].value_counts(normalize=True).to_dict()
    
    global_kpis['total_customers'] = len(df)
    
    if 'monetary_total' in df.columns:
        global_kpis['total_sales_eur'] = df['monetary_total'].sum()
        
    return global_kpis


def compute_cluster_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean KPIs per cluster. Returns DataFrame: rows=clusters, cols=KPIs + size."""
    valid_kpis = [kpi for kpi in NUMERICAL_KPIS if kpi in df.columns]
    cluster_kpis = df.groupby('cluster_id')[valid_kpis].mean()
    cluster_sizes = df.groupby('cluster_id').size()
    cluster_kpis.insert(0, 'n_customers', cluster_sizes)
    cluster_kpis.insert(1, 'pct_customers', cluster_kpis['n_customers'] / len(df) * 100)
    return cluster_kpis


def build_delta_table(cluster_kpis: pd.DataFrame, global_kpis: dict) -> pd.DataFrame:
    """Build long-format delta table: cluster × KPI × delta_abs × delta_pct.

    Clusters sorted by monetary_total descending.
    """
    rows = []
    for cluster_id, row in cluster_kpis.iterrows():
        for kpi in NUMERICAL_KPIS:
            global_val = global_kpis.get(kpi, np.nan)
            cluster_val = row[kpi]
            delta_abs = cluster_val - global_val
            delta_pct = ((cluster_val / global_val) - 1) * 100 if global_val != 0 else np.nan
            rows.append({
                'cluster_id': cluster_id,
                'kpi': kpi,
                'global_avg': round(global_val, 4),
                'cluster_value': round(cluster_val, 4),
                'delta_abs': round(delta_abs, 4),
                'delta_pct': round(delta_pct, 2),
            })
    delta_df = pd.DataFrame(rows)

    # Sort clusters by monetary_total descending
    cluster_order = cluster_kpis.sort_values('monetary_total', ascending=False).index.tolist()
    delta_df['cluster_id'] = pd.Categorical(delta_df['cluster_id'], categories=cluster_order, ordered=True)
    delta_df = delta_df.sort_values(['cluster_id', 'kpi']).reset_index(drop=True)

    return delta_df


def get_notable_deltas(delta_df: pd.DataFrame, threshold: float = 30.0) -> pd.DataFrame:
    """Return rows where |delta_pct| > threshold."""
    return delta_df[delta_df['delta_pct'].abs() > threshold].copy()


def export_delta_table_md(delta_df: pd.DataFrame, output_path: str) -> None:
    """Export delta table as formatted Markdown file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# KPI Delta Table — Clusters vs. Global Average\n\n")
        f.write(delta_df.to_markdown(index=False))


def compute_distinguishing_features(df: pd.DataFrame, features: list) -> dict:
    """Return top distinguishing features per cluster via Cohen's d.

    For each cluster, computes Cohen's d = (cluster_mean - rest_mean) / rest_std
    (comparing the cluster to the rest of the population)
    and returns a dict: cluster_id → DataFrame sorted by |Cohen's d| descending.
    """
    result = {}
    cluster_groups = df.groupby('cluster_id')
    
    for cluster_id, cluster_df in cluster_groups:
        rest_df = df[df['cluster_id'] != cluster_id]
        
        if rest_df.empty:
            cluster_mean = cluster_df[features].mean()
            cohens_d = pd.Series(0.0, index=features)
        else:
            cluster_mean = cluster_df[features].mean()
            rest_mean = rest_df[features].mean()
            rest_std = rest_df[features].std().fillna(0.0)
            
            cohens_d = (cluster_mean - rest_mean) / (rest_std + 1e-10)
            
        feat_df = cohens_d.abs().sort_values(ascending=False).reset_index()
        feat_df.columns = ['feature', 'cohens_d_abs']
        feat_df['cohens_d'] = cohens_d[feat_df['feature'].values].values
        result[cluster_id] = feat_df
    return result


def rank_by_clv(cluster_kpis: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters by monetary_total (estimated CLV) descending and assign tier.

    Returns a copy sorted by monetary_total desc with a 'clv_tier' column
    (Top / Mid / Low tertiles).
    """
    ranked = cluster_kpis.sort_values('monetary_total', ascending=False).copy()
    n = len(ranked)
    if n >= 3:
        # Use rank(method='first') to ensure unique bin edges for qcut
        ranked['clv_tier'] = pd.qcut(
            ranked['monetary_total'].rank(method='first'), q=3, labels=['Low', 'Mid', 'Top']
        )
    else:
        # Not enough clusters for tertiles — assign based on rank order
        tiers = ['Top', 'Mid', 'Low']
        ranked['clv_tier'] = [tiers[i] if i < len(tiers) else 'Low' for i in range(n)]
    return ranked


def identify_high_potential_segments(cluster_kpis: pd.DataFrame) -> dict:
    """Identify top 3 priority, high-potential, and investment-worthy segments.

    Returns dict with keys:
    - 'top3_priority': list of cluster IDs (top 3 by CLV)
    - 'high_potential': list of cluster IDs (above-median size AND above-median CLV)
    - 'investment_worthy': list of cluster IDs (above-median CLV, below-median size)
    """
    sorted_kpis = cluster_kpis.sort_values('monetary_total', ascending=False)
    top3 = sorted_kpis.index[:3].tolist()

    median_size = cluster_kpis['pct_customers'].median()
    median_clv = cluster_kpis['monetary_total'].median()

    high_potential = cluster_kpis[
        (cluster_kpis['pct_customers'] >= median_size) &
        (cluster_kpis['monetary_total'] >= median_clv)
    ].index.tolist()

    investment_worthy = cluster_kpis[
        (cluster_kpis['monetary_total'] >= median_clv) &
        (cluster_kpis['pct_customers'] < median_size)
    ].index.tolist()

    return {
        'top3_priority': top3,
        'high_potential': high_potential,
        'investment_worthy': investment_worthy,
    }
