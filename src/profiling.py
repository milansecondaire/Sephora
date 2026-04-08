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

    For each cluster, computes Cohen's d = (cluster_mean - global_mean) / global_std
    and returns a dict: cluster_id → DataFrame sorted by |Cohen's d| descending.
    """
    global_mean = df[features].mean()
    global_std = df[features].std()
    result = {}
    for cluster_id in df['cluster_id'].unique():
        cluster_df = df[df['cluster_id'] == cluster_id]
        cluster_mean = cluster_df[features].mean()
        cohens_d = (cluster_mean - global_mean) / (global_std + 1e-10)
        feat_df = cohens_d.abs().sort_values(ascending=False).reset_index()
        feat_df.columns = ['feature', 'cohens_d_abs']
        feat_df['cohens_d'] = cohens_d[feat_df['feature'].values].values
        result[cluster_id] = feat_df
    return result
