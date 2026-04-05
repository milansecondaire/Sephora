"""Generate figures for Deliverable 2."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

OUT = "_bmad-output/implementation-artifacts/figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("data/processed/customers_features_clustered.csv")

# ── Figure 1: Per-cluster KPI radar / heatmap delta ──
global_means = {}
kpis = ['monetary_total','frequency','recency_days','avg_basket_size_eur','discount_rate']
kpi_labels = ['Monetary (€)','Frequency','Recency (d)','Basket (€)','Discount %']
for c in kpis:
    global_means[c] = df[c].mean()

clusters_to_show = [c for c in sorted(df['kmeans_label'].unique()) if c != 4]  # exclude micro-cluster 4
per_cluster = df[df['kmeans_label'].isin(clusters_to_show)].groupby('kmeans_label')[kpis].mean()
delta_pct = ((per_cluster - pd.Series(global_means)) / pd.Series(global_means) * 100).round(1)

fig, ax = plt.subplots(figsize=(10, 4.5))
im = sns.heatmap(delta_pct.T, annot=True, fmt=".0f", cmap="RdYlGn", center=0, 
                  linewidths=0.5, ax=ax, cbar_kws={"label": "% vs. Global Avg"})
ax.set_yticklabels(kpi_labels, rotation=0, fontsize=9)
ax.set_xlabel("Cluster", fontsize=10)
ax.set_title("KPI Delta vs. Global Average (% deviation)", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/d2_fig1_kpi_delta.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/d2_fig1_kpi_delta.png", dpi=300, bbox_inches="tight")
print("Fig1 saved.")
plt.close()

# ── Figure 2: Cluster composition bar (product affinity + channel) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Product affinity
axe_cols = ['axe_make_up_ratio','axe_skincare_ratio','axe_fragrance_ratio','axe_haircare_ratio','axe_others_ratio']
axe_labels = ['Make Up','Skincare','Fragrance','Haircare','Others']
axe_colors = ['#A855F7','#FF6B8A','#F59E0B','#10B981','#6B7280']
per_cluster_axe = df[df['kmeans_label'].isin(clusters_to_show)].groupby('kmeans_label')[axe_cols].mean()
per_cluster_axe.columns = axe_labels
per_cluster_axe.plot(kind='bar', stacked=True, ax=ax1, color=axe_colors, edgecolor='white')
ax1.set_title("Product Affinity Mix", fontsize=11, fontweight="bold")
ax1.set_ylabel("Share")
ax1.set_xlabel("Cluster")
ax1.legend(fontsize=7, loc='upper right')
ax1.tick_params(axis='x', rotation=0)

# Channel mix
ch_cols = ['store_ratio','estore_ratio','click_collect_ratio']
ch_labels = ['Store','E-store','Click&Collect']
ch_colors = ['#4C72B0','#DD8452','#55A868']
per_cluster_ch = df[df['kmeans_label'].isin(clusters_to_show)].groupby('kmeans_label')[ch_cols].mean()
per_cluster_ch.columns = ch_labels
per_cluster_ch.plot(kind='bar', stacked=True, ax=ax2, color=ch_colors, edgecolor='white')
ax2.set_title("Channel Mix", fontsize=11, fontweight="bold")
ax2.set_ylabel("Share")
ax2.set_xlabel("Cluster")
ax2.legend(fontsize=7, loc='upper right')
ax2.tick_params(axis='x', rotation=0)

fig.suptitle("Cluster Profiles — Product & Channel", fontsize=13, y=1.02) 
fig.tight_layout()
fig.savefig(f"{OUT}/d2_fig2_cluster_profiles.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/d2_fig2_cluster_profiles.png", dpi=300, bbox_inches="tight")
print("Fig2 saved.")
plt.close()

# ── Figure 3: Elbow + Silhouette compact (reuse existing elbow data) ──
from src.clustering import evaluate_kmeans_k_range
from src.preprocessing import preprocess_for_clustering
X_scaled = preprocess_for_clustering(df)
print("Running k-range evaluation...")
metrics = evaluate_kmeans_k_range(X_scaled, k_range=range(2, 16))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
ax1.plot(metrics['k'], metrics['inertia'], 'b-o', markersize=4)
ax1.axvline(x=9, color='orange', linestyle='--', label='k=9 (selected)')
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Method", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8)

ax2.plot(metrics['k'], metrics['silhouette'], 'g-o', markersize=4)
ax2.axvline(x=9, color='orange', linestyle='--', label='k=9')
ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT}/d2_fig3_elbow_silhouette.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/d2_fig3_elbow_silhouette.png", dpi=300, bbox_inches="tight")
print("Fig3 saved.")
plt.close()

print("All D2 figures generated.")
