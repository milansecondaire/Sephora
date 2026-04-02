"""Generate all figures for Deliverable 1 - Data Mastery."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
FIGDIR = "_bmad-output/implementation-artifacts/figures"

df = pd.read_csv("data/BDD#7_Database_Albert_School_Sephora.csv", encoding='utf-8-sig',
                  dtype={'anonymized_card_code': str, 'anonymized_Ticket_ID': str},
                  parse_dates=['transactionDate'])

# ----- Fix typo -----
df['Axe_Desc'] = df['Axe_Desc'].replace('MAEK UP', 'MAKE UP')

# ----- Customer-level aggregation -----
cust = df.groupby('anonymized_card_code').agg(
    total_sales=('salesVatEUR', 'sum'),
    total_txn=('anonymized_Ticket_ID', 'nunique'),
    total_lines=('salesVatEUR', 'count'),
    total_discount=('discountEUR', 'sum'),
    total_qty=('quantity', 'sum'),
    last_purchase=('transactionDate', 'max'),
    first_purchase=('transactionDate', 'min'),
).reset_index()

ref = pd.Timestamp('2025-12-31')
cust['recency'] = (ref - cust['last_purchase']).dt.days
cust['avg_basket'] = cust['total_sales'] / cust['total_txn']
cust['tenure'] = (cust['last_purchase'] - cust['first_purchase']).dt.days

# Demographics
demo = df.groupby('anonymized_card_code').agg(
    gender=('gender', 'first'),
    age=('age', 'first'),
    status=('status', 'last'),
    age_gen=('age_generation', 'first'),
).reset_index()
cust = cust.merge(demo, on='anonymized_card_code')

# Channel share
ch = df.groupby(['anonymized_card_code', 'channel'])['salesVatEUR'].sum().unstack(fill_value=0)
ch['estore_share'] = ch.get('estore', 0) / (ch.sum(axis=1)).replace(0, np.nan)
cust = cust.merge(ch[['estore_share']], left_on='anonymized_card_code', right_index=True, how='left')
cust['estore_share'] = cust['estore_share'].fillna(0)

# Product axis dominant
axis_spend = df.groupby(['anonymized_card_code', 'Axe_Desc'])['salesVatEUR'].sum().reset_index()
idx_max = axis_spend.groupby('anonymized_card_code')['salesVatEUR'].idxmax()
dom_axis = axis_spend.loc[idx_max, ['anonymized_card_code', 'Axe_Desc']].rename(columns={'Axe_Desc': 'dominant_axis'})
cust = cust.merge(dom_axis, on='anonymized_card_code', how='left')

status_map = {2: 'BRONZE', 3: 'SILVER', 4: 'GOLD'}
cust['status_label'] = cust['status'].map(status_map)

# =====================================================================
# FIGURE 1: Monthly revenue trend with transaction count
# =====================================================================
monthly = df.groupby(df['transactionDate'].dt.to_period('M')).agg(
    revenue=('salesVatEUR', 'sum'),
    txn=('anonymized_Ticket_ID', 'nunique'),
)
monthly.index = monthly.index.to_timestamp()

fig, ax1 = plt.subplots(figsize=(5.5, 2.5))
color1, color2 = '#2c3e50', '#e74c3c'
ax1.bar(monthly.index, monthly['revenue']/1e3, width=20, color=color1, alpha=0.7, label='Revenue')
ax1.set_ylabel('Revenue (k\\u20ac)', fontsize=8)
ax1.tick_params(axis='both', labelsize=7)
ax1.set_xlabel('')
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly['txn']/1e3, color=color2, marker='o', markersize=3, linewidth=1.2, label='Transactions')
ax2.set_ylabel('Transactions (k)', fontsize=8, color=color2)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=7)
fig.legend(loc='upper left', fontsize=7, bbox_to_anchor=(0.12, 0.95))
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
plt.title('Monthly Revenue and Transaction Volume (2025)', fontsize=9, pad=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/d1_fig1_monthly_trend.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIGDIR}/d1_fig1_monthly_trend.png", bbox_inches='tight', dpi=300)
plt.close()
print("Fig 1 done")

# =====================================================================
# FIGURE 2: Customer spend distribution (log scale) with loyalty tiers
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 2.5))
positive_cust = cust[cust['total_sales'] > 0]
for tier, color in [('BRONZE', '#cd7f32'), ('SILVER', '#C0C0C0'), ('GOLD', '#FFD700')]:
    subset = positive_cust[positive_cust['status_label'] == tier]
    ax.hist(np.log10(subset['total_sales']), bins=60, alpha=0.55, label=f'{tier} (n={len(subset):,})', color=color, edgecolor='none')
ax.set_xlabel('Total Customer Spend (\\u20ac, log\\u2081\\u2080 scale)', fontsize=8)
ax.set_ylabel('Number of Customers', fontsize=8)
ax.tick_params(axis='both', labelsize=7)
# Custom x ticks
xticks = [0, 1, 2, 3, 4]
ax.set_xticks(xticks)
ax.set_xticklabels(['1', '10', '100', '1k', '10k'], fontsize=7)
ax.legend(fontsize=7)
plt.title('Customer Lifetime Spend Distribution by Loyalty Tier', fontsize=9, pad=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/d1_fig2_spend_distribution.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIGDIR}/d1_fig2_spend_distribution.png", bbox_inches='tight', dpi=300)
plt.close()
print("Fig 2 done")

# =====================================================================
# FIGURE 3: RFM Scatter (Recency vs Frequency, size=monetary)
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 2.8))
sample = cust[(cust['total_sales'] > 0)].sample(n=min(5000, len(cust)), random_state=42)
sizes = np.clip(sample['total_sales'] / sample['total_sales'].quantile(0.95) * 40, 3, 80)
scatter = ax.scatter(sample['recency'], sample['total_txn'],
                     c=np.log1p(sample['total_sales']), cmap='viridis',
                     s=sizes, alpha=0.4, edgecolors='none')
ax.set_xlabel('Recency (days since last purchase)', fontsize=8)
ax.set_ylabel('Frequency (# transactions)', fontsize=8)
ax.tick_params(axis='both', labelsize=7)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('log(1+Spend)', fontsize=7)
cbar.ax.tick_params(labelsize=6)
plt.title('RFM Landscape: Recency vs Frequency (color = Spend)', fontsize=9, pad=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/d1_fig3_rfm_scatter.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIGDIR}/d1_fig3_rfm_scatter.png", bbox_inches='tight', dpi=300)
plt.close()
print("Fig 3 done")

# =====================================================================
# FIGURE 4: Product axis revenue share + channel breakdown (stacked bar)
# =====================================================================
axis_channel = df.groupby(['Axe_Desc', 'channel'])['salesVatEUR'].sum().unstack(fill_value=0)
axis_channel = axis_channel.loc[axis_channel.sum(axis=1).sort_values(ascending=True).index]
# Remove NOT SPECIFIED
axis_channel = axis_channel.drop('NOT SPECIFIED', errors='ignore')

fig, ax = plt.subplots(figsize=(5.5, 2.5))
axis_channel_k = axis_channel / 1e3
bars_store = ax.barh(axis_channel_k.index, axis_channel_k.get('store', 0), color='#3498db', label='Store')
bars_estore = ax.barh(axis_channel_k.index, axis_channel_k.get('estore', 0),
                       left=axis_channel_k.get('store', 0), color='#e67e22', label='Estore')
ax.set_xlabel('Revenue (k\\u20ac)', fontsize=8)
ax.tick_params(axis='both', labelsize=7)
ax.legend(fontsize=7, loc='lower right')
plt.title('Revenue by Product Axis and Channel', fontsize=9, pad=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/d1_fig4_axis_channel.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIGDIR}/d1_fig4_axis_channel.png", bbox_inches='tight', dpi=300)
plt.close()
print("Fig 4 done")

# =====================================================================
# FIGURE 5: Loyalty tier x average basket x estore share (bubble)
# =====================================================================
tier_stats = cust.groupby('status_label').agg(
    avg_basket_mean=('avg_basket', 'median'),
    estore_share_mean=('estore_share', 'mean'),
    count=('anonymized_card_code', 'count'),
    total_sales_mean=('total_sales', 'median'),
).reindex(['BRONZE', 'SILVER', 'GOLD'])

fig, ax = plt.subplots(figsize=(4.5, 2.8))
colors = {'BRONZE': '#cd7f32', 'SILVER': '#808080', 'GOLD': '#FFD700'}
for tier in tier_stats.index:
    row = tier_stats.loc[tier]
    ax.scatter(row['avg_basket_mean'], row['estore_share_mean']*100,
               s=row['count']/80, color=colors[tier], alpha=0.7, edgecolors='k', linewidth=0.5)
    ax.annotate(f"{tier}\n(n={int(row['count']):,})", (row['avg_basket_mean'], row['estore_share_mean']*100),
                fontsize=7, ha='center', va='bottom', xytext=(0, 8), textcoords='offset points')
ax.set_xlabel('Median Basket (\\u20ac)', fontsize=8)
ax.set_ylabel('Mean Estore Share (%)', fontsize=8)
ax.tick_params(axis='both', labelsize=7)
plt.title('Loyalty Tier Profiles: Basket Size vs Digital Adoption', fontsize=9, pad=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/d1_fig5_tier_profile.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIGDIR}/d1_fig5_tier_profile.png", bbox_inches='tight', dpi=300)
plt.close()
print("Fig 5 done")

print("ALL FIGURES DONE.")
