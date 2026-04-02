import pandas as pd
import numpy as np

df = pd.read_csv("data/BDD#7_Database_Albert_School_Sephora.csv", encoding='utf-8-sig',
                  dtype={'anonymized_card_code': str, 'anonymized_Ticket_ID': str, 'anonymized_first_purchase_id': str},
                  parse_dates=['transactionDate', 'first_purchase_dt', 'subscription_date'])

print("="*60)
print("SECTION 1: DATASET OVERVIEW")
print("="*60)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {df.shape[1]}")
print(f"Unique customers: {df['anonymized_card_code'].nunique():,}")
print(f"Unique tickets: {df['anonymized_Ticket_ID'].nunique():,}")
print(f"Columns: {list(df.columns)}")
print()
print("Dtypes:")
print(df.dtypes)
print()

# Date range
print(f"Transaction date range: {df['transactionDate'].min()} to {df['transactionDate'].max()}")
print(f"First purchase date range: {df['first_purchase_dt'].min()} to {df['first_purchase_dt'].max()}")
print()

# Rows per customer stats
rows_per_cust = df.groupby('anonymized_card_code').size()
print(f"Rows per customer: mean={rows_per_cust.mean():.1f}, median={rows_per_cust.median():.0f}, max={rows_per_cust.max()}")

# Tickets per customer stats
tickets_per_cust = df.groupby('anonymized_card_code')['anonymized_Ticket_ID'].nunique()
print(f"Tickets per customer: mean={tickets_per_cust.mean():.1f}, median={tickets_per_cust.median():.0f}, max={tickets_per_cust.max()}")

# Lines per ticket
lines_per_ticket = df.groupby('anonymized_Ticket_ID').size()
print(f"Lines per ticket: mean={lines_per_ticket.mean():.1f}, median={lines_per_ticket.median():.0f}, max={lines_per_ticket.max()}")

print()
print("="*60)
print("SECTION 2: DATA QUALITY")
print("="*60)

# Missing values
print("Missing values per column:")
mv = df.isnull().sum()
mv_pct = (mv / len(df) * 100).round(2)
for col in df.columns:
    if mv[col] > 0:
        print(f"  {col}: {mv[col]:,} ({mv_pct[col]}%)")
print(f"  Columns with NO missing: {(mv == 0).sum()} / {len(df.columns)}")
print()

# Typo in Axe_Desc
print("Axe_Desc value counts:")
print(df['Axe_Desc'].value_counts())
print()

# Age issues
print(f"Age == 0 (unknown): {(df['age'] == 0).sum():,} rows ({(df['age'] == 0).sum()/len(df)*100:.1f}%)")
age_valid = df[df['age'] > 0]['age']
print(f"Valid age range: {age_valid.min()} to {age_valid.max()}")
print(f"Age < 15 (suspect): {(df['age'].between(1, 14)).sum()} rows")
print()

# Gender
print("Gender value counts:")
print(df['gender'].value_counts())
print()

# Status (loyalty)
print("Status (loyalty) value counts:")
print(df['status'].value_counts())
print()

# Channel
print("Channel value counts:")
print(df['channel'].value_counts())
print()

# store_type_app
print("store_type_app value counts:")
print(df['store_type_app'].value_counts())
print()

# Country
print("Country value counts:")
print(df['countryIsoCode'].value_counts())
print()

# Click & Collect detection
cc_mask = (df['channel'] == 'estore') & (~df['store_type_app'].isin(['ESTORE', 'WEB', 'MOBILE', 'APP', 'CSC']))
print(f"Click & Collect rows: {cc_mask.sum():,} ({cc_mask.sum()/len(df)*100:.1f}%)")
print()

# Negative sales
print(f"Negative salesVatEUR rows: {(df['salesVatEUR'] < 0).sum():,} ({(df['salesVatEUR'] < 0).sum()/len(df)*100:.1f}%)")
print(f"Zero salesVatEUR rows: {(df['salesVatEUR'] == 0).sum():,}")
print()

print("="*60)
print("SECTION 3: KEY NUMBERS (MONETARY)")
print("="*60)
print(f"Total revenue (salesVatEUR): {df['salesVatEUR'].sum():,.2f} EUR")
print(f"Mean salesVatEUR per row: {df['salesVatEUR'].mean():.2f} EUR")
print(f"Median salesVatEUR per row: {df['salesVatEUR'].median():.2f} EUR")
print(f"Total discount: {df['discountEUR'].sum():,.2f} EUR")
print(f"Mean discount rate: {(df['discountEUR'].sum()/df['salesVatEUR'].sum()*100):.2f}%")
print()

# Per customer aggregation
cust = df.groupby('anonymized_card_code').agg(
    total_sales=('salesVatEUR', 'sum'),
    total_transactions=('anonymized_Ticket_ID', 'nunique'),
    total_lines=('salesVatEUR', 'count'),
    total_discount=('discountEUR', 'sum'),
    total_quantity=('quantity', 'sum'),
    last_purchase=('transactionDate', 'max'),
    first_purchase=('transactionDate', 'min'),
).reset_index()

ref_date = pd.Timestamp('2025-12-31')
cust['recency_days'] = (ref_date - cust['last_purchase']).dt.days
cust['avg_basket'] = cust['total_sales'] / cust['total_transactions']
cust['tenure_days'] = (cust['last_purchase'] - cust['first_purchase']).dt.days

print("PER-CUSTOMER STATS:")
print(f"Number of customers: {len(cust):,}")
for col in ['total_sales', 'total_transactions', 'total_lines', 'avg_basket', 'recency_days', 'total_quantity', 'total_discount', 'tenure_days']:
    desc = cust[col].describe()
    print(f"  {col}: mean={desc['mean']:.1f}, median={desc['50%']:.1f}, std={desc['std']:.1f}, min={desc['min']:.1f}, max={desc['max']:.1f}")

print()

# Channel mix per customer
channel_mix = df.groupby(['anonymized_card_code', 'channel']).size().unstack(fill_value=0)
if 'store' in channel_mix.columns and 'estore' in channel_mix.columns:
    channel_mix['total'] = channel_mix.sum(axis=1)
    channel_mix['store_pct'] = channel_mix['store'] / channel_mix['total'] * 100
    channel_mix['estore_pct'] = channel_mix['estore'] / channel_mix['total'] * 100
    print("Channel mix (% of rows per customer):")
    print(f"  Mean store %: {channel_mix['store_pct'].mean():.1f}%")
    print(f"  Mean estore %: {channel_mix['estore_pct'].mean():.1f}%")
    # Customers who are store-only vs estore-only vs omnichannel
    store_only = (channel_mix['estore'] == 0).sum()
    estore_only = (channel_mix['store'] == 0).sum()
    omni = len(channel_mix) - store_only - estore_only
    print(f"  Store-only customers: {store_only:,} ({store_only/len(channel_mix)*100:.1f}%)")
    print(f"  Estore-only customers: {estore_only:,} ({estore_only/len(channel_mix)*100:.1f}%)")
    print(f"  Omnichannel customers: {omni:,} ({omni/len(channel_mix)*100:.1f}%)")
print()

# Product axis distribution
print("="*60)
print("SECTION 4: PRODUCT & MARKET INSIGHTS")
print("="*60)
print("Revenue by Axe_Desc:")
axis_rev = df.groupby('Axe_Desc')['salesVatEUR'].sum().sort_values(ascending=False)
for ax, rev in axis_rev.items():
    print(f"  {ax}: {rev:,.0f} EUR ({rev/df['salesVatEUR'].sum()*100:.1f}%)")
print()

print("Revenue by Market_Desc:")
mkt_rev = df.groupby('Market_Desc')['salesVatEUR'].sum().sort_values(ascending=False)
for mk, rev in mkt_rev.items():
    print(f"  {mk}: {rev:,.0f} EUR ({rev/df['salesVatEUR'].sum()*100:.1f}%)")
print()

# Loyalty tier distribution per CUSTOMER (most recent status)
cust_status = df.sort_values('transactionDate').groupby('anonymized_card_code')['status'].last()
status_map = {1: 'No Fid', 2: 'BRONZE', 3: 'SILVER', 4: 'GOLD'}
cust_status_label = cust_status.map(status_map)
print("Customer distribution by loyalty tier (last known status):")
for s, c in cust_status_label.value_counts().sort_index().items():
    print(f"  {s}: {c:,} ({c/len(cust_status)*100:.1f}%)")
print()

# Age generation distribution  
print("Age generation distribution:")
cust_gen = df.groupby('anonymized_card_code')['age_generation'].first()
print(cust_gen.value_counts())
print()

# Gender distribution
cust_gender = df.groupby('anonymized_card_code')['gender'].first()
gender_map = {1: 'Men', 2: 'Women', 99999: 'Unknown'}
cust_gender_label = cust_gender.map(gender_map)
print("Customer gender distribution:")
print(cust_gender_label.value_counts())
print()

# Monthly revenue trend
print("="*60)
print("SECTION 5: TIME TRENDS")
print("="*60)
monthly = df.groupby(df['transactionDate'].dt.to_period('M')).agg(
    revenue=('salesVatEUR', 'sum'),
    transactions=('anonymized_Ticket_ID', 'nunique'),
    customers=('anonymized_card_code', 'nunique')
)
print("Monthly revenue, transactions, unique customers:")
for idx, row in monthly.iterrows():
    print(f"  {idx}: {row['revenue']:,.0f} EUR | {row['transactions']:,} txn | {row['customers']:,} cust")
print()

# RFM Segment distribution
print("="*60)
print("SECTION 6: EXISTING RFM SEGMENTS")
print("="*60)
print("RFM_Segment_ID value counts (transaction rows):")
print(df['RFM_Segment_ID'].value_counts().sort_index())
print()

# Number of unique brands
print(f"Number of unique brands: {df['Brand_Desc'].nunique()}")
print("Top 10 brands by revenue:")
brand_rev = df.groupby('Brand_Desc')['salesVatEUR'].sum().sort_values(ascending=False).head(10)
for b, r in brand_rev.items():
    print(f"  {b}: {r:,.0f} EUR")
print()

print("DONE.")
