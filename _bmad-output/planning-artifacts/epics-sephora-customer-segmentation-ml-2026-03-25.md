---
project: 'Edram — Sephora Customer Segmentation ML'
author: 'Milan'
date: '2026-03-25'
version: '1.0'
prd: 'prd-sephora-customer-segmentation-ml-2026-03-25.md'
---

# Epics & User Stories  
## Advanced Customer Segmentation & Persona Creation — Sephora (Use Case 2)

---

## Global Definition of Done

A story is **Done** when:
- [ ] Code runs end-to-end without errors
- [ ] All output cells are visible and interpretable in the notebook
- [ ] Analytical choices are explained in a Markdown cell immediately above/below the code
- [ ] Results are coherent with domain expectations (sanity-checked)
- [ ] Any data quality issue encountered is documented
- [ ] Random seeds are fixed (`random_state=42`) where applicable

---

## Epic Map

| Epic | Title | Goal |
|---|---|---|
| E1 | Data Foundation | Clean, aggregated, customer-level feature matrix ready for ML |
| E2 | Exploratory Data Analysis | Full understanding of the data before modeling |
| E3 | Feature Engineering & Preprocessing | Optimal ML-ready feature set |
| E4 | Clustering Models | Multi-algorithm evaluation and final model selection |
| E5 | Segment Profiling & Quantification | Statistical characterization of each cluster vs. global average |
| E6 | Personas & Marketing Recommendations | Actionable persona cards and campaign guidance |

---

## E1 — Data Foundation

> **Epic Goal:** Produce a clean, validated, customer-level feature matrix from the raw transaction-level CSV — the single source of truth for all downstream analysis and modeling.

**Value Statement:**  
*As a Data Analyst, I need a reliable, de-duplicated, customer-level dataset so that every analytical and modeling result rests on a verified, reproducible foundation.*

**Dependencies:** None — this is the entry point of the pipeline.

---

### US-1.1 — Load & Validate Raw Dataset

**As a** Data Analyst,  
**I want to** load the raw CSV and perform an initial quality check,  
**so that** I know exactly what I'm working with before any transformation.

**Acceptance Criteria:**
- [ ] File loads without encoding error (use `encoding='utf-8-sig'`)
- [ ] Total row count and column count displayed (expected: 34 columns)
- [ ] Each column's dtype inferred and printed (`df.dtypes`)
- [ ] `anonymized_card_code` parsed as **string** (not float) to avoid precision loss from scientific notation
- [ ] `transactionDate` and `first_purchase_dt` parsed as `datetime` objects
- [ ] Sample of 5 rows displayed for visual inspection

**Technical Notes:**
- Use `pd.read_csv(..., dtype={'anonymized_card_code': str, 'anonymized_Ticket_ID': str, 'anonymized_first_purchase_id': str})`
- Apply `parse_dates=['transactionDate', 'first_purchase_dt', 'subscription_date']`

---

### US-1.2 — Data Quality Assessment & Cleaning

**As a** Data Analyst,  
**I want to** identify and fix all known data quality issues,  
**so that** downstream features are computed on reliable, consistent data.

**Acceptance Criteria:**
- [ ] Missing value count and rate per column displayed (`df.isnull().sum()`)
- [ ] Typo corrected: `Axe_Desc == 'MAEK UP'` → `'MAKE UP'`
- [ ] `age == 0` or age < 15 flagged as missing (`np.nan`) — do not drop the row
- [ ] `gender == 99999` replaced with `'Unknown'`
- [ ] `gender` recoded: `1 → 'Men'`, `2 → 'Women'`
- [ ] `status` recoded: `1 → 'No Fid'`, `2 → 'BRONZE'`, `3 → 'SILVER'`, `4 → 'GOLD'`
- [ ] Click & Collect orders identified: `channel == 'estore'` AND `store_type_app == 'STORE'` → add column `is_click_collect = True`
- [ ] Markdown cell summarizes all transformations applied and rows affected

**Technical Notes:**
- Create a clean copy: `df_clean = df.copy()`; never mutate the original loaded frame
- The C&C detection rule: `(df['channel'] == 'estore') & (~df['store_type_app'].isin(['ESTORE', 'WEB', 'MOBILE', 'APP', 'CSC']))`

---

### US-1.3 — Customer-Level Aggregation

**As a** Data Analyst,  
**I want to** collapse transaction-level rows into one row per customer,  
**so that** each observation in the feature matrix represents a unique individual.

**Acceptance Criteria:**
- [ ] Output dataframe has exactly one row per `anonymized_card_code`
- [ ] Number of unique customers printed
- [ ] The following aggregations computed per customer:

| Output Column | Source | Aggregation |
|---|---|---|
| `total_transactions` | `anonymized_Ticket_ID` | `nunique` |
| `total_lines` | rows | `count` |
| `total_sales_eur` | `salesVatEUR` | `sum` |
| `avg_sales_eur` | `salesVatEUR` | `mean` |
| `total_discount_eur` | `discountEUR` | `sum` |
| `total_quantity` | `quantity` | `sum` |
| `last_purchase_date` | `transactionDate` | `max` |
| `first_purchase_date` | `transactionDate` | `min` (or `first_purchase_dt`) |
| `loyalty_status` | `status` | `last` (most recent status) |
| `age` | `age` | `first` (stable per customer) |
| `age_category` | `age_category` | `first` |
| `age_generation` | `age_generation` | `first` |
| `gender` | `gender` | `first` |
| `country` | `countryIsoCode` | `first` |
| `customer_city` | `customer_city` | `first` |
| `subscription_date` | `subscription_date` | `first` |
| `channel_recruitment` | `channel_recruitment` | `first` |
| First purchase fields | `salesVatEUR_first_purchase`, etc. | `first` |

- [ ] Aggregation logic documented in Markdown cell

**Technical Notes:**
- Use `groupby('anonymized_card_code').agg({...})` with a defined agg dict
- For list-type first-purchase columns (`Axe_Desc_first_purchase`, etc.), take `first` — they are the same across all rows of the same customer

---

### US-1.4 — RFM Feature Computation

**As a** Data Analyst,  
**I want to** compute Recency, Frequency, and Monetary features per customer,  
**so that** the backbone of behavioral segmentation is established.

**Acceptance Criteria:**
- [ ] Reference date set to `2025-12-31`
- [ ] `recency_days` = (2025-12-31 − `last_purchase_date`).days; integer, ≥ 0
- [ ] `frequency` = `total_transactions` (number of unique tickets)
- [ ] `monetary_total` = `total_sales_eur` (sum of all purchases)
- [ ] `monetary_avg` = `avg_sales_eur` (mean basket in EUR)
- [ ] All four features have no null values
- [ ] Summary statistics printed (`df[['recency_days','frequency','monetary_total','monetary_avg']].describe()`)

---

### US-1.5 — Behavioral & Channel Feature Computation

**As a** Data Analyst,  
**I want to** compute purchase behavior and channel preference features,  
**so that** the clustering captures shopping patterns beyond pure RFM.

**Acceptance Criteria:**
- [ ] `avg_basket_size_eur` = `total_sales_eur` / `total_transactions`
- [ ] `avg_units_per_basket` = `total_quantity` / `total_transactions`
- [ ] `discount_rate` = `total_discount_eur` / `total_sales_eur` (capped at 1.0; 0 where no discount)
- [ ] `store_ratio` = share of transactions with `channel == 'store'` (excluding C&C)
- [ ] `estore_ratio` = share of `channel == 'estore'` (excluding C&C)
- [ ] `click_collect_ratio` = share of C&C transactions
- [ ] `dominant_channel` = argmax of (store_ratio, estore_ratio, click_collect_ratio)
- [ ] `nb_unique_brands` = number of distinct brands purchased
- [ ] `nb_unique_stores` = number of distinct `store_code_name` visited

---

### US-1.6 — Product Affinity Feature Computation

**As a** Data Analyst,  
**I want to** compute product axis and market tier share features per customer,  
**so that** the clustering captures product preferences.

**Acceptance Criteria:**
- [ ] For each of `['MAKE UP', 'SKINCARE', 'FRAGRANCE', 'HAIRCARE', 'OTHERS']`:  
  `axe_{name}_ratio` = share of `salesVatEUR` in that axis / `total_sales_eur`
- [ ] For each of `['SELECTIVE', 'EXCLUSIVE', 'SEPHORA', 'OTHERS']`:  
  `market_{name}_ratio` = share of `salesVatEUR` in that market
- [ ] `dominant_axe` = axis with highest spend share
- [ ] `dominant_market` = market tier with highest spend share
- [ ] `axis_diversity` = number of distinct axes purchased (1–5)
- [ ] All ratio columns sum to 1.0 ± 0.001 per customer (assert checked)

---

### US-1.7 — Lifecycle Feature Computation

**As a** Data Analyst,  
**I want to** compute customer lifecycle and loyalty features,  
**so that** the clustering can distinguish new, active, and mature customers.

**Acceptance Criteria:**
- [ ] `subscription_tenure_days` = (2025-12-31 − `subscription_date`).days; NaN where missing
- [ ] `loyalty_numeric` = ordinal encoding: No Fid=0, BRONZE=1, SILVER=2, GOLD=3
- [ ] `is_new_customer` = 1 if `first_purchase_date` ≥ 2025-01-01 (first purchase within 2025), else 0
- [ ] `first_purchase_axe` = dominant axis from `Axe_Desc_first_purchase` (parse list string, take first element)
- [ ] `first_purchase_channel` = `channel_recruitment` (already scalar)
- [ ] `first_purchase_amount` = `salesVatEUR_first_purchase`

---

## E2 — Exploratory Data Analysis

> **Epic Goal:** Build a complete statistical picture of the customer base — distributions, correlations, behavioral patterns — before any modeling, to validate assumptions and detect potential issues.

**Value Statement:**  
*As a Marketing Manager, I want to understand the existing customer landscape so that modeling choices are grounded in business reality.*

**Dependencies:** E1 fully complete (customer-level feature matrix available).

---

### US-2.1 — Univariate Distribution Analysis

**As a** Data Analyst,  
**I want to** visualize the distribution of every key feature,  
**so that** I can identify skewness, outliers, and encoding issues.

**Acceptance Criteria:**
- [ ] Histogram + KDE for each numerical feature: recency, frequency, monetary_total, monetary_avg, avg_basket_size, discount_rate, subscription_tenure_days
- [ ] Bar chart for each categorical feature: loyalty_status, gender, dominant_channel, dominant_axe, dominant_market, country, age_generation
- [ ] Summary stats table (mean, median, std, min, max, % missing) for all features
- [ ] Date range of `transactionDate` confirmed and printed (expected: Jan–Dec 2025)
- [ ] Number of unique customers printed (answers OQ-4)

---

### US-2.2 — Correlation & Redundancy Analysis

**As a** Data Analyst,  
**I want to** measure correlations between features,  
**so that** I can remove redundant variables before clustering.

**Acceptance Criteria:**
- [ ] Pearson correlation heatmap for all numerical features
- [ ] Pairs with |r| > 0.85 listed explicitly with a decision: keep / drop / transform
- [ ] Spearman correlation computed for ordinal features
- [ ] At minimum document the expected high correlation: `monetary_total` ↔ `monetary_avg` ↔ `avg_basket_size`

---

### US-2.3 — RFM Space Visualization

**As a** Marketing Manager,  
**I want to** see the customer population in RFM space,  
**so that** I can develop an intuition for natural groupings.

**Acceptance Criteria:**
- [ ] 2D scatter plots: Recency vs. Frequency, Recency vs. Monetary, Frequency vs. Monetary
- [ ] Points colored by existing `RFM_Segment_ID` to orient the reader
- [ ] Points colored by `loyalty_status` as a second view
- [ ] Log-scale applied where distributions are heavily right-skewed
- [ ] Observations about natural cluster visibility documented in Markdown

---

### US-2.4 — Channel, Product & Brand Analysis

**As a** Marketing Manager,  
**I want to** understand channel mix and product affinities at the population level,  
**so that** I can set marketing expectations for segment output.

**Acceptance Criteria:**
- [ ] Pie/bar chart: store vs. estore vs. Click & Collect share
- [ ] Stacked bar: product axis mix by channel
- [ ] Stacked bar: market tier mix (SELECTIVE / EXCLUSIVE / SEPHORA) by loyalty status
- [ ] Top 20 brands by total sales EUR (horizontal bar chart)
- [ ] Sales EUR by `age_generation` and `gender`
- [ ] Monthly transaction volume chart (seasonality check)

---

### US-2.5 — Outlier Analysis & Treatment Decision

**As a** Data Analyst,  
**I want to** identify extreme customers (very high spenders, very high frequency),  
**so that** I can decide whether to include, cap, or exclude them from clustering.

**Acceptance Criteria:**
- [ ] Box plots for `monetary_total`, `frequency`, `recency_days`
- [ ] Customers beyond 99th percentile on `monetary_total` listed with their key stats
- [ ] Decision documented: cap at 99th percentile (Winsorization) OR keep OR separate analysis
- [ ] Chosen strategy applied and a column `is_outlier` created for traceability

---

### US-2.6 — Loyalty & Lifecycle Analysis

**As a** CRM Manager,  
**I want to** understand how loyalty tiers correlate with purchase behavior,  
**so that** I can interpret future segment loyalty distributions.

**Acceptance Criteria:**
- [ ] Box plots of `monetary_avg`, `frequency`, `recency_days` by `loyalty_status` (No Fid / BRONZE / SILVER / GOLD)
- [ ] Proportion of customers per tier printed
- [ ] Mean `subscription_tenure_days` per status
- [ ] `is_new_customer` rate per loyalty tier
- [ ] Key insight summarized in Markdown: e.g., "GOLD customers spend Nx more on average than No Fid"

---

## E3 — Feature Engineering & Preprocessing

> **Epic Goal:** Produce the final ML-ready feature matrix: selected, scaled, and optionally dimension-reduced features that maximize cluster quality.

**Value Statement:**  
*As a Data Scientist, I want an optimal, properly scaled feature set so that clustering algorithms are not distorted by scale differences or noise.*

**Dependencies:** E1 complete; E2 insights inform feature selection decisions.

---

### US-3.1 — Feature Audit & Selection (REVISED)

**As a** Data Scientist,  
**I want to** audit ALL features in the customer matrix and only discard truly useless ones,  
**so that** the model preserves maximum behavioral information.

**Acceptance Criteria:**
- [ ] Every feature from `customers_features.csv` classified as: **keep**, **drop** (with rationale), or **transform** (encoding type)
- [ ] Only drop: zero-variance features, exact duplicate columns, raw date columns, and traceability flags
- [ ] Define 4 feature groups in `src/config.py`: `FEATURES_DROP`, `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT`, `FEATURES_FREQUENCY`
- [ ] At least one feature from each of the 6 families: RFM, Behavior, Product, Channel, Sociodemographic, Lifecycle
- [ ] Markdown decision table with every feature and its fate

**Decision Table (53 → 40 retained features):**

| Dropped (13 features) | Reason |
|---|---|
| `is_new_customer` | Constant=1 (zero variance) |
| `total_sales_eur`, `avg_sales_eur` | Exact duplicates of `monetary_total`, `monetary_avg` |
| `salesVatEUR_first_purchase` | Exact duplicate of `first_purchase_amount` |
| `first_purchase_channel` | Exact duplicate of `channel_recruitment` |
| `monetary_total_capped`, `frequency_capped` | Winsorized duplicates |
| `loyalty_status` | String version of `loyalty_numeric` |
| `Axe_Desc_first_purchase` | Raw version — `first_purchase_axe` retained |
| `last_purchase_date`, `first_purchase_date`, `subscription_date` | Raw dates — derivatives retained |
| `is_outlier` | Traceability flag, not behavioral |

---

### US-3.2 — Imputation & Missing Indicators (REVISED)

**As a** Data Scientist,  
**I want to** handle missing values with type-appropriate strategies AND create missing indicators for high-missingness features,  
**so that** missingness patterns become informative rather than destructive.

**Acceptance Criteria:**
- [ ] Missing rate per feature printed
- [ ] Continuous features with NaN → **median imputation** (age: 13.4%, subscription_tenure_days: 0.8%, first_purchase_amount: 62.7%)
- [ ] Categorical features with NaN → **`'Unknown'` fill** (channel_recruitment: 62.7%, age_category: 30.5%, age_generation: 32.5%, first_purchase_axe: 62.7%, customer_city: 5.2%)
- [ ] **Missing indicators created** before imputation: `has_age_info` (binary), `has_first_purchase_info` (binary)
- [ ] Infinite values in ratio columns cleaned (inf → 0.0)
- [ ] `first_purchase_axe` cleaned: 2118 raw values → 6 primary axes (MAKE UP, SKINCARE, FRAGRANCE, HAIRCARE, OTHERS, Unknown)
- [ ] Zero NaN confirmed in output

---

### US-3.3 — Multi-Encoding & Scaling Pipeline (REVISED)

**As a** Data Scientist,  
**I want to** apply the correct encoding per feature type and then scale all features uniformly,  
**so that** categorical, ordinal, and continuous features all contribute properly to distance-based clustering.

**Acceptance Criteria:**
- [ ] **Frequency encoding** applied to `customer_city` (~12K unique → 1 frequency column)
- [ ] **One-Hot encoding** applied to 9 low-cardinality categoricals: `gender`, `dominant_channel`, `dominant_axe`, `dominant_market`, `country`, `channel_recruitment`, `age_category`, `age_generation`, `first_purchase_axe` → 37 dummy columns
- [ ] **StandardScaler** applied to ALL resulting features (continuous + encoded)
- [ ] Rationale documented: why StandardScaler after encoding (zero-mean, unit-variance for K-Means distances)
- [ ] `X_scaled` shape printed: expected `(64469, 70)` — 30 continuous + 37 dummies + 2 missing indicators + 1 frequency
- [ ] `preprocess_for_clustering()` single-function pipeline in `src/preprocessing.py`
- [ ] No NaN, no Inf in final output

---

### US-3.4 — PCA — Variance Analysis (REVISED)

**As a** Data Scientist,  
**I want to** run PCA on the 70-feature scaled matrix,  
**so that** I understand explained variance and can decide whether to reduce dimensionality before clustering.

**Acceptance Criteria:**
- [ ] PCA fitted on `X_scaled` (70 features) with `n_components = min(n_features, 30)` — increased from 20 due to richer feature set
- [ ] Cumulative explained variance curve plotted (x = n components, y = % variance explained)
- [ ] Number of components explaining ≥ 80% and ≥ 90% of variance identified
- [ ] Top 3 components' loading bar charts produced (which features contribute most — expect one-hot dummies to appear)
- [ ] Decision documented: cluster on PCA components OR on full scaled features? (with rationale)
- [ ] If PCA used: `X_pca` matrix created with chosen n_components
- [ ] Note: with 70 features including one-hot dummies, PCA reduction is likely more beneficial than with the old 21-feature set

---

### US-3.5 — UMAP 2D Visualization

**As a** Data Scientist,  
**I want to** project customers onto a 2D plane with UMAP,  
**so that** I can visually assess whether clusters exist before running the algorithms.

**Acceptance Criteria:**
- [ ] UMAP fitted with `n_neighbors=15`, `min_dist=0.1`, `random_state=42`
- [ ] 2D scatter plot produced, colored by `RFM_Segment_ID`
- [ ] 2D scatter plot produced, colored by `loyalty_status`
- [ ] 2D scatter plot produced, colored by `dominant_axe`
- [ ] Qualitative observation in Markdown: "Natural cluster structure is [visible / not visible] in 2D"
- [ ] `umap-learn` installed and import confirmed; `random_state` fixed

---

## E4 — Clustering Models

> **Epic Goal:** Evaluate multiple clustering algorithms systematically, validate them with internal metrics, and select the best model for segment assignment.

**Value Statement:**  
*As a Data Scientist, I want a rigorous multi-algorithm comparison so that the final segmentation rests on the most appropriate method for this data.*

**Dependencies:** E3 complete (`X_scaled` or `X_pca` available).

---

### US-4.1 — K-Means — Optimal k Selection

**As a** Data Scientist,  
**I want to** find the optimal number of clusters for K-Means,  
**so that** the elbow and silhouette criteria jointly inform the final k choice.

**Acceptance Criteria:**
- [ ] K-Means evaluated for k = 2 to 30 (OQ-6: target 10–30 segments)
- [ ] Inertia (within-cluster sum of squares) plotted vs. k — elbow method
- [ ] Silhouette score plotted vs. k
- [ ] Davies-Bouldin score plotted vs. k  
- [ ] Top 3 candidate values of k identified with scores tabulated
- [ ] `random_state=42`, `n_init=10`, `max_iter=300`

**Technical Notes:**
- Consider computing only up to k=30 to keep runtime reasonable
- Use `KMeans` from `sklearn.cluster`

---

### US-4.2 — K-Means — Final Run & Assignment

**As a** Data Scientist,  
**I want to** run the final K-Means with the chosen k,  
**so that** each customer receives a cluster label.

**Acceptance Criteria:**
- [ ] Final K-Means run with `k = k_optimal` (chosen from US-4.1)
- [ ] Cluster labels stored as `df_customers['kmeans_label']`
- [ ] Cluster size distribution printed (count and % per cluster)
- [ ] No cluster smaller than 0.5% of the base (flag if so)
- [ ] Silhouette Score, Davies-Bouldin, Calinski-Harabasz printed for this final model

---

### US-4.3 — Agglomerative Hierarchical Clustering

**As a** Data Scientist,  
**I want to** run hierarchical clustering and compare it to K-Means,  
**so that** the segmentation is not biased toward a single algorithm's assumptions.

**Acceptance Criteria:**
- [ ] Dendrogram plotted (truncated to last 30 merges for readability)
- [ ] `AgglomerativeClustering` with `linkage='ward'` run for same k as K-Means optimal
- [ ] Cluster labels stored as `df_customers['hclust_label']`
- [ ] Silhouette, DB, CH scores computed and added to comparison table (US-4.5)
- [ ] Visual comparison: UMAP colored by K-Means labels vs. Hierarchical labels side-by-side

---

### US-4.4 — Gaussian Mixture Models

**As a** Data Scientist,  
**I want to** run a GMM (soft clustering) and compare it to hard clustering methods,  
**so that** I can assess whether probabilistic assignment improves interpretability.

**Acceptance Criteria:**
- [ ] `GaussianMixture` with `n_components = k_optimal`, `covariance_type='full'`, `random_state=42`
- [ ] Cluster labels stored as `df_customers['gmm_label']`
- [ ] BIC and AIC curves plotted over k range (additional model selection tools)
- [ ] Silhouette, DB, CH scores added to comparison table (US-4.5)

---

### US-4.5 — Algorithm Comparison & Final Selection

**As a** Data Scientist,  
**I want to** compare all algorithms on a common metrics table,  
**so that** the final model choice is transparent and justified.

**Acceptance Criteria:**
- [ ] Comparison table with columns: Algorithm, k, Silhouette ↑, Davies-Bouldin ↓, Calinski-Harabasz ↑, Min cluster size %, Notes
- [ ] Best algorithm identified based on combined score
- [ ] Chosen model's cluster labels stored as `df_customers['final_cluster']`
- [ ] Markdown cell: 3–5 sentences justifying the choice
- [ ] `df_customers` exported to `data/processed/customers_with_clusters.csv`

---

### US-4.6 — Cluster Stability Validation

**As a** Data Scientist,  
**I want to** test the stability of the final clustering on data subsamples,  
**so that** I can confirm segments are not artifacts of the exact sample.

**Acceptance Criteria:**
- [ ] Run final algorithm on 5 bootstrapped 80% subsamples
- [ ] ARI (Adjusted Rand Index) computed between each subsample result and full-data result
- [ ] Mean ARI ≥ 0.70 (acceptable stability threshold)
- [ ] Results reported in a table: subsample #, ARI score, cluster count
- [ ] If ARI < 0.70: document the instability and revisit k or algorithm

---

## E5 — Segment Profiling & Quantification

> **Epic Goal:** Produce a complete, quantified characterization of each cluster — all KPIs, delta vs. global average, and statistical confirmation that the segments are genuinely different.

**Value Statement:**  
*As a Marketing Manager, I want every segment described with hard numbers and a comparison to the global average so that I can prioritize investment with confidence.*

**Dependencies:** E4 complete (`final_cluster` labels available on `df_customers`).

---

### US-5.1 — Global KPI Baseline

**As a** Marketing Manager,  
**I want to** know the global average for each KPI across the full customer base,  
**so that** every segment can be benchmarked against it.

**Acceptance Criteria:**
- [ ] The following KPIs computed at global level:

| KPI | Column(s) |
|---|---|
| Avg Sales Value (€) | `monetary_avg` |
| Purchase Frequency | `frequency` |
| Avg Basket Size (€) | `avg_basket_size_eur` |
| Avg Units per Basket | `avg_units_per_basket` |
| Recency (days) | `recency_days` |
| Channel Mix (%) | store / estore / C&C ratios |
| Product Axis Mix (%) | axe_* ratios |
| Avg Discount Rate (%) | `discount_rate` |
| Loyalty Status Distribution | % per status |
| CLV Estimate (€) | `monetary_total` |

- [ ] Global baseline displayed as a formatted table
- [ ] Total number of customers and total sales (€) printed

---

### US-5.2 — Per-Cluster KPI Computation

**As a** Marketing Manager,  
**I want to** see the same 10 KPIs computed per cluster,  
**so that** I can understand each segment's behavior in absolute terms.

**Acceptance Criteria:**
- [ ] All 10 KPIs from US-5.1 computed per `final_cluster`
- [ ] Cluster size (n customers and %) added as first column
- [ ] Results displayed as a matrix: rows = clusters, columns = KPIs
- [ ] Heatmap visualization of the KPI matrix (normalized per column) for quick pattern reading

---

### US-5.3 — Delta Table (Clusters vs. Global Average)

**As a** Marketing Manager,  
**I want to** see how each cluster deviates from the global average — in absolute and relative terms,  
**so that** I can immediately identify above-average and below-average segments.

**Acceptance Criteria:**
- [ ] For each numerical KPI: `delta_abs = cluster_value - global_avg`; `delta_pct = (cluster_value / global_avg - 1) * 100`
- [ ] Delta table formatted as: Cluster | KPI | Global Avg | Cluster Value | Delta Abs | Delta %
- [ ] Color-coded display: green if positive delta (above avg), red if negative
- [ ] Clusters sorted by `monetary_total` descending (highest CLV first)
- [ ] Notable deltas (|delta %| > 30%) highlighted in narrative summary

---

### US-5.4 — Top Distinguishing Features per Cluster

**As a** Data Scientist,  
**I want to** identify which features most strongly define each cluster,  
**so that** persona narratives are grounded in statistical evidence.

**Acceptance Criteria:**
- [ ] For each cluster: compute mean of each feature vs. rest-of-population mean
- [ ] Rank features by absolute standardized difference (Cohen's d): `(cluster_mean - global_mean) / global_std`
- [ ] Top 5 distinguishing features listed per cluster (positive and negative)
- [ ] Bar chart per cluster: top 5 features, x-axis = Cohen's d

---

### US-5.5 — Statistical Validation of Segment Differences

**As a** Data Scientist,  
**I want to** statistically confirm that clusters are genuinely different,  
**so that** the segmentation is not the result of random noise.

**Acceptance Criteria:**
- [ ] Kruskal-Wallis test run on each of the 10 KPIs across all clusters
- [ ] p-values tabulated; threshold p < 0.05
- [ ] At least 7 of the 10 KPIs show significant differences (p < 0.05)
- [ ] Post-hoc Dunn test (or Mann-Whitney pairwise) for KPIs that pass Kruskal-Wallis
- [ ] Results summary: "X/10 KPIs significantly differ across segments at p < 0.05"

---

### US-5.6 — CLV Ranking & High-Potential Segment Identification

**As a** Head of Marketing,  
**I want to** see segments ranked by business value,  
**so that** I know where to focus marketing investment first.

**Acceptance Criteria:**
- [ ] Segments ranked by `monetary_total` (estimated CLV) descending
- [ ] 2×2 prioritization matrix plotted: x = segment size (% of base), y = avg CLV; bubbles sized by total revenue contribution
- [ ] "High potential" quadrant defined: above-median size AND above-median CLV
- [ ] Top 3 priority segments identified and labeled
- [ ] "Investment-worthy" segments (high CLV, small size) flagged for loyalty upgrade campaigns

---

## E6 — Personas & Marketing Recommendations

> **Epic Goal:** Transform statistical cluster profiles into marketing-ready personas and concrete campaign recommendations that the Marketing team can act on immediately.

**Value Statement:**  
*As a Marketing Manager, I want named, narrative personas with specific campaign guidance so that every segment activates with a tailored strategy.*

**Dependencies:** E5 complete (profiling, delta table, and CLV ranking available).

---

### US-6.1 — Persona Naming & Archetype Definition

**As a** Marketing Manager,  
**I want to** give each cluster a memorable name and archetype,  
**so that** the personas are usable in briefs and presentations without referencing cluster numbers.

**Acceptance Criteria:**
- [ ] Each cluster receives a name that reflects its dominant behavioral trait (e.g., "The Fragrance Loyalist", "The Budget Hunter", "The Digital Explorer", "The GOLD Skincare Devotee")
- [ ] Each name is unique, ≤ 4 words, and marketing-friendly
- [ ] A one-sentence archetype description accompanies each name
- [ ] Naming rationale tied explicitly to the top distinguishing features from US-5.4
- [ ] All names documented in a summary table: Cluster ID | Name | Archetype

---

### US-6.2 — Persona Card Production

**As a** Marketing Manager,  
**I want to** have a structured persona card per segment,  
**so that** any team member can quickly understand a segment without reading the full analysis.

**Acceptance Criteria (per card):**
- [ ] Persona name and cluster ID
- [ ] Size: n customers and % of base
- [ ] Top 3 behavioral traits with quantified delta vs. average (e.g., "+48% higher basket size")
- [ ] Dominant channel, dominant product axis, dominant market tier
- [ ] Loyalty status distribution (% No Fid / BRONZE / SILVER / GOLD)
- [ ] CLV tier (Top / Mid / Low)
- [ ] 2–3 sentence plain-language narrative usable in a brief
- [ ] Cards formatted as Markdown tables in the notebook AND exported as a standalone Markdown file

---

### US-6.3 — Marketing Recommendations per Segment

**As a** Marketing Manager,  
**I want to** receive specific marketing activation guidance per persona,  
**so that** I can brief campaign teams without additional data analysis.

**Acceptance Criteria (per persona):**
- [ ] Objective: Acquire / Retain / Upsell / Re-engage (one primary objective per segment)
- [ ] Recommended primary channel: in-store / estore / push notification / email
- [ ] Suggested offer type: discovery (new product axis), loyalty tier upgrade, discount, premium experience
- [ ] Suggested frequency of communication: weekly / biweekly / monthly
- [ ] KPI to track for this segment: the single most relevant metric to measure campaign success
- [ ] Recommendations grounded in segment profile data (no generic advice)

---

### US-6.4 — Segment Prioritization Matrix

**As a** Head of Marketing,  
**I want to** see all segments in a single strategic view,  
**so that** I can allocate marketing budget across segments.

**Acceptance Criteria:**
- [ ] 2×2 scatter plot: x = segment size (% base), y = avg CLV (€), bubble size = total revenue contribution
- [ ] Four quadrants labeled: "Grow" (large, high CLV), "Nurture" (small, high CLV), "Volume" (large, low CLV), "Monitor" (small, low CLV)
- [ ] Each persona name displayed on the plot as a label
- [ ] Narrative: 5–8 sentences mapping each quadrant to a strategic priority

---

### US-6.5 — AI-Generated Personalized Content Opportunity

**As a** Marketing Manager,  
**I want to** know which segment(s) would benefit most from AI-generated personalized content,  
**so that** I can champion an automation pilot with the right audience.

**Acceptance Criteria:**
- [ ] Criteria for "AI content fit" defined (e.g., high frequency + high estore ratio + clear single product axis preference)
- [ ] Each segment scored against the criteria
- [ ] Top 1–2 segments identified with justification
- [ ] Suggested content type: product recommendation email / personalized homepage / push notification
- [ ] Expected lift quantified as a hypothesis: "If we deliver {content type} to {persona name}, we expect {KPI} to improve by approximately {range}% based on the segment's current profile"

---

## Appendix — Story Dependency Graph

```
US-1.1 → US-1.2 → US-1.3 → US-1.4 ─┐
                                      ├─→ E2 (EDA)
                         US-1.5 ──────┤
                         US-1.6 ──────┤
                         US-1.7 ──────┘
                                      │
                                      ↓
                              E3 (Feature Eng. — REVISED)
                  US-3.1 (audit) → US-3.2 (impute+indicators)
                                        ↓
                              US-3.3 (encode+scale) → US-3.4 (PCA) → US-3.5 (UMAP)
                  [3.1–3.3 unified in preprocess_for_clustering()]
                                                              │
                                                              ↓
                                                     E4 (Clustering)
                                          US-4.1 → US-4.2
                                          US-4.3
                                          US-4.4
                                               └──→ US-4.5 → US-4.6
                                                              │
                                                              ↓
                                                    E5 (Profiling)
                                         US-5.1 → US-5.2 → US-5.3
                                         US-5.4 → US-5.5 → US-5.6
                                                              │
                                                              ↓
                                                    E6 (Personas)
                                         US-6.1 → US-6.2 → US-6.3
                                         US-6.4 → US-6.5
```

---

### E3 Course Correction Log (2026-03-31)

**Trigger:** Review revealed the old preprocessing pipeline (US-3.1–3.3) discarded 32+ useful features by applying only StandardScaler to 21 numeric features. Categorical, ordinal, and high-missing features were all excluded.

**Changes made:**
- **US-3.1** rewritten: comprehensive audit of all 53 features. Only 13 truly useless ones dropped (zero variance, exact duplicates, raw dates, flags). 40 features retained.
- **US-3.2** rewritten: type-appropriate imputation (median vs. Unknown), missing indicators created (`has_age_info`, `has_first_purchase_info`), inf cleaned, `first_purchase_axe` simplified from 2118 values to 6 clean categories.
- **US-3.3** rewritten: multi-encoding pipeline — Frequency Encoding for `customer_city` (12K→1), One-Hot for 9 categoricals (→37 dummies), then StandardScaler on all 70 features.
- **US-3.4** adapted: `n_components` increased to 30 to account for 70-feature input.
- **US-3.5** unchanged: UMAP input is `X_cluster` regardless.
- **Code implementation:** `preprocess_for_clustering()` added to `src/preprocessing.py`. Old functions retained for backward compatibility.

---

*Epics document — Edram Project — Sephora Use Case 2 — March 25, 2026 (Revised March 31, 2026)*
