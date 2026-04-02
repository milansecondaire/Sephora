---
stepsCompleted: ['discovery', 'analysis', 'content', 'review']
inputDocuments: ['data/BDD#7_Database_Albert_School_Sephora.csv', 'use-case-2-brief.md']
workflowType: 'prd'
project: 'Edram'
author: 'Milan'
date: '2026-03-25'
version: '1.0'
status: 'Draft'
---

# Product Requirements Document  
## Advanced Customer Segmentation & Persona Creation — Sephora (Use Case 2)

**Author:** Milan  
**Date:** March 25, 2026  
**Version:** 1.0  
**Status:** Draft  
**Project:** Edram — Albert School B2 BDD

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [Scope](#4-scope)
5. [User Personas & Stakeholders](#5-user-personas--stakeholders)
6. [Functional Requirements](#6-functional-requirements)
7. [Data Requirements](#7-data-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Technical Architecture Overview](#9-technical-architecture-overview)
10. [Epics & User Stories](#10-epics--user-stories)
11. [Constraints & Assumptions](#11-constraints--assumptions)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Open Questions](#13-open-questions)

---

## 1. Executive Summary

Sephora's customer base displays extreme behavioral diversity that classical rule-based segmentation cannot adequately capture. This project delivers a **machine-learning-powered customer segmentation pipeline** that identifies homogeneous, actionable profiles using unsupervised clustering algorithms applied to transactional and demographic data.

The output is a set of **data-driven customer personas** — enriched archetypes combining statistical cluster characteristics with marketing-ready descriptions — that the Marketing team can immediately operationalize for personalized campaigns, targeted communications, and resource prioritization.

The dataset covers **34 variables** per transaction including RFM indicators, purchase behavior, channel preference (in-store vs. estore), product axis (Skincare, Haircare, Make-up, Fragrance), market tier (Selective, Exclusive, Sephora Own-Brand), brand, loyalty status, and sociodemographic proxies. Value is measured by comparing each segment's KPIs against the global customer average.

---

## 2. Problem Statement

### 2.1 Context

Sephora operates a large, multi-channel loyalty customer base transacting across physical stores and its estore. The existing RFM-segment classification (`RFM_Segment_ID`) provides a rudimentary starting point but does not reflect the full complexity of purchase behaviors, channel mix, product affinities, and lifecycle stages.

Marketing activations applied uniformly across the base — or via coarse RFM buckets — underperform because they do not resonate with customers' specific needs and preferences.

### 2.2 Root Cause

| Pain Point | Current Gap |
|---|---|
| One-size communications | No granular behavioral segments |
| Low offer relevance | No product affinity profiling |
| Missed high-potential customers | No identification of growth segments |
| No ROI benchmark | No baseline to measure incremental value |
| Reactive marketing | No predictive personas to anticipate needs |

### 2.3 Opportunity

By applying unsupervised Machine Learning (clustering) on the full feature space available in the transactional dataset, it is possible to:

- Discover **latent behavioral groups** invisible to manual rules
- Quantify within-cluster homogeneity and between-cluster separation
- Profile each cluster across all key dimensions to create **rich, interpretable personas**
- Measure the **incremental value** of each segment vs. the global average

---

## 3. Goals & Success Metrics

### 3.1 Business Goals

| # | Goal | Rationale |
|---|---|---|
| G1 | Identify **10 to 30** distinct, actionable customer segments (final count driven by clustering metrics and marketing operability) | Range agreed with marketing stakeholders; quality over quantity — prefer fewer stable segments over many unstable ones |
| G2 | Enable segment-specific campaign targeting | Increase conversion and relevance per communication |
| G3 | Prioritize high-potential segments for investment | Focus budget on clusters with highest incremental revenue potential |
| G4 | Produce shareable persona cards | Bridge analytics and marketing teams |
| G5 | Establish a repeatable segmentation framework | Enable quarterly or annual refresh |

### 3.2 ML / Analytical Goals

| # | Goal | Metric |
|---|---|---|
| A1 | High intra-cluster cohesion | Silhouette Score ≥ 0.35 |
| A2 | Good inter-cluster separation | Davies-Bouldin Index as low as possible |
| A3 | Stability of segments | Cluster composition variance < 10% on bootstrapped samples |
| A4 | Statistical significance of cluster differences | ANOVA / Kruskal-Wallis p < 0.05 on key KPIs |
| A5 | Full feature coverage | All 6 dimension families represented in clustering |

### 3.3 Business Value KPIs (Compared to Global Average)

Each segment must be quantified on these KPIs, with delta vs. average clearly stated:

| KPI | Definition |
|---|---|
| Average Sales Value (€) | Mean `salesVatEUR` per customer |
| Purchase Frequency | Number of transactions per customer per period |
| Average Basket Size (€) | `salesVatEUR` / number of transactions |
| Units per Basket | Mean `quantity` per transaction |
| Recency (days) | Days since last transaction |
| Channel Mix (%) | Share of `store` vs. `estore` |
| Product Axis Mix (%) | Distribution across SKINCARE, HAIRCARE, MAKE UP, FRAGRANCE |
| Average Discount Rate (%) | `discountEUR` / `salesVatEUR` |
| Loyalty Status Distribution | Share of statuses 2, 3, 4 |
| Customer Lifetime Value (CLV) Estimate | Total spend over observed period |

> **Quantification mandate:** All segments must report each KPI as an absolute value **and** as a % delta vs. the global customer average, e.g., *"+42% above average basket size"*.

---

## 4. Scope

### 4.1 In Scope

- **Exploratory Data Analysis (EDA)** on the full Sephora dataset
- **Feature engineering** to derive customer-level aggregated features from transaction-level data
- **Dimensionality reduction** (PCA, UMAP) for visualization and optional pre-processing
- **Clustering models**: K-Means, DBSCAN, Hierarchical/Agglomerative, and Gaussian Mixture Models (evaluation of multiple algorithms)
- **Cluster validation** using internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Segment profiling**: statistical characterization of each cluster on all KPIs
- **Persona card creation**: narrative marketing personas per segment
- **Benchmark vs. global average**: delta table for each segment
- **Recommendations** on marketing activations per segment

### 4.2 Out of Scope

- Real-time or production model deployment (batch scoring only)
- Predictive churn or propensity models (separate use case)
- A/B testing of campaigns (follow-on work)
- Integration with CRM or marketing automation platforms

---

## 5. User Personas & Stakeholders

### 5.1 Primary Users

| Persona | Role | Needs from this PRD |
|---|---|---|
| **Marketing Manager** | Activates campaigns and communications | Clear segment descriptions, actionable targeting criteria, KPI benchmarks |
| **CRM / Loyalty Manager** | Manages loyalty program and personalization | Segment profiles with loyalty status distribution, recommended actions |
| **Data Analyst / Data Scientist** | Builds and maintains the model | Clear feature requirements, ML algorithm guidance, evaluation criteria |

### 5.2 Secondary Stakeholders

| Persona | Interest |
|---|---|
| **Head of Marketing** | ROI justification; high-level segment story |
| **Category Managers** | Segment affinity to their product axis |
| **Digital Team** | estore segment identification |

---

## 6. Functional Requirements

### FR-1: Data Preparation & Feature Engineering

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | The system shall aggregate transaction-level data to customer-level features | Must Have |
| FR-1.2 | Compute RFM features: Recency (days since last purchase), Frequency (# transactions), Monetary (total / average salesVatEUR) | Must Have |
| FR-1.3 | Compute channel features: dominant channel, store / estore / Click&Collect ratio, `store_type_app` distribution (APP, CSC, ESTORE, MOBILE, STORE, WEB) | Must Have |
| FR-1.4 | Compute product affinity features: share of spend per Axe_Desc, dominant product axis, number of axes purchased | Must Have |
| FR-1.5 | Compute brand diversity: number of unique brands, share of top brand | Should Have |
| FR-1.6 | Compute discount sensitivity: average discount %, discount transaction rate | Should Have |
| FR-1.7 | Encode sociodemographic proxies: age category, gender, store city, customer city where available | Should Have |
| FR-1.8 | Compute loyalty and lifecycle features: loyalty status, subscription tenure, first purchase characteristics | Should Have |
| FR-1.9 | Handle missing values with documented imputation strategy | Must Have |
| FR-1.10 | Normalize / standardize features before clustering | Must Have |

### FR-2: Exploratory Data Analysis

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | Produce univariate distributions for all key features | Must Have |
| FR-2.2 | Produce correlation matrix and identify redundant features | Must Have |
| FR-2.3 | Visualize customer distribution across RFM dimensions | Must Have |
| FR-2.4 | Identify and document outlier handling strategy | Must Have |
| FR-2.5 | Explore additional KPIs beyond the base list in section 3.3 | Should Have |

### FR-3: Dimensionality Reduction

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Apply PCA to quantify variance explained and reduce dimensionality if beneficial | Must Have |
| FR-3.2 | Apply UMAP or t-SNE for 2D cluster visualization | Should Have |
| FR-3.3 | Document the explained variance and component loadings | Must Have |

### FR-4: Clustering

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | Evaluate a minimum of 3 clustering algorithms (K-Means, Hierarchical, and one additional) | Must Have |
| FR-4.2 | Use elbow method and silhouette curve to determine optimal k for K-Means; evaluate range k = [2..30] to accommodate the target of 10–30 segments | Must Have |
| FR-4.3 | Produce a dendrogram for hierarchical clustering | Should Have |
| FR-4.4 | Evaluate DBSCAN for noise detection / outlier customers | Could Have |
| FR-4.5 | Select the best model based on validation metrics and interpretability | Must Have |
| FR-4.6 | Document algorithm selection rationale | Must Have |

### FR-5: Cluster Validation & Quality

| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | Compute Silhouette Score for the final segmentation | Must Have |
| FR-5.2 | Compute Davies-Bouldin Index | Must Have |
| FR-5.3 | Compute Calinski-Harabasz Index | Should Have |
| FR-5.4 | Perform statistical testing (ANOVA / Kruskal-Wallis) to confirm segment differences on key KPIs | Must Have |
| FR-5.5 | Test cluster stability via bootstrapping or cross-validation subsample | Should Have |

### FR-6: Segment Profiling & Quantification

| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | Compute all KPIs from section 3.3 for each cluster | Must Have |
| FR-6.2 | Compute global average benchmarks for each KPI | Must Have |
| FR-6.3 | Produce a Delta Table: each segment's KPI vs. global average (absolute and %) | Must Have |
| FR-6.4 | Identify the top 3 defining features per cluster | Must Have |
| FR-6.5 | Rank segments by estimated Customer Lifetime Value | Must Have |
| FR-6.6 | Identify the highest-potential segment(s) for marketing investment | Must Have |

### FR-7: Persona Cards

| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | Produce one persona card per cluster | Must Have |
| FR-7.2 | Each card must include: persona name, archetype description, key behavioral traits, preferred channel, product affinity, value tier, and recommended marketing action | Must Have |
| FR-7.3 | Persona names must be memorable and marketing-friendly (e.g., "The Fragrance Loyalist", "The Digital Explorer") | Should Have |
| FR-7.4 | Cards available as visualizations (charts + text) | Should Have |

### FR-8: Recommendations

| ID | Requirement | Priority |
|---|---|---|
| FR-8.1 | Propose specific marketing actions per segment (campaign type, channel, offer type) | Must Have |
| FR-8.2 | Propose content personalization approach per segment | Should Have |
| FR-8.3 | Identify segments where automated, AI-generated personalized content would be most impactful | Could Have |

---

## 7. Data Requirements

### 7.1 Dataset Overview

**Source file:** `data/BDD#7_Database_Albert_School_Sephora.csv`  
**Granularity:** Transaction-level (one row per purchase line)  
**Unit of analysis:** Customer (aggregated to `anonymized_card_code`)

### 7.2 Available Variables

#### Transaction & Customer Core

| Variable | Type | Description | Used For |
|---|---|---|---|
| `anonymized_card_code` | int (hashed) | Unique customer identifier — hashed ID used to identify the customer across bases | Aggregation key |
| `anonymized_Ticket_ID` | int (hashed) | Unique transaction identifier — hashed | Deduplication |
| `countryIsoCode` | string | Country in which the purchase was made | Segmentation dimension |
| `transactionDate` | date | Date of purchase (based on invoice) | Recency computation |
| `salesVatEUR` | decimal | Amount spent in EUR (incl. VAT) | Monetary, basket value |
| `discountEUR` | decimal | Discount applied in EUR | Discount sensitivity |
| `quantity` | int | Quantity of products purchased | Units per basket |

#### Loyalty & Sociodemographic

| Variable | Type | Description | Used For |
|---|---|---|---|
| `status` | int | Loyalty status at the moment of the purchase: **1 = No Fid / 2 = BRONZE / 3 = SILVER (200 pts) / 4 = GOLD (1000 pts)**. Note: France launched MySephora in Sept. 2025; historically 1 pt = 1€, now additional criteria apply. | Loyalty feature |
| `RFM_Segment_ID` | int | Pre-computed segment based on RFM (see `RFM Segment ID` reference) | Validation benchmark |
| `age` | int | Customer's age at the moment of the purchase (0 = unknown) | Sociodemographic feature |
| `age_category` | string | Customer's age category at the moment of the purchase | Sociodemographic feature |
| `age_generation` | string | Customer's age generation at the moment of the purchase | Sociodemographic feature |
| `gender` | int | Customer's gender — **1 = Men / 2 = Women** (99999 = not declared) | Sociodemographic feature |
| `subscription_date` | date | Date on which the customer subscribed to the loyalty program | Customer tenure |
| `customer_city` | string | City where the customer declared to live | Geographic feature |

#### Channel & Store

| Variable | Type | Description | Used For |
|---|---|---|---|
| `channel` | string | Channel of purchase: **`estore`** or **`store`**. Note: if `channel = estore` but `store_code_name` is not labeled as ESTORE, it is a **Click & Collect** purchase | Channel preference, C&C detection |
| `store_type_app` | string | Store declination on the app channel: **APP / CSC (Customer Services) / ESTORE / MOBILE / STORE / WEB** | Granular channel analysis |
| `store_code_name` | string | Association of the store code and its name | Store geography |
| `store_city` | string | City of the store | Geographic feature |
| `subscription_store_code` | int | Store code on which the customer subscribed | Recruitment geography |

#### Product

| Variable | Type | Description | Used For |
|---|---|---|---|
| `materialCode` | int | ID of the product purchased | Product-level analysis |
| `Axe_Desc` | string | Axis of the product purchased: **Make Up / Skincare / Fragrance / Haircare / Others** | Product affinity |
| `Market_Desc` | string | Market of the product: **SELECTIVE** (not only at Sephora) / **EXCLUSIVE** (only at Sephora) / **SEPHORA** (Sephora collection) | Value tier affinity |
| `brand` | string | Brand of the product purchased | Brand loyalty |

#### First Purchase (Lifecycle)

| Variable | Type | Description | Used For |
|---|---|---|---|
| `first_purchase_dt` | date | Date of the cardholder's first purchase | Lifecycle stage, tenure |
| `anonymized_first_purchase_id` | int (hashed) | Hashed ID of the first transaction | Deduplication |
| `channel_recruitment` | string | Channel on which the first purchase was made | Acquisition channel |
| `salesVatEUR_first_purchase` | decimal | Amount spent on the first purchase in EUR | Onboarding behavior |
| `discountEUR_first_purchase` | decimal | Discount applied on the first purchase total sales in EUR | Onboarding discount sensitivity |
| `quantity_first_purchase` | int | Quantity of products purchased on the first purchase | Onboarding basket size |
| `materialCode_first_purchase` | list\<int\> | List of product IDs purchased on the first purchase | First product affinity |
| `Axe_Desc_first_purchase` | list\<string\> | List of product axes purchased on the first purchase: Make Up / Skincare / Fragrance / Haircare / Others | Initial category preference |
| `Market_Desc_first_purchase` | list\<string\> | List of product markets purchased on the first purchase: SELECTIVE / EXCLUSIVE / SEPHORA | Initial market preference |
| `brand_first_purchase` | list\<string\> | List of brands purchased on the first purchase | Initial brand preference |

### 7.3 Known Data Quality Issues

| Issue | Column(s) | Proposed Handling |
|---|---|---|
| Missing values in customer city | `customer_city` | Impute with `store_city` or mark as `Unknown` |
| Missing first purchase details | `first_purchase_dt`, `brand_first_purchase` | Exclude from first-purchase features or create a "no first purchase data" category |
| Gender code 99999 | `gender` | Treat as `Unknown` — not declared by customer |
| Typo in product axis | `Axe_Desc` = `MAEK UP` | Normalize to `MAKE UP` before any processing |
| Age = 0 or very low age | `age` | Confirmed: 0 = unknown / not declared; very low ages also considered unreliable. Flag as missing; use `age_category` / `age_generation` as primary sociodemographic proxies for these customers |
| Floating-point customer IDs | `anonymized_card_code` | Parse as string to avoid precision loss from scientific notation (e.g., `-8.17068E+18`) |
| Click & Collect detection | `channel`, `store_code_name` | When `channel = estore` AND store is not labeled ESTORE → flag as Click & Collect; treat as a **third channel** for analysis |
| First purchase fields as lists | `materialCode_first_purchase`, `Axe_Desc_first_purchase`, `Market_Desc_first_purchase`, `brand_first_purchase` | Parse stringified lists; extract dominant/first element for scalar features |

---

## 8. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | **Reproducibility** — All experiments must be reproducible | Fixed random seeds; versioned environment (`requirements.txt`) |
| NFR-2 | **Documentation** — All analytical choices must be documented inline | Jupyter notebooks with explanatory markdown cells |
| NFR-3 | **Interpretability** — Segments must be explainable to a non-technical marketing audience | Persona cards + plain-language summaries accompany all technical outputs |
| NFR-4 | **Performance** — EDA and clustering pipeline must complete in < 30 minutes on a standard laptop | Optimize with sampling for initial exploration if needed |
| NFR-5 | **Data Privacy** — All data remains anonymized; no re-identification attempts | Use only `anonymized_card_code`; never cross-reference external datasets |
| NFR-6 | **Modularity** — Code organized in reusable functions / classes | Single Jupyter notebook or structured Python scripts per pipeline stage |
| NFR-7 | **Visualization quality** — Charts must be publication-quality | Use matplotlib/seaborn with consistent color palette; min 300 DPI export |

---

## 9. Technical Architecture Overview

### 9.1 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Dimensionality reduction | scikit-learn (PCA), umap-learn |
| Clustering | scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture) |
| Validation metrics | scikit-learn (silhouette_score, davies_bouldin_score, calinski_harabasz_score) |
| Statistical tests | scipy.stats |
| Notebook environment | Jupyter Notebook / JupyterLab |

### 9.2 Data Pipeline

```
Raw CSV (transaction-level)
        │
        ▼
1. Data Loading & Quality Check
        │
        ▼
2. Data Cleaning (normalize categories, handle missing values)
        │
        ▼
3. Customer Aggregation (transaction → customer-level features)
        │
        ▼
4. Feature Engineering (RFM, channel mix, product affinity, etc.)
        │
        ▼
5. EDA (distributions, correlations, outliers)
        │
        ▼
6. Feature Scaling + Optional Dimensionality Reduction (PCA / UMAP)
        │
        ▼
7. Clustering (K-Means + alternatives; optimal k selection)
        │
        ▼
8. Cluster Validation (Silhouette, DB, CH, statistical tests)
        │
        ▼
9. Segment Profiling (KPIs per cluster + delta vs. global average)
        │
        ▼
10. Persona Card Generation
        │
        ▼
11. Marketing Recommendations
```

### 9.3 Output Artifacts

| Artifact | Format | Location |
|---|---|---|
| Cleaned, aggregated customer dataset | `.csv` | `data/processed/` |
| Cluster assignment per customer | `.csv` | `data/processed/` |
| Segment profiling report | `.md` / `.html` | `_bmad-output/implementation-artifacts/` |
| Persona cards | `.md` + visual | `_bmad-output/implementation-artifacts/` |
| Delta KPI table (segments vs. global average) | `.md` table | `_bmad-output/implementation-artifacts/` |
| Jupyter notebooks (EDA, clustering, profiling) | `.ipynb` | project root |

---

## 10. Epics & User Stories

> **Full detail:** See [epics-sephora-customer-segmentation-ml-2026-03-25.md](epics-sephora-customer-segmentation-ml-2026-03-25.md) for complete user stories, acceptance criteria, technical notes, and dependency graph.

### Summary

| Epic | Stories | Goal |
|---|---|---|
| E1 — Data Foundation | US-1.1 → US-1.7 | Clean customer-level feature matrix |
| E2 — Exploratory Data Analysis | US-2.1 → US-2.6 | Full statistical picture of the base |
| E3 — Feature Engineering & Preprocessing | US-3.1 → US-3.5 | ML-ready scaled feature set + PCA/UMAP |
| E4 — Clustering Models | US-4.1 → US-4.6 | Multi-algorithm comparison + final model |
| E5 — Segment Profiling & Quantification | US-5.1 → US-5.6 | KPI delta tables + statistical validation |
| E6 — Personas & Recommendations | US-6.1 → US-6.5 | Persona cards + marketing activation guide |

### Epic 1 — Data Foundation

> *"As a Data Analyst, I need clean, customer-level features so that I can build reliable clustering models."*

**Stories:**

| Story | Description | AC |
|---|---|---|
| US-1.1 | Load and validate raw CSV | File loads without errors; row count and column count confirmed; data types validated |
| US-1.2 | Clean data quality issues | Typo `MAEK UP` → `MAKE UP`; age=0 flagged; gender 99999 → `Unknown`; missing values documented |
| US-1.3 | Aggregate to customer level | One row per `anonymized_card_code`; all transaction-level metrics aggregated (sum, mean, count, first, last) |
| US-1.4 | Compute RFM features | `recency_days`, `frequency`, `monetary_total`, `monetary_avg` computed and validated |
| US-1.5 | Compute behavioral features | `avg_basket_size`, `units_per_basket`, `discount_rate`, `channel_mix`, `product_axis_shares` computed |
| US-1.6 | Encode categorical features | Loyalty status, gender, dominant channel, dominant axis encoded appropriately |

### Epic 2 — Exploratory Analysis

> *"As a Marketing Manager, I want to understand the existing customer distribution so that I can validate modeling assumptions."*

**Stories:**

| Story | Description | AC |
|---|---|---|
| US-2.1 | EDA on all key features | Distribution plots for all numerical features; frequency charts for categoricals |
| US-2.2 | Correlation analysis | Heatmap produced; highly correlated feature pairs documented |
| US-2.3 | RFM distribution visualization | 3D scatter plot + 2D projections of RFM space; existing RFM segments displayed |
| US-2.4 | Channel and product analysis | Bar charts of channel split; product axis purchases by channel; top brands |
| US-2.5 | Outlier analysis | Outlier customers identified (e.g., extremely high spenders); strategy documented (keep vs. exclude) |

### Epic 3 — Clustering Model

> *"As a Data Scientist, I want to evaluate multiple clustering algorithms so that I can select the one that best fits the data."*

**Stories:**

| Story | Description | AC |
|---|---|---|
| US-3.1 | Feature scaling | All features normalized (MinMaxScaler or StandardScaler); rationale documented |
| US-3.2 | PCA analysis | Cumulative variance explained chart; number of components to retain decided |
| US-3.3 | UMAP 2D visualization | 2D customer map produced; colored by initial RFM segment for orientation |
| US-3.4 | K-Means with elbow + silhouette | Range k=[2..10] evaluated; optimal k selected; cluster assignments stored |
| US-3.5 | Hierarchical clustering | Dendrogram plotted; comparison with K-Means results |
| US-3.6 | Algorithm comparison | Table comparing Silhouette, DB, CH scores for all tested algorithms |
| US-3.7 | Final model selection | Best algorithm documented with rationale; final cluster labels assigned to customers |

### Epic 4 — Segment Profiling & Quantification

> *"As a Marketing Manager, I want a complete KPI profile for each segment compared to the global average so that I can prioritize my marketing efforts."*

**Stories:**

| Story | Description | AC |
|---|---|---|
| US-4.1 | Global average computation | All 10 KPIs from section 3.3 computed at global level |
| US-4.2 | Cluster KPI computation | All 10 KPIs computed per cluster |
| US-4.3 | Delta table | Table showing each KPI per cluster with absolute value and % vs. global average |
| US-4.4 | Top distinguishing features | For each cluster, top 3 features that best differentiate it from others (ANOVA / feature importance) |
| US-4.5 | CLV ranking | Segments ranked by estimated Customer Lifetime Value; high-potential segments highlighted |
| US-4.6 | Statistical validation | Kruskal-Wallis confirms significant differences between segments on at least 5 KPIs |

### Epic 5 — Personas & Recommendations

> *"As a Marketing Manager, I want ready-to-use persona cards and campaign recommendations so that I can immediately activate the segments."*

**Stories:**

| Story | Description | AC |
|---|---|---|
| US-5.1 | Persona card per segment | Card includes: name, archetype, top 3 behavioral traits, channel preference, product affinity, value tier, size (% of base) |
| US-5.2 | Persona narrative | 2–3 sentence plain-language description per persona usable in briefs and presentations |
| US-5.3 | Marketing recommendations | Per segment: recommended campaign type (acquisition/retention/upsell), preferred channel, offer type (discount, product discovery, loyalty) |
| US-5.4 | Segment prioritization matrix | 2×2 matrix: segment value (CLV) vs. segment size; quadrant-based priority logic |
| US-5.5 | Automated content opportunity | Identify ≥1 segment where AI-generated personalized content (e.g., product recommendation emails) delivers highest expected lift |

---

## 11. Constraints & Assumptions

### Constraints

| # | Constraint |
|---|---|
| C1 | Dataset is static — no real-time data pipeline is required |
| C2 | Anonymized data only — no re-identification or external data enrichment |
| C3 | Academic context — production deployment is out of scope |
| C4 | No labeled ground truth — evaluation is entirely unsupervised |
| C5 | Dataset covers the **full year 2025** (Jan 1 – Dec 31, 2025); recency computed relative to Dec 31, 2025 |
| C6 | No additional variables available — the 34 columns in the CSV represent the complete feature space (confirmed by Sephora) |

### Assumptions

| # | Assumption |
|---|---|
| A1 | The `anonymized_card_code` uniquely identifies a customer across all transactions |
| A2 | Loyalty `status` values 2, 3, 4 represent ascending loyalty tiers |
| A3 | `gender` code **1 = Men, 2 = Women** (confirmed by official data dictionary); 99999 = not declared |
| A7 | Recency is computed relative to **December 31, 2025** (end of the observation period) |
| A8 | Status code **1 = No Fid** may be present in the full dataset even though not observed in the initial 10k-row sample |
| A4 | The dataset is a representative sample of the Sephora customer base |
| A5 | Transactions with missing `first_purchase_dt` are not first-time purchasers in the observation window |
| A6 | `salesVatEUR` represents gross sales (before discount); net sales = `salesVatEUR - discountEUR` |

---

## 12. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| R1: Insufficient cluster separation (low silhouette score) | Medium | High | Test multiple feature subsets; consider separate models for sub-populations |
| R2: High missing rate in sociodemographic fields | Medium | Medium | Use proxy variables (store city, existing RFM); exclude from primary model if too sparse |
| R3: Imbalanced cluster sizes (one very large cluster) | Medium | Medium | Use DBSCAN or GMM as alternatives; consider hierarchical splitting |
| R4: Persona outputs not actionable for marketing | Low | High | Involve marketing stakeholders in persona naming and recommendation session |
| R5: Overfitting to noise in high-dimensional space | Low | High | Apply PCA before clustering; validate on bootstrapped samples |
| R6: Data represents only a partial sample | Unknown | Medium | Document explicitly; note limitations in conclusions |

---

## 13. Resolved Questions

All questions have been answered prior to implementation start.

| # | Question | Answer | Impact on PRD |
|---|---|---|---|
| OQ-1 | Mapping of `status` loyalty codes | **1 = No Fid, 2 = BRONZE, 3 = SILVER (200 pts), 4 = GOLD (1000 pts).** France launched MySephora in Sept. 2025; historically 1 pt = 1€, now additional criteria apply. Status 1 (No Fid) may appear in the full dataset. | Section 7.2 updated; status 1 added as possible value; loyalty feature encoding updated |
| OQ-2 | Date range of the dataset | **Full year 2025** | Constraint C5 updated; recency computed relative to Dec 31, 2025 |
| OQ-3 | `age = 0` meaning | **Unknown / not declared** — low ages also considered unreliable | Data quality rule confirmed: age = 0 → missing; use `age_category` / `age_generation` as proxies (Section 7.3) |
| OQ-4 | Number of unique customers | **To be computed during EDA** (US-1.3) | No change to requirements |
| OQ-5 | Additional variables available? | **No** — the 34 columns in the CSV are the complete feature space | Constraint C6 added: no external feature enrichment possible |
| OQ-6 | Target number of segments | **10 to 30 segments, depending on results** — final count driven by clustering metrics and marketing operability | Goal G1 updated; FR-4.2 range extended |
| OQ-7 | Preferred clustering algorithm? | **Compare all algorithms** — no pre-selected approach | FR-4.1 confirmed: minimum 3 algorithms evaluated and compared |

---

## Appendix A — Dimension Families for Feature Engineering

The following 6 dimension families must each be represented by at least one feature in the clustering input:

| Family | Example Features |
|---|---|
| **RFM** | `recency_days`, `frequency`, `monetary_avg`, `monetary_total` |
| **Purchase Behavior** | `avg_basket_size`, `units_per_basket`, `discount_rate`, `nb_unique_brands` |
| **Purchase Recurrence** | `dominant_axe`, `axis_diversity_index`, `brand_loyalty_rate`, `nb_axes_purchased` |
| **Channel** | `dominant_channel` (store/estore/click&collect), `store_ratio`, `estore_ratio`, `click_collect_ratio`, `store_type_app_dominant`, `nb_unique_stores` |
| **Sociodemographic** | `gender`, `age_category`, `age_generation`, `store_city` |
| **Lifecycle** | `loyalty_status`, `subscription_tenure_days`, `first_purchase_axe`, `first_purchase_amount` |

---

## Appendix B — Persona Card Template

```
┌──────────────────────────────────────────────────────────────┐
│  SEGMENT {N} — "{Persona Name}"                              │
│  Size: {N}% of customer base                                 │
├──────────────────────────────────────────────────────────────┤
│  ARCHETYPE: {One-line description}                           │
├──────────────────────────────────────────────────────────────┤
│  KEY TRAITS:                                                 │
│  • {Trait 1 with value vs. avg}                              │
│  • {Trait 2 with value vs. avg}                              │
│  • {Trait 3 with value vs. avg}                              │
├──────────────────────────────────────────────────────────────┤
│  CHANNEL:        {Dominant channel}                          │
│  PRODUCT AXIS:   {Top 1-2 axes}                              │
│  VALUE TIER:     {High / Medium / Low vs. avg}               │
│  LOYALTY STATUS: {Distribution}                              │
├──────────────────────────────────────────────────────────────┤
│  MARKETING ACTION:                                           │
│  {Recommended campaign type + channel + offer}               │
└──────────────────────────────────────────────────────────────┘
```

---

*PRD created by John (PM Agent) — BMAD Method — Edram Project — March 25, 2026*
