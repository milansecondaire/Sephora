# Story 5.1: Global KPI Baseline

Status: done

## Story

As a Marketing Manager,
I want to know the global average for each KPI across the full customer base,
so that every segment can be benchmarked against it.

## Acceptance Criteria

1. The following KPIs computed at global level: Avg Sales Value (€) `monetary_avg`, Purchase Frequency `frequency`, Avg Basket Size (€) `avg_basket_size_eur`, Avg Units per Basket `avg_units_per_basket`, Recency (days) `recency_days`, Channel Mix (%) store/estore/C&C ratios, Product Axis Mix (%) axe_* ratios, Avg Discount Rate (%) `discount_rate`, Loyalty Status Distribution % per status, CLV Estimate (€) `monetary_total`
2. Global baseline displayed as a formatted table
3. Total number of customers and total sales (€) printed

## Tasks / Subtasks

- [x] Task 1 — Implement `compute_global_kpis()` in `src/profiling.py` (AC: 1–3)
  - [x] Compute all 10 KPI groups as a dict or Series
  - [x] Return as formatted dict usable as baseline
- [x] Task 2 — Create `03_profiling.ipynb` and add E5 setup + US-5.1 section (AC: 2, 3)
  - [x] Load `customers_with_clusters.csv`
  - [x] Call `compute_global_kpis(df_customers)`
  - [x] Display formatted baseline table
  - [x] Print total customers + total sales

## Dev Notes

### Architecture Guardrails

**Module:** `src/profiling.py` — create this module.  
**Notebook:** `03_profiling.ipynb` — create this notebook. E5 section starts here.  
**Dependency:** E4 complete — `data/processed/customers_with_clusters.csv` must exist.

**Notebook setup cell:**
```python
from src.config import *
import numpy as np
np.random.seed(RANDOM_STATE)
import pandas as pd

df_customers = pd.read_csv(DATA_PROCESSED_PATH + "customers_with_clusters.csv",
                           index_col='anonymized_card_code')
print(f"Customers: {len(df_customers):,} | Clusters: {df_customers['cluster_id'].nunique()}")
```

**profiling.py skeleton:**
```python
# src/profiling.py
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
    global_kpis = df[NUMERICAL_KPIS].mean().to_dict()
    # Loyalty distribution
    global_kpis['loyalty_distribution'] = df['loyalty_status'].value_counts(normalize=True).to_dict()
    global_kpis['total_customers'] = len(df)
    global_kpis['total_sales_eur'] = df['monetary_total'].sum()
    return global_kpis
```

**Architecture note:** `profiling.py` input is the clustered DataFrame; output feeds `kpi_delta_table.md` (architecture D3.2).

### Previous Story Output (E4 complete)
- `data/processed/customers_with_clusters.csv` with `cluster_id` column

### Output of This Story
- `global_kpis` dict — baseline for all delta calculations in US-5.2/5.3
- `03_profiling.ipynb` created

### References
- [epics — US-5.1](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — profiling.py module boundary](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None

### Completion Notes List
- Task 1: Created `src/profiling.py` with `NUMERICAL_KPIS` list and `compute_global_kpis()` — computes mean of 15 numerical KPIs, loyalty status distribution, total customers, total sales.
- Task 1 tests: 9 unit tests in `tests/test_profiling.py` — all passing (returns dict, contains all KPIs, values are means, loyalty sums to 1, total customers/sales correct, single-row edge case).
- Task 2: Created `03_profiling.ipynb` with E5 setup cell + US-5.1 section (baseline table, loyalty distribution table, total customers & sales print).
- Note: `customers_with_clusters.csv` exported with `index=False` in E4 — notebook loads without `index_col`.
- Pre-existing failure: `test_config.py::TestFeatureCategories::test_required_features_present` (market_others_ratio mismatch) — not related to this story.
- **Code Review Fixes**: 
  - Updated `03_profiling.ipynb` to format percentage metrics properly (`.2%`).
  - Added `IPython.display` missing import.
  - Made `compute_global_kpis` robust against missing columns.
  - Created `pytest.ini` for smooth test discovery.

### File List
- `src/profiling.py` (created/updated)
- `tests/test_profiling.py` (created)
- `03_profiling.ipynb` (created/updated)
- `02_clustering.ipynb` (updated - tracked changes)
- `first-analysis.ipynb` (updated - tracked changes)
- `pytest.ini` (created)
