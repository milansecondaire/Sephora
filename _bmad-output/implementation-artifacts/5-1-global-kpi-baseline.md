# Story 5.1: Global KPI Baseline

Status: ready-for-dev

## Story

As a Marketing Manager,
I want to know the global average for each KPI across the full customer base,
so that every segment can be benchmarked against it.

## Acceptance Criteria

1. The following KPIs computed at global level: Avg Sales Value (€) `monetary_avg`, Purchase Frequency `frequency`, Avg Basket Size (€) `avg_basket_size_eur`, Avg Units per Basket `avg_units_per_basket`, Recency (days) `recency_days`, Channel Mix (%) store/estore/C&C ratios, Product Axis Mix (%) axe_* ratios, Avg Discount Rate (%) `discount_rate`, Loyalty Status Distribution % per status, CLV Estimate (€) `monetary_total`
2. Global baseline displayed as a formatted table
3. Total number of customers and total sales (€) printed

## Tasks / Subtasks

- [ ] Task 1 — Implement `compute_global_kpis()` in `src/profiling.py` (AC: 1–3)
  - [ ] Compute all 10 KPI groups as a dict or Series
  - [ ] Return as formatted dict usable as baseline
- [ ] Task 2 — Create `03_profiling.ipynb` and add E5 setup + US-5.1 section (AC: 2, 3)
  - [ ] Load `customers_with_clusters.csv`
  - [ ] Call `compute_global_kpis(df_customers)`
  - [ ] Display formatted baseline table
  - [ ] Print total customers + total sales

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
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to create:
- `src/profiling.py` (create — add `compute_global_kpis()`, `NUMERICAL_KPIS`)
- `03_profiling.ipynb` (create — add notebook setup + US-5.1 section)
