# Story 2.1: Univariate Distribution Analysis

Status: done

## Story

As a Data Analyst,
I want to visualize the distribution of every key feature,
so that I can identify skewness, outliers, and encoding issues before modeling.

## Acceptance Criteria

1. Histogram + KDE for each numerical feature: `recency_days`, `frequency`, `monetary_total`, `monetary_avg`, `avg_basket_size_eur`, `discount_rate`, `subscription_tenure_days`
2. Bar chart for each categorical feature: `loyalty_status`, `gender`, `dominant_channel`, `dominant_axe`, `dominant_market`, `country`, `age_generation`
3. Summary stats table (mean, median, std, min, max, % missing) for all features
4. Date range of `transactionDate` confirmed and printed (expected: Jan–Dec 2025)
5. Number of unique customers printed

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_numerical_distributions()` in `src/visualization.py` (AC: 1)
  - [x] Iterate over numerical feature list; produce subplot grid of histograms + KDE
  - [x] Save figure to `_bmad-output/implementation-artifacts/figures/distributions_numerical.png`
- [x] Task 2 — Implement `plot_categorical_distributions()` in `src/visualization.py` (AC: 2)
  - [x] Iterate over categorical feature list; produce bar charts in subplot grid
  - [x] Save figure to `figures/distributions_categorical.png`
- [x] Task 3 — Compute and display summary stats table (AC: 3)
  - [x] Build stats DataFrame: mean, median, std, min, max, % missing per feature
  - [x] Display as styled table in notebook
- [x] Task 4 — Add notebook section in `first-analysis.ipynb` (AC: 4, 5)
  - [x] Load `customers_features.csv` if not already in memory
  - [x] Print unique customers, date range from `last_purchase_date`
  - [x] Call visualization functions and display inline
  - [x] Markdown cell with initial observations

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` — create if it doesn't exist yet.  
**Notebook:** `01_eda.ipynb` — E2 section starts here.  
**Dependency:** E1 must be complete — reads `data/processed/customers_features.csv`.

**Notebook entry point for E2:**
```python
import pandas as pd
from src.config import DATA_PROCESSED_PATH
df_customers = pd.read_csv(DATA_PROCESSED_PATH + "customers_features.csv", 
                            index_col='anonymized_card_code')
print(f"Customers loaded: {len(df_customers):,}")
```

**Visualization module pattern (from architecture):**
```python
# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGSIZE_BAR, FIGSIZE_SCATTER, FIGURE_DPI, OUTPUT_PATH
import os

def plot_numerical_distributions(df: pd.DataFrame, features: list, save_path: str = None) -> None:
    """Plot histograms + KDE for numerical features. Save to figures/ if save_path provided."""
    figures_dir = OUTPUT_PATH + "figures/"
    os.makedirs(figures_dir, exist_ok=True)
    ...
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.tight_layout()    # always before savefig
    plt.show()
```

**Figure standards (from architecture):**
- Default figsize: `(12, 6)` for bar/line, `(10, 8)` for scatter
- DPI: 300 for all `savefig()` calls
- Always call `plt.tight_layout()` before `savefig()`
- Titles and axis labels in **English**

**Numerical features to plot:** `['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'avg_basket_size_eur', 'discount_rate', 'subscription_tenure_days']`  
**Categorical features to plot:** `['loyalty_status', 'gender', 'dominant_channel', 'dominant_axe', 'dominant_market', 'country', 'age_generation']`

### Previous Story Output (1-7 / E1 complete)
- `data/processed/customers_features.csv` — load at start of E2
- Full feature matrix available

### Output of This Story
- `figures/distributions_numerical.png`
- `figures/distributions_categorical.png`
- `01_eda.ipynb` with E2.1 analysis section

### References
- [epics — US-2.1](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — Visualization Patterns](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — Module: visualization.py](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No issues encountered.

### Completion Notes List
- Created `src/visualization.py` with `plot_numerical_distributions()`, `plot_categorical_distributions()`, `compute_summary_stats()`
- 28 unit tests written in `tests/test_visualization.py` — all passing
- Notebook sections added: E2 header, E2-1a/b/c/d cells + markdown observations
- Figures saved to `_bmad-output/implementation-artifacts/figures/`

### File List
Files created:
- `src/visualization.py`
- `tests/test_visualization.py`

Files modified:
- `first-analysis.ipynb` (added E2 header + US-2.1 section, updated cell `#VSC-91a9fb64` to use `transactionDate`)

### Code Review Fixes List
- Handled KDE exception in `plot_numerical_distributions` to warn instead of dropping.
- Rotated x-axis labels with `set_ha('right')` for `plot_categorical_distributions`.
- Added `nunique` field to `compute_summary_stats`.
- Updated notebook Cell `E2-1d` to use explicit `transactionDate` to satisfy AC4.
