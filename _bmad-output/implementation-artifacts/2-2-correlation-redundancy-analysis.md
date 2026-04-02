# Story 2.2: Correlation & Redundancy Analysis

Status: done

## Story

As a Data Analyst,
I want to measure correlations between features,
so that I can remove redundant variables before clustering.

## Acceptance Criteria

1. Pearson correlation heatmap for all numerical features
2. Pairs with |r| > 0.85 listed explicitly with a decision: keep / drop / transform
3. Spearman correlation computed for ordinal features (`loyalty_numeric`, `axis_diversity`)
4. At minimum document the expected high correlation: `monetary_total` ↔ `monetary_avg` ↔ `avg_basket_size_eur`

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_correlation_heatmap()` in `src/visualization.py` (AC: 1)
  - [x] Compute Pearson correlation matrix for all numeric columns
  - [x] Render seaborn heatmap with annotations
  - [x] Save to `figures/correlation_heatmap.png`
- [x] Task 2 — Identify high-correlation pairs (AC: 2)
  - [x] Extract pairs where `|r| > 0.85` from correlation matrix (upper triangle)
  - [x] Display as table: Feature A | Feature B | r | Decision
- [x] Task 3 — Spearman for ordinal features (AC: 3)
  - [x] Compute Spearman correlation for `['loyalty_numeric', 'axis_diversity', 'is_new_customer']`
  - [x] Print or display result
- [x] Task 4 — Add notebook section in `first-analysis.ipynb` (AC: 4)
  - [x] Call functions, display outputs
  - [x] Markdown cell: list high-correlation pairs and explicit keep/drop/transform decisions

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E2 section, after US-2.1.

**High-correlation pair extraction:**
```python
import numpy as np
def get_high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    corr = df.select_dtypes('number').corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = upper.stack()
    return high_corr[high_corr.abs() > threshold].reset_index()
```

**Expected high-correlation pairs** (document in Markdown even if not statistically confirmed):
- `monetary_total` ↔ `monetary_avg` ↔ `avg_basket_size_eur`
- `frequency` ↔ `total_lines`
- `total_sales_eur` ↔ `monetary_total` (same column, just renamed)

**This story's decisions feed US-3.1** (Feature Selection). Document clearly in Markdown which features will be dropped.

### Previous Story Output (2-1)
- `df_customers` loaded from `customers_features.csv`
- `src/visualization.py` exists

### Output of This Story
- `figures/correlation_heatmap.png`
- Documented correlation decisions in notebook Markdown cell

### References
- [epics — US-2.2](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No issues encountered.

### Completion Notes List
- Added `plot_correlation_heatmap()` and `get_high_correlation_pairs()` to `src/visualization.py`
- Tests: `TestPlotCorrelationHeatmap` (2 tests), `TestGetHighCorrelationPairs` (4 tests) — all passing
- Notebook cells E2-2a/b/c + markdown decisions for high-correlation pairs
- Spearman correlation computed for ordinal features

### File List
Files modified:
- `src/visualization.py` (added `plot_correlation_heatmap`, `get_high_correlation_pairs`)
- `tests/test_visualization.py` (added correlation tests)
- `first-analysis.ipynb` (added US-2.2 section)

### Code Review Fixes List
- Applied `.style.background_gradient(cmap="RdBu_r")` to Spearman correlation dataframe in notebook cell E2-2c for better readability.
