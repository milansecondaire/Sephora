# Story 2.3: RFM Space Visualization

Status: done

## Story

As a Marketing Manager,
I want to see the customer population in RFM space,
so that I can develop an intuition for natural groupings.

## Acceptance Criteria

1. 2D scatter plots: Recency vs. Frequency, Recency vs. Monetary, Frequency vs. Monetary
2. Points colored by existing `RFM_Segment_ID` column (if present) to orient the reader
3. Points colored by `loyalty_status` as a second view
4. Log-scale applied where distributions are heavily right-skewed (monetary, frequency)
5. Observations about natural cluster visibility documented in Markdown

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_rfm_scatter()` in `src/visualization.py` (AC: 1–4)
  - [x] Create 3 scatter plots: R vs F, R vs M, F vs M
  - [x] Color by `loyalty_status`; if `RFM_Segment_ID` column exists, add a second row with that coloring
  - [x] Apply log scale to frequency and monetary axes
  - [x] Save to `figures/rfm_scatter_loyalty.png` and `figures/rfm_scatter_rfmseg.png`
- [x] Task 2 — Add notebook section in `first-analysis.ipynb` (AC: 5)
  - [x] Check if `RFM_Segment_ID` is available in `df_customers`
  - [x] Call `plot_rfm_scatter(df_customers)`
  - [x] Markdown cell: qualitative observations about natural cluster visibility

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E2 section, after US-2.2.

**RFM_Segment_ID note:** The raw CSV may contain an `RFM_Segment_ID` column (check during E1-US-1.1 column list). If not present in `df_customers`, color by `loyalty_status` only. Handle gracefully:
```python
color_col = 'RFM_Segment_ID' if 'RFM_Segment_ID' in df_customers.columns else 'loyalty_status'
```

**Log scale pattern:**
```python
ax.set_xscale('log')  # for frequency, monetary axes only — NOT recency
ax.set_xlabel('Frequency (log scale)')
```

**Palette:** Use `SEGMENT_COLORS` from `src/config.py` for consistent coloring.

**Sampling for readability:** If customer count > 50,000, sample 10,000 points for scatter:
```python
plot_df = df_customers.sample(min(10000, len(df_customers)), random_state=RANDOM_STATE)
```

### Previous Story Output (2-2)
- `df_customers` with all features including `recency_days`, `frequency`, `monetary_total`
- `src/visualization.py` exists

### Output of This Story
- `figures/rfm_scatter_loyalty.png`
- `figures/rfm_scatter_rfmseg.png` (if `RFM_Segment_ID` available)

### References
- [epics — US-2.3](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — Visualization Patterns](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No issues encountered.

### Completion Notes List
- Added `plot_rfm_scatter()` with log-scale, loyalty/RFM_Segment_ID coloring, 10k sampling for large datasets
- Tests: `TestPlotRfmScatter` (4 tests) — all passing incl. conditional RFM_Segment_ID handling
- Notebook cell E2-3a + markdown observations on natural cluster visibility

### File List
Files modified:
- `src/visualization.py` (added `plot_rfm_scatter()`)
- `tests/test_visualization.py` (added RFM scatter tests)
- `first-analysis.ipynb` (added US-2.3 section)

### Code Review Fixes List
- Standardized the scatterplot palette to use `SEGMENT_COLORS` from `/src/config.py`.
- Added `nonpositive="clip"` keyword argument strictly to `ax.set_xscale("log")` and `ax.set_yscale("log")` to prevent log-negative value warnings with zero-dollar purchases or negative returns.
