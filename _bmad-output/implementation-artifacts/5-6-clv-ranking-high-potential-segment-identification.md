# Story 5.6: CLV Ranking & High-Potential Segment Identification

Status: done

## Story

As a Head of Marketing,
I want to see segments ranked by business value,
so that I know where to focus marketing investment first.

## Acceptance Criteria

1. Segments ranked by `monetary_total` (estimated CLV) descending
2. 2×2 prioritization matrix plotted: x = segment size (% of base), y = avg CLV; bubbles sized by total revenue contribution
3. "High potential" quadrant defined: above-median size AND above-median CLV
4. Top 3 priority segments identified and labeled
5. "Investment-worthy" segments (high CLV, small size) flagged for loyalty upgrade campaigns

## Tasks / Subtasks

- [x] Task 1 — Implement `rank_by_clv()` in `src/profiling.py` (AC: 1)
  - [x] Sort `cluster_kpis_df` by `monetary_total` desc
  - [x] Add `clv_tier` column: Top/Mid/Low (tertiles)
  - [x] Print CLV ranking table
- [x] Task 2 — Implement `plot_priority_matrix()` in `src/visualization.py` (AC: 2–5)
  - [x] Scatter plot: x = pct_customers, y = monetary_total (avg CLV), bubble size = total revenue
  - [x] Add quadrant lines at medians
  - [x] Label quadrants: "Grow" / "Nurture" / "Volume" / "Monitor"
  - [x] Add persona name labels (if available; else cluster ID)
  - [x] Save to `figures/priority_matrix.png`
- [x] Task 3 — Identify high-potential and investment-worthy segments (AC: 4, 5)
  - [x] Print top 3 priority segments by CLV
  - [x] Flag investment-worthy: high CLV + small size
- [x] Task 4 — Add notebook section in `03_profiling.ipynb` (E5 final section)
  - [x] E5 summary Markdown: all segments ranked with tier assignments
  - [x] Gate statement: "Profiling complete — proceeding to E6 Personas"

## Dev Notes

### Architecture Guardrails

**Module:** `src/profiling.py` + `src/visualization.py`.  
**Notebook:** `03_profiling.ipynb` — E5 final section. After this story, E5 is complete.

**CLV ranking:**
```python
def rank_by_clv(cluster_kpis: pd.DataFrame) -> pd.DataFrame:
    ranked = cluster_kpis.sort_values('monetary_total', ascending=False).copy()
    n = len(ranked)
    ranked['clv_tier'] = pd.cut(range(n), bins=3, labels=['Top', 'Mid', 'Low'])
    # Or simpler tertile:
    ranked['clv_tier'] = pd.qcut(ranked['monetary_total'], q=3, labels=['Low', 'Mid', 'Top'])
    return ranked
```

**Priority matrix:**
```python
def plot_priority_matrix(cluster_kpis: pd.DataFrame, save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    x = cluster_kpis['pct_customers']
    y = cluster_kpis['monetary_total']
    size = cluster_kpis['n_customers'] * cluster_kpis['monetary_total'] / 1000
    ax.scatter(x, y, s=size, alpha=0.7)
    # Quadrant lines
    ax.axvline(x.median(), linestyle='--', color='gray')
    ax.axhline(y.median(), linestyle='--', color='gray')
    # Labels
    for idx, row in cluster_kpis.iterrows():
        ax.annotate(str(idx), (row['pct_customers'], row['monetary_total']))
    ...
```

**Architecture D3.2:** `figures/priority_matrix.png` saved to `_bmad-output/implementation-artifacts/figures/`.

### Previous Story Output (5-5)
- `cluster_kpis_df` with all KPI means
- `delta_df` and full profiling results

### Output of This Story (E5 Final Output)
- `ranked_clusters_df` with `clv_tier`
- `figures/priority_matrix.png`
- E5 complete

### References
- [epics — US-5.6](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D3.2 Output File Locations](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None — all tasks implemented cleanly.
- [AI-Review] Fixed missing `duplicates='drop'` safety in `pd.qcut` by switching to `rank(method='first')`.
- [AI-Review] Added safety `ZeroDivisionError` fallback in bubble size calculation.
- [AI-Review] Added 2 new tests to ensure crash resiliency.

### Completion Notes List
- `rank_by_clv()`: sorts clusters desc by monetary_total, assigns clv_tier via pd.qcut tertiles (fallback for <3 clusters)
- `identify_high_potential_segments()`: returns top3_priority, high_potential (above-median size+CLV), investment_worthy (high CLV, small size)
- `plot_priority_matrix()`: bubble scatter with median quadrant lines, 4 labeled quadrants (Grow/Nurture/Volume/Monitor), cluster annotations
- Notebook: 5 new cells (1 heading, 2 code, 1 viz, 1 gate markdown)
- Tests: 8 tests for rank_by_clv, 8 tests for identify_high_potential_segments, 6 tests for plot_priority_matrix
- Full suite: 425 passed, 0 failed

### File List
- `src/profiling.py` — added `rank_by_clv()`, `identify_high_potential_segments()`
- `src/visualization.py` — added `plot_priority_matrix()`
- `tests/test_profiling.py` — added TestRankByClv (9 tests), TestIdentifyHighPotentialSegments (8 tests)
- `tests/test_visualization.py` — added TestPlotPriorityMatrix (7 tests)
- `03_profiling.ipynb` — added US-5.6 section (CLV ranking, segment identification, priority matrix, E5 gate)
