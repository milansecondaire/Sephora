# Story 5.6: CLV Ranking & High-Potential Segment Identification

Status: ready-for-dev

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

- [ ] Task 1 — Implement `rank_by_clv()` in `src/profiling.py` (AC: 1)
  - [ ] Sort `cluster_kpis_df` by `monetary_total` desc
  - [ ] Add `clv_tier` column: Top/Mid/Low (tertiles)
  - [ ] Print CLV ranking table
- [ ] Task 2 — Implement `plot_priority_matrix()` in `src/visualization.py` (AC: 2–5)
  - [ ] Scatter plot: x = pct_customers, y = monetary_total (avg CLV), bubble size = total revenue
  - [ ] Add quadrant lines at medians
  - [ ] Label quadrants: "Grow" / "Nurture" / "Volume" / "Monitor"
  - [ ] Add persona name labels (if available; else cluster ID)
  - [ ] Save to `figures/priority_matrix.png`
- [ ] Task 3 — Identify high-potential and investment-worthy segments (AC: 4, 5)
  - [ ] Print top 3 priority segments by CLV
  - [ ] Flag investment-worthy: high CLV + small size
- [ ] Task 4 — Add notebook section in `03_profiling.ipynb` (E5 final section)
  - [ ] E5 summary Markdown: all segments ranked with tier assignments
  - [ ] Gate statement: "Profiling complete — proceeding to E6 Personas"

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
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/profiling.py` (add `rank_by_clv()`)
- `src/visualization.py` (add `plot_priority_matrix()`)
- `03_profiling.ipynb` (add US-5.6 section + E5 summary)
