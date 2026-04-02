# Story 5.3: Delta Table (Clusters vs. Global Average)

Status: ready-for-dev

## Story

As a Marketing Manager,
I want to see how each cluster deviates from the global average in absolute and relative terms,
so that I can immediately identify above-average and below-average segments.

## Acceptance Criteria

1. For each numerical KPI: `delta_abs = cluster_value - global_avg`; `delta_pct = (cluster_value / global_avg - 1) * 100`
2. Delta table columns: Cluster | KPI | Global Avg | Cluster Value | Delta Abs | Delta %
3. Color-coded display: green if positive delta (above avg), red if negative
4. Clusters sorted by `monetary_total` descending (highest CLV first)
5. Notable deltas (|delta %| > 30%) highlighted in narrative summary
6. Delta table exported as `_bmad-output/implementation-artifacts/kpi_delta_table.md`

## Tasks / Subtasks

- [ ] Task 1 — Implement `build_delta_table()` in `src/profiling.py` (AC: 1–2)
  - [ ] Compute delta_abs and delta_pct for each cluster × KPI combination
  - [ ] Return long-format DataFrame
- [ ] Task 2 — Sort and highlight (AC: 4, 5)
  - [ ] Sort clusters by `monetary_total` desc
  - [ ] Identify notable deltas
- [ ] Task 3 — Export `kpi_delta_table.md` (AC: 6)
  - [ ] Write formatted Markdown table to `OUTPUT_PATH + "kpi_delta_table.md"`
- [ ] Task 4 — Color-coded display in notebook (AC: 3)
  - [ ] Use pandas Styler with `background_gradient` or conditional formatting
- [ ] Task 5 — Add notebook section in `03_profiling.ipynb` (AC: 5)
  - [ ] Narrative summary cell of notable deltas

## Dev Notes

### Architecture Guardrails

**Module:** `src/profiling.py` — add to existing module.  
**Notebook:** `03_profiling.ipynb` — E5 section, after US-5.2.  
**Output file path:** `OUTPUT_PATH + "kpi_delta_table.md"` (from `src/config.py`).

**Delta table function:**
```python
def build_delta_table(cluster_kpis: pd.DataFrame, global_kpis: dict) -> pd.DataFrame:
    """Build long-format delta table: cluster × KPI × delta_abs × delta_pct."""
    rows = []
    for cluster_id, row in cluster_kpis.iterrows():
        for kpi in NUMERICAL_KPIS:
            global_val = global_kpis.get(kpi, np.nan)
            cluster_val = row[kpi]
            delta_abs = cluster_val - global_val
            delta_pct = ((cluster_val / global_val) - 1) * 100 if global_val != 0 else np.nan
            rows.append({
                'cluster_id': cluster_id,
                'kpi': kpi,
                'global_avg': round(global_val, 4),
                'cluster_value': round(cluster_val, 4),
                'delta_abs': round(delta_abs, 4),
                'delta_pct': round(delta_pct, 2),
            })
    return pd.DataFrame(rows)
```

**Markdown export:**
```python
import os
def export_delta_table_md(delta_df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# KPI Delta Table — Clusters vs. Global Average\n\n")
        f.write(delta_df.to_markdown(index=False))
```

**Architecture D3.2:** `kpi_delta_table.md` saved to `_bmad-output/implementation-artifacts/`.

### Previous Story Output (5-2)
- `cluster_kpis_df`
- `global_kpis` dict

### Output of This Story
- `delta_df`: long-format delta table
- `_bmad-output/implementation-artifacts/kpi_delta_table.md`

### References
- [epics — US-5.3](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D3.2 Output File Locations](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/profiling.py` (add `build_delta_table()`, `export_delta_table_md()`)
- `03_profiling.ipynb` (add US-5.3 section)

Files to create:
- `_bmad-output/implementation-artifacts/kpi_delta_table.md`
