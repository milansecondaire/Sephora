# Story 5.4: Top Distinguishing Features per Cluster

Status: done

## Story

As a Data Scientist,
I want to identify which features most strongly define each cluster,
so that persona narratives are grounded in statistical evidence.

## Acceptance Criteria

1. For each cluster: compute mean of each feature vs. rest-of-population mean
2. Rank features by absolute standardized difference (Cohen's d): `(cluster_mean - global_mean) / global_std`
3. Top 5 distinguishing features listed per cluster (positive = above average; negative = below)
4. Bar chart per cluster: top 5 features, x-axis = Cohen's d

## Tasks / Subtasks

- [x] Task 1 — Implement `compute_distinguishing_features()` in `src/profiling.py` (AC: 1–3)
  - [x] For each cluster: compute Cohen's d per feature
  - [x] Return dict: cluster_id → sorted features DataFrame with Cohen's d
- [x] Task 2 — Implement `plot_distinguishing_features()` in `src/visualization.py` (AC: 4)
  - [x] One horizontal bar chart per cluster (top 5 pos + top 5 neg)
  - [x] Save to `figures/distinguishing_features_cluster_{n}.png` per cluster
- [x] Task 3 — Add notebook section in `03_profiling.ipynb`
  - [x] Display top 5 features per cluster as tables
  - [x] Display bar charts

## Dev Notes

### Architecture Guardrails

**Module:** `src/profiling.py` + `src/visualization.py`.  
**Notebook:** `03_profiling.ipynb` — E5 section, after US-5.3.

**Cohen's d computation:**
```python
def compute_distinguishing_features(df: pd.DataFrame, features: list) -> dict:
    """Return top distinguishing features per cluster via Cohen's d."""
    global_mean = df[features].mean()
    global_std = df[features].std()
    result = {}
    for cluster_id in df['cluster_id'].unique():
        cluster_df = df[df['cluster_id'] == cluster_id]
        cluster_mean = cluster_df[features].mean()
        cohens_d = (cluster_mean - global_mean) / (global_std + 1e-10)
        result[cluster_id] = cohens_d.abs().sort_values(ascending=False).reset_index()
        result[cluster_id].columns = ['feature', 'cohens_d_abs']
        result[cluster_id]['cohens_d'] = (cluster_mean - global_mean)[result[cluster_id]['feature'].values].values / (global_std[result[cluster_id]['feature'].values].values + 1e-10)
    return result
```

**This story's output directly feeds E6 persona writing** — the top distinguishing features are the foundation for persona name choices and narrative descriptions.

**Features to use:** `CLUSTERING_FEATURES` or `NUMERICAL_KPIS` — use the same feature set that was used for clustering.

### Previous Story Output (5-3)
- `delta_df` available
- `df_customers` with `cluster_id` and all features

### Output of This Story
- `distinguishing_features`: dict per cluster of top distinguishing features
- `figures/distinguishing_features_cluster_{n}.png` for each cluster

### References
- [epics — US-5.4](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None — clean implementation, all tests passed first try.

### Code Review Action Items
- [x] [HIGH] AC1 fix: use rest_mean and rest_std from non-cluster customers instead of global population.
- [x] [MEDIUM] Notebook fix: changed `print` to `display` for HTML table rendering.
- [x] [LOW] Performance: Refactored `compute_distinguishing_features` to use `df.groupby('cluster_id')` instead of full dataset masking.
- [x] [MEDIUM] Git state adjusted manually (files added via terminal).

### Completion Notes List
- Task 1: `compute_distinguishing_features()` added to `src/profiling.py` — computes Cohen's d per feature per cluster, returns dict of sorted DataFrames. 9 unit tests added covering formula correctness, sorting, edge cases.
- Task 2: `plot_distinguishing_features()` added to `src/visualization.py` — horizontal bar chart per cluster (top 5 pos + top 5 neg), green/red coloring, saves to figures/. 5 unit tests added.
- Task 3: 3 cells added to `03_profiling.ipynb` after US-5.3: markdown header, table display, bar chart generation.
- Full test suite: 383 passed, 1 pre-existing failure (test_config.py — unrelated to US-5.4).

### File List
Files modified:
- `src/profiling.py` — added `compute_distinguishing_features()`
- `src/visualization.py` — added `plot_distinguishing_features()`
- `03_profiling.ipynb` — added US-5.4 section (3 cells)
- `tests/test_profiling.py` — added 9 tests for `compute_distinguishing_features()`
- `tests/test_visualization.py` — added 5 tests for `plot_distinguishing_features()`

Files generated:
- `figures/distinguishing_features_cluster_{n}.png` — one per cluster
