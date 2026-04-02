# Story 4.3: Agglomerative Hierarchical Clustering

Status: done

## Story

As a Data Scientist,
I want to run hierarchical clustering and compare it to K-Means,
so that the segmentation is not biased toward a single algorithm's assumptions.

## Acceptance Criteria

1. Dendrogram plotted (truncated to last 30 merges for readability)
2. `AgglomerativeClustering` with `linkage='ward'` run for same k as K-Means optimal
3. Cluster labels stored as `df_customers['hclust_label']`
4. Silhouette, DB, CH scores computed and added to comparison table (to be finalized in US-4.5)
5. Visual comparison: UMAP colored by K-Means labels vs. Hierarchical labels side-by-side

## Tasks / Subtasks

- [x] Task 1 — Implement `run_hierarchical()` in `src/clustering.py` (AC: 2–4)
  - [x] Fit `AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')`
  - [x] Return labels
- [x] Task 2 — Plot dendrogram (AC: 1)
  - [x] Use `scipy.cluster.hierarchy.dendrogram` with `truncate_mode='lastp', p=30`
  - [x] Save to `figures/dendrogram.png`
- [x] Task 3 — Side-by-side UMAP comparison (AC: 5)
  - [x] Reuse `umap_embedding` from E3; color by `kmeans_label` and `hclust_label`
  - [x] Save to `figures/umap_kmeans_vs_hclust.png`
- [x] Task 4 — Add notebook section in `02_clustering.ipynb`
  - [x] Collect metrics into `comparison_rows` list (used by US-4.5)

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — add to existing module.  
**Notebook:** `02_clustering.ipynb` — E4 section, after US-4.2.

**AgglomerativeClustering does NOT accept `random_state`** — no seed needed.

**Dendrogram requires scipy and a fitted linkage matrix:**
```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def plot_dendrogram(X: pd.DataFrame, save_path: str = None) -> None:
    Z = linkage(X, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Clustering Dendrogram (last 30 merges)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.show()
```

**Performance note:** `linkage()` on large datasets (>50k rows) is very slow (O(n²)). Sample if needed:
```python
X_sample = X_cluster.sample(min(10000, len(X_cluster)), random_state=RANDOM_STATE)
# Dendrogram on sample; fit AgglomerativeClustering on full data
```

**`AgglomerativeClustering` on full data is fine** — it doesn't build a full distance matrix for `ward` linkage.

**Comparison table:** Initialize in notebook before US-4.3, append to it across US-4.3, 4.4, 4.5:
```python
comparison_results = []
# After each algorithm, append:
comparison_results.append({
    'algorithm': 'KMeans', 'k': k_optimal, 'silhouette': ..., 'db': ..., 'ch': ...
})
```

### Previous Story Output (4-2)
- `df_customers['kmeans_label']`
- `umap_embedding` (from E3)
- `k_optimal`

### Output of This Story
- `df_customers['hclust_label']`
- `figures/dendrogram.png`
- `figures/umap_kmeans_vs_hclust.png`
- Hierarchical scores appended to `comparison_results`

### References
- [epics — US-4.3](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Sonnet 4.6

### Debug Log References
— none —

### Completion Notes List
- `run_hierarchical()` added to `src/clustering.py` using `AgglomerativeClustering(linkage='ward')`.
- `plot_dendrogram()` added to `src/visualization.py`; samples ≤10k rows for performance (scipy linkage is O(n²)).
- `plot_umap_kmeans_vs_hclust()` added to `src/visualization.py`; side-by-side 2-panel figure saved to `figures/umap_kmeans_vs_hclust.png`.
- 3 notebook cells added to `02_clustering.ipynb` after US-4.2 section (markdown header + 2 code cells).
- `comparison_results` list initialised in notebook with KMeans + Hierarchical entries (ready for US-4.4 & 4.5).
- 12 new unit tests written across `TestRunHierarchical`, `TestPlotDendrogram`, `TestPlotUmapKmeansVsHclust` — all 37 clustering tests pass.

### File List
Files modified:
- `src/clustering.py` — added `run_hierarchical()`
- `src/visualization.py` — added `plot_dendrogram()`, `plot_umap_kmeans_vs_hclust()`
- `tests/test_clustering.py` — added `TestRunHierarchical`, `TestPlotDendrogram`, `TestPlotUmapKmeansVsHclust`
- `02_clustering.ipynb` — added US-4.3 section (3 cells)
- `src/visualization.py` (add `plot_dendrogram()`)
- `02_clustering.ipynb` (add US-4.3 section)
