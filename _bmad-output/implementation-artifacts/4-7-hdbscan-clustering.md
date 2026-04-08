# Story 4.7: HDBSCAN Clustering

Status: done

## Story

As a Data Scientist,
I want to run HDBSCAN (Hierarchical Density-Based Spatial Clustering) on the customer feature matrix,
so that density-based natural groupings are tested as a complement to partition-based methods (K-Means, GMM).

## Acceptance Criteria

1. `run_hdbscan()` implemented in `src/clustering.py` using `sklearn.cluster.HDBSCAN` (sklearn >= 1.3 — available at 1.8.0)
2. A `min_cluster_size` sweep (100, 200, 500, 1000) run to identify the best value (maximise silhouette on non-noise points; minimise noise %)
3. Final HDBSCAN run with chosen parameters; cluster labels stored as `df_customers['hdbscan_label']` (noise = -1 kept as-is)
4. Noise statistics printed: absolute count and % of total customers labelled as noise
5. Silhouette, DB, CH scores computed **on non-noise points only**; appended to `comparison_results` with `'noise_pct'` field
6. UMAP scatter coloured by `hdbscan_label` saved to `figures/hdbscan_umap.png`
7. HDBSCAN run logged to MLflow as `"hdbscan-final"` with params/metrics schema consistent with US-4.2, 4.3, 4.4

## Tasks / Subtasks

- [x] Task 1 — Implement `run_hdbscan()` in `src/clustering.py` (AC: 1, 3)
  - [x] `from sklearn.cluster import HDBSCAN`
  - [x] Signature: `run_hdbscan(X, min_cluster_size=500, min_samples=5) -> tuple[ndarray, HDBSCAN]`
  - [x] Return `(labels, fitted_model)` — noise points have label `-1`
- [x] Task 2 — `min_cluster_size` sweep to choose best parameter (AC: 2)
  - [x] Sweep values: `[100, 200, 500, 1000]`
  - [x] For each: count clusters found (excl. noise), noise %, silhouette on non-noise (sample 10K if needed)
  - [x] Print summary table; choose value with best silhouette AND noise % < 10%
  - [x] If all configs exceed 10% noise, take the one with lowest noise %
- [x] Task 3 — Final HDBSCAN run + label assignment (AC: 3, 4)
  - [x] Run with chosen `min_cluster_size`, `min_samples=5`, `cluster_selection_method='eom'`
  - [x] `df_customers['hdbscan_label'] = labels`
  - [x] Print: `n_clusters`, `n_noise`, `noise_pct`
- [x] Task 4 — Compute metrics on non-noise points (AC: 5)
  - [x] `mask = labels != -1`; apply to `X_scaled` and `labels` before metric computation
  - [x] Guard: if fewer than 2 distinct clusters in non-noise → log NaN for silhouette, skip DB/CH
  - [x] Append to `comparison_results`
- [x] Task 5 — UMAP scatter plot (AC: 6)
  - [x] Reuse `umap_embedding` already computed in E3/E4 previous steps
  - [x] Color by `hdbscan_label`; noise points (-1) coloured grey
  - [x] Save to `figures/hdbscan_umap.png`
- [x] Task 6 — MLflow logging (AC: 7)
  - [x] `mlflow.start_run(run_name="hdbscan-final")`
  - [x] Log params: `min_cluster_size`, `min_samples`, `cluster_selection_method`, `n_features=33`
  - [x] Log metrics: `n_clusters`, `noise_pct`, `silhouette`, `davies_bouldin`, `calinski_harabasz`
- [x] Task 7 — Add notebook section in `02_clustering.ipynb` after US-4.4 GMM section

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — add `run_hdbscan()` to the existing module; do NOT create a new file.  
**Notebook:** `02_clustering.ipynb` — new section `## US-4.7 — HDBSCAN`, inserted after the GMM section (US-4.4) and before the comparison section (US-4.5).  
**Input data:** `X_scaled` — shape **(64469, 33)**, StandardScaler applied, no PCA.

**`sklearn.cluster.HDBSCAN` — usage (sklearn 1.8.0):**
```python
from sklearn.cluster import HDBSCAN

def run_hdbscan(
    X: pd.DataFrame,
    min_cluster_size: int = 500,
    min_samples: int = 5,
) -> tuple:
    """Fit HDBSCAN. Returns (labels ndarray, fitted model).
    Noise points are labelled -1.
    """
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',  # 'eom' (default) vs 'leaf'
    )
    labels = model.fit_predict(X)
    return labels, model
```

**Noise handling — mandatory pattern:**
```python
mask = labels != -1
X_no_noise = X_scaled.values[mask]
labels_no_noise = labels[mask]
n_clusters = len(set(labels_no_noise))
noise_pct = (~mask).mean() * 100

if n_clusters >= 2:
    sil_kwargs = {"random_state": RANDOM_STATE}
    if len(X_no_noise) > 10_000:
        sil_kwargs["sample_size"] = 10_000
    sil = silhouette_score(X_no_noise, labels_no_noise, **sil_kwargs)
    db = davies_bouldin_score(X_no_noise, labels_no_noise)
    ch = calinski_harabasz_score(X_no_noise, labels_no_noise)
else:
    sil = db = ch = float('nan')
```

**UMAP scatter with noise coloring:**
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.cm.tab10
unique_labels = sorted(set(hdbscan_labels))
for lbl in unique_labels:
    mask_lbl = hdbscan_labels == lbl
    color = 'grey' if lbl == -1 else cmap(lbl % 10)
    label_str = 'Noise' if lbl == -1 else f'Cluster {lbl}'
    ax.scatter(umap_embedding[mask_lbl, 0], umap_embedding[mask_lbl, 1],
               c=[color], s=2, alpha=0.5, label=label_str)
ax.set_title('HDBSCAN — UMAP projection')
ax.legend(markerscale=4, loc='best')
fig.tight_layout()
fig.savefig('figures/hdbscan_umap.png', dpi=150)
```

**`comparison_results` dict must include `'noise_pct'`** — the `build_comparison_table()` function uses `pd.DataFrame(comparison_results)`, so adding an extra key is non-breaking. Existing entries (KMeans, Hierarchical, GMM) simply get `NaN` for that column.

**`min_cluster_size` guidance:**
- 64K customers → 1% = 640 customers → `min_cluster_size=500` is a reasonable default
- Too small (100) → many micro-clusters and high noise %
- Too large (1000+) → under-segmentation
- Let the sweep decide; document chosen value in notebook markdown cell

**`min_samples` = 5** — standard default; controls density sensitivity (higher = more conservative, more noise).

**Performance:** HDBSCAN on 64K × 33 with sklearn 1.8 is fast (~10–30s). No subsampling needed.

**MLflow schema (consistent with US-4.2, 4.3, 4.4):**
```python
with mlflow.start_run(run_name="hdbscan-final"):
    mlflow.log_params({
        'algorithm': 'hdbscan',
        'min_cluster_size': min_cluster_size,
        'min_samples': 5,
        'cluster_selection_method': 'eom',
        'n_features': 33,
    })
    mlflow.log_metrics({
        'n_clusters': n_clusters,
        'noise_pct': noise_pct,
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
    })
```

### Previous Story Output (4-4 GMM)

- `df_customers['kmeans_label']`, `df_customers['hclust_label']`, `df_customers['gmm_label']`
- `comparison_results` list with 3 entries (KMeans, Hierarchical, GMM)
- `k_optimal` (int)
- `X_scaled` — DataFrame shape (64469, 33)
- `umap_embedding` — ndarray shape (64469, 2), computed in E3 / reused across E4

### Output of This Story

- `df_customers['hdbscan_label']` — int array, -1 = noise
- `figures/hdbscan_umap.png`
- HDBSCAN entry appended to `comparison_results`
- MLflow run `"hdbscan-final"`

### References

- [epics — E4 Clustering Models](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- sklearn 1.8 HDBSCAN docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 via GitHub Copilot

### Debug Log References

N/A

### Completion Notes List

- Task 1: `run_hdbscan()` added to `src/clustering.py` — imports `sklearn.cluster.HDBSCAN`, signature `(X, min_cluster_size=500, min_samples=5) -> tuple`, `cluster_selection_method='eom'`, labels cast to `np.intp`
- Task 1 Tests: 10 unit tests in `TestRunHdbscan` class covering returns, params, noise labels, fitting, integer dtype, eom method
- Task 2: Notebook cell with `min_cluster_size` sweep [100, 200, 500, 1000], selection logic (best silhouette + noise < 10%)
- Task 3: Final HDBSCAN run cell, label assignment to `df_customers['hdbscan_label']`, noise stats printed
- Task 4: Metrics cell with non-noise masking, guard for < 2 clusters, appended to `comparison_results` with `noise_pct` field
- Task 5: UMAP scatter cell with per-label coloring, noise in grey, saved to `figures/hdbscan_umap.png`
- Task 6: MLflow logging cell with `run_name="hdbscan-final"`, params/metrics schema consistent with US-4.2/4.3/4.4
- Task 7: All 6 cells + markdown header inserted in `02_clustering.ipynb` after US-4.4 GMM section and before US-4.5 comparison
- Updated US-4.5 comparison cell to include HDBSCAN entry, and `ALGO_TO_LABEL_COL` mapping updated with HDBSCAN
- Full test suite: 437/437 pass, 0 regressions

### File List

- `src/clustering.py` — added `run_hdbscan()`, added `HDBSCAN` import
- `tests/test_clustering.py` — added `TestRunHdbscan` class (10 tests)
- `03_profiling.ipynb` — Profiling metadata updated implicitly via Jupyter saving changes
- `02_clustering.ipynb` — added US-4.7 HDBSCAN section (1 markdown + 5 code cells), updated US-4.5 comparison cell + ALGO_TO_LABEL_COL

### Review Follow-ups (AI)
- [x] [HIGH] Filter out `-1` (noise) labels in `build_comparison_table()` calculation of `min_cluster_pct` for HDBSCAN results
- [x] [MEDIUM] Prevent completely redefining and overwriting `comparison_results` array in US-4.5 Task 1 of notebook
- [x] [MEDIUM] Document hidden `03_profiling.ipynb` changes inside the File List
- [x] [LOW] Use `tab20` (20 colors max) rather than `tab10` in US-4.7 Task 5 UMAP plot to easily support more clusters
