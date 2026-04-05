# Story R1.4 — Suppression de la PCA du pipeline de clustering

Status: done

## Story

As a Marketing Manager,
I want clustering to run directly on the scaled features without PCA dimensionality reduction,
so that each cluster can be explained feature-by-feature without abstract components.

## Acceptance Criteria

1. No call to `apply_pca()` appears in the clustering flow of `02_clustering.ipynb`
2. All clustering algorithms (KMeans, Hierarchical, GMM) receive `X_scaled` directly as input — not `X_pca`
3. The old US-3.4 PCA notebook section is replaced by a Markdown cell that explicitly states the decision not to use PCA for clustering and the marketing rationale
4. `apply_pca()` and `apply_umap()` functions in `src/preprocessing.py` are retained as-is (not deleted) — they may still be used for ad-hoc visualization
5. All clustering metric computations (silhouette, DB, CH) use the same `X_scaled` input
6. If a variable `X_pca` was previously created in the notebook, it is removed or commented out — clustering code must reference `X_scaled`

## Tasks / Subtasks

- [x] Task 1 — Replace the PCA notebook section (AC: 1, 3, 6)
  - [x] Find the existing US-3.4 / PCA section in `02_clustering.ipynb`
  - [x] Replace it with a single Markdown cell containing the no-PCA decision rationale
  - [x] Remove or comment out any code cells that create `X_pca` and pass it to clustering functions
  - [x] Ensure all subsequent clustering code references `X_scaled` (not `X_pca`)

- [x] Task 2 — Verify all clustering function calls use `X_scaled` (AC: 2, 5)
  - [x] Search notebook for: `X_pca`, `apply_pca`, `n_components`, `pca.fit_transform` in clustering cells
  - [x] For each occurrence in a clustering context: replace the input with `X_scaled`
  - [x] Verify `evaluate_kmeans_k_range(X_scaled, ...)` — not `X_pca`
  - [x] Verify `run_kmeans_final(X_scaled, k)` — not `X_pca`
  - [x] Verify `run_hierarchical(X_scaled, k)` — not `X_pca`

- [x] Task 3 — Confirm `apply_pca()` intact in `src/preprocessing.py` (AC: 4)
  - [x] Read `src/preprocessing.py` — confirm `apply_pca()` and `apply_umap()` are present
  - [x] Make NO changes to these functions

## Dev Notes

### Architecture Guardrails

**Notebook only:** `02_clustering.ipynb` — this story is exclusively a notebook change. No `src/` files are modified.
**Do NOT delete** `apply_pca()` or `apply_umap()` from `src/preprocessing.py`.

### What changes in the notebook flow

```
BEFORE (old pipeline):
  X_scaled (70 features)
      → apply_pca() → X_pca (20-30 components)
          → KMeans / Hierarchical / GMM

AFTER (new pipeline):
  X_scaled (43 features)
      → KMeans / Hierarchical / GMM   ← directly
```

### Performance note

64 469 customers × 43 features: KMeans with `n_init=10`, `max_iter=300` runs in well under 60 seconds on a standard laptop. No performance concern with skipping PCA.

### Umap visualization note

If the notebook previously used `X_pca` as input to `apply_umap()` for 2D visualization, switch it to `X_scaled`:
```python
umap_embedding = apply_umap(X_scaled, n_neighbors=15, min_dist=0.1)
```
UMAP can handle 43 features natively.

### Previous Story Output
- R1.2 complete: `X_scaled` of shape `(64469, 43)` produced by `preprocess_for_clustering()`
- R1.3 complete: correlation circle uses its own internal PCA — this does not affect clustering flow

### Output of This Story
- `02_clustering.ipynb` — PCA section replaced by decision Markdown cell; all clustering cells use `X_scaled`

### References
- [EPIC-R1](_bmad-output/implementation-artifacts/EPIC-R1-feature-refonte-and-mlflow.md) — US-R1.4

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (GitHub Copilot)

### Debug Log References
_none_

### Completion Notes List
- Deleted 4 cells: PCA apply code, PCA variance plot, PCA loadings plot, old PCA decision markdown
- Replaced E3.4 markdown with no-PCA decision rationale
- Replaced PCA component selection cell with simple `print(X_scaled.shape)` confirmation
- Global `X_cluster` → `X_scaled` replacement across all clustering cells (silhouette, KMeans, hierarchical, UMAP)
- Updated TOC (cell 0) and E3 Summary table (cell 24)
- `apply_pca()` and `apply_umap()` untouched in `src/preprocessing.py`
- 288 tests pass (0 failures)

### File List
Files modified:
- `02_clustering.ipynb` — PCA section replaced, all clustering inputs now use `X_scaled`
