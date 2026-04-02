# Story 3.4: PCA — Variance Analysis (REVISED)

Status: done

## Story

As a Data Scientist,
I want to run PCA on the 70-feature scaled matrix,
so that I understand explained variance and can decide whether to reduce dimensionality before clustering.

## Acceptance Criteria

1. PCA fitted on `X_scaled` (70 features) with `n_components = min(n_features, 30)` — increased from 20 due to richer feature set
2. Cumulative explained variance curve plotted (x = n components, y = % variance explained)
3. Number of components explaining ≥ 80% and ≥ 90% of variance identified
4. Top 3 components' loading bar charts produced (which features contribute most — expect one-hot dummies to appear)
5. Decision documented: cluster on PCA components OR on full scaled features? (with rationale)
6. If PCA used: `X_pca` matrix created with chosen `n_components`
7. Note: with 70 features including one-hot dummies, PCA reduction is likely more beneficial than with the old 21-feature set

## Tasks / Subtasks

- [x] Task 1 — Implement `apply_pca()` in `src/preprocessing.py` (AC: 1, 6)
  - [x] Fit PCA with `n_components=min(n_features, 30)`, `random_state=RANDOM_STATE`
  - [x] Return PCA object + `X_pca` DataFrame (if n_components chosen)
- [x] Task 2 — Implement `plot_pca_variance()` in `src/visualization.py` (AC: 2, 3, 4)
  - [x] Cumulative explained variance plot
  - [x] Mark 80% and 90% thresholds
  - [x] Bar charts for loadings of top 3 components
  - [x] Save to `figures/pca_variance.png` and `figures/pca_loadings.png`
- [x] Task 3 — Add notebook section in `02_clustering.ipynb` (AC: 5, 7)
  - [x] Display PCA results
  - [x] Markdown cell: decision — cluster on PCA or full scaled features?
  - [x] Note the impact of one-hot encoded dummies on PCA components

## Dev Notes

### Architecture Guardrails

**Module:** `src/preprocessing.py` + `src/visualization.py`.
**Notebook:** `02_clustering.ipynb` — E3 section, after US-3.3.

**REVISED: Input is now 70 features** (was 21). This changes the PCA analysis significantly:
- One-hot encoded dummies (37 columns) are binary 0/1 → PCA will need more components to explain variance
- `n_components` bumped to `min(n_features, 30)` to capture enough variance
- PCA reduction is MORE beneficial now: 70 → ~15-25 components likely explains 90%+

**PCA function signature:**
```python
from sklearn.decomposition import PCA
from src.config import RANDOM_STATE

def apply_pca(X: pd.DataFrame, n_components: int = None) -> tuple[pd.DataFrame, PCA]:
    """Fit PCA. If n_components is None, use min(n_features, 30) for analysis.
    Returns (X_pca DataFrame, fitted PCA object).
    """
    n = n_components or min(X.shape[1], 30)
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    components = pca.fit_transform(X)
    col_names = [f'PC{i+1}' for i in range(n)]
    return pd.DataFrame(components, index=X.index, columns=col_names), pca
```

**`random_state=RANDOM_STATE` MUST be passed to PCA** — reproducibility requirement.

**Decision guidance (updated for 70 features):**
- If ≥ 80% variance explained by <30 components → PCA components preferred (reduces noise from sparse one-hot dummies)
- Otherwise → use full `X_scaled` directly for clustering
- With 70 features, PCA is strongly recommended to avoid curse of dimensionality

### Previous Story Output (3-3 revised)
- `X_scaled`: (64469, 70) standardized feature DataFrame from `preprocess_for_clustering()`

### Output of This Story
- `X_pca` DataFrame (if PCA chosen) or confirmed use of `X_scaled`
- `X_cluster`: the variable name to use in E4 (either `X_pca` or `X_scaled`)
- `figures/pca_variance.png`, `figures/pca_loadings.png`

### References
- [epics — US-3.4 (REVISED)](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (GitHub Copilot)

### Debug Log References
No issues encountered.

### Completion Notes List
- `apply_pca()` added to `src/preprocessing.py` — fits PCA with `n_components=min(n_features, 30)`, `random_state=RANDOM_STATE`, returns `(X_pca DataFrame, PCA object)`
- `plot_pca_variance()` added to `src/visualization.py` — cumulative variance curve with 80%/90% threshold lines and annotations
- `plot_pca_loadings()` added to `src/visualization.py` — horizontal bar charts of top feature loadings for first N principal components
- 13 unit tests added in `tests/test_preprocessing.py` for `apply_pca()` (types, defaults, explicit components, index preservation, column names, reproducibility, variance properties)
- 8 unit tests added in `tests/test_visualization.py` for `plot_pca_variance()` and `plot_pca_loadings()` (figure return, save, subplots, capping)
- Notebook section E3.4 added with PCA fit, variance plot, loadings plot, decision markdown, and `X_cluster` creation
- Decision D3.4: Cluster on PCA components (reduces curse of dimensionality from 70 features, collapses sparse one-hot dummies)
- 1 pre-existing test failure in `test_feature_engineer_clustering.py` (unrelated to US-3.4 — tests raw `has_age_info` values after StandardScaler)

### File List
Files modified:
- `src/preprocessing.py` — added `apply_pca()`, imported `PCA` and `RANDOM_STATE`
- `src/visualization.py` — added `plot_pca_variance()` and `plot_pca_loadings()`, imported `PCA`
- `tests/test_preprocessing.py` — added 13 PCA tests (7 test classes)
- `tests/test_visualization.py` — added 8 PCA visualization tests (2 test classes)
- `02_clustering.ipynb` — added 6 cells (E3.4 section: markdown intro, PCA fit, variance plot, loadings plot, decision markdown, X_cluster creation)

### Code Review Updates
- `apply_pca` type signature updated to naturally accept float for fractional explained variance threshold cases.
- Notebook cells refactored to safely handle scenarios where n80/n90 thresholds might not be reached within the fitted number of components, preventing `IndexError`. Tests pass and workflow closed.
