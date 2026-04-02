# Story 4.1: K-Means — Optimal k Selection

Status: done

## Story

As a Data Scientist,
I want to find the optimal number of clusters for K-Means,
so that the elbow and silhouette criteria jointly inform the final k choice.

## Acceptance Criteria

1. K-Means evaluated for k = 2 to 30 (`K_RANGE` from config)
2. Inertia (within-cluster sum of squares) plotted vs. k — elbow method
3. Silhouette score plotted vs. k
4. Davies-Bouldin score plotted vs. k
5. Top 3 candidate values of k identified with scores tabulated
6. `random_state=RANDOM_STATE`, `n_init=10`, `max_iter=300`

## Tasks / Subtasks

- [x] Task 1 — Implement `evaluate_kmeans_k_range()` in `src/clustering.py` (AC: 1, 6)
  - [x] Loop over `K_RANGE`; fit KMeans; collect inertia, silhouette, DB scores
  - [x] Return results dict or DataFrame
- [x] Task 2 — Implement `plot_elbow_curves()` in `src/visualization.py` (AC: 2, 3, 4)
  - [x] 3-panel plot: inertia, silhouette, Davies-Bouldin vs. k
  - [x] Save to `figures/elbow_kmeans.png`
- [x] Task 3 — Identify top 3 k candidates (AC: 5)
  - [x] Rank by silhouette; flag elbow visually
  - [x] Display candidate table: k | Inertia | Silhouette ↑ | DB ↓
- [x] Task 4 — Add notebook section in `02_clustering.ipynb` (E4 section starts here)
  - [x] Call evaluation + plot functions
  - [x] Markdown cell: top 3 k candidates with justification

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — create this module.  
**Notebook:** `02_clustering.ipynb` — E4 section starts here.  
**Dependency:** `X_cluster` from E3 must be available.

**clustering.py skeleton:**
```python
# src/clustering.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.config import RANDOM_STATE, K_RANGE
import pandas as pd

def evaluate_kmeans_k_range(X: pd.DataFrame) -> pd.DataFrame:
    """Evaluate KMeans for all k in K_RANGE. Returns metrics DataFrame."""
    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        results.append({
            'k': k,
            'inertia': km.inertia_,
            'silhouette': silhouette_score(X, labels, sample_size=10000, random_state=RANDOM_STATE),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(results)
```

**Runtime note:** `silhouette_score` on large datasets is slow. Use `sample_size=10000` parameter to cap computation time while keeping representative scores.

**`K_RANGE = range(2, 31)` from `src/config.py`** — never hardcode k range.

**Architecture D2.2:** The dictionary pattern for clustering algorithms will be used in US-4.5. For this story, only KMeans evaluation is needed.

### Previous Story Output (E3 complete)
- `X_cluster`: scaled (or PCA) feature matrix, index=`anonymized_card_code`

### Output of This Story
- `kmeans_metrics_df`: metrics table for k=2..30
- `figures/elbow_kmeans.png`
- `k_optimal` variable in notebook (chosen from top 3 candidates)

### References
- [epics — US-4.1](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D2.2 Multi-Algorithm Evaluation Pattern](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 via GitHub Copilot

### Debug Log References
All 253 tests passing (17 new + 236 existing).

### Completion Notes List
- `evaluate_kmeans_k_range()`: loops k=2..30, collects inertia/silhouette/DB/CH. Uses `sample_size=10000` for silhouette on large datasets.
- `get_top_k_candidates()`: ranks by silhouette desc, returns top N.
- `plot_elbow_curves()`: 3-panel figure (inertia, silhouette, DB) with best-k vertical markers.
- Notebook E4 section: 4 cells (markdown header, evaluation, plot, top-3 table + markdown summary).
- Fixed code review items (used nlargest in candidates, fixed type hinting).

### File List
Files created:
- `src/clustering.py` — `evaluate_kmeans_k_range()`, `get_top_k_candidates()`
- `tests/test_clustering.py` — 17 tests (3 classes)

Files modified:
- `src/visualization.py` — added `plot_elbow_curves()`
- `02_clustering.ipynb` — added E4 header + US-4.1 section (5 cells)
