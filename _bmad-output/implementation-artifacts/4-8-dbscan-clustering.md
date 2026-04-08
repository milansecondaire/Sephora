# Story 4.8: DBSCAN Clustering

Status: ready-for-dev

## Story

As a Data Scientist,
I want to run DBSCAN on the customer feature matrix,
so that a second density-based approach is evaluated alongside HDBSCAN and partition-based models.

## Acceptance Criteria

1. `plot_kdistance()` implemented in `src/clustering.py` to guide `eps` selection via the k-distance graph
2. `run_dbscan()` implemented in `src/clustering.py` using `sklearn.cluster.DBSCAN`
3. `eps` chosen from the k-distance graph elbow; `min_samples=5` fixed; both values documented in a notebook markdown cell
4. Final DBSCAN run with chosen `eps`; cluster labels stored as `df_customers['dbscan_label']` (noise = -1 kept as-is)
5. Noise statistics printed: absolute count and % of total customers labelled as noise
6. If noise % > 30%, a second run with a larger `eps` value tried and compared; final `eps` chosen documented
7. Silhouette, DB, CH scores computed **on non-noise points only**; appended to `comparison_results` with `'noise_pct'` field
8. k-distance graph saved to `figures/dbscan_kdistance.png`; UMAP scatter saved to `figures/dbscan_umap.png`
9. DBSCAN run logged to MLflow as `"dbscan-final"` with consistent params/metrics schema

## Tasks / Subtasks

- [ ] Task 1 — Implement `plot_kdistance()` in `src/clustering.py` (AC: 1, 8)
  - [ ] Signature: `plot_kdistance(X, k=5, save_path=None) -> matplotlib.Figure`
  - [ ] Use `sklearn.neighbors.NearestNeighbors(n_neighbors=k)` on a subsample (max 10K rows) for speed
  - [ ] Sort k-th distances ascending and plot; elbow = candidate `eps`
  - [ ] Save to `figures/dbscan_kdistance.png`
- [ ] Task 2 — Implement `run_dbscan()` in `src/clustering.py` (AC: 2, 4)
  - [ ] Signature: `run_dbscan(X, eps=1.5, min_samples=5) -> tuple[ndarray, DBSCAN]`
  - [ ] Return `(labels, fitted_model)` — noise points have label `-1`
- [ ] Task 3 — eps tuning via k-distance graph (AC: 3)
  - [ ] Call `plot_kdistance(X_scaled, k=5)` in notebook cell
  - [ ] Visually identify elbow → choose initial `eps` candidate
  - [ ] Document chosen `eps` value and rationale in markdown cell
- [ ] Task 4 — Final DBSCAN run + noise check + optional re-run (AC: 4, 5, 6)
  - [ ] Run `run_dbscan(X_scaled, eps=eps_chosen, min_samples=5)`
  - [ ] `df_customers['dbscan_label'] = labels`
  - [ ] Print: `n_clusters`, `n_noise`, `noise_pct`
  - [ ] If `noise_pct > 30%`: increase eps by 20–50% and re-run; document both attempts; pick final eps
- [ ] Task 5 — Compute metrics on non-noise points (AC: 7)
  - [ ] `mask = labels != -1`; apply to `X_scaled` and `labels` before metric computation
  - [ ] Guard: if fewer than 2 distinct clusters in non-noise → log NaN for all metrics
  - [ ] Append to `comparison_results`:
    ```python
    comparison_results.append({
        'algorithm': 'DBSCAN',
        'k': n_clusters_found,
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
        'noise_pct': noise_pct,
    })
    ```
- [ ] Task 6 — UMAP scatter plot (AC: 8)
  - [ ] Reuse `umap_embedding` from E3/E4
  - [ ] Color by `dbscan_label`; noise points (-1) coloured grey
  - [ ] Save to `figures/dbscan_umap.png`
- [ ] Task 7 — MLflow logging (AC: 9)
  - [ ] `mlflow.start_run(run_name="dbscan-final")`
  - [ ] Log params: `eps`, `min_samples`, `n_features=33`
  - [ ] Log metrics: `n_clusters`, `noise_pct`, `silhouette`, `davies_bouldin`, `calinski_harabasz`
- [ ] Task 8 — Add notebook section in `02_clustering.ipynb` after US-4.7 HDBSCAN section

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — add `plot_kdistance()` and `run_dbscan()` to the existing module; do NOT create a new file.  
**Notebook:** `02_clustering.ipynb` — new section `## US-4.8 — DBSCAN`, inserted after the HDBSCAN section (US-4.7) and before the comparison section (US-4.5).  
**Input data:** `X_scaled` — shape **(64469, 33)**, StandardScaler applied, no PCA.

**k-distance graph — mandatory for eps selection:**
```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def plot_kdistance(
    X: pd.DataFrame,
    k: int = 5,
    n_samples: int = 10_000,
    save_path: str | None = None,
):
    """Plot sorted k-th nearest-neighbour distances (k-distance graph).
    The elbow of the curve is the candidate eps for DBSCAN.

    Args:
        X: Feature matrix (n_samples, n_features).
        k: Neighbour count — should equal DBSCAN min_samples.
        n_samples: Subsample size for speed.
        save_path: Optional figure save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    X_sub = X.values[idx]

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_sub)
    distances, _ = nn.kneighbors(X_sub)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_distances, linewidth=1.5)
    ax.set_xlabel("Points triés par distance croissante")
    ax.set_ylabel(f"Distance au {k}e plus proche voisin (eps candidat)")
    ax.set_title(f"k-distance graph (k={k}) — Identifier le coude pour choisir eps")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
```

**DBSCAN function:**
```python
from sklearn.cluster import DBSCAN

def run_dbscan(
    X: pd.DataFrame,
    eps: float = 1.5,
    min_samples: int = 5,
) -> tuple:
    """Fit DBSCAN. Returns (labels ndarray, fitted model).
    Noise points are labelled -1.

    Args:
        X: Feature matrix (n_samples, n_features).
        eps: Neighbourhood radius — tune via k-distance graph.
        min_samples: Minimum points to form a core point.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)
    return labels, model
```

**Noise handling — identical pattern to HDBSCAN (US-4.7):**
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

**eps guidance for StandardScaled data (33 features):**
- Euclidean distances in 33D after StandardScaler: typical inter-point distances ≈ 3–8
- Expected elbow range: `eps ∈ [1.0, 4.0]`
- Start with elbow value; if noise % > 30%, increase eps by 20–50% and re-run
- If DBSCAN produces only 1 cluster + extreme noise (> 50%), the algorithm is not suited to this manifold: document and still log metrics as NaN; the comparison will reflect this honestly

**UMAP scatter with noise coloring (same pattern as HDBSCAN):**
```python
fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.cm.tab10
unique_labels = sorted(set(dbscan_labels))
for lbl in unique_labels:
    mask_lbl = dbscan_labels == lbl
    color = 'grey' if lbl == -1 else cmap(lbl % 10)
    label_str = 'Noise' if lbl == -1 else f'Cluster {lbl}'
    ax.scatter(umap_embedding[mask_lbl, 0], umap_embedding[mask_lbl, 1],
               c=[color], s=2, alpha=0.5, label=label_str)
ax.set_title('DBSCAN — UMAP projection')
ax.legend(markerscale=4, loc='best')
fig.tight_layout()
fig.savefig('figures/dbscan_umap.png', dpi=150)
```

**MLflow schema (consistent with US-4.2, 4.3, 4.4, 4.7):**
```python
with mlflow.start_run(run_name="dbscan-final"):
    mlflow.log_params({
        'algorithm': 'dbscan',
        'eps': eps_chosen,
        'min_samples': 5,
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

**⚠️ Impact on US-4.5 (Algorithm Comparison):** After completing this story, `comparison_results` will contain 5 entries: KMeans, Hierarchical, GMM, HDBSCAN, DBSCAN. The `build_comparison_table()` function in `src/clustering.py` must be able to locate `hdbscan_label` and `dbscan_label` columns. The current logic uses name-based heuristics — either:
- Pass label column names explicitly in each `comparison_results` dict entry, OR
- Patch `build_comparison_table()` to also handle `hdbscan` and `dbscan` naming

The dev agent for US-4.5 must reconcile this. Worth adding a `'label_col'` key to each dict for explicit resolution:
```python
comparison_results.append({
    ...
    'label_col': 'dbscan_label',  # explicit column reference
})
```

### Previous Story Output (4-7 HDBSCAN)

- `df_customers['hdbscan_label']`
- `comparison_results` list with 4 entries (KMeans, Hierarchical, GMM, HDBSCAN)
- `k_optimal`, `X_scaled` (64469, 33), `umap_embedding` (64469, 2)

### Output of This Story

- `df_customers['dbscan_label']` — int array, -1 = noise
- `figures/dbscan_kdistance.png`
- `figures/dbscan_umap.png`
- DBSCAN entry appended to `comparison_results` (5 entries total)
- MLflow run `"dbscan-final"`

### References

- [epics — E4 Clustering Models](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- sklearn 1.8 DBSCAN docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- sklearn 1.8 HDBSCAN docs (for noise handling comparison): https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 via GitHub Copilot

### Debug Log References

### Completion Notes List

### File List
