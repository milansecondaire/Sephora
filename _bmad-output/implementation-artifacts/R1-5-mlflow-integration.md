# Story R1.5 — Intégration MLflow pour le tracking expérimental

Status: done

## Story

As a Data Scientist,
I want every clustering experiment tracked with MLflow (parameters, metrics, artifacts),
so that each run is reproducible, comparable, and auditable.

## Acceptance Criteria

1. `mlflow` is importable in the project environment; a `pip install mlflow` instruction appears in a notebook Setup cell
2. A `log_clustering_run()` utility function is added to `src/clustering.py`
3. MLflow experiment named `"sephora-customer-segmentation"` is initialised in the notebook Setup cell
4. The K-scan loop (`evaluate_kmeans_k_range`) is wrapped in a MLflow parent run `"kmeans-k-scan"` with one nested run per value of k, each logging: `k`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, `inertia`
5. The final KMeans run is logged in a separate MLflow run `"kmeans-final"` with: `k` (chosen), `algorithm="kmeans"`, `n_features`, `random_state`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, `min_cluster_pct`, `max_cluster_pct`; and the cluster label CSV as an artifact
6. The Hierarchical clustering run is logged in a run `"hierarchical-final"` with the same metrics schema
7. `mlruns/` is added to `.gitignore` if not already present
8. A notebook Markdown cell documents how to view results: `mlflow ui` command and expected URL

## Tasks / Subtasks

- [x] Task 1 — Add `log_clustering_run()` to `src/clustering.py` (AC: 2)
  - [x] Function signature:
    ```python
    def log_clustering_run(
        run_name: str,
        params: dict,
        metrics: dict,
        artifacts: dict | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        """Log a clustering run to MLflow. Returns the run_id."""
    ```
  - [x] Inside: `with mlflow.start_run(run_name=run_name, nested=parent_run_id is not None) as run:`
  - [x] `mlflow.log_params(params)`
  - [x] `mlflow.log_metrics(metrics)`
  - [x] For each `(name, path)` in `artifacts.items()`: `mlflow.log_artifact(path, artifact_path=name)`
  - [x] Return `run.info.run_id`
  - [x] Add `import mlflow` at top of `src/clustering.py`

- [x] Task 2 — Setup MLflow experiment in notebook (AC: 1, 3)
  - [x] Add (or update) the Setup cell in `02_clustering.ipynb`:
    ```python
    import mlflow
    mlflow.set_experiment("sephora-customer-segmentation")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    ```
  - [x] Add a comment: `# Run 'mlflow ui' in terminal then open http://127.0.0.1:5000`

- [x] Task 3 — Wrap K-scan loop with MLflow (AC: 4)
  - [x] Locate the cell that calls `evaluate_kmeans_k_range()` in `02_clustering.ipynb`
  - [x] Replace with a cell that runs KMeans for each k individually and logs each as a nested run:
    ```python
    from src.clustering import log_clustering_run
    from src.config import K_RANGE, RANDOM_STATE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import mlflow

    metrics_rows = []
    with mlflow.start_run(run_name="kmeans-k-scan") as parent_run:
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels, random_state=RANDOM_STATE,
                                   sample_size=10_000 if len(X_scaled) > 10_000 else None)
            db  = davies_bouldin_score(X_scaled, labels)
            ch  = calinski_harabasz_score(X_scaled, labels)
            with mlflow.start_run(run_name=f"kmeans-k{k}", nested=True):
                mlflow.log_params({"k": k, "algorithm": "kmeans"})
                mlflow.log_metrics({"silhouette": sil, "davies_bouldin": db,
                                    "calinski_harabasz": ch, "inertia": km.inertia_})
            metrics_rows.append({"k": k, "inertia": km.inertia_,
                                  "silhouette": sil, "davies_bouldin": db,
                                  "calinski_harabasz": ch})
    metrics_df = pd.DataFrame(metrics_rows)
    ```
  - [x] Add a Markdown cell above this loop explaining what is being logged

- [x] Task 4 — Log final KMeans run (AC: 5)
  - [x] After `k_optimal` is chosen and `run_kmeans_final()` is called:
    ```python
    from src.clustering import log_clustering_run
    import tempfile, os

    labels_final, km_final = run_kmeans_final(X_scaled, k_optimal)
    df_customers["kmeans_label"] = labels_final

    cluster_sizes = pd.Series(labels_final).value_counts(normalize=True)
    label_csv = "data/processed/kmeans_labels.csv"
    df_customers[["kmeans_label"]].to_csv(label_csv)

    with mlflow.start_run(run_name="kmeans-final"):
        mlflow.log_params({
            "k": k_optimal, "algorithm": "kmeans",
            "n_features": X_scaled.shape[1], "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics({
            "silhouette": silhouette_score(X_scaled, labels_final,
                              sample_size=10_000, random_state=RANDOM_STATE),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels_final),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, labels_final),
            "min_cluster_pct": float(cluster_sizes.min()),
            "max_cluster_pct": float(cluster_sizes.max()),
        })
        mlflow.log_artifact(label_csv, artifact_path="labels")
    ```

- [x] Task 5 — Log Hierarchical clustering run (AC: 6)
  - [x] After `run_hierarchical()`:
    ```python
    hclust_labels = run_hierarchical(X_scaled, k_optimal)
    df_customers["hclust_label"] = hclust_labels
    hclust_csv = "data/processed/hclust_labels.csv"
    df_customers[["hclust_label"]].to_csv(hclust_csv)

    with mlflow.start_run(run_name="hierarchical-final"):
        mlflow.log_params({
            "k": k_optimal, "algorithm": "agglomerative_ward",
            "n_features": X_scaled.shape[1],
        })
        mlflow.log_metrics({
            "silhouette": silhouette_score(X_scaled, hclust_labels,
                              sample_size=10_000, random_state=RANDOM_STATE),
            "davies_bouldin": davies_bouldin_score(X_scaled, hclust_labels),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, hclust_labels),
        })
        mlflow.log_artifact(hclust_csv, artifact_path="labels")
    ```

- [x] Task 6 — Update `.gitignore` (AC: 7)
  - [x] Check if `mlruns/` is already in `.gitignore`
  - [x] If not, add `mlruns/` on a new line

- [x] Task 7 — Add `mlflow ui` documentation Markdown cell (AC: 8)
  - [x] Add at the end of the Setup section:
    ```
    ### Viewing MLflow Results
    After running the notebook, launch the MLflow UI from the project root:
    ```
    mlflow ui
    ```
    Then open: http://127.0.0.1:5000
    Navigate to experiment "sephora-customer-segmentation" to compare runs.
    ```

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — add `import mlflow` at top + `log_clustering_run()` function.
**Notebook:** `02_clustering.ipynb` — wrap existing clustering cells.
**File:** `.gitignore` — add `mlruns/`.
**Do NOT** create a separate MLflow module — keep it inside `src/clustering.py`.

### MLflow tracking structure

```
Experiment: sephora-customer-segmentation
│
├── Run: kmeans-k-scan  (parent)
│   ├── Run: kmeans-k2   → params: k=2  | metrics: silhouette, db, ch, inertia
│   ├── Run: kmeans-k3   → ...
│   └── Run: kmeans-k30  → ...
│
├── Run: kmeans-final    → params + metrics + artifact (labels CSV)
│
└── Run: hierarchical-final → params + metrics + artifact (labels CSV)
```

### mlflow version

Use whatever `pip install mlflow` installs (no version pin needed). Minimum: 2.0.0.

### Previous Story Output
- R1.4 complete: notebook clustering cells use `X_scaled` directly

### Output of This Story
- `src/clustering.py` — `log_clustering_run()` added + `import mlflow`
- `02_clustering.ipynb` — Setup cell, wrapped K-scan loop, final run logging cells
- `.gitignore` — `mlruns/` line added

### References
- [EPIC-R1](_bmad-output/implementation-artifacts/EPIC-R1-feature-refonte-and-mlflow.md) — US-R1.5

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (Amelia — Dev Agent)

### Debug Log References
_none_

### Completion Notes List
- `import mlflow` added to `src/clustering.py`
- `log_clustering_run()` utility added with params/metrics/artifacts support + nested runs
- Notebook Setup cell: `mlflow.set_experiment("sephora-customer-segmentation")` + tracking URI print
- K-scan cell replaced: inline KMeans loop with parent run `kmeans-k-scan` + nested child per k
- Final KMeans cell: logs `kmeans-final` run with params, metrics, min/max cluster %, labels CSV artifact
- Hierarchical cell: logs `hierarchical-final` run with params, metrics, labels CSV artifact
- `.gitignore` created with `mlruns/`
- Markdown cell added after Setup with `mlflow ui` instructions
- 6 new tests in `TestLogClusteringRun` — all 43 tests pass (0 failures)

### File List
Files modified:
- `src/clustering.py` — added `import mlflow` + `log_clustering_run()`
- `02_clustering.ipynb` — Setup cell (mlflow init), K-scan MLflow wrapping, kmeans-final logging, hierarchical-final logging, MLflow UI doc cell
- `.gitignore` — created with `mlruns/`
- `tests/test_clustering.py` — added `TestLogClusteringRun` (6 tests)
