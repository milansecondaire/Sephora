# Story R1.3 — Cercle des corrélations (décision humaine)

Status: done

## Story

As a Data Analyst,
I want to display a correlation circle on the scaled features, colored by marketing category,
so that Milan can visually identify remaining redundant variables and decide which to remove.

## Acceptance Criteria

1. A `plot_correlation_circle()` function is added to `src/visualization.py`
2. The function accepts `X_scaled` (DataFrame), `feature_categories` (dict from `FEATURE_CATEGORIES`) and an optional `save_path`
3. Each feature arrow is colored according to its marketing category using `CATEGORY_COLORS`; `dominant_*` one-hot dummies are colored by the base feature's category
4. The unit circle is drawn and the two PCA axes are labeled with their explained variance percentage
5. A legend showing the 6 category colors is displayed
6. Variable pairs with cosine similarity > 0.90 (angle < 26°) are automatically printed to stdout as potential redundancies for Milan to review
7. The figure is saved to `figures/correlation_circle.png`
8. A notebook cell calls the function and displays the figure inline after the preprocessing step
9. ⚠️ PCA here is used ONLY for the visualization projection — it must NOT be applied to `X_scaled` for clustering purposes

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_correlation_circle()` in `src/visualization.py` (AC: 1–7)
  - [x] Fit `PCA(n_components=2, random_state=42)` on `X_scaled`
  - [x] Extract loadings matrix: `loadings = pca.components_.T` — shape `(n_features, 2)`
  - [x] Draw unit circle: `ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='grey', linewidth=1))`
  - [x] For each feature, draw an arrow from `(0,0)` to `(loadings[i, 0], loadings[i, 1])` using `ax.annotate()`
  - [x] Color each arrow by category: look up the base feature name in `feature_categories`; for one-hot dummies (e.g. `gender_Women`), strip the suffix and look up `gender` in categories
  - [x] Annotate each arrow endpoint with the feature name (`fontsize=7`, `ha='center'`)
  - [x] Add legend with `matplotlib.patches.Patch` per category
  - [x] Print to stdout: "Potential redundancies (cosine sim > 0.90):" followed by pairs
  - [x] Set `ax.set_xlim(-1.1, 1.1)`, `ax.set_ylim(-1.1, 1.1)`, `ax.set_aspect('equal')`
  - [x] Label axes: `f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"`, `f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"`
  - [x] Save if `save_path` provided
- [x] Task 2 — Add notebook cell in `02_clustering.ipynb` (AC: 8)
  - [x] Add code cell after the preprocessing / `X_scaled` cell:
    ```python
    from src.visualization import plot_correlation_circle
    from src.config import FEATURE_CATEGORIES, CATEGORY_COLORS
    plot_correlation_circle(
        X_scaled,
        feature_categories=FEATURE_CATEGORIES,
        category_colors=CATEGORY_COLORS,
        save_path="figures/correlation_circle.png"
    )
    ```
  - [x] Add a Markdown cell immediately below explaining: "Inspect this chart. Arrows pointing in the same direction (angle < 26°) indicate potentially redundant features. Decide which to remove before clustering."

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` — add `plot_correlation_circle()` after the existing `plot_correlation_heatmap()` function.
**Notebook:** `02_clustering.ipynb` — insert after the E3.3 preprocessing cell, before the clustering section.
**No modifications** to `src/preprocessing.py` or `src/config.py`.

### Implementation details

```python
def plot_correlation_circle(
    X_scaled: pd.DataFrame,
    feature_categories: dict,
    category_colors: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Correlation circle: projects original features onto PC1/PC2 plane.
    PCA used ONLY for visualization — not for clustering.
    """
    from sklearn.decomposition import PCA
    import matplotlib.patches as mpatches

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    loadings = pca.components_.T   # shape: (n_features, 2)

    # Build reverse map: feature_name -> category
    feat_to_cat = {}
    for cat, feats in feature_categories.items():
        for f in feats:
            feat_to_cat[f] = cat

    def _get_cat(col_name):
        # Try exact match first, then strip OHE suffix
        if col_name in feat_to_cat:
            return feat_to_cat[col_name]
        for base in feat_to_cat:
            if col_name.startswith(base + "_"):
                return feat_to_cat[base]
        return "other"

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='grey', linewidth=1, linestyle='--'))
    ax.axhline(0, color='lightgrey', linewidth=0.5)
    ax.axvline(0, color='lightgrey', linewidth=0.5)

    cols = X_scaled.columns.tolist()
    for i, col in enumerate(cols):
        x, y = loadings[i, 0], loadings[i, 1]
        cat = _get_cat(col)
        color = category_colors.get(cat, "#999999")
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
        )
        ax.text(x * 1.05, y * 1.05, col, fontsize=6.5, ha='center', color=color)

    # Print high-similarity pairs
    from itertools import combinations
    import numpy as np
    print("Potential redundancies (cosine sim > 0.90):")
    found = False
    for i, j in combinations(range(len(cols)), 2):
        sim = np.dot(loadings[i], loadings[j]) / (
            np.linalg.norm(loadings[i]) * np.linalg.norm(loadings[j]) + 1e-9
        )
        if sim > 0.90:
            print(f"  {cols[i]}  ↔  {cols[j]}  (sim={sim:.3f})")
            found = True
    if not found:
        print("  None found.")

    # Legend
    handles = [mpatches.Patch(color=category_colors[c], label=c) for c in category_colors]
    ax.legend(handles=handles, loc='lower right', fontsize=9)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Cercle des corrélations — features colorées par catégorie", fontsize=13)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return fig
```

### Handling one-hot dummy column names
OHE produces columns like `gender_Women`, `dominant_axe_MAKE UP`, `country_FR`.
The `_get_cat()` helper strips the suffix to find the base feature in `FEATURE_CATEGORIES`.
If no match is found, the arrow is colored `"#999999"` (grey) and labeled "other".

### Previous Story Output
- R1.1 complete: `FEATURE_CATEGORIES`, `CATEGORY_COLORS` in `src/config.py`
- R1.2 complete: `preprocess_for_clustering()` produces `X_scaled` of shape `(64469, 43)`

### Output of This Story
- `src/visualization.py` — `plot_correlation_circle()` function added
- `02_clustering.ipynb` — correlation circle cell added (code + explanatory Markdown)
- `figures/correlation_circle.png` — saved figure

### References
- [EPIC-R1](_bmad-output/implementation-artifacts/EPIC-R1-feature-refonte-and-mlflow.md) — US-R1.3

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
_none_

### Completion Notes List
- Task 1: Added `plot_correlation_circle()` to `src/visualization.py` after `get_high_correlation_pairs()`. Function fits PCA(2) for visualization only, draws arrows colored by FEATURE_CATEGORIES, prints cosine sim > 0.90 pairs.
- Task 1 tests: 8 unit tests added to `tests/test_visualization.py` — returns figure, saves to disk, unit circle drawn, axis labels with variance %, equal aspect, legend with all 6 categories, redundancy stdout, OHE column handling.
- Task 2: Inserted Markdown cell (R1.3 header + instructions) + code cell after E3.3 preprocessing cell in `02_clustering.ipynb`, before E3.4 PCA.
- All 288 project tests pass, zero regressions.

### File List
- `src/visualization.py` — added `plot_correlation_circle()` (AC 1-7, 9)
- `tests/test_visualization.py` — added `TestPlotCorrelationCircle` class (8 tests)
- `02_clustering.ipynb` — added R1.3 markdown cell + code cell (AC 8)
