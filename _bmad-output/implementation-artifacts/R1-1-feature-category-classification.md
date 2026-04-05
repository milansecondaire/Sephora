# Story R1.1 — Classification des variables en 6 catégories marketing

Status: done

## Story

As a Marketing Manager,
I want every feature classified into one of 6 marketing-actionable categories,
so that I know exactly what each feature measures and how it supports targeting decisions.

## Acceptance Criteria

1. A `FEATURE_CATEGORIES` dictionary is added to `src/config.py` with exactly 6 keys: `"profil"`, `"valeur"`, `"affinite_produit"`, `"comportement"`, `"canal"`, `"dates"`
2. Every feature that will be retained after R1.2 (the 25 continuous + 5 one-hot groups) appears in exactly one category list
3. The dictionary is consistent with the updated `FEATURES_CONTINUOUS` and `FEATURES_ONEHOT` lists (no feature listed in `FEATURE_CATEGORIES` that is in `FEATURES_DROP`)
4. A Markdown cell in `02_clustering.ipynb` displays the full classification table grouped by category
5. The category colors dict `CATEGORY_COLORS` is also added to `src/config.py` for use in the correlation circle (R1.3)

## Tasks / Subtasks

- [x] Task 1 — Add `FEATURE_CATEGORIES` to `src/config.py` (AC: 1–3)
  - [x] Define `FEATURE_CATEGORIES` dict with 6 category keys
  - [x] `"profil"` → `["age", "gender", "country", "loyalty_numeric"]`
  - [x] `"valeur"` → `["recency_days", "frequency", "monetary_total", "monetary_avg", "avg_basket_size_eur", "discount_rate"]`
  - [x] `"affinite_produit"` → `["axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio", "axe_haircare_ratio", "axe_others_ratio", "dominant_axe", "market_selective_ratio", "market_exclusive_ratio", "market_sephora_ratio", "market_others_ratio", "dominant_market", "axis_diversity"]`
  - [x] `"comportement"` → `["avg_units_per_basket", "nb_unique_brands", "nb_unique_stores"]`
  - [x] `"canal"` → `["store_ratio", "estore_ratio", "click_collect_ratio", "dominant_channel"]`
  - [x] `"dates"` → `["subscription_tenure_days"]`
  - [x] Add `CATEGORY_COLORS` dict: `{"profil": "#4C72B0", "valeur": "#DD8452", "affinite_produit": "#55A868", "comportement": "#C44E52", "canal": "#8172B2", "dates": "#937860"}`
- [x] Task 2 — Add classification table Markdown cell to `02_clustering.ipynb` (AC: 4)
  - [x] Add a new Markdown cell at the top of Section E3 (Feature Engineering) with a table showing all retained features grouped by category
  - [x] Each row: Feature name | Category | Type (Continuous / One-Hot) | Marketing rationale (1 sentence)

## Dev Notes

### Architecture Guardrails

**Module:** `src/config.py` — add after the existing `FEATURES_FREQUENCY` block.
**Notebook:** `02_clustering.ipynb` — add Markdown cell at the start of the E3 / Feature Audit section.
**No new Python functions needed** — this story is purely config + documentation.

### Complete `FEATURE_CATEGORIES` mapping

```python
FEATURE_CATEGORIES = {
    "profil": [
        "age", "gender", "country", "loyalty_numeric",
    ],
    "valeur": [
        "recency_days", "frequency", "monetary_total", "monetary_avg",
        "avg_basket_size_eur", "discount_rate",
    ],
    "affinite_produit": [
        "axe_make_up_ratio", "axe_skincare_ratio", "axe_fragrance_ratio",
        "axe_haircare_ratio", "axe_others_ratio",
        "dominant_axe",
        "market_selective_ratio", "market_exclusive_ratio",
        "market_sephora_ratio", "market_others_ratio",
        "dominant_market", "axis_diversity",
    ],
    "comportement": [
        "avg_units_per_basket", "nb_unique_brands", "nb_unique_stores",
    ],
    "canal": [
        "store_ratio", "estore_ratio", "click_collect_ratio", "dominant_channel",
    ],
    "dates": [
        "subscription_tenure_days",
    ],
}

CATEGORY_COLORS = {
    "profil":           "#4C72B0",
    "valeur":           "#DD8452",
    "affinite_produit": "#55A868",
    "comportement":     "#C44E52",
    "canal":            "#8172B2",
    "dates":            "#937860",
}
```

### Rationale for each category

| Category | Rationale |
|---|---|
| **Profil** | Who the customer is — static socio-demographic traits used to personalize message tone and channel |
| **Valeur** | How much the customer is worth — drives investment priority per segment |
| **Affinité Produit** | What the customer buys — drives product recommendation and cross-sell |
| **Comportement** | How the customer shops — informs promotionnal mechanics (multi-brand, basket depth) |
| **Canal** | Where the customer shops — determines which activation channel to use (CRM push, display, in-store) |
| **Dates** | Loyalty seniority — proxy for engagement maturity and churn risk |

### Note on `dominant_*` variables
`dominant_axe`, `dominant_market`, `dominant_channel` are categorical (One-Hot encoded). Their category in `FEATURE_CATEGORIES` must match the base feature name, not the dummy column names. The correlation circle (R1.3) will handle the mapping.

### Previous Story Output
- `src/config.py` — `FEATURES_DROP`, `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT`, `FEATURES_FREQUENCY` already defined (Story 3.1)

### Output of This Story
- `src/config.py` — `FEATURE_CATEGORIES`, `CATEGORY_COLORS` added
- `02_clustering.ipynb` — Feature classification Markdown table visible at start of E3

### References
- [EPIC-R1](_bmad-output/implementation-artifacts/EPIC-R1-feature-refonte-and-mlflow.md) — US-R1.1

## Dev Agent Record

### Agent Model Used
Claude Sonnet 4.6

### Debug Log References
_none_

### Completion Notes List
- Task 1: `FEATURE_CATEGORIES` (6 keys, 30 features) et `CATEGORY_COLORS` (6 hex) ajoutés à `src/config.py` après le bloc `FEATURES_FREQUENCY`.
- Task 2: Cellule Markdown `R1.1 — Feature Classification by Marketing Category` insérée à l'indice 2 du notebook (avant E3.1), contenant une table 30 lignes avec Feature | Category | Type | Marketing rationale.
- 7 tests unitaires créés dans `tests/test_config.py` → `TestFeatureCategories` (tous verts).
- 0 régression introduite (1 échec PCA pré-existant non lié à R1.1).

### File List
Files modified:
- `src/config.py` — `FEATURE_CATEGORIES`, `CATEGORY_COLORS` ajoutés
- `02_clustering.ipynb` — cellule Markdown R1.1 insérée (index 2) et stylisée avec sauts de lignes par catégorie
- `tests/test_config.py` — `TestFeatureCategories` (7 tests) ajoutée et améliorée avec check exhaustif (total 30 features)
- `_bmad-output/implementation-artifacts/figures/umap_2d_loyalty_status.png` — implicitement générée lors des tests
