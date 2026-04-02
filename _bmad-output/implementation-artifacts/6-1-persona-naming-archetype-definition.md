# Story 6.1: Persona Naming & Archetype Definition

Status: ready-for-dev

## Story

As a Marketing Manager,
I want to give each cluster a memorable name and archetype,
so that the personas are usable in briefs and presentations without referencing cluster numbers.

## Acceptance Criteria

1. Each cluster receives a name reflecting its dominant behavioral trait (e.g., "The Fragrance Loyalist", "The Budget Hunter")
2. Each name is unique, ≤ 4 words, and marketing-friendly
3. A one-sentence archetype description accompanies each name
4. Naming rationale tied explicitly to the top distinguishing features from US-5.4
5. All names documented in a summary table: Cluster ID | Name | Archetype

## Tasks / Subtasks

- [ ] Task 1 — Create `src/persona_generator.py` with persona registry (AC: 1–4)
  - [ ] Define `PERSONA_REGISTRY` dict: cluster_id → {name, archetype, rationale}
  - [ ] Names derived from dominant `Axe`, `Channel`, `loyalty_status`, `CLV tier`
- [ ] Task 2 — Create `04_personas.ipynb` and add E6 setup + naming section (AC: 5)
  - [ ] Load `customers_with_clusters.csv`
  - [ ] Display naming summary table
  - [ ] Markdown cell explaining each name's derivation from data

## Dev Notes

### Architecture Guardrails

**Module:** `src/persona_generator.py` — create this module.  
**Notebook:** `04_personas.ipynb` — create this notebook. E6 section starts here.

**Notebook setup:**
```python
from src.config import *
import numpy as np
np.random.seed(RANDOM_STATE)
import pandas as pd

df_customers = pd.read_csv(DATA_PROCESSED_PATH + "customers_with_clusters.csv",
                           index_col='anonymized_card_code')
# Load profiling outputs
cluster_kpis = pd.read_csv(...)  # or recompute
```

**persona_generator.py skeleton:**
```python
# src/persona_generator.py
# PERSONA_REGISTRY is populated after E5 analysis reveals cluster profiles.
# The Dev Agent must populate this dict based on actual cluster characteristics.

PERSONA_REGISTRY = {}
# Example structure (to be filled with real cluster analysis):
# {
#     0: {
#         'name': 'The GOLD Skincare Devotee',
#         'archetype': 'A high-spending, loyalty-program active customer whose basket is dominated by Skincare products.',
#         'rationale': 'Cluster 0 shows Cohen\'s d > 1.5 for axe_skincare_ratio and loyalty_numeric=GOLD.',
#         'clv_tier': 'Top',
#     },
#     ...
# }
```

**Naming rules (from epics):**
- Name ≤ 4 words
- Marketing-friendly (no data jargon)
- Reflects DOMINANT feature from Cohen's d analysis
- Examples: "The Fragrance Loyalist", "The Budget Hunter", "The Digital Explorer", "The New Convert"

**This story is partially data-driven:** The Dev Agent must look at actual cluster profiles from `cluster_kpis_df` and `distinguishing_features` outputs (E5) to assign names. The naming output is a judgment call informed by data.

### Previous Story Output (E5 complete)
- `data/processed/customers_with_clusters.csv`
- `cluster_kpis_df`, `distinguishing_features` dict (E5 outputs)

### Output of This Story
- `PERSONA_REGISTRY` populated in `src/persona_generator.py`
- `04_personas.ipynb` created

### References
- [epics — US-6.1](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — persona_generator.py module](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to create:
- `src/persona_generator.py` (create — with `PERSONA_REGISTRY` stub)
- `04_personas.ipynb` (create — add notebook setup + E6.1 section)
