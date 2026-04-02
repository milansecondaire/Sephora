# Story 6.5: AI-Generated Personalized Content Opportunity

Status: ready-for-dev

## Story

As a Digital Marketing Lead,
I want to know which segments are best suited for AI-generated personalized content,
so that I can prioritize AI tooling investments.

## Acceptance Criteria

1. Define AI content fit criteria: high `purchase_frequency` + high `estore_ratio` + clear single dominant feature axis
2. Score each segment on the AI content fit criteria (composite score: 0–10)
3. Identify top 1–2 segments best suited for personalized AI content
4. Suggest content type per top segment (product recommendation email, homepage AI banner, tailored push notification)
5. State expected lift hypothesis (e.g., "+X% on estore conversion rate") grounded in segment data
6. Final E6 Markdown summary cell with gate statement: "Analysis complete — all personas and recommendations produced."

## Tasks / Subtasks

- [ ] Task 1 — Define and compute AI fit score (AC: 1–2)
  - [ ] Add `score_ai_content_fit()` to `src/recommendations.py`
  - [ ] Inputs: cluster_kpis_df, weights for each criterion
  - [ ] Normalize each component to [0, 10] and compute weighted average
- [ ] Task 2 — Identify top segments and map content type (AC: 3–4)
  - [ ] Select top 1–2 clusters by AI fit score
  - [ ] Map to content type based on dominant channel and feature axis
- [ ] Task 3 — Formulate lift hypotheses (AC: 5)
  - [ ] For each top segment: derive expected lift from estore_ratio and frequency benchmarks
  - [ ] Write as Markdown in notebook: "Segment X has Y% estore ratio, above Z% average → expect +W% conversion on personalized email"
- [ ] Task 4 — Final E6 summary section (AC: 6)
  - [ ] Markdown summary cell: objectives achieved, personas listed, next steps
  - [ ] Gate statement: `# Analysis complete — all personas and recommendations produced.`

## Dev Notes

### Architecture Guardrails

**Module:** `src/recommendations.py` (add `score_ai_content_fit()`).  
**Notebook:** `04_personas.ipynb` — final section of E6 and of the entire project.

**Function signature:**
```python
# src/recommendations.py
def score_ai_content_fit(
    cluster_kpis_df: pd.DataFrame,
    global_kpis: dict,
) -> pd.DataFrame:
    """
    Computes AI content fit score for each cluster.
    
    Criteria:
    - purchase_frequency normalized (higher = better for personalization frequency)
    - estore_ratio normalized (higher = better, digital channel = AI content delivery)
    - feature_concentration: 1 dominant feature axis (higher concentration = better targeting)
    
    Returns cluster_kpis_df with added column 'ai_fit_score' (0–10 float).
    """
    ...
```

**Scoring approach (simple, explainable):**
```python
# 3 criteria weighted equally (can be tuned)
score = (
    0.4 * normalize(frequency) +
    0.4 * normalize(estore_ratio) +
    0.2 * normalize(feature_concentration)
) * 10
```

**Content type mapping:**
| Profile | Content Type |
|---|---|
| High frequency + high estore | Personalized product recommendation email |
| High estore + low frequency | AI homepage banner (retargeting) |
| High store + high frequency | In-app push notification (loyalty app) |

**Lift hypothesis format:**
> "Segment [name] has an estore ratio of [X]% vs. global average [Y]%. AI-personalized product recommendation emails targeting their top categories (fragrance, skincare) are expected to yield a [+Z]% improvement in estore conversion rate."

**Final gate statement** (AC-6) — exact string to use as Markdown heading:
```
# Analysis complete — all personas and recommendations produced.
```

### E6 Full Story Arc (notebook outline)
```
04_personas.ipynb:
  US-6.1 — Persona naming + archetype definition
  US-6.2 — Persona card production
  US-6.3 — Marketing recommendations per segment
  US-6.4 — Segment prioritization matrix
  US-6.5 — AI content fit scoring + final summary  ← this story
```

### Previous Story Output (6-4)
- Priority matrix figure saved
- `cluster_kpis_df` with full profiling data
- Recommendations built per persona

### Output of This Story
- `cluster_kpis_df` with `ai_fit_score` column
- AI content strategy Markdown cells in `04_personas.ipynb`
- Final gate cell: "Analysis complete — all personas and recommendations produced."

### References
- [epics — US-6.5](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture module boundaries](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/recommendations.py` (add `score_ai_content_fit()`)
- `04_personas.ipynb` (add US-6.5 section + final summary cell)
