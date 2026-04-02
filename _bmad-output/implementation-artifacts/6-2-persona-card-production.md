# Story 6.2: Persona Card Production

Status: ready-for-dev

## Story

As a Marketing Manager,
I want to have a structured persona card per segment,
so that any team member can quickly understand a segment without reading the full analysis.

## Acceptance Criteria

Per card:
1. Persona name and cluster ID
2. Size: n customers and % of base
3. Top 3 behavioral traits with quantified delta vs. average (e.g., "+48% higher basket size")
4. Dominant channel, dominant product axis, dominant market tier
5. Loyalty status distribution (% No Fid / BRONZE / SILVER / GOLD)
6. CLV tier (Top / Mid / Low)
7. 2–3 sentence plain-language narrative usable in a brief
8. Cards formatted as Markdown tables in notebook AND exported as standalone Markdown files in `_bmad-output/implementation-artifacts/personas/`

## Tasks / Subtasks

- [ ] Task 1 — Implement `generate_persona_card()` in `src/persona_generator.py` (AC: 1–7)
  - [ ] Accepts cluster_id + cluster_kpis + global_kpis + delta_df + PERSONA_REGISTRY
  - [ ] Returns a dict with all card fields
- [ ] Task 2 — Implement `render_persona_md()` in `src/persona_generator.py` (AC: 8)
  - [ ] Format card dict as Markdown
  - [ ] Save to `_bmad-output/implementation-artifacts/personas/segment_{n}.md`
- [ ] Task 3 — Add notebook section in `04_personas.ipynb`
  - [ ] Loop over all clusters; generate + render cards
  - [ ] Display cards inline in notebook

## Dev Notes

### Architecture Guardrails

**Module:** `src/persona_generator.py` — add to existing module.  
**Notebook:** `04_personas.ipynb` — E6 section, after US-6.1.

**Card generation function:**
```python
import os

def generate_persona_card(
    cluster_id: int,
    cluster_kpis: pd.DataFrame,
    global_kpis: dict,
    delta_df: pd.DataFrame,
    persona_registry: dict,
) -> dict:
    row = cluster_kpis.loc[cluster_id]
    persona = persona_registry[cluster_id]
    
    # Top 3 delta traits (|delta_pct| largest)
    cluster_deltas = delta_df[delta_df['cluster_id'] == cluster_id]
    top_deltas = cluster_deltas.nlargest(3, 'delta_pct')[['kpi', 'delta_pct']]
    
    return {
        'cluster_id': cluster_id,
        'name': persona['name'],
        'archetype': persona['archetype'],
        'n_customers': int(row['n_customers']),
        'pct_customers': round(row['pct_customers'], 1),
        'top_traits': top_deltas.to_dict('records'),
        'dominant_channel': ...,  # from cluster_kpis
        'dominant_axe': ...,
        'dominant_market': ...,
        'clv_tier': persona['clv_tier'],
        'narrative': persona.get('narrative', ''),
    }
```

**Output directory:**
```python
PERSONAS_DIR = OUTPUT_PATH + "personas/"
os.makedirs(PERSONAS_DIR, exist_ok=True)
# Example output: _bmad-output/implementation-artifacts/personas/segment_0.md
```

**Markdown card template:**
```markdown
# Persona: {name}
**Cluster ID:** {cluster_id} | **Size:** {n_customers:,} ({pct_customers}%)

## Archetype
{archetype}

## Top 3 Behavioral Traits
| Trait | Delta vs. Average |
|---|---|
| {trait1} | {delta_pct1:+.1f}% |
| {trait2} | {delta_pct2:+.1f}% |
| {trait3} | {delta_pct3:+.1f}% |

## Profile
- **Dominant Channel:** {dominant_channel}
- **Dominant Product Axis:** {dominant_axe}
- **CLV Tier:** {clv_tier}

## Narrative
{narrative}
```

### Previous Story Output (6-1)
- `PERSONA_REGISTRY` populated
- `cluster_kpis_df`, `delta_df`, `global_kpis` from E5

### Output of This Story
- `_bmad-output/implementation-artifacts/personas/segment_{n}.md` for each cluster

### References
- [epics — US-6.2](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D3.1 Persona Card Format](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D3.2 Output: personas/segment_{n}.md](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/persona_generator.py` (add `generate_persona_card()`, `render_persona_md()`)
- `04_personas.ipynb` (add US-6.2 section)

Files created by this story:
- `_bmad-output/implementation-artifacts/personas/segment_{n}.md` (one per cluster)
