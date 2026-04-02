# Story 6.3: Marketing Recommendations per Segment

Status: ready-for-dev

## Story

As a Marketing Manager,
I want to receive specific marketing activation guidance per persona,
so that I can brief campaign teams without additional data analysis.

## Acceptance Criteria

Per persona:
1. Objective: Acquire / Retain / Upsell / Re-engage (one primary objective)
2. Recommended primary channel: in-store / estore / push notification / email
3. Suggested offer type: discovery, loyalty tier upgrade, discount, premium experience
4. Suggested frequency of communication: weekly / biweekly / monthly
5. KPI to track: the single most relevant metric to measure campaign success
6. Recommendations grounded in segment profile data (no generic advice)

## Tasks / Subtasks

- [ ] Task 1 — Implement `build_marketing_recommendations()` in `src/recommendations.py` (AC: 1–6)
  - [ ] Create `src/recommendations.py` module
  - [ ] For each persona: derive objective/channel/offer/frequency/KPI from cluster profile
  - [ ] Return dict: cluster_id → recommendation fields
- [ ] Task 2 — Add to persona cards in `04_personas.ipynb` (AC: 6)
  - [ ] Append recommendations section to each persona card display
  - [ ] Update `segment_{n}.md` files with recommendations section

## Dev Notes

### Architecture Guardrails

**Module:** `src/recommendations.py` — create this module.  
**Notebook:** `04_personas.ipynb` — E6 section, after US-6.2.

**recommendations.py skeleton:**
```python
# src/recommendations.py
import pandas as pd

OBJECTIVE_MAP = {
    'Top': 'Retain',    # High CLV → focus on retention
    'Mid': 'Upsell',    # Mid CLV → grow their value
    'Low': 'Re-engage', # Low CLV → reactivation needed
    # Override for new customers regardless of tier:
    'new_customer': 'Acquire',
}

def build_marketing_recommendations(
    cluster_id: int,
    cluster_profile: dict,   # from PERSONA_REGISTRY + cluster_kpis row
    global_kpis: dict,
) -> dict:
    """Build marketing recommendation for one cluster."""
    # Channel recommendation: match dominant_channel
    # Offer type: inferred from loyalty tier + CLV tier
    # Frequency: higher for high-frequency customers
    ...
```

**Recommendation logic (data-driven, not hardcoded):**
- `dominant_channel == 'store'` → recommended channel: in-store event / loyalty touchpoint
- `dominant_channel == 'estore'` → recommended channel: email + push notification
- `dominant_channel == 'click_collect'` → recommended channel: multi-channel (both)
- `clv_tier == 'Top'` + `loyalty_status dominant = GOLD` → objective: Retain; offer: exclusive preview
- `clv_tier == 'Low'` + high `recency_days` → objective: Re-engage; offer: "We miss you" discount
- `is_new_customer` high → objective: Acquire more like this; offer: welcome program

**Frequency guidance:**
- `frequency` (avg) > median → monthly (already buying often)
- `frequency` < median AND `recency_days` > 90 → weekly re-engagement

### Previous Story Output (6-2)
- `PERSONA_REGISTRY` with names and archetypes
- `cluster_kpis_df`, persona cards generated

### Output of This Story
- `recommendations`: dict per cluster
- `segment_{n}.md` files updated with recommendations section
- `src/recommendations.py` created

### References
- [epics — US-6.3](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — recommendations.py module](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to create:
- `src/recommendations.py` (create — add `build_marketing_recommendations()`)

Files to modify:
- `04_personas.ipynb` (add US-6.3 section)
- `_bmad-output/implementation-artifacts/personas/segment_{n}.md` (add recommendations)
