"""
Agent #5: Fiscal Impact Estimator (v2 — Day 6 prompt rewrite)

Day 5 v1 hung in the retry loop for 16+ minutes on BBB ch01. Three changes
in v2 to make it ship clean:

1. Domain enum: 12 -> 6 categories. Less "right answer" pressure on the model
   under truncation; broader buckets are easier to pick correctly.
2. max_tokens 6000 -> 4000. Cap items 40 -> 25. Tighter budget = faster fail
   if validation issues, less wall clock per attempt.
3. Totals aggregation pulled OUT of the LLM. The model returns items[] only;
   totals_by_domain and grand_total_usd are computed in Python from the
   items array. The LLM was using its limited token budget on arithmetic
   and aggregation logic that python.sum() handles trivially. This is the
   biggest win.

Two-tier vision design preserved from v1:
  - Text pass extracts dollar-amount items from prose
  - vision_pages_suggested[] flags pages where the bill formats numbers as
    tables (rate schedules, allocation matrices). Orchestrator routes
    those to the vision endpoint separately.

The agent itself is text-only; vision call is the orchestrator's
responsibility. Per-page vision invocation is heavy (~2K image tokens)
and shouldn't fire on every chunk.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from .base import AgentBase, SPINE_ENDPOINT


# 6 categories, deliberately broad. The model picks the dominant theme;
# downstream code can re-bucket more finely from the items[] field if needed.
DOMAIN_CATEGORIES = {
    "agriculture-conservation-energy",   # USDA, EPA, DOE, Forest, climate
    "health-human-services",             # HHS, Medicare, Medicaid, education
    "infrastructure-housing-transport",  # housing, roads, water, broadband
    "tax-and-revenue",                   # tax credits, deductions, IRS
    "defense-immigration-law",           # DOD, DHS, immigration, justice
    "research-administration-other",     # NSF, generic admin, catch-all
}


class LineItem(BaseModel):
    amount_usd: float = Field(description="Dollar amount in USD as a number (no $, no commas)")
    purpose: str = Field(description="Brief paraphrase of what the funds are for, <=20 words")
    recipient_class: str = Field(description="Kind of recipient (e.g. 'Forest Service program', 'state grants')")
    fiscal_years: list[int] = Field(default_factory=list, description="FY ints, e.g. [2024, 2025]")
    domain: str = Field(description=" | ".join(sorted(DOMAIN_CATEGORIES)))
    bill_section: Optional[str] = Field(default=None)
    source_page: Optional[int] = Field(default=None)
    table_formatted: bool = Field(default=False, description="True if amount is in a table the model thinks vision could clarify")

    @field_validator("domain")
    @classmethod
    def _valid_domain(cls, v: str) -> str:
        if v not in DOMAIN_CATEGORIES:
            raise ValueError(f"domain must be one of {sorted(DOMAIN_CATEGORIES)}, got {v!r}")
        return v

    @field_validator("amount_usd")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"amount_usd must be >= 0, got {v}")
        return v


class FiscalImpactOutput(BaseModel):
    chunk_id: str
    items: list[LineItem] = Field(default_factory=list)
    vision_pages_suggested: list[int] = Field(default_factory=list)
    note: Optional[str] = Field(default=None)
    # totals_by_domain and grand_total_usd are NOT requested from the LLM.
    # They get computed in Python after validation passes; see compute_totals().
    totals_by_domain: Optional[dict] = Field(default=None, description="Computed post-hoc; not LLM-emitted")
    grand_total_usd: Optional[float] = Field(default=None, description="Computed post-hoc; not LLM-emitted")
    model_config = {"extra": "allow"}


def compute_totals(output: FiscalImpactOutput) -> FiscalImpactOutput:
    """Post-process: compute totals_by_domain and grand_total_usd from items[].

    Keeps the LLM out of arithmetic. Mutates output in place AND returns it.
    """
    by_domain: dict[str, dict] = {}
    grand = 0.0
    for item in output.items:
        d = item.domain
        if d not in by_domain:
            by_domain[d] = {"total_usd": 0.0, "item_count": 0}
        by_domain[d]["total_usd"] += item.amount_usd
        by_domain[d]["item_count"] += 1
        grand += item.amount_usd
    output.totals_by_domain = by_domain
    output.grand_total_usd = grand
    return output


class FiscalImpactEstimator(AgentBase):
    """Text-pass appropriations extractor on spine.

    Items extraction only -- aggregation handled in Python via compute_totals().
    """
    name = "fiscal_impact_estimator"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 4000
    output_schema = FiscalImpactOutput

    def system_prompt(self) -> str:
        return (
            "You are a federal budget analyst extracting structured appropriations data "
            "from US legislative bills. For each dollar amount, you record: amount, purpose, "
            "recipient class, fiscal years, domain category.\n\n"
            "FRAMING: most chunks have 5-25 distinct appropriations. Some have zero "
            "(procedural-only). Find the real numbers and structure them. Do not invent "
            "or duplicate.\n\n"
            "WHAT COUNTS AS A LINE ITEM:\n"
            "- Direct appropriations: '$10,000,000,000 for hazardous fuels reduction'\n"
            "- Authorizations: 'There are authorized to be appropriated $500,000,000...'\n"
            "- Funding ceilings: 'not to exceed $250,000,000 in any fiscal year'\n"
            "- Tax expenditures: 'a credit equal to $X per qualified...'\n"
            "WHAT DOES NOT COUNT:\n"
            "- Citations to dollar amounts in OTHER laws (e.g. references to 26 USC limits)\n"
            "- Threshold values that are not appropriations (e.g. '$50,000 income limit')\n"
            "- Open-ended 'such sums as may be necessary' without a number — OMIT, do not guess\n\n"
            "DOMAINS (pick exactly one per item):\n"
            "  agriculture-conservation-energy : USDA, EPA, DOE, Forest, climate, water\n"
            "  health-human-services           : HHS, Medicare, Medicaid, education, workforce\n"
            "  infrastructure-housing-transport: HUD, DOT, broadband, public works, housing\n"
            "  tax-and-revenue                 : tax credits, deductions, IRS, revenue provisions\n"
            "  defense-immigration-law         : DOD, DHS, justice, immigration, veterans\n"
            "  research-administration-other   : NSF, generic admin, catch-all\n\n"
            "TABLE FLAGS: if the bill has a table or rate schedule (parallel columns of "
            "numbers), set table_formatted=true on the affected items AND add the bill page "
            "number to vision_pages_suggested. Tax bracket schedules are the canonical case.\n\n"
            "Cap items at 25. Prefer the largest/most-consequential when over-cap.\n\n"
            "Return a single JSON object matching the schema. Do NOT include totals; those "
            "are computed downstream from items[]."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        return f"""Extract appropriations from this bill chunk.

Chunk ID: {chunk_id}
Structural marker: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "items": [
    {{
      "amount_usd": 10000000000,
      "purpose": "Hazardous fuels reduction in wildland-urban interface.",
      "recipient_class": "Forest Service program",
      "fiscal_years": [2022, 2023, 2024, 2025, 2026],
      "domain": "agriculture-conservation-energy",
      "bill_section": "SEC. 11001",
      "source_page": 5,
      "table_formatted": false
    }}
  ],
  "vision_pages_suggested": [],
  "note": null
}}

Reminders:
- amount_usd: number, no $ sign, no commas
- domain MUST be one of: agriculture-conservation-energy, health-human-services,
  infrastructure-housing-transport, tax-and-revenue, defense-immigration-law,
  research-administration-other
- items[] empty is fine for procedural chunks; use note to explain
- Cap at 25 items; pick the most consequential
- Do NOT compute totals; just extract items
- Return ONLY the JSON object. No commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""