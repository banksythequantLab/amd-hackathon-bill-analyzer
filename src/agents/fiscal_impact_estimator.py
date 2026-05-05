"""
Agent #5: Fiscal Impact Estimator

Reads a bill chunk and extracts structured appropriations data:
  - per line item: amount, recipient class, fiscal year, purpose
  - aggregated totals by domain category
  - flags items where amount appears in a TABLE (not prose) for vision augmentation

Two-tier design:
  Tier 1: TEXT pass on the spine model. Reads the full chunk, extracts
    all appropriations expressible as prose ("$10,000,000,000 for X").
    Most appropriations in BBB-2021 / HR1 are this shape.
  Tier 2 (optional): VISION pass. The agent identifies pages where the
    bill formats appropriations as tables (rate schedules, allocation
    matrices, etc.). A separate caller can route those pages to the
    vision endpoint for structured extraction. The text-pass output
    flags these pages so downstream code knows where to call vision.

The agent itself is text-only; it returns "vision_pages_suggested" with
page numbers the spine thinks contain table-formatted appropriations.
The caller decides whether to act on those suggestions.

Why this split:
  Vision calls are expensive (a single page is ~2K image tokens through
  Qwen3-VL). Asking the spine "which pages have tables?" is cheap because
  the chunk is already cached via APC. The text agent does the cheap
  routing decision; vision is invoked only when worthwhile.
"""
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .base import AgentBase, SPINE_ENDPOINT


# Aggregation buckets. Each line item gets categorized into one of these.
DOMAIN_CATEGORIES = {
    "agriculture-conservation",
    "energy-climate",
    "health-medicare-medicaid",
    "education-workforce",
    "housing-community",
    "transportation-infrastructure",
    "tax-credits",
    "defense-veterans",
    "immigration-border",
    "research-science",
    "general-administration",
    "other",
}


class LineItem(BaseModel):
    amount_usd: float = Field(description="Dollar amount in USD as a number (no $, no commas)")
    purpose: str = Field(description="Brief paraphrase of what the funds are for, <=25 words")
    recipient_class: str = Field(
        description=(
            "What kind of recipient. Examples: 'Forest Service program', 'state grants', "
            "'tribal governments', 'national lab consortium', 'veterans of the Vietnam War'. "
            "Distinct from Pork Finder's 'specific named entity' field."
        )
    )
    fiscal_years: list[int] = Field(
        default_factory=list,
        description="Fiscal years the appropriation covers, e.g. [2024, 2025, 2026]",
    )
    domain: str = Field(description=" | ".join(sorted(DOMAIN_CATEGORIES)))
    bill_section: Optional[str] = Field(default=None, description="Bill section ID like 'SEC. 11001'")
    source_page: Optional[int] = Field(default=None, description="Page in the original PDF, if known")
    table_formatted: bool = Field(
        default=False,
        description="True if the agent thinks this amount is part of a table/schedule that vision could clarify",
    )

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


class DomainTotal(BaseModel):
    domain: str
    total_usd: float
    item_count: int


class FiscalImpactOutput(BaseModel):
    chunk_id: str
    items: list[LineItem] = Field(default_factory=list)
    totals_by_domain: list[DomainTotal] = Field(default_factory=list)
    grand_total_usd: float = Field(default=0.0)
    vision_pages_suggested: list[int] = Field(
        default_factory=list,
        description=(
            "Page numbers where the chunk contains table-formatted appropriations "
            "that text-only extraction may have missed. The orchestrator can route "
            "these pages to the vision endpoint."
        ),
    )
    note: Optional[str] = Field(default=None)
    model_config = {"extra": "allow"}


class FiscalImpactEstimator(AgentBase):
    """Text-pass appropriations extractor on spine.

    Routes vision augmentation decisions to the orchestrator via
    vision_pages_suggested. Doesn't call vision itself; that's the
    orchestrator's job because page->image conversion + base64 encoding
    is heavy I/O and shouldn't live in an agent.
    """
    name = "fiscal_impact_estimator"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 6000
    output_schema = FiscalImpactOutput

    def system_prompt(self) -> str:
        return (
            "You are a federal budget analyst extracting structured appropriations data "
            "from US legislative bills. For each dollar amount in the chunk, you record: "
            "the amount, the purpose, the kind of recipient, fiscal years covered, and "
            "the bill section.\n\n"
            "FRAMING: most legislative chunks contain 5-30 distinct appropriations. Some "
            "have zero (procedural-only chunks). Your job is to find the real numbers and "
            "structure them, not to invent or duplicate.\n\n"
            "WHAT COUNTS AS A LINE ITEM:\n"
            "- Direct appropriations: '$10,000,000,000 for hazardous fuels reduction'\n"
            "- Authorizations: 'There are authorized to be appropriated $500,000,000...'\n"
            "- Funding limits: 'not to exceed $250,000,000 in any fiscal year'\n"
            "- Tax expenditures: 'a credit equal to $X per qualified...'\n"
            "WHAT DOES NOT COUNT:\n"
            "- Citations to dollar amounts in OTHER laws (e.g. references to 26 USC limits)\n"
            "- Threshold values that aren't appropriations (e.g. '$50,000 income limit')\n"
            "- Historical figures cited for context\n\n"
            "DOMAIN ASSIGNMENT: pick the BEST single category from the enum. If a line item "
            "spans two domains, pick the dominant one. 'other' is a real choice — use it for "
            "items that don't fit cleanly.\n\n"
            "TABLE FLAGS: if you see a structured table or rate schedule in the bill text "
            "(parallel columns of numbers, dense numerical layout that's hard to parse from "
            "text alone, references like 'see table at end of section'), set table_formatted "
            "to true on the affected items AND add the bill page number to "
            "vision_pages_suggested. Tax bracket schedules, formula tables, and allocation "
            "matrices typically warrant this.\n\n"
            "AGGREGATION: after extracting items, compute totals_by_domain (sum amount_usd "
            "per domain bucket) and grand_total_usd (sum of all items). Be honest about "
            "the precision: if amounts are 'such sums as may be necessary' (open-ended), "
            "OMIT the item rather than guessing.\n\n"
            "Return a single JSON object matching the schema. Cap items at 40."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        page_hint = context.get("page_range", "")
        page_str = f"\nPage range: {page_hint}" if page_hint else ""
        return f"""Extract appropriations from this bill chunk.

Chunk ID: {chunk_id}
Structural marker: {title_marker}{page_str}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "items": [
    {{
      "amount_usd": 10000000000,
      "purpose": "Hazardous fuels reduction projects in the wildland-urban interface.",
      "recipient_class": "Forest Service program",
      "fiscal_years": [2022, 2023, 2024, 2025, 2026],
      "domain": "agriculture-conservation",
      "bill_section": "SEC. 11001",
      "source_page": 5,
      "table_formatted": false
    }}
  ],
  "totals_by_domain": [
    {{"domain": "agriculture-conservation", "total_usd": 12500000000, "item_count": 3}}
  ],
  "grand_total_usd": 12500000000,
  "vision_pages_suggested": [2222],
  "note": null
}}

Reminders:
- amount_usd is a number, no $ sign, no commas
- domain MUST be one of: agriculture-conservation, energy-climate, health-medicare-medicaid,
  education-workforce, housing-community, transportation-infrastructure, tax-credits,
  defense-veterans, immigration-border, research-science, general-administration, other
- If you can't find any appropriations, return items: [] and grand_total_usd: 0
- Compute totals_by_domain yourself from the items you extracted
- Set vision_pages_suggested to page numbers with table-formatted appropriations
- Return ONLY the JSON. No commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""