"""
Agent #7: Stakeholder Tracer (per-chunk extractor)

Map step of a map-reduce that traces named entities across all chunks of a bill.
Each chunk-level run extracts entities mentioned + their role + the section ID
where they appear. A separate reducer (not yet built) will merge cross-chunk
results into a single stakeholder map.

Why split: a 200K-token chunk doesn't fit anywhere alongside the OTHER chunks
of the same bill. Map-reduce is the honest pattern. Each map call benefits
from APC because the chunk text is already cached from the prior agents.

Four stakeholder buckets:
  - federal_agencies : USDA, EPA, DOE, Forest Service, IRS, etc.
  - programs        : named programs/funds (Collaborative Forest Landscape
                       Restoration Program, Healthy Soils Initiative, etc.)
  - geographies     : states, counties, watersheds, federal lands named in
                       the bill text
  - named_recipients : specific organizations/companies the bill names

Distinct from Pork Finder (which looks at SUSPECT named recipients only)
and from Fiscal Impact Estimator (which looks at amount + recipient_class
not specific entities). This agent surfaces all entities for downstream
analysis - "who shows up in this bill, where, in what role."
"""
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .base import AgentBase, SPINE_ENDPOINT


STAKEHOLDER_KINDS = {
    "federal-agency",
    "program",
    "geography",
    "named-recipient",
    "congressional-committee",
    "advisory-body",
    "other",
}


class StakeholderMention(BaseModel):
    name: str = Field(description="Canonical name as referenced in the bill")
    kind: str = Field(description=" | ".join(sorted(STAKEHOLDER_KINDS)))
    role: str = Field(description="Brief description of the entity's role here, <=20 words")
    bill_section: Optional[str] = Field(default=None, description="Section ID where this mention occurs")
    mention_count: int = Field(default=1, description="How many times this entity appears in the chunk")

    @field_validator("kind")
    @classmethod
    def _valid_kind(cls, v: str) -> str:
        if v not in STAKEHOLDER_KINDS:
            raise ValueError(f"kind must be one of {sorted(STAKEHOLDER_KINDS)}, got {v!r}")
        return v

    @field_validator("mention_count")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"mention_count must be >= 1, got {v}")
        return v


class StakeholderTracerOutput(BaseModel):
    chunk_id: str
    stakeholders: list[StakeholderMention] = Field(default_factory=list)
    note: Optional[str] = Field(default=None)
    model_config = {"extra": "allow"}


class StakeholderTracer(AgentBase):
    """Per-chunk stakeholder extractor. Map step of cross-chunk tracing.

    On spine because chunk text is APC-warm from prior agents. Could move
    to reasoner once we have a reducer that operates on the maps - the
    reducer fits in 32K easily.
    """
    name = "stakeholder_tracer"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 5000
    output_schema = StakeholderTracerOutput

    def system_prompt(self) -> str:
        return (
            "You are a legislative analyst building a stakeholder map for a US "
            "legislative bill. You scan a bill chunk and extract every named "
            "entity that has a SUBSTANTIVE role in the chunk's provisions.\n\n"
            "STAKEHOLDER KINDS:\n"
            "1. federal-agency : USDA, EPA, DOE, Forest Service, IRS, FEMA, "
            "    Corps of Engineers, Bureau of Reclamation, etc.\n"
            "2. program : named programs or funds. Examples: Collaborative "
            "    Forest Landscape Restoration Program, Healthy Soils "
            "    Initiative, Civilian Climate Corps, Rural Energy for "
            "    America Program. NOT generic phrases like 'the program' "
            "    or 'this initiative'.\n"
            "3. geography : states, counties, cities, watersheds, federal "
            "    lands, tribal lands. NOT vague references like 'rural "
            "    America' or 'underserved communities'.\n"
            "4. named-recipient : specific organizations, universities, "
            "    companies, non-profits explicitly named as receiving "
            "    funds or authority.\n"
            "5. congressional-committee : House/Senate committees named in "
            "    the bill (Appropriations, Agriculture, Armed Services, "
            "    etc.).\n"
            "6. advisory-body : task forces, advisory councils, blue ribbon "
            "    panels, etc. created or referenced.\n"
            "7. other : substantive entity that doesn't fit the above. Use "
            "    sparingly.\n\n"
            "FILTERING RULES:\n"
            "- DO NOT list generic terms ('the Secretary', 'the Department', "
            "'the public') as stakeholders. They are roles, not entities.\n"
            "- DO NOT list a stakeholder unless they have a substantive role "
            "(authorized to act, receive funds, be consulted, etc.). Pure "
            "name-drops in findings clauses can be skipped.\n"
            "- Cap stakeholders at 60 per chunk. Most chunks have 10-30.\n"
            "- mention_count: estimate, doesn't need to be exact.\n\n"
            "Return a single JSON object matching the schema. Empty "
            "stakeholders is acceptable for procedural chunks."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        return f"""Trace named stakeholders in this bill chunk.

Chunk ID: {chunk_id}
Structural marker: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "stakeholders": [
    {{
      "name": "Forest Service",
      "kind": "federal-agency",
      "role": "Authorized to award grants and administer the Collaborative Forest Landscape Restoration Program.",
      "bill_section": "SEC. 11001",
      "mention_count": 12
    }}
  ],
  "note": null
}}

Reminders:
- kind must be one of: federal-agency, program, geography, named-recipient,
  congressional-committee, advisory-body, other
- Skip generic role-references ("the Secretary", "the Department") - those
  are positions, not stakeholders
- Empty stakeholders is fine for procedural-only chunks
- Cap at 60 entries
- Return ONLY the JSON. No commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""