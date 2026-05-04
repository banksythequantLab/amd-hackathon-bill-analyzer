"""Agent: Conflict Spotter — flags contradictory provisions within a chunk."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from .base import AgentBase, SPINE_ENDPOINT


class Conflict(BaseModel):
    description: str = Field(description="One-sentence description of the inconsistency")
    provision_a: str = Field(description="First provision (paraphrased, <=25 words)")
    provision_b: str = Field(description="Second provision (paraphrased, <=25 words)")
    conflict_type: str = Field(description="rule-vs-rule | effective-date | definition-divergence | amount-mismatch | authority-overlap | other")
    severity: str = Field(description="high | medium | low")


class ConflictSpotterOutput(BaseModel):
    chunk_id: str
    conflicts: list[Conflict] = Field(default_factory=list)
    note: Optional[str] = Field(default=None)
    model_config = {"extra": "allow"}


class ConflictSpotter(AgentBase):
    """Surfaces internally contradictory provisions within a single chunk."""
    name = "conflict_spotter"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 6000
    output_schema = ConflictSpotterOutput

    def system_prompt(self) -> str:
        return (
            "You are a legislative drafting analyst. You scan a chunk of a US legislative "
            "bill for internally inconsistent provisions: two clauses that prescribe "
            "different rules for the same situation, conflicting effective dates, "
            "definitions that diverge from how a term is used elsewhere, or amounts that "
            "don't reconcile.\n\n"
            "What COUNTS as a conflict:\n"
            "- Two provisions both purport to govern the same fact pattern with different rules\n"
            "- Effective dates that are inconsistent\n"
            "- A defined term used inconsistently with its definition\n"
            "- Authorization amounts that don't add up to the totals stated elsewhere\n"
            "- Two agencies given overlapping authority with no priority rule\n\n"
            "What does NOT count:\n"
            "- Different rules for different situations (that's coverage, not conflict)\n"
            "- Phased rollouts where dates differ on purpose\n"
            "- Cross-references to the same section\n\n"
            "Always return a single JSON object matching the requested schema. "
            "Cap your output at 15 conflicts. If you find none, return an empty list."
        )

    def user_prompt(self, chunk_text, chunk_id, title_marker="(unknown)", **context):
        return f"""Scan this bill chunk for internally inconsistent provisions.

Chunk ID: {chunk_id}
Structural marker: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "conflicts": [
    {{
      "description": "One-sentence summary of the inconsistency.",
      "provision_a": "First provision paraphrased, <=25 words. SEC. 10101(a) says...",
      "provision_b": "Second provision paraphrased, <=25 words. SEC. 10204(b) says...",
      "conflict_type": "rule-vs-rule",
      "severity": "high"
    }}
  ],
  "note": null
}}

Important:
- Cap conflicts at 15.
- "conflict_type" must be one of: rule-vs-rule, effective-date, definition-divergence, amount-mismatch, authority-overlap, other.
- "severity" must be one of: high, medium, low.
- BOTH provisions MUST be <=25 words. Reference the section number if visible.
- If you find no genuine conflicts, return an empty conflicts array — do not invent.
- Return ONLY the JSON object, no commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""