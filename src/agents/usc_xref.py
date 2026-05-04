"""
Agent #4: USC Cross-Reference

Identifies USC citations in a bill chunk, fetches each from the LMDB, and
reports a citation map: {bill_passage -> {usc_section, heading, current_text_excerpt}}

Two-pass design:
  Pass 1: LLM identifies all USC citations in the chunk text. Returns a list
          of {citation, context, relevance}. We DON'T have the LLM regurgitate
          the existing USC text — that's what the tool is for.
  Pass 2: Local code (no LLM) calls fetch_usc for each citation, attaches
          the existing-statute heading/excerpt, and bundles into the final output.

This pattern is critical: it grounds the LLM's output in actual statutory text
rather than the LLM's recollection (which would hallucinate freely).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentBase, REASONER_ENDPOINT


class IdentifiedCitation(BaseModel):
    citation: str = Field(description="The citation as it appears in the bill, e.g. '26 USC 401(k)'")
    bill_context: str = Field(description="A short (1-2 sentence) excerpt of the surrounding bill text")
    relevance: str = Field(description="Why the bill is invoking this section (amend / repeal / cross-reference / definitions)")


class CrossReferenceOutput(BaseModel):
    chunk_id: str
    citations: list[IdentifiedCitation] = Field(default_factory=list)
    note: Optional[str] = Field(default=None, description="Optional note about citation density, gaps, etc.")


class UscCrossReference(AgentBase):
    """Pass 1 of USC cross-reference: identify citations only."""
    name = "usc_cross_reference"
    target_endpoint = REASONER_ENDPOINT
    target_model = "reasoner"
    temperature = 0.0
    max_tokens = 8000  # Reasoner emits a long <think> chain before JSON
    output_schema = CrossReferenceOutput

    def system_prompt(self) -> str:
        return (
            "You are a legislative analyst expert in identifying citations to the US Code. "
            "You receive a chunk of a US legislative bill and identify every citation it makes "
            "to the US Code (USC). Citations may appear as '26 USC 401', '26 U.S.C. § 401(k)', "
            "'section 401 of the Internal Revenue Code of 1986', '26 USC 1', etc. "
            "For each citation, report: the citation in canonical form (e.g. '26 USC 401(k)'), "
            "a 1-2 sentence excerpt of bill context around it, and the bill's intent toward that "
            "section (amend, repeal, cross-reference, definitions, transitional rule, etc.). "
            "Do NOT speculate about what the cited section says — your job is purely to identify "
            "the citation. A separate tool will fetch the existing statute. "
            "Return a single JSON object matching the requested schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        return f"""Identify every USC citation in this bill chunk.

Chunk ID: {chunk_id}
Structural marker: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "citations": [
    {{
      "citation": "26 USC 401(k)",
      "bill_context": "Brief 1-2 sentence quote/paraphrase of where this appears in the bill.",
      "relevance": "amend" 
    }}
  ],
  "note": null
}}

Important:
- Cap citations at 30 total. If the chunk has more, focus on the most consequential.
- The "relevance" field should be one of: amend, repeal, cross-reference, definitions, transitional, other.
- Skip purely procedural references (e.g. "1 USC 1" rules of construction) unless the bill specifically modifies them.
- Return ONLY the JSON object, no commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""


# ----------------------------------------------------------------------
# Pass 2: enrich identified citations with USC data via fetch_usc tool
# ----------------------------------------------------------------------
def enrich_with_usc(crossref: dict, fetcher) -> dict:
    """Enrich each citation with the actual USC text via fetch_usc.

    Mutates `crossref` in place by adding 'usc_data' to each citation entry.
    Returns the same dict for chaining.
    """
    for c in crossref.get("citations", []):
        record = fetcher(c["citation"])
        if record is None:
            c["usc_data"] = None
            c["resolution_status"] = "not_found"
            continue
        # Compact the heavy fields for downstream agents
        c["usc_data"] = {
            "title": record["title"],
            "section": record["section"],
            "heading": record["heading"],
            "text_excerpt": record["text"][:600] + ("..." if len(record["text"]) > 600 else ""),
            "source_url": record["source_url"],
            "release_point": record["release_point"],
        }
        c["resolution_status"] = "ok"
    return crossref
