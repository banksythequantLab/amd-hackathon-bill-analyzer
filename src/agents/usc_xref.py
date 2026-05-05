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

from .base import AgentBase, SPINE_ENDPOINT


class IdentifiedCitation(BaseModel):
    citation: str = Field(description="The citation as it appears in the bill, e.g. '26 USC 401(k)'")
    bill_context: str = Field(description="A short (1-2 sentence) excerpt of the surrounding bill text")
    relevance: str = Field(description="Why the bill is invoking this section (amend / repeal / cross-reference / definitions)")


class CrossReferenceOutput(BaseModel):
    chunk_id: str
    citations: list[IdentifiedCitation] = Field(default_factory=list)
    note: Optional[str] = Field(default=None, description="Optional note about citation density, gaps, etc.")

    # Allow the truncation-recovery flag from base.extract_json without breaking validation.
    # When True, the citations list is partial (the LLM was cut off at max_tokens).
    model_config = {"extra": "allow"}


class UscCrossReference(AgentBase):
    """Pass 1 of USC cross-reference: identify citations only.

    Runs on the SPINE endpoint (262K context) because bill chunks routinely
    exceed the reasoner's 32K window. APC means the second agent against the
    same chunk gets a warm KV cache and runs much faster than the first.
    """
    name = "usc_cross_reference"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 8000   # ~30 citations with bill_context excerpts ≈ 6-7K tokens
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
- The "bill_context" field MUST be 30 words or fewer. Just enough context to recognize where in the bill the citation appears. DO NOT quote whole statutory paragraphs.
- Skip purely procedural references (e.g. "1 USC 1" rules of construction) unless the bill specifically modifies them.
- Return ONLY the JSON object, no commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""


# ----------------------------------------------------------------------
# Pass 2: enrich identified citations with USC data via fetch_usc tool
# ----------------------------------------------------------------------
def _strip_subparagraph(citation: str) -> str:
    """Strip sub-paragraph notation: '16 U.S.C. 6542(d)(1)' -> '16 U.S.C. 6542'.
    LMDB stores at section level only. Falling back to bare section is the
    right semantics for citation enrichment - downstream Citation Validator
    can flag if the bill's claimed sub-paragraph doesn't exist in the section.
    """
    import re as _re
    return _re.sub(r"\([^)]*\)(?:\([^)]*\))*\s*$", "", citation).strip()


def _dedup_citations(citations: list) -> list:
    """Drop duplicate (citation, bill_context) pairs left by truncation-recovery.
    The Day 3 BBB run produced 13 copies of '16 U.S.C. 7655d' with identical
    bill_context because the model truncated mid-citation and recovery duplicated
    the trailing partial entry. This collapses such duplicates."""
    seen = set()
    out = []
    for c in citations:
        key = (c.get("citation", ""), (c.get("bill_context", "") or "")[:100])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def enrich_with_usc(crossref: dict, fetcher) -> dict:
    """Enrich each citation with the actual USC text via fetch_usc.

    Mutates `crossref` in place by adding 'usc_data' to each citation entry.
    Two-tier lookup: first try the citation as written, then strip
    sub-paragraphs and retry. If both miss, mark as not_found.
    Also dedupes citations on (citation, first 100 chars of bill_context) to
    drop truncation-recovery artifacts.
    Returns the same dict for chaining.
    """
    # Dedup BEFORE enrichment so we don't pay LMDB lookups for duplicates.
    citations = crossref.get("citations", [])
    deduped = _dedup_citations(citations)
    if len(deduped) < len(citations):
        crossref.setdefault("_pipeline_notes", []).append(
            f"deduped {len(citations)} -> {len(deduped)} citations (truncation-recovery artifacts)"
        )
    crossref["citations"] = deduped

    for c in crossref["citations"]:
        # Tier 1: try the citation as written
        cit_str = c["citation"]
        record = fetcher(cit_str)
        resolution = "ok"

        # Tier 2: if the citation has sub-paragraphs, strip and retry at section level
        if record is None:
            stripped = _strip_subparagraph(cit_str)
            if stripped and stripped != cit_str:
                record = fetcher(stripped)
                if record is not None:
                    resolution = "ok-section-level"
                    c["resolved_to"] = stripped

        if record is None:
            c["usc_data"] = None
            c["resolution_status"] = "not_found"
            continue

        c["usc_data"] = {
            "title": record["title"],
            "section": record["section"],
            "heading": record["heading"],
            "text_excerpt": record["text"][:600] + ("..." if len(record["text"]) > 600 else ""),
            "source_url": record["source_url"],
            "release_point": record["release_point"],
        }
        c["resolution_status"] = resolution
    return crossref
