"""Agent: Citation Validator - audits USC cross-references for accuracy.

Different shape from the chunk-level agents: this works at citation-level,
running per-citation calls against the reasoner endpoint (32K context is
plenty for a single citation + 600 char USC text excerpt + bill context).

Designed to be the second-pass auditor for usc_xref output. The xref agent
identifies citations and the LMDB enrichment fetches the actual USC text;
this validator decides whether the citation is well-formed and whether the
bill's claimed intent (amend/repeal/cross-reference/etc.) matches what the
statute actually says.

This is a per-citation agent, NOT a per-chunk agent. The orchestrator (or a
runner script) loops over xref output citations and calls this agent for
each one. Each call is small (~3-5K input tokens) and fast (~2-5 seconds
generation).

Output schema:
  {
    "citation": "7 U.S.C. 2013(a)",
    "verdict": "valid" | "format-error" | "wrong-section" | "intent-mismatch" | "unverifiable",
    "confidence": "high" | "medium" | "low",
    "issues": ["string", ...],
    "suggested_fix": "string or null"
  }

Day 3 net-new: this is agent #5 in the system. Adds a real layer of
trustworthiness to the citation pipeline that the report consumer can
actually rely on.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from .base import AgentBase, REASONER_ENDPOINT


VALID_VERDICTS = {
    "valid",            # Citation looks good; matches the bill's claimed intent
    "format-error",     # Citation string is malformed (wrong title number, missing section, etc.)
    "wrong-section",    # The cited section exists but doesn't match what the bill says it does
    "intent-mismatch",  # The bill claims to amend/repeal X but the actual USC text says otherwise
    "unverifiable",     # Can't tell from available info; needs human review
}


class ValidationResult(BaseModel):
    citation: str = Field(description="The citation as identified in the bill")
    verdict: str = Field(description=" | ".join(sorted(VALID_VERDICTS)))
    confidence: str = Field(description="high | medium | low")
    issues: list[str] = Field(default_factory=list, description="Specific problems found, if any")
    suggested_fix: Optional[str] = Field(default=None, description="If the citation has a fixable error, what should it be")
    model_config = {"extra": "allow"}

    @field_validator("verdict")
    @classmethod
    def _valid_verdict(cls, v: str) -> str:
        if v not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}, got '{v}'")
        return v

    @field_validator("confidence")
    @classmethod
    def _valid_conf(cls, v: str) -> str:
        v_low = v.lower().strip()
        if v_low not in {"high", "medium", "low"}:
            raise ValueError(f"confidence must be high/medium/low, got '{v}'")
        return v_low


class CitationValidator(AgentBase):
    """Per-citation auditor for usc_xref output.

    Runs on the reasoner endpoint (Qwen3-32B-FP8, 32K context). Reasoner is the
    right home because:
      1. Per-citation calls don't need spine's 262K context
      2. Reasoner has explicit reasoning chain support, useful for legal-correctness
         questions where step-by-step matters
      3. Frees spine for chunk-level agents

    Caller wraps each {citation, bill_context, usc_data} tuple from xref output
    in a per-call invocation.
    """
    name = "citation_validator"
    target_endpoint = REASONER_ENDPOINT
    target_model = "reasoner"
    temperature = 0.0
    max_tokens = 2000  # ~1500 tokens of reasoning + JSON output
    output_schema = ValidationResult

    def system_prompt(self) -> str:
        return (
            "You are a legal-reference auditor specializing in US Code citations within "
            "federal legislation. You receive one citation at a time, drawn from a bill, "
            "along with two pieces of context:\n\n"
            "  1. bill_context: a short excerpt from the bill showing how the citation appears\n"
            "  2. usc_data: the actual USC section heading and text excerpt, fetched from a\n"
            "     local LMDB index of the United States Code\n\n"
            "Your job: decide whether the citation is well-formed and whether the bill's\n"
            "claimed intent (amend, repeal, cross-reference, definitions, etc.) matches\n"
            "what the actual USC section says.\n\n"
            "Possible verdicts:\n"
            "  - valid: citation is well-formed and the bill's intent matches the USC text\n"
            "  - format-error: citation string is malformed (e.g. '26 USC 401k' missing\n"
            "    parens, wrong title number, etc.)\n"
            "  - wrong-section: citation exists but doesn't match what the bill claims\n"
            "    (e.g. bill says 'amends 26 USC 401(k)(7)' but no such subsection exists)\n"
            "  - intent-mismatch: bill claims to amend X but the USC text shows X is about\n"
            "    something different from what the bill is doing\n"
            "  - unverifiable: insufficient information to tell. Use sparingly.\n\n"
            "Be conservative. 'valid' is the right answer when:\n"
            "  - The citation format is correct\n"
            "  - The USC text excerpt is broadly consistent with the bill_context\n"
            "  - The bill's claimed intent is plausible given the USC content\n\n"
            "Don't flag false positives. Bills routinely cross-reference USC sections in\n"
            "ways that look terse but are valid. Only flag things that are actually wrong.\n\n"
            "Return JSON only. No commentary."
        )

    def user_prompt(self, chunk_text: str = "", chunk_id: str = "", **context):
        # citation_validator runs at citation level not chunk level. Conform to
        # AgentBase.user_prompt(chunk_text, chunk_id, **context) signature so
        # base.run() can call us positionally; pull our actual fields from kwargs.
        citation = context.get("citation", chunk_id or "")
        bill_context = context.get("bill_context", "")
        relevance = context.get("relevance", "")
        usc_heading = context.get("usc_heading", "")
        usc_text = context.get("usc_text", "")
        return f"""Audit this USC citation from a federal bill.

CITATION FROM BILL:
  Citation: {citation}
  Bill claims relevance: {relevance}
  Bill context excerpt: "{bill_context}"

ACTUAL USC SECTION (from LMDB index):
  Heading: {usc_heading}
  Text excerpt: "{usc_text}"

Decide whether the citation is well-formed and matches the bill's claimed
intent. Return a JSON object:

{{
  "citation": "{citation}",
  "verdict": "valid",
  "confidence": "high",
  "issues": [],
  "suggested_fix": null
}}

Verdict must be one of: valid, format-error, wrong-section, intent-mismatch, unverifiable.
Confidence must be: high, medium, or low.
issues: list of specific problems (empty if verdict is valid).
suggested_fix: a corrected citation string if the issue is fixable, else null.

Return ONLY the JSON object.
"""