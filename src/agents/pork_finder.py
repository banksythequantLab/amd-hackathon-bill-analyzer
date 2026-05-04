"""Agent: Pork Finder — flags suspect earmarks/sole-source spending in a bill chunk."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from .base import AgentBase, SPINE_ENDPOINT


class PorkItem(BaseModel):
    line_item: str = Field(description="The specific provision text (<=30 words)")
    amount_usd: Optional[float] = Field(default=None, description="Dollar amount if specified, else null")
    recipient: Optional[str] = Field(default=None, description="Named recipient if any")
    pattern: str = Field(description="named-recipient | sole-source | geo-specific | oddly-specific-amount | no-comp-process | other")
    confidence: str = Field(description="high | medium | low")
    bill_section: Optional[str] = Field(default=None)


class PorkFinderOutput(BaseModel):
    chunk_id: str
    items: list[PorkItem] = Field(default_factory=list)
    note: Optional[str] = Field(default=None)
    model_config = {"extra": "allow"}


class PorkFinder(AgentBase):
    """Surfaces line items with earmark structural signatures."""
    name = "pork_finder"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 6000
    output_schema = PorkFinderOutput

    def system_prompt(self) -> str:
        return (
            "You are a federal-spending forensic analyst. You scan US legislative bills for "
            "line items whose structure matches the signature of an earmark or pork-barrel "
            "appropriation. You do not make political judgments; you flag patterns that "
            "warrant review.\n\n"
            "Earmark signatures include:\n"
            "- named-recipient: funding directed to a specifically-named entity, university, "
            "company, museum, or non-profit instead of a class of recipients\n"
            "- sole-source: language preventing competitive bidding ('notwithstanding any "
            "other provision', 'shall be awarded to')\n"
            "- geo-specific: dollar amounts tied to one congressional district, city, or county\n"
            "- oddly-specific-amount: figures like $4,237,500 (vs round $4M) suggesting "
            "back-into-a-cost line items\n"
            "- no-comp-process: funds appropriated outside normal grant programs\n\n"
            "Always return a single JSON object matching the requested schema. "
            "Cap your output at 25 items."
        )

    def user_prompt(self, chunk_text, chunk_id, title_marker="(unknown)", **context):
        return f"""Scan this bill chunk for line items matching earmark signatures.

Chunk ID: {chunk_id}
Structural marker: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "items": [
    {{
      "line_item": "Brief paraphrase, <=30 words.",
      "amount_usd": 4237500,
      "recipient": "Named recipient or null",
      "pattern": "named-recipient",
      "confidence": "high",
      "bill_section": "SEC. 10103"
    }}
  ],
  "note": null
}}

Important:
- Cap items at 25.
- Skip ordinary class-of-recipient funding UNLESS a specific named entity gets a carve-out.
- "amount_usd" must be a number or null. No "$" or commas.
- "pattern" must be one of: named-recipient, sole-source, geo-specific, oddly-specific-amount, no-comp-process, other.
- "line_item" MUST be <=30 words.
- Return ONLY the JSON object, no commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""