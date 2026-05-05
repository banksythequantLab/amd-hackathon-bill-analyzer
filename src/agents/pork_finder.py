"""Agent: Pork Finder - flags suspect earmarks/sole-source spending in a bill chunk.

Day 3 prompt rewrite (2026-05-04 evening):

Background: The Day 2 run on BBB-2021 ch01 produced 27 items, ALL tagged
'named-recipient' with recipient=null. That's the model picking the most
prominent pattern label from the prompt without actually checking whether
the named-recipient definition applied. Real result was: model surfaced
27 ordinary class-of-recipient appropriations that the prompt's own rules
told it to skip.

Root causes:
  1. The pattern enum was front-loaded with 'named-recipient' which biased
     the classifier toward that label.
  2. The prompt described what to look for but didn't show worked examples
     of each pattern vs ordinary appropriations to skip.
  3. The instruction "skip ordinary class-of-recipient funding UNLESS a
     specific named entity gets a carve-out" lived in a long bullet list,
     easy to lose in 200K tokens of context.
  4. No hard cross-validation: a 'named-recipient' classification with
     recipient=null is structurally invalid but the schema didn't enforce
     it.

This rewrite:
  - Pattern enum now leads with 'ordinary-class' which is the FILTER label,
    so the model has somewhere clean to put non-pork items if it must
    classify them. Combined with an explicit instruction to OMIT such items.
  - Each pattern label has a concrete example IN the prompt (a real-sounding
    bill snippet + why it qualifies).
  - The instruction "if recipient is null, the pattern CANNOT be
    named-recipient" appears as a hard rule with the schema, not buried.
  - Empty result is explicitly endorsed: "If nothing in the chunk meets the
    pork-signature definitions below, return items: []. Most bill chunks
    will have zero or one or two real flags; large lists are suspicious."
  - Pydantic schema now enforces the recipient/pattern cross-rule via a
    model_validator so the agent's retry loop catches the model's own
    inconsistencies before they reach downstream agents.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from .base import AgentBase, SPINE_ENDPOINT


VALID_PATTERNS = {
    "ordinary-class",       # NEW: not pork; the filter label
    "named-recipient",      # specific entity named in the bill
    "sole-source",          # competitive-bidding bypass language
    "geo-specific",         # tied to one district/city/county
    "oddly-specific-amount",# unusual precision (e.g. $4,237,500)
    "no-comp-process",      # outside normal grant/appropriation channels
    "other",                # something looks off but doesn't fit above
}


class PorkItem(BaseModel):
    line_item: str = Field(description="The specific provision text (<=30 words)")
    amount_usd: Optional[float] = Field(default=None, description="Dollar amount if specified, else null")
    recipient: Optional[str] = Field(default=None, description="Named recipient if any. Required when pattern == named-recipient.")
    pattern: str = Field(description=" | ".join(sorted(VALID_PATTERNS)))
    confidence: str = Field(description="high | medium | low")
    bill_section: Optional[str] = Field(default=None)
    why_flagged: Optional[str] = Field(default=None, description="Brief reason this matches the pattern signature, <=20 words.")

    @field_validator("pattern")
    @classmethod
    def _valid_pattern(cls, v: str) -> str:
        if v not in VALID_PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(VALID_PATTERNS)}, got '{v}'")
        return v

    @field_validator("confidence")
    @classmethod
    def _valid_confidence(cls, v: str) -> str:
        v_low = v.lower().strip()
        if v_low not in {"high", "medium", "low"}:
            raise ValueError(f"confidence must be high/medium/low, got '{v}'")
        return v_low

    @model_validator(mode="after")
    def _named_recipient_requires_recipient(self):
        # Hard cross-validation: named-recipient with no recipient is structurally invalid.
        # This caught the entire Day 2 BBB ch01 failure mode.
        if self.pattern == "named-recipient" and not (self.recipient and self.recipient.strip()):
            raise ValueError(
                "pattern='named-recipient' but recipient is empty/null. "
                "If no specific entity is named in the bill, the correct pattern is "
                "'ordinary-class' (and the item should be omitted from items[])."
            )
        return self


class PorkFinderOutput(BaseModel):
    chunk_id: str
    items: list[PorkItem] = Field(default_factory=list)
    note: Optional[str] = Field(default=None, description="Optional analyst note (e.g. 'no flags found in this chunk').")
    model_config = {"extra": "allow"}


class PorkFinder(AgentBase):
    """Surfaces line items with earmark structural signatures.

    Critical: the right answer for many bill chunks is items: []. Class-of-recipient
    appropriations (e.g. '$10B for hazardous fuels reduction') are NOT pork —
    they're how Congress normally funds programs. Pork has structural signatures:
    a named entity, a geo-specific carve-out, sole-source language, or oddly
    precise amounts. Without one of those signatures the line item should be
    omitted, not flagged.
    """
    name = "pork_finder"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 6000
    output_schema = PorkFinderOutput

    def system_prompt(self) -> str:
        return (
            "You are a federal-spending forensic analyst. You scan US legislative bills "
            "for line items whose STRUCTURE matches the signature of an earmark or "
            "pork-barrel appropriation. You do not make political judgments. You flag "
            "patterns that warrant review.\n\n"
            "CRITICAL FRAMING: most line items in a bill are NOT pork. They are ordinary "
            "class-of-recipient appropriations: '$10B for hazardous fuels reduction' or "
            "'$3B for urban forestry programs.' These fund a category of work, not a "
            "specific entity, and should be OMITTED from your output. The correct answer "
            "for a typical agriculture/appropriations title is zero, one, or two flags. "
            "Large lists of flags are a sign you're over-classifying.\n\n"
            "PATTERN DEFINITIONS WITH EXAMPLES:\n\n"
            "1. named-recipient: bill names a specific entity, university, company, "
            "museum, or non-profit. ALWAYS requires a non-empty recipient field.\n"
            "   Example flag:   '$5,000,000 to the Lawrence Welk Museum in Strasburg, "
            "North Dakota, for renovation' -> recipient='Lawrence Welk Museum'\n"
            "   Example NON-flag: '$10B for hazardous fuels reduction projects' -> "
            "OMIT (no entity named; class of work)\n\n"
            "2. sole-source: language explicitly bypasses competitive bidding. "
            "Triggers: 'notwithstanding any other provision of law', 'shall be awarded "
            "to [X] without competition', 'sole-source contract'.\n"
            "   Example flag: 'Notwithstanding any other provision of law, the Secretary "
            "shall award $40,000,000 to [contractor]'\n"
            "   Example NON-flag: 'The Secretary shall award grants on a competitive basis' "
            "-> OMIT (normal procurement)\n\n"
            "3. geo-specific: dollar amount tied to one congressional district, city, or "
            "county.\n"
            "   Example flag: '$2,500,000 for the construction of a community center in "
            "the city of Buffalo, New York' -> pattern='geo-specific', recipient=null OK\n"
            "   Example NON-flag: '$2.5B in grants distributed to states by formula' -> "
            "OMIT (formula grant, not a carve-out)\n\n"
            "4. oddly-specific-amount: figures with unusual precision suggesting "
            "back-into-a-cost line items. Round numbers ($1B, $500M, $100M) are NORMAL "
            "and not pork. Suspicious: $4,237,500 / $12,873,200 / $87,500,000.\n"
            "   Example flag: '$87,237,500 for the National Center for [...]'\n"
            "   Example NON-flag: '$10,000,000,000 for vegetation management' -> OMIT\n\n"
            "5. no-comp-process: funds appropriated outside the normal grant or "
            "appropriation process for that domain.\n"
            "6. other: something looks off but doesn't fit above. Use sparingly. Always "
            "fill why_flagged.\n\n"
            "HARD RULES:\n"
            "- If recipient is null/empty, pattern CANNOT be 'named-recipient'.\n"
            "- If a line item is just 'class-of-X funding', OMIT it from items[].\n"
            "- Round-number amounts ($Xb, $YM with no odd cents) are NEVER "
            "'oddly-specific-amount'.\n"
            "- Empty items[] is the right answer when the chunk has no real flags. "
            "Set note to 'no pork-signature line items found in this chunk' in that case.\n\n"
            "Return a single JSON object matching the requested schema. Cap items at 15."
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
      "line_item": "Brief paraphrase of the bill text, <=30 words.",
      "amount_usd": 4237500,
      "recipient": "Named entity, OR null if pattern is not named-recipient",
      "pattern": "named-recipient",
      "confidence": "high",
      "bill_section": "SEC. 10103",
      "why_flagged": "Specific reason this matches the pattern (<=20 words)."
    }}
  ],
  "note": null
}}

Reminders:
- items[] should be 0-15 entries. Empty is fine and often correct.
- Do not include ordinary class-of-recipient appropriations.
- pattern='named-recipient' REQUIRES a non-empty recipient string. If you can't fill
  recipient with a real entity name from the bill, OMIT the item.
- amount_usd is a JSON number (no '$', no commas), or null.
- pattern must be one of: named-recipient, sole-source, geo-specific,
  oddly-specific-amount, no-comp-process, other. Do NOT use 'ordinary-class' in
  output -- that's the filter label, omit those items entirely.
- Return ONLY the JSON object. No commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""