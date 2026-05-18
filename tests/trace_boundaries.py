"""Trace why TITLE VII didn't split at Subtitle B-Health.

Run find_boundaries on the full HR1 PDF and check the dedupe logic.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking.smart_chunker import (  # noqa: E402
    extract_text_with_pages,
    find_boundaries,
    BOUNDARY_PATTERNS,
)

HR1 = ROOT / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"

text, page_starts = extract_text_with_pages(HR1)
print(f"Full text: {len(text):,} chars")

# Run each pattern in isolation first to see what's there
print("\n--- RAW pattern hits across full bill ---")
import re
for label, regex in BOUNDARY_PATTERNS:
    hits = list(regex.finditer(text))
    print(f"  {label:10s}: {len(hits)} hits")

# Now the deduped boundaries
boundaries = find_boundaries(text, page_starts)
print(f"\n--- deduped boundaries: {len(boundaries)} ---")
# Filter to those between char 200_000 and 600_000 (the TITLE VII region)
title_vii_region = [b for b in boundaries if 200_000 <= b.char_offset <= 700_000]
print(f"In approx-TITLE-VII region: {len(title_vii_region)}")
for b in title_vii_region:
    print(f"  @{b.char_offset:>7} p.{b.page:>4} {b.label:10s} {b.marker:20s} | {b.full_line[:80]}")

print("\n--- Bill spans char-offset breakdown ---")
for b in boundaries:
    if b.label in ("DIVISION", "TITLE"):
        print(f"  @{b.char_offset:>7} p.{b.page:>4} {b.label:10s} {b.marker:20s} | {b.full_line[:70]}")
