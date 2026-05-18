"""Why didn't smart_chunker pick up Subtitle B—Health inside TITLE VII?"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402

HR1 = ROOT / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"

# Pull pages 87..219 (TITLE VII per the chunker output)
text_parts = []
with pdfplumber.open(HR1) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        if 87 <= i <= 219:
            t = page.extract_text() or ""
            text_parts.append(t + "\n\n")

title_vii = "".join(text_parts)

# Find every 'Subtitle' occurrence
for m in re.finditer(r"Subtitle", title_vii):
    idx = m.start()
    snip = title_vii[max(0, idx-3):idx+30]
    # Render as bytes to see the actual chars including em-dashes
    print(f"@{idx:>7}: {repr(snip)}")
print()

# Production regex from smart_chunker
prod_pat = re.compile(r"^\s*(Subtitle\s+([A-Z]|[0-9]+))(?:\b|$)", re.MULTILINE)
hits = list(prod_pat.finditer(title_vii))
print(f"Production regex Subtitle matches in TITLE VII: {len(hits)}")
for h in hits:
    print(f"  @{h.start()}: {h.group(1)}")
