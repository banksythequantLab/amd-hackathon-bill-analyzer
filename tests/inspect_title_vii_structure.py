"""Inspect HR1 TITLE VII for sub-structure that the chunker is missing.

Goal: confirm whether TITLE VII has SEC./Sec./Subchapter/PART/Chapter
markers we can split on, so we can teach smart_chunker about them
without changing the existing TITLE/Subtitle behavior.
"""
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
print(f"TITLE VII text length: {len(title_vii):,} chars")
print()

patterns = [
    ("DIVISION",        re.compile(r"^\s*DIVISION\s+([A-Z]|[IVXLCDM]+)\b", re.MULTILINE)),
    ("TITLE",           re.compile(r"^\s*TITLE\s+[IVXLCDM]+\b", re.MULTILINE | re.IGNORECASE)),
    ("Subtitle",        re.compile(r"^\s*Subtitle\s+[A-Z]\b", re.MULTILINE)),
    ("CHAPTER (UPPER)", re.compile(r"^\s*CHAPTER\s+\d+\b", re.MULTILINE)),
    ("Chapter",         re.compile(r"^\s*Chapter\s+\d+\b", re.MULTILINE)),
    ("PART (UPPER)",    re.compile(r"^\s*PART\s+[IVXLCDM]+\b", re.MULTILINE)),
    ("Part",            re.compile(r"^\s*Part\s+[IVXLCDM]+\b", re.MULTILINE)),
    ("SUBCHAPTER",      re.compile(r"^\s*Subchapter\s+[A-Z]\b", re.MULTILINE | re.IGNORECASE)),
    ("SEC. (top)",      re.compile(r"^SEC\.\s+\d{5}\b", re.MULTILINE)),
    ("Sec. (top)",      re.compile(r"^Sec\.\s+\d{5}\b", re.MULTILINE)),
    ("SEC. all",        re.compile(r"^\s*SEC\.\s+\d+\b", re.MULTILINE)),
    ("Sec. all",        re.compile(r"^\s*Sec\.\s+\d+\b", re.MULTILINE)),
]

for name, pat in patterns:
    hits = list(pat.finditer(title_vii))
    print(f"{name:20s}: {len(hits):>4} hits")
    if hits and len(hits) <= 8:
        for h in hits[:8]:
            # snip line
            line_end = title_vii.find("\n", h.start())
            line = title_vii[h.start():line_end if line_end > 0 else h.end() + 80].strip()
            print(f"   @char {h.start():>7}: {line[:90]}")
    elif hits:
        for h in hits[:3]:
            line_end = title_vii.find("\n", h.start())
            line = title_vii[h.start():line_end if line_end > 0 else h.end() + 80].strip()
            print(f"   @char {h.start():>7}: {line[:90]}")
        print(f"   ...")
        for h in hits[-2:]:
            line_end = title_vii.find("\n", h.start())
            line = title_vii[h.start():line_end if line_end > 0 else h.end() + 80].strip()
            print(f"   @char {h.start():>7}: {line[:90]}")
