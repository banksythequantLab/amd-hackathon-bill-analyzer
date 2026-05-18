"""Smoke-test the USC LMDB directly via FetchUsc (no HTTP yet).

If this works, we know:
  - lmdb python library is installed and works on Windows
  - The 378 MB data.mdb is readable
  - Normalized lookup works
  - We have specimens to use as HTTP server test cases
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.tools.fetch_usc import FetchUsc

LMDB = REPO / "data" / "usc.lmdb"
print(f"LMDB: {LMDB}")
print(f"  data.mdb size: {(LMDB / 'data.mdb').stat().st_size / 1024 / 1024:.1f} MB")

fetcher = FetchUsc(str(LMDB))

# Specimens spanning common citation formats agents will produce.
test_citations = [
    "42 USC 1395dd",      # EMTALA. Used by app.py health check.
    "26:401(k)",           # Normalized form
    "26 U.S.C. § 401",    # Pedantic form with section sign
    "26 USC 401",          # Plain
    "42 U.S.C. 1395dd",   # USC with section number that has letters
    "5 USC 552a",          # Privacy Act — single section, letter suffix
    "1:1",                 # Title 1, Section 1 (boundary)
    "26:401",              # Plain title:section
    "99999 USC 99999",    # Definitely doesn't exist
    "totally garbage",     # Non-citation string
]

print()
print(f"{'CITATION':<28}  {'HIT?':<5}  {'HEADING':<60}")
print("-" * 100)
for c in test_citations:
    t0 = time.perf_counter()
    rec = fetcher(c)
    elapsed_us = (time.perf_counter() - t0) * 1_000_000
    if rec:
        heading = (rec.get("heading") or "(no heading)").replace("\n", " ")[:60]
        text_len = len(rec.get("text", ""))
        print(f"{c:<28}  HIT    {heading:<60}  ({elapsed_us:>6.0f} us, text={text_len:,}b)")
    else:
        print(f"{c:<28}  miss   --                                                            ({elapsed_us:>6.0f} us)")

print()
print(f"stats: {fetcher.stats()}")
fetcher.close()

# Bonus: count total keys (peek at the LMDB stats)
import lmdb
env = lmdb.open(str(LMDB), readonly=True, lock=False, subdir=True)
with env.begin() as txn:
    s = txn.stat()
print(f"LMDB entries: {s['entries']:,}")
env.close()
