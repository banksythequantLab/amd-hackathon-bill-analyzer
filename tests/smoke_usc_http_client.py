"""End-to-end smoke test of the HttpFetchUsc client against the local server.

Validates that app.py's USC enrichment path -- which uses HttpFetchUsc when
USC_LMDB_HTTP is set -- now works on Johnson.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.tools.http_fetch_usc import HttpFetchUsc, get_fetcher

print("=== Test 1: HttpFetchUsc direct against local server")
fetcher = HttpFetchUsc("http://127.0.0.1:8004")

cases = [
    ("42 USC 1395dd",   True),
    ("26 USC 401",       True),
    ("5 USC 552a",       True),
    ("1:1",              True),
    ("99999 USC 99999", False),
    ("totally garbage",  False),
]
for cite, expect_hit in cases:
    rec = fetcher(cite)
    got_hit = rec is not None
    status = "OK " if got_hit == expect_hit else "FAIL"
    heading = (rec.get("heading") if rec else "(no record)")[:55].replace("\n", " ")
    print(f"  [{status}] {cite:<22} expect_hit={expect_hit}  got={got_hit}  {heading}")
print(f"  stats: {fetcher.stats()}")
fetcher.close()

print()
print("=== Test 2: get_fetcher() auto-selects HTTP when http_url is set")
auto = get_fetcher(http_url="http://127.0.0.1:8004")
print(f"  picked: {type(auto).__name__}")
rec = auto("42 USC 1395dd")
print(f"  EMTALA lookup: hit={rec is not None}, heading={(rec.get('heading') if rec else None)!r}")
auto.close()

print()
print("=== Test 3: get_fetcher() auto-selects local when only local_path given")
local = get_fetcher(local_path=str(REPO / "data" / "usc.lmdb"))
print(f"  picked: {type(local).__name__}")
rec = local("42 USC 1395dd")
print(f"  EMTALA lookup: hit={rec is not None}, heading={(rec.get('heading') if rec else None)!r}")
local.close()

print()
print("=== Test 4: get_fetcher() with neither => None (graceful degrade)")
none = get_fetcher()
print(f"  picked: {none!r}")
print()
print("All tests done.")
