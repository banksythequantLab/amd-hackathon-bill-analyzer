"""End-to-end: USC xref agent + HTTP-server-backed enrichment on HR1 ch01.

Proves:
  1. xref agent works on the 3090 fork's spine (Instruct-2507 @ 64K ctx)
  2. enrich_with_usc + HttpFetchUsc + our :8004 server hydrate citations
  3. Citations get resolution_status='ok' with real heading + text payloads

Output: eval/hr1-ch01-xref-via-http-server.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tiktoken  # noqa: E402

from src.agents.usc_xref import UscCrossReference, enrich_with_usc  # noqa: E402
from src.tools.http_fetch_usc import HttpFetchUsc  # noqa: E402
from src.chunking.smart_chunker import (  # noqa: E402
    extract_text_with_pages,
    find_boundaries,
    pack_chunks,
)


HR1 = REPO / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"
OUT = REPO / "eval" / "hr1-ch01-xref-via-http-server.json"
USC_HTTP_URL = "http://127.0.0.1:8004"


def main() -> int:
    # Re-derive ch01 deterministically with the same chunker run.
    print("[1/3] Re-chunking HR1 at max_tokens=50500 ...")
    text, page_starts = extract_text_with_pages(HR1)
    enc = tiktoken.get_encoding("cl100k_base")
    boundaries = find_boundaries(text, page_starts)
    chunks = pack_chunks(text, boundaries, page_starts, 50_500, enc)
    ch01 = chunks[0]
    print(f"   ch01: {ch01.marker}  ({ch01.tokens:,} cl100k tokens)")

    # Run xref agent.
    print()
    print(f"[2/3] Running xref agent on ch01 via spine ...")
    agent = UscCrossReference()
    print(f"   endpoint={agent.target_endpoint}  model={agent.target_model}")
    t0 = time.perf_counter()
    result = agent.run(
        chunk_text=ch01.text,
        chunk_id="ch01",
        title_marker=ch01.marker_label,
        max_retries=1,
    )
    elapsed = time.perf_counter() - t0
    print(f"   elapsed={elapsed:.1f}s  prompt={result.prompt_tokens:,}  "
          f"completion={result.completion_tokens:,}")
    if result.errors:
        print(f"   [ERR] {result.errors}")
        print(f"   raw (first 800): {result.raw_response[:800]}")
        return 1
    citations_pre = result.output.get("citations", []) if isinstance(result.output, dict) else []
    print(f"   citations extracted: {len(citations_pre)}")

    # Enrich via HTTP server.
    print()
    print(f"[3/3] Enriching {len(citations_pre)} citations via HTTP at {USC_HTTP_URL} ...")
    fetcher = HttpFetchUsc(USC_HTTP_URL)
    t1 = time.perf_counter()
    enriched = enrich_with_usc(result.output, fetcher)
    enrich_elapsed = time.perf_counter() - t1
    ok_count = sum(1 for c in enriched.get("citations", []) if c.get("resolution_status") == "ok")
    print(f"   enrich_with_usc: {ok_count}/{len(citations_pre)} resolved in {enrich_elapsed*1000:.0f}ms")
    print(f"   HttpFetchUsc stats: {fetcher.stats()}")
    fetcher.close()

    # Print a few examples.
    print()
    print(f"Sample (up to 5) resolved citations:")
    for i, c in enumerate(enriched.get("citations", [])[:5], 1):
        status = c.get("resolution_status", "?")
        cite = c.get("citation_text") or c.get("usc_citation") or c.get("citation") or "?"
        heading = (c.get("usc_heading") or c.get("heading") or "")[:55].replace("\n", " ")
        print(f"   {i}. [{status:<5}] {cite:<28}  {heading}")

    # Save full report.
    report = {
        "test": "hr1_ch01_xref_via_http_server",
        "spine": agent.target_endpoint,
        "usc_http_url": USC_HTTP_URL,
        "chunk": {
            "chunk_id": "ch01",
            "marker": ch01.marker,
            "tokens_cl100k": ch01.tokens,
        },
        "xref_elapsed_s": round(elapsed, 2),
        "xref_prompt_tokens": result.prompt_tokens,
        "xref_completion_tokens": result.completion_tokens,
        "n_citations_extracted": len(citations_pre),
        "n_citations_resolved": ok_count,
        "enrich_elapsed_ms": round(enrich_elapsed * 1000, 1),
        "fetcher_stats": fetcher.stats(),
        "output": enriched,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ARTIFACT] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
