"""
Smoke test: Plain-English Summarizer + USC Cross-Reference on BBB-2021 chunk 1.

Goal: verify the per-chunk agent pattern works end-to-end against the live
cloud endpoints. Uses chunk 1 (TITLE I - AGRICULTURE, 246K tokens).

This is the PROOF that the architecture works. If both agents return well-
formed structured output and the USC enrichment finds at least some valid
sections, the pattern is good and we can multiply across the other 12 agents.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add the repo to sys.path so we can import src.*
REPO_ROOT = Path(r"B:\amd-hackathon-bill-analyzer")
sys.path.insert(0, str(REPO_ROOT))

from src.agents.summarizer import PlainEnglishSummarizer
from src.agents.usc_xref import UscCrossReference, enrich_with_usc
from src.tools.fetch_usc import FetchUsc

CHUNKS_PATH = Path(r"B:\hackathon-build\chunks-bbb-full.json")
USC_LMDB = Path(r"B:\amd-hackathon-bill-analyzer\data\usc.lmdb")
OUT_DIR = Path(r"B:\hackathon-build\agent-smoke")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print(f"[smoke] Loading chunks from {CHUNKS_PATH} ...")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"[smoke]   {len(chunks)} chunks loaded")

    chunk = chunks[0]  # ch01 TITLE I - AGRICULTURE
    print(f"[smoke] Target: {chunk['chunk_id']} {chunk['marker']} "
          f"({chunk['tokens']:,} tokens, pp.{chunk['start_page']}-{chunk['end_page']})")

    # ----------------------------------------------------------------
    # Agent 1: Plain-English Summarizer (spine endpoint, long context)
    # ----------------------------------------------------------------
    print(f"\n[smoke] === Plain-English Summarizer (spine, long-context) ===")
    summarizer = PlainEnglishSummarizer()
    t0 = time.perf_counter()
    summary = summarizer.run(
        chunk_text=chunk["text"],
        chunk_id=chunk["chunk_id"],
        title_marker=chunk["marker_label"],
    )
    dt = time.perf_counter() - t0
    print(f"[smoke]   elapsed: {dt:.1f}s  ({summary.elapsed_ms:.0f}ms inside agent)")
    print(f"[smoke]   tokens:  prompt={summary.prompt_tokens:,}  completion={summary.completion_tokens:,}")
    if summary.errors:
        print(f"[smoke]   ERRORS: {summary.errors}")
        return 1
    print(f"[smoke]   --- output ---")
    print(json.dumps(summary.output, indent=2)[:2000])
    (OUT_DIR / "summarizer-ch01.json").write_text(
        json.dumps(summary.output, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------
    # Agent 2: USC Cross-Reference (spine endpoint, full chunk)
    #
    # USC now runs on spine (262K context). This means it gets the FULL chunk
    # AND benefits from APC: the summarizer just primed spine with this same
    # chunk text, so USC's prefill should be largely a cache hit. Watch the
    # second agent's TTFT — it should be dramatically faster than the first.
    # ----------------------------------------------------------------
    print(f"\n[smoke] === USC Cross-Reference (spine, full chunk, APC warm) ===")
    xref = UscCrossReference()
    t0 = time.perf_counter()
    crossref = xref.run(
        chunk_text=chunk["text"],
        chunk_id=chunk["chunk_id"],
        title_marker=chunk["marker_label"],
    )
    dt = time.perf_counter() - t0
    print(f"[smoke]   elapsed: {dt:.1f}s")
    print(f"[smoke]   APC-WARMUP COMPARISON: summarizer was {summary.elapsed_ms/1000:.1f}s, "
          f"USC xref was {dt:.1f}s")
    print(f"[smoke]   tokens:  prompt={crossref.prompt_tokens:,}  completion={crossref.completion_tokens:,}")
    if crossref.errors:
        print(f"[smoke]   ERRORS: {crossref.errors}")
        return 1
    citations = crossref.output.get("citations", [])
    print(f"[smoke]   citations identified: {len(citations)}")
    for c in citations[:5]:
        print(f"     - {c.get('citation','?'):<25} relevance={c.get('relevance','?')}")

    # ----------------------------------------------------------------
    # Pass 2: Enrich citations with the actual USC text from LMDB
    # ----------------------------------------------------------------
    print(f"\n[smoke] === USC Enrichment (LMDB lookups, no LLM) ===")
    fetcher = FetchUsc(USC_LMDB)
    t0 = time.perf_counter()
    enriched = enrich_with_usc(crossref.output, fetcher)
    dt = time.perf_counter() - t0
    print(f"[smoke]   elapsed: {dt*1000:.1f}ms for {len(citations)} lookups")
    print(f"[smoke]   stats: {fetcher.stats()}")
    fetcher.close()

    resolved = [c for c in enriched["citations"] if c.get("resolution_status") == "ok"]
    not_found = [c for c in enriched["citations"] if c.get("resolution_status") == "not_found"]
    print(f"[smoke]   resolved: {len(resolved)} / {len(citations)}  "
          f"(unresolved: {len(not_found)})")

    if resolved:
        print(f"\n[smoke]   --- first 3 enriched citations ---")
        for c in resolved[:3]:
            data = c["usc_data"]
            print(f"     {c['citation']:<25} -> {data['title']} USC {data['section']}")
            print(f"       {data['heading'][:80]}")
            print(f"       {data['text_excerpt'][:200].replace(chr(10), ' ')}")

    (OUT_DIR / "usc-xref-ch01.json").write_text(
        json.dumps(enriched, indent=2), encoding="utf-8")

    print(f"\n[smoke] === Pattern validated ===")
    print(f"[smoke] Outputs:")
    print(f"   {OUT_DIR / 'summarizer-ch01.json'}")
    print(f"   {OUT_DIR / 'usc-xref-ch01.json'}")

    # ---- Print the summary cleanly for the demo screenshot ----
    print(f"\n{'=' * 60}")
    print(f"BBB-2021 Chunk 1 (TITLE I - AGRICULTURE)")
    print(f"{'=' * 60}")
    out = summary.output
    print(f"\nIn one sentence: {out['one_sentence_summary']}")
    print(f"\nKey points:")
    for b in out["bullets"]:
        print(f"  - {b}")
    print(f"\nAffected groups:")
    for g in out.get("affected_groups", []):
        print(f"  - {g}")
    print(f"\n{'=' * 60}")
    print(f"USC Cross-References: {len(citations)} identified, {len(resolved)} resolved")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
