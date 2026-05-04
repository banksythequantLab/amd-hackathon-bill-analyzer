"""
End-to-end smoke for the first two agents.

Reads chunk ch01 from BBB-2021's chunked output, runs:
  1. Plain-English Summarizer (spine, no tools)
  2. USC Cross-Reference Pass 1 (reasoner, no tools)
  3. enrich_with_usc Pass 2 (local LMDB lookup, no LLM)

Saves results to B:\hackathon-build\agent-smoke\.

Usage:
    python tests/smoke_first_two_agents.py [--chunk-id ch01] [--bill bbb]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.summarizer import PlainEnglishSummarizer
from src.agents.usc_xref import UscCrossReference, enrich_with_usc
from src.tools.fetch_usc import FetchUsc


CHUNK_FILES = {
    "bbb":  Path(r"B:\hackathon-build\chunks-bbb.json"),
    "hr1":  Path(r"B:\hackathon-build\chunks-hr1.json"),
    "ndaa": Path(r"B:\hackathon-build\chunks-ndaa.json"),
}

# We chunked with --summary-only so chunks-bbb.json has text_preview not text.
# For real testing we need the full text. Re-chunk without --summary-only here.
FULL_CHUNK_FILES = {
    "bbb":  Path(r"B:\hackathon-build\chunks-bbb-full.json"),
    "hr1":  Path(r"B:\hackathon-build\chunks-hr1-full.json"),
    "ndaa": Path(r"B:\hackathon-build\chunks-ndaa-full.json"),
}

USC_LMDB = Path(r"B:\amd-hackathon-bill-analyzer\data\usc.lmdb")


def load_chunk(bill: str, chunk_id: str) -> dict:
    p = FULL_CHUNK_FILES[bill]
    if not p.exists():
        # Fallback: re-run chunker without --summary-only
        print(f"[smoke] full chunks file missing — re-running chunker for {bill}")
        from src.chunking.smart_chunker import (
            extract_text_with_pages, find_boundaries, pack_chunks, count_tokens
        )
        import tiktoken
        bill_pdf = {
            "bbb":  Path(r"B:\amd-hackathon-bill-analyzer\tests\fixtures\build_back_better_2021_hr5376.pdf"),
            "hr1":  Path(r"B:\amd-hackathon-bill-analyzer\tests\fixtures\one_big_beautiful_bill_2025_hr1.pdf"),
            "ndaa": Path(r"B:\amd-hackathon-bill-analyzer\tests\fixtures\fy24_ndaa_hr2670.pdf"),
        }[bill]
        text, page_starts = extract_text_with_pages(bill_pdf)
        enc = tiktoken.get_encoding("cl100k_base")
        boundaries = find_boundaries(text, page_starts)
        chunks = pack_chunks(text, boundaries, page_starts, 250_000, enc)
        from dataclasses import asdict
        data = [asdict(c) for c in chunks]
        p.write_text(json.dumps(data), encoding="utf-8")
        print(f"[smoke] wrote {p}")

    chunks = json.loads(p.read_text(encoding="utf-8"))
    for c in chunks:
        if c["chunk_id"] == chunk_id:
            return c
    raise SystemExit(f"chunk_id {chunk_id} not found in {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bill", default="bbb", choices=list(CHUNK_FILES))
    ap.add_argument("--chunk-id", default="ch01")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(r"B:\hackathon-build\agent-smoke"))
    ap.add_argument("--skip-summarizer", action="store_true")
    ap.add_argument("--skip-xref", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] Loading chunk {args.chunk_id} from {args.bill} ...")
    chunk = load_chunk(args.bill, args.chunk_id)
    chunk_text = chunk["text"]
    title_marker = chunk["marker_label"]
    print(f"[smoke]   marker: {title_marker}")
    print(f"[smoke]   tokens: {chunk['tokens']:,}")
    print(f"[smoke]   chars:  {chunk['char_count']:,}")
    print(f"[smoke]   pages:  {chunk['start_page']}-{chunk['end_page']}")

    # ----- Agent 1: Plain-English Summarizer -----
    if not args.skip_summarizer:
        print(f"\n[smoke] === Plain-English Summarizer ===")
        agent = PlainEnglishSummarizer()
        t0 = time.perf_counter()
        result = agent.run(chunk_text, args.chunk_id, title_marker=title_marker)
        elapsed = time.perf_counter() - t0
        print(f"[smoke]   elapsed: {elapsed:.1f}s")
        print(f"[smoke]   prompt tokens:     {result.prompt_tokens:,}")
        print(f"[smoke]   completion tokens: {result.completion_tokens:,}")
        if result.errors:
            print(f"[smoke]   ERRORS: {result.errors}")
            (args.out_dir / f"summarizer-raw-{args.bill}-{args.chunk_id}.txt").write_text(
                result.raw_response, encoding="utf-8")
            print(f"[smoke]   Raw response saved for inspection.")
            return 2
        out_file = args.out_dir / f"summary-{args.bill}-{args.chunk_id}.json"
        out_file.write_text(json.dumps(result.output, indent=2), encoding="utf-8")
        print(f"[smoke]   wrote: {out_file}")
        print(f"\n[smoke] One-sentence summary:")
        print(f"   {result.output.get('one_sentence_summary')}")
        print(f"\n[smoke] Bullets:")
        for i, b in enumerate(result.output.get("bullets", []), 1):
            print(f"   {i}. {b}")
        print(f"\n[smoke] Affected groups: {result.output.get('affected_groups')}")

    # ----- Agent 2: USC Cross-Reference -----
    if not args.skip_xref:
        print(f"\n[smoke] === USC Cross-Reference (Pass 1: identify) ===")
        agent = UscCrossReference()
        t0 = time.perf_counter()
        result = agent.run(chunk_text, args.chunk_id, title_marker=title_marker)
        elapsed = time.perf_counter() - t0
        print(f"[smoke]   elapsed: {elapsed:.1f}s")
        print(f"[smoke]   prompt tokens:     {result.prompt_tokens:,}")
        print(f"[smoke]   completion tokens: {result.completion_tokens:,}")
        if result.errors:
            print(f"[smoke]   ERRORS: {result.errors}")
            (args.out_dir / "xref-raw.txt").write_text(result.raw_response, encoding="utf-8")
            return 3
        crossref = result.output
        print(f"[smoke]   citations identified: {len(crossref.get('citations', []))}")

        print(f"\n[smoke] === USC Cross-Reference (Pass 2: enrich via fetch_usc) ===")
        if not USC_LMDB.exists():
            print(f"[smoke]   USC LMDB not at {USC_LMDB} — skipping enrichment (Pass 1 still saved)")
            (args.out_dir / f"xref-{args.bill}-{args.chunk_id}.json").write_text(
                json.dumps(crossref, indent=2), encoding="utf-8")
            return 0
        fetcher = FetchUsc(USC_LMDB)
        t0 = time.perf_counter()
        enriched = enrich_with_usc(crossref, fetcher)
        elapsed = time.perf_counter() - t0
        stats = fetcher.stats()
        fetcher.close()
        print(f"[smoke]   enrichment elapsed: {elapsed*1000:.1f}ms")
        print(f"[smoke]   fetch_usc stats: {stats}")

        out_file = args.out_dir / f"xref-{args.bill}-{args.chunk_id}.json"
        out_file.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        print(f"[smoke]   wrote: {out_file}")

        # Print a few representative citations
        print(f"\n[smoke] First 5 citations:")
        for c in enriched.get("citations", [])[:5]:
            status = c.get("resolution_status")
            usc = c.get("usc_data")
            heading = usc.get("heading", "(no heading)") if usc else "—"
            print(f"   {c['citation']:<30s} relevance={c.get('relevance'):<15s} {status:<10s} {heading[:50]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
