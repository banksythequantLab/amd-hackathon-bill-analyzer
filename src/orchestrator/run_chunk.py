"""
Orchestrator (lightweight, sequential): runs N agents against a single chunk
and emits a combined report.

Why sequential and not parallel-async:
  * vLLM serves one request at a time per spine endpoint (the "Running: 1
    reqs" we see in the logs). True parallel agent calls would just queue
    behind each other on the spine engine, gaining nothing in wall-clock
    while complicating the code path.
  * APC delivers the speedup we want: agent N+1 hits a warm prefix cache
    from agent N because they share the bill chunk. Sequential is what
    actually exercises that.

The orchestrator's job is therefore:
  1. Load the chunk
  2. Run each agent in turn, collecting timings + APC stats
  3. Run enrich_with_usc on the xref output (local, free)
  4. Persist a combined report.json with all four agent outputs +
     timing summary + (eventually) cross-agent reconciliation hints

Usage:
    python -m src.orchestrator.run_chunk --bill bbb --chunk-id ch01

Outputs:
    B:\\hackathon-build\\agent-smoke\\report-{bill}-{chunk_id}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on stdout/stderr so em-dashes and other non-ASCII characters
# from the bill chunk's title_marker don't crash print() under Windows
# default cp1252 encoding (silent process death without a useful traceback).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.summarizer       import PlainEnglishSummarizer
from src.agents.usc_xref         import UscCrossReference, enrich_with_usc
from src.agents.pork_finder      import PorkFinder
from src.agents.conflict_spotter import ConflictSpotter
from src.tools.fetch_usc         import FetchUsc


CHUNK_FILES = {
    "bbb":  Path(r"B:\hackathon-build\chunks-bbb-full.json"),
    "hr1":  Path(r"B:\hackathon-build\chunks-hr1-full.json"),
    "ndaa": Path(r"B:\hackathon-build\chunks-ndaa-full.json"),
}
# 3090 FORK: USC_LMDB derived from this file's location (was hardcoded
# to old fork). parents[2] = .../src/orchestrator/run_chunk.py -> repo root.
USC_LMDB = Path(__file__).resolve().parents[2] / "data" / "usc.lmdb"
DEFAULT_OUT = Path(r"B:\hackathon-build\agent-smoke")

AGENTS = [
    ("summarizer",       PlainEnglishSummarizer),
    ("usc_cross_ref",    UscCrossReference),
    ("pork_finder",      PorkFinder),
    ("conflict_spotter", ConflictSpotter),
]


def load_chunk(bill: str, chunk_id: str) -> dict:
    chunks = json.loads(CHUNK_FILES[bill].read_text(encoding="utf-8"))
    for c in chunks:
        if c["chunk_id"] == chunk_id:
            return c
    raise SystemExit(f"chunk_id {chunk_id} not found in {CHUNK_FILES[bill]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bill", default="bbb", choices=list(CHUNK_FILES))
    ap.add_argument("--chunk-id", default="ch01")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="agent names to skip (e.g. --skip pork_finder)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunk = load_chunk(args.bill, args.chunk_id)
    chunk_text = chunk["text"]
    title_marker = chunk["marker_label"]

    print(f"[orch] bill={args.bill} chunk={args.chunk_id} "
          f"tokens={chunk['tokens']:,} pages={chunk['start_page']}-{chunk['end_page']}",
          flush=True)
    print(f"[orch] marker: {title_marker}", flush=True)

    report = {
        "bill": args.bill,
        "chunk_id": args.chunk_id,
        "title_marker": title_marker,
        "chunk_tokens_cl100k": chunk["tokens"],
        "chunk_pages": [chunk["start_page"], chunk["end_page"]],
        "agents": {},
        "timings_s": {},
        "totals": {},
    }

    grand_t0 = time.perf_counter()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    failed = []

    for name, AgentClass in AGENTS:
        if name in args.skip:
            print(f"[orch] skip {name}", flush=True)
            continue
        print(f"\n[orch] === {name} ===", flush=True)
        agent = AgentClass()
        t0 = time.perf_counter()
        result = agent.run(chunk_text, args.chunk_id, title_marker=title_marker)
        elapsed = time.perf_counter() - t0
        print(f"[orch]   elapsed={elapsed:.1f}s "
              f"prompt_toks={result.prompt_tokens:,} "
              f"completion_toks={result.completion_tokens:,}", flush=True)
        if result.errors:
            print(f"[orch]   ERRORS: {result.errors}", flush=True)
            failed.append(name)

        report["agents"][name] = {
            "output": result.output,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "errors": result.errors,
        }
        report["timings_s"][name] = round(elapsed, 2)
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens

    # Pass 2 of USC Cross-Reference: enrich identified citations via LMDB
    if "usc_cross_ref" in report["agents"] and USC_LMDB.exists():
        xref_out = report["agents"]["usc_cross_ref"]["output"]
        if xref_out and isinstance(xref_out, dict) and "citations" in xref_out:
            print(f"\n[orch] === enrich_with_usc (Pass 2: LMDB lookups) ===", flush=True)
            fetcher = FetchUsc(USC_LMDB)
            t0 = time.perf_counter()
            enrich_with_usc(xref_out, fetcher)
            enrich_elapsed = (time.perf_counter() - t0) * 1000
            stats = fetcher.stats()
            fetcher.close()
            print(f"[orch]   enrichment elapsed={enrich_elapsed:.1f}ms stats={stats}",
                  flush=True)
            report["agents"]["usc_cross_ref"]["enrichment"] = {
                "elapsed_ms": round(enrich_elapsed, 2),
                "lmdb_stats": stats,
            }

    grand_elapsed = time.perf_counter() - grand_t0
    report["totals"] = {
        "wall_clock_s": round(grand_elapsed, 2),
        "prompt_tokens_total": total_prompt_tokens,
        "completion_tokens_total": total_completion_tokens,
        "agents_run": len(report["agents"]),
        "agents_failed": failed,
    }

    print(f"\n[orch] ====== SUMMARY ======", flush=True)
    print(f"[orch]   total wall-clock: {grand_elapsed:.1f}s", flush=True)
    print(f"[orch]   total prompt tokens: {total_prompt_tokens:,}", flush=True)
    print(f"[orch]   total completion tokens: {total_completion_tokens:,}", flush=True)
    print(f"[orch]   agents run: {len(report['agents'])}", flush=True)
    if failed:
        print(f"[orch]   agents failed: {failed}", flush=True)

    out_file = args.out_dir / f"report-{args.bill}-{args.chunk_id}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[orch] wrote {out_file}", flush=True)

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
