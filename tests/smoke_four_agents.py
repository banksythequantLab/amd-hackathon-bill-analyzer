"""Smoke test for all 4 agents on a single chunk — captures APC reuse timing."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
USC_LMDB = Path(r"B:\amd-hackathon-bill-analyzer\data\usc.lmdb")

def load_chunk(bill, chunk_id):
    p = CHUNK_FILES[bill]
    if not p.exists():
        raise SystemExit(f"chunks file missing: {p}")
    chunks = json.loads(p.read_text(encoding="utf-8"))
    for c in chunks:
        if c["chunk_id"] == chunk_id:
            return c
    raise SystemExit(f"chunk_id {chunk_id} not found")

def run_agent(label, agent_cls, chunk_text, chunk_id, title_marker, out_dir, metrics):
    print(f"\n[smoke] === {label} ===", flush=True)
    agent = agent_cls()
    t0 = time.perf_counter()
    result = agent.run(chunk_text, chunk_id, title_marker=title_marker)
    elapsed = time.perf_counter() - t0
    print(f"[smoke]   elapsed: {elapsed:.1f}s", flush=True)
    print(f"[smoke]   tokens:  prompt={result.prompt_tokens:,}  completion={result.completion_tokens:,}", flush=True)
    if result.errors:
        print(f"[smoke]   ERRORS: {result.errors}", flush=True)
        (out_dir / f"{agent.name}-raw-{chunk_id}.txt").write_text(result.raw_response or "", encoding="utf-8")
        metrics[agent.name] = {"elapsed_s": round(elapsed,2), "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "errors": result.errors, "ok": False}
        return None
    out_file = out_dir / f"{agent.name}-{chunk_id}.json"
    out_file.write_text(json.dumps(result.output, indent=2), encoding="utf-8")
    print(f"[smoke]   wrote: {out_file}", flush=True)
    metrics[agent.name] = {"elapsed_s": round(elapsed,2), "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "ok": True}
    return result.output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bill", default="bbb", choices=list(CHUNK_FILES))
    ap.add_argument("--chunk-id", default="ch01")
    ap.add_argument("--out-dir", type=Path, default=Path(r"B:\hackathon-build\agent-smoke"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] Loading chunk {args.chunk_id} from {args.bill} ...", flush=True)
    chunk = load_chunk(args.bill, args.chunk_id)
    chunk_text = chunk["text"]
    title_marker = chunk["marker_label"]
    print(f"[smoke]   marker: {title_marker}", flush=True)
    print(f"[smoke]   tokens: {chunk['tokens']:,}", flush=True)
    print(f"[smoke]   chars:  {chunk['char_count']:,}", flush=True)

    metrics = {"bill": args.bill, "chunk_id": args.chunk_id, "title_marker": title_marker, "chunk_tokens_cl100k": chunk["tokens"], "chunk_pages": [chunk["start_page"], chunk["end_page"]]}

    summary = run_agent("Plain-English Summarizer", PlainEnglishSummarizer, chunk_text, args.chunk_id, title_marker, args.out_dir, metrics)
    crossref = run_agent("USC Cross-Reference", UscCrossReference, chunk_text, args.chunk_id, title_marker, args.out_dir, metrics)
    pork = run_agent("Pork Finder", PorkFinder, chunk_text, args.chunk_id, title_marker, args.out_dir, metrics)
    conflicts = run_agent("Conflict Spotter", ConflictSpotter, chunk_text, args.chunk_id, title_marker, args.out_dir, metrics)

    if crossref and USC_LMDB.exists():
        print(f"\n[smoke] === USC Enrichment ===", flush=True)
        fetcher = FetchUsc(USC_LMDB)
        t0 = time.perf_counter()
        enriched = enrich_with_usc(crossref, fetcher)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = fetcher.stats()
        fetcher.close()
        print(f"[smoke]   enrichment elapsed: {elapsed_ms:.1f}ms", flush=True)
        print(f"[smoke]   stats: {stats}", flush=True)
        out_file = args.out_dir / f"usc_cross_reference-enriched-{args.chunk_id}.json"
        out_file.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        metrics["usc_enrichment"] = {"elapsed_ms": round(elapsed_ms,2), "calls": stats["calls"], "hits": stats["hits"], "misses": stats["misses"], "hit_rate": round(stats["hits"]/max(1,stats["calls"]),3)}

    metrics_file = args.out_dir / f"metrics-{args.bill}-{args.chunk_id}.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[smoke] Metrics roll-up: {metrics_file}", flush=True)

    print(f"\n[smoke] === Per-agent timing roll-up ===", flush=True)
    print(f"  {'agent':<35s} {'elapsed_s':>10s} {'prompt_tok':>12s} {'completion':>12s}", flush=True)
    for name in ["plain_english_summarizer","usc_cross_reference","pork_finder","conflict_spotter"]:
        m = metrics.get(name)
        if not m: continue
        flag = "OK" if m.get("ok") else "FAIL"
        print(f"  [{flag}] {name:<33s} {m.get('elapsed_s','?'):>9}s {m.get('prompt_tokens',0):>12,} {m.get('completion_tokens',0):>12,}", flush=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())