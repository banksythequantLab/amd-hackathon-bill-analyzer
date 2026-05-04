"""Run a single agent against a chunk. One python process per agent."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AGENT_MAP = {
    "summarizer":  ("src.agents.summarizer",       "PlainEnglishSummarizer"),
    "xref":        ("src.agents.usc_xref",         "UscCrossReference"),
    "pork":        ("src.agents.pork_finder",      "PorkFinder"),
    "conflict":    ("src.agents.conflict_spotter", "ConflictSpotter"),
}

CHUNK_FILES = {
    "bbb":  Path(r"B:\hackathon-build\chunks-bbb-full.json"),
    "hr1":  Path(r"B:\hackathon-build\chunks-hr1-full.json"),
    "ndaa": Path(r"B:\hackathon-build\chunks-ndaa-full.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(AGENT_MAP))
    ap.add_argument("--bill", default="bbb", choices=list(CHUNK_FILES))
    ap.add_argument("--chunk-id", default="ch01")
    ap.add_argument("--out-dir", type=Path, default=Path(r"B:\hackathon-build\agent-smoke"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mod_name, cls_name = AGENT_MAP[args.agent]
    import importlib
    mod = importlib.import_module(mod_name)
    AgentClass = getattr(mod, cls_name)

    # Load chunk
    chunks = json.loads(CHUNK_FILES[args.bill].read_text(encoding="utf-8"))
    chunk = next((c for c in chunks if c["chunk_id"] == args.chunk_id), None)
    if not chunk:
        print(f"[run] chunk_id {args.chunk_id} not found", flush=True)
        return 1

    chunk_text = chunk["text"]
    title_marker = chunk["marker_label"]
    print(f"[run] agent={args.agent} bill={args.bill} chunk={args.chunk_id} tokens={chunk['tokens']:,}", flush=True)

    agent = AgentClass()
    t0 = time.perf_counter()
    result = agent.run(chunk_text, args.chunk_id, title_marker=title_marker)
    elapsed = time.perf_counter() - t0
    print(f"[run] elapsed={elapsed:.1f}s prompt_toks={result.prompt_tokens:,} completion_toks={result.completion_tokens:,}", flush=True)

    if result.errors:
        print(f"[run] ERRORS: {result.errors}", flush=True)
        (args.out_dir / f"{agent.name}-raw-{args.chunk_id}.txt").write_text(result.raw_response or "", encoding="utf-8")

    out_file = args.out_dir / f"{agent.name}-{args.chunk_id}.json"
    out_file.write_text(json.dumps(result.output if result.output else {"errors": result.errors}, indent=2), encoding="utf-8")

    metric = {
        "agent": agent.name,
        "bill": args.bill,
        "chunk_id": args.chunk_id,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "ok": not bool(result.errors),
    }
    (args.out_dir / f"{agent.name}-metric-{args.chunk_id}.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
    print(f"[run] wrote {out_file.name} + metric. ok={metric['ok']}", flush=True)
    return 0 if metric["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())