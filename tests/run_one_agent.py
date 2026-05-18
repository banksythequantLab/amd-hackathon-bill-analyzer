"""Run a single agent against a chunk. One python process per agent.

Local (Vesper) defaults: chunks under B:\hackathon-build\, out under same.
Cloud-side defaults (Day 3): when invoked with the env vars below, runs
against /root/bills/, /root/agent-smoke/, and the localhost endpoints.

Env var overrides (optional):
  BILL_ANALYZER_CHUNKS_DIR    - directory containing chunks-{bill}-full.json
  BILL_ANALYZER_OUT_DIR       - directory to write agent outputs
  BILL_ANALYZER_USC_LMDB      - path to the USC LMDB
  BILL_ANALYZER_SPINE_URL     - spine endpoint URL (default falls through to base.py: http://127.0.0.1:11434/v1 on Johnson; was http://165.245.134.1:8001/v1 on AMD)
  BILL_ANALYZER_REASONER_URL  - reasoner endpoint URL (same Ollama instance, alias 'reasoner')
  BILL_ANALYZER_VISION_URL    - vision endpoint URL (TBD per TODO #4: llava:7b or qwen2.5-vl:7b)

Local example (Windows):
  python tests/run_one_agent.py --agent summarizer --bill bbb --chunk-id ch01

3090 fork example (Johnson, hitting local Ollama):
  python tests/run_one_agent.py --agent summarizer --bill bbb --chunk-id ch01

Cloud example (Linux on AMD instance, hitting localhost):
  BILL_ANALYZER_CHUNKS_DIR=/root/bills \
  BILL_ANALYZER_OUT_DIR=/root/agent-smoke \
  BILL_ANALYZER_USC_LMDB=/root/usc/usc.lmdb \
  BILL_ANALYZER_SPINE_URL=http://localhost:8001/v1 \
  BILL_ANALYZER_REASONER_URL=http://localhost:8003/v1 \
  /root/repo/.venv/bin/python -u tests/run_one_agent.py --agent summarizer --bill bbb --chunk-id ch01
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Endpoint overrides MUST be applied before any agent module is imported,
# because base.py reads SPINE_ENDPOINT etc. at module-import time and the
# agent classes capture target_endpoint as class attributes from those.
def _apply_endpoint_env_overrides() -> None:
    """Patch src.agents.base endpoint constants from env vars, if set."""
    import importlib
    base = importlib.import_module("src.agents.base")
    spine = os.environ.get("BILL_ANALYZER_SPINE_URL")
    reas  = os.environ.get("BILL_ANALYZER_REASONER_URL")
    vis   = os.environ.get("BILL_ANALYZER_VISION_URL")
    if spine:
        base.SPINE_ENDPOINT = spine
    if reas:
        base.REASONER_ENDPOINT = reas
    if vis:
        base.VISION_ENDPOINT = vis

_apply_endpoint_env_overrides()

AGENT_MAP = {
    "summarizer":  ("src.agents.summarizer",       "PlainEnglishSummarizer"),
    "xref":        ("src.agents.usc_xref",         "UscCrossReference"),
    "pork":        ("src.agents.pork_finder",      "PorkFinder"),
    "conflict":    ("src.agents.conflict_spotter", "ConflictSpotter"),
    "fiscal":      ("src.agents.fiscal_impact_estimator", "FiscalImpactEstimator"),
    "stakeholder": ("src.agents.stakeholder_tracer", "StakeholderTracer"),
    "podcast":     ("src.agents.podcast_generator", "PodcastGenerator"),
    "relay":       ("src.agents.prompt_relay_author", "PromptRelayAuthor"),
}

DEFAULT_CHUNKS_DIR = Path(os.environ.get("BILL_ANALYZER_CHUNKS_DIR", r"B:\hackathon-build"))
DEFAULT_OUT_DIR    = Path(os.environ.get("BILL_ANALYZER_OUT_DIR",    r"B:\hackathon-build\agent-smoke"))

CHUNK_FILE_NAMES = {
    "bbb":  "chunks-bbb-full.json",
    "hr1":  "chunks-hr1-full.json",
    "ndaa": "chunks-ndaa-full.json",
}


def chunks_path(bill: str) -> Path:
    return DEFAULT_CHUNKS_DIR / CHUNK_FILE_NAMES[bill]


def re_target_agent_to_endpoint(AgentClass) -> None:
    """If env vars set new endpoints, agent class attributes need re-targeting too.
    They were captured at class-definition time before our env override could land.
    """
    import importlib
    base = importlib.import_module("src.agents.base")
    # Each agent picks its endpoint by name. Update the captured class attr.
    name_to_const = {
        "spine":    base.SPINE_ENDPOINT,
        "reasoner": base.REASONER_ENDPOINT,
        "vision":   base.VISION_ENDPOINT,
    }
    if AgentClass.target_model in name_to_const:
        AgentClass.target_endpoint = name_to_const[AgentClass.target_model]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(AGENT_MAP))
    ap.add_argument("--bill", default="bbb", choices=list(CHUNK_FILE_NAMES))
    ap.add_argument("--chunk-id", default="ch01")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--report-file", type=Path, default=None,
                    help="For podcast agent: path to a report JSON to use as input instead of a bill chunk.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mod_name, cls_name = AGENT_MAP[args.agent]
    import importlib
    mod = importlib.import_module(mod_name)
    AgentClass = getattr(mod, cls_name)
    re_target_agent_to_endpoint(AgentClass)

    # Podcast agent uses a report JSON as input, not a bill chunk.
    if args.agent in ("podcast", "relay"):
        if args.report_file is None:
            print("[run] ERROR: --report-file required for the podcast agent", flush=True)
            return 1
        if not args.report_file.exists():
            print(f"[run] ERROR: report file not found: {args.report_file}", flush=True)
            return 1
        report_text = args.report_file.read_text(encoding="utf-8")
        chunk_text = report_text
        title_marker = f"report:{args.report_file.name}"
        print(f"[run] agent={args.agent} bill={args.bill} chunk={args.chunk_id} report_chars={len(report_text):,}", flush=True)
        print(f"[run] endpoint={AgentClass.target_endpoint}", flush=True)
        agent = AgentClass()
        t0 = time.perf_counter()
        result = agent.run(chunk_text, args.chunk_id, title_marker=title_marker, bill=args.bill)
        elapsed = time.perf_counter() - t0
        print(f"[run] elapsed={elapsed:.1f}s prompt_toks={result.prompt_tokens:,} completion_toks={result.completion_tokens:,}", flush=True)
        if result.errors:
            print(f"[run] ERRORS: {result.errors}", flush=True)
            (args.out_dir / f"{agent.name}-raw-{args.chunk_id}.txt").write_text(result.raw_response or "", encoding="utf-8")
        out_file = args.out_dir / f"{agent.name}-{args.chunk_id}.json"
        out_file.write_text(json.dumps(result.output if result.output else {"errors": result.errors}, indent=2), encoding="utf-8")
        metric = {
            "agent": agent.name, "bill": args.bill, "chunk_id": args.chunk_id,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "ok": not bool(result.errors),
        }
        (args.out_dir / f"{agent.name}-metric-{args.chunk_id}.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
        print(f"[run] wrote {out_file.name} + metric. ok={metric['ok']}", flush=True)
        return 0 if metric["ok"] else 2
    chunks_file = chunks_path(args.bill)
    if not chunks_file.exists():
        print(f"[run] ERROR: chunks file not found: {chunks_file}", flush=True)
        return 1

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunk = next((c for c in chunks if c["chunk_id"] == args.chunk_id), None)
    if not chunk:
        print(f"[run] chunk_id {args.chunk_id} not found in {chunks_file}", flush=True)
        return 1

    chunk_text = chunk["text"]
    title_marker = chunk["marker_label"]
    print(f"[run] agent={args.agent} bill={args.bill} chunk={args.chunk_id} tokens={chunk['tokens']:,}", flush=True)
    print(f"[run] endpoint={AgentClass.target_endpoint}", flush=True)

    agent = AgentClass()
    t0 = time.perf_counter()
    result = agent.run(chunk_text, args.chunk_id, title_marker=title_marker)
    elapsed = time.perf_counter() - t0
    print(f"[run] elapsed={elapsed:.1f}s prompt_toks={result.prompt_tokens:,} completion_toks={result.completion_tokens:,}", flush=True)

    if result.errors:
        print(f"[run] ERRORS: {result.errors}", flush=True)
        (args.out_dir / f"{agent.name}-raw-{args.chunk_id}.txt").write_text(result.raw_response or "", encoding="utf-8")

    # Pass 2 enrichment for the xref agent: attach LMDB-resolved USC text to each citation.
    # Without this, downstream Citation Validator sees bare citations and can't audit.
    if args.agent == "xref" and result.output and not result.errors:
        try:
            from src.agents.usc_xref import enrich_with_usc
            from src.tools.fetch_usc import FetchUsc
            # 3090 FORK: default LMDB path derived from this file's location
            # (was hardcoded to B:\amd-hackathon-bill-analyzer\... old fork).
            _default_lmdb = str(Path(__file__).resolve().parents[1] / "data" / "usc.lmdb")
            lmdb_path = os.environ.get("BILL_ANALYZER_USC_LMDB", _default_lmdb)
            fetcher = FetchUsc(lmdb_path)
            t1 = time.perf_counter()
            result.output = enrich_with_usc(result.output, fetcher)
            enrich_elapsed = time.perf_counter() - t1
            ok_count = sum(1 for c in result.output.get("citations", []) if c.get("resolution_status") == "ok")
            total = len(result.output.get("citations", []))
            print(f"[run] enrich_with_usc: {ok_count}/{total} resolved in {enrich_elapsed*1000:.1f}ms (LMDB at {lmdb_path})", flush=True)
        except Exception as e:
            print(f"[run] WARN: enrich_with_usc failed: {type(e).__name__}: {e}", flush=True)

    # Pass 2 for the fiscal agent: compute totals_by_domain + grand_total_usd
    # in Python from the items[] the LLM returned. Keeping the LLM out of
    # arithmetic - it was burning tokens on aggregation in v1 and timing out.
    if args.agent == "fiscal" and result.output and not result.errors:
        try:
            from src.agents.fiscal_impact_estimator import FiscalImpactOutput, compute_totals
            validated = FiscalImpactOutput(**result.output)
            validated = compute_totals(validated)
            result.output = validated.model_dump()
            n_items = len(result.output.get("items", []))
            grand = result.output.get("grand_total_usd") or 0.0
            n_vision = len(result.output.get("vision_pages_suggested", []))
            print(f"[run] compute_totals: {n_items} items, grand=${grand:,.0f}, vision_pages_suggested={n_vision}", flush=True)
        except Exception as e:
            print(f"[run] WARN: compute_totals failed: {type(e).__name__}: {e}", flush=True)

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