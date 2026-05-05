"""Run citation_validator across all citations in a usc_cross_reference output file.

Usage:
    python tests/validate_citations.py --xref-file <path> --out-file <path>

Designed to run on the cloud instance (where reasoner endpoint is localhost:8003)
or locally. Each citation is one reasoner call, ~2-5 seconds. 142 citations
=> ~5-10 minutes total. Each call hits the SAME 32K context window so APC will
help on subsequent calls if the system prompt + user prompt template stays
constant (which it does in this design).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _apply_endpoint_env_overrides() -> None:
    """Patch base.py endpoint constants from env vars if set (mirrors run_one_agent.py)."""
    import importlib
    base = importlib.import_module("src.agents.base")
    for env_var, attr in [
        ("BILL_ANALYZER_SPINE_URL", "SPINE_ENDPOINT"),
        ("BILL_ANALYZER_REASONER_URL", "REASONER_ENDPOINT"),
        ("BILL_ANALYZER_VISION_URL", "VISION_ENDPOINT"),
    ]:
        v = os.environ.get(env_var)
        if v:
            setattr(base, attr, v)

_apply_endpoint_env_overrides()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xref-file", type=Path, required=True,
                    help="Path to usc_cross_reference-*.json output file")
    ap.add_argument("--out-file", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Validate only the first N citations (for smoke testing)")
    args = ap.parse_args()

    from src.agents.citation_validator import CitationValidator
    import importlib
    base = importlib.import_module("src.agents.base")
    CitationValidator.target_endpoint = base.REASONER_ENDPOINT

    xref = json.loads(args.xref_file.read_text(encoding="utf-8"))
    citations = xref.get("citations", [])
    if args.limit:
        citations = citations[: args.limit]

    print(f"[validate] target endpoint: {CitationValidator.target_endpoint}", flush=True)
    print(f"[validate] citations to audit: {len(citations)}", flush=True)

    agent = CitationValidator()
    results = []
    summary = {"valid": 0, "format-error": 0, "wrong-section": 0, "intent-mismatch": 0, "unverifiable": 0, "errored": 0}

    t0 = time.perf_counter()
    for i, c in enumerate(citations):
        usc = c.get("usc_data") or {}
        usc_heading = usc.get("heading", "")
        usc_text = (usc.get("text_excerpt") or "")[:600]  # cap to keep prompt small
        try:
            result = agent.run(
                citation=c.get("citation", ""),
                bill_context=c.get("bill_context", ""),
                relevance=c.get("relevance", ""),
                usc_heading=usc_heading,
                usc_text=usc_text,
            )
            if result.errors:
                summary["errored"] += 1
                results.append({"citation": c.get("citation"), "errored": True, "errors": result.errors})
            else:
                out = result.output
                verdict = out.get("verdict", "errored")
                summary[verdict] = summary.get(verdict, 0) + 1
                results.append(out)
            if i < 3 or i % 20 == 0:
                print(f"  [{i+1}/{len(citations)}] {c.get('citation', '')[:40]} -> {result.output.get('verdict', 'err') if result.output else 'err'}", flush=True)
        except Exception as e:
            summary["errored"] += 1
            results.append({"citation": c.get("citation"), "errored": True, "exception": str(e)})

    elapsed = time.perf_counter() - t0
    print(f"\n[validate] DONE  elapsed={elapsed:.1f}s  avg={elapsed/max(1,len(citations)):.2f}s/citation", flush=True)
    print(f"[validate] summary: {summary}", flush=True)

    out_data = {
        "source_xref_file": str(args.xref_file),
        "citations_audited": len(citations),
        "elapsed_s": round(elapsed, 2),
        "summary": summary,
        "results": results,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[validate] wrote {args.out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
