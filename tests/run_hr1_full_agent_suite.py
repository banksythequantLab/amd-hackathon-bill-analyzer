"""Run every agent in the headline/podcast/slide pipeline against HR1 ch01.

GOAL: Prove that the 3090 fork (qwen3:30b-a3b-instruct-2507 + qwen3-vl)
can run the full agent suite without thinking-block truncation, JSON
parse failures, or other regressions vs the AMD baseline.

ORDER OF OPERATIONS:
  1. summarizer (already validated, re-run for context chain)
  2. usc_cross_reference (already validated, re-run for context chain)
  3. pork_finder       \
  4. conflict_spotter   \
  5. fiscal_impact      } These 4 are independent of upstream outputs
  6. stakeholder_tracer/
  7. citation_validator (reasoner alias)
  8. headline_ranker (typically takes podcast headlines)
  9. podcast_headlines_generator (takes summarizer output)
 10. podcast_script_writer (takes summarizer + headlines)
 11. podcast_generator (takes script)
 12. prompt_relay_author (visual relay; takes script)
 13. slide_prompt_generator (takes script)
 14. wan_motion_prompt_generator (takes script)
 15. youtube_metadata_generator (takes summarizer + script + headlines)
 16. slide_critic (vision model; takes a rendered slide image -- skipped here,
                   needs image asset which we don't have for HR1 yet)

For each agent we record: elapsed, prompt_tokens, completion_tokens,
output, any errors. Output to eval/hr1-ch01-full-agent-suite.json.
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tiktoken  # noqa: E402

from src.chunking.smart_chunker import (  # noqa: E402
    extract_text_with_pages,
    find_boundaries,
    pack_chunks,
)
from src.agents.summarizer import PlainEnglishSummarizer  # noqa: E402
from src.agents.usc_xref import UscCrossReference, enrich_with_usc  # noqa: E402
from src.agents.pork_finder import PorkFinder  # noqa: E402
from src.agents.conflict_spotter import ConflictSpotter  # noqa: E402
from src.agents.fiscal_impact_estimator import FiscalImpactEstimator  # noqa: E402
from src.agents.stakeholder_tracer import StakeholderTracer  # noqa: E402
from src.agents.citation_validator import CitationValidator  # noqa: E402
from src.agents.podcast_headlines_generator import PodcastHeadlinesGenerator  # noqa: E402
from src.agents.headline_ranker import HeadlineRanker  # noqa: E402
from src.agents.podcast_script_writer import PodcastScriptWriter  # noqa: E402
from src.agents.podcast_generator import PodcastGenerator  # noqa: E402
from src.agents.prompt_relay_author import PromptRelayAuthor  # noqa: E402
from src.agents.slide_prompt_generator import SlidePromptGenerator  # noqa: E402
from src.agents.wan_motion_prompt_generator import WanMotionPromptGenerator  # noqa: E402
from src.agents.youtube_metadata_generator import YouTubeMetadataGenerator  # noqa: E402
from src.tools.http_fetch_usc import HttpFetchUsc  # noqa: E402


HR1 = REPO / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"
OUT = REPO / "eval" / "hr1-ch01-full-agent-suite.json"
USC_HTTP_URL = "http://127.0.0.1:8004"


def get_ch01():
    """Re-derive ch01 deterministically -- same chunker run as fork-v2."""
    text, page_starts = extract_text_with_pages(HR1)
    enc = tiktoken.get_encoding("cl100k_base")
    boundaries = find_boundaries(text, page_starts)
    chunks = pack_chunks(text, boundaries, page_starts, 50_500, enc)
    return chunks[0]


def run_agent(label: str, agent_obj, chunk_text: str, chunk_id: str,
              context: dict | None = None, **extra) -> dict:
    """Run one agent.run() call and return a uniform result record."""
    print(f"[RUN ] {label} ({agent_obj.target_model}, {agent_obj.max_tokens} max_tok)")
    t0 = time.perf_counter()
    try:
        result = agent_obj.run(
            chunk_text=chunk_text,
            chunk_id=chunk_id,
            context=context or {},
            **extra,
        )
        elapsed = time.perf_counter() - t0
        if result.errors:
            print(f"       [ERR] {result.errors}  raw_preview={result.raw_response[:200]!r}")
            return {
                "agent": label,
                "ok": False,
                "elapsed_s": round(elapsed, 2),
                "errors": result.errors,
                "raw_preview": result.raw_response[:600],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        out = result.output if isinstance(result.output, dict) else {}
        # Brief headline of what came back so it's visible in the log.
        if isinstance(out, dict):
            keys = list(out.keys())[:5]
            sample = None
            for v in out.values():
                if isinstance(v, list) and v:
                    sample = f"first-of-list ({len(v)}): {str(v[0])[:80]}"
                    break
                if isinstance(v, str):
                    sample = f"str: {v[:80]}"
                    break
            print(f"       [OK ] {elapsed:.1f}s  prompt={result.prompt_tokens:,}  "
                  f"compl={result.completion_tokens:,}  keys={keys}  {sample or ''}")
        return {
            "agent": label,
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "output": out,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()
        print(f"       [EXC] {type(e).__name__}: {e}  after {elapsed:.1f}s")
        print(tb)
        return {
            "agent": label,
            "ok": False,
            "elapsed_s": round(elapsed, 2),
            "exception": f"{type(e).__name__}: {e}",
            "traceback": tb,
        }


def main() -> int:
    ch01 = get_ch01()
    print(f"=== HR1 ch01: {ch01.marker}  ({ch01.tokens:,} cl100k tokens, pp.{ch01.start_page}-{ch01.end_page})")
    print(f"=== Spine model: qwen3:30b-a3b-instruct-2507-q4_K_M @ 64K ctx, KV q8_0")
    print(f"=== USC HTTP: {USC_HTTP_URL}")
    print()

    chunk_text = ch01.text
    chunk_id = "ch01"
    title_marker = ch01.marker_label
    all_results: list[dict] = []
    context: dict[str, Any] = {"title_marker": title_marker}
    t_start = time.perf_counter()

    # 1. summarizer
    r = run_agent("summarizer", PlainEnglishSummarizer(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["summarizer"] = r["output"]

    # 2. usc_cross_reference (+ enrichment via HTTP server)
    r = run_agent("usc_cross_reference", UscCrossReference(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    if r.get("ok"):
        fetcher = HttpFetchUsc(USC_HTTP_URL)
        t1 = time.perf_counter()
        enriched = enrich_with_usc(r["output"], fetcher)
        r["enrich_elapsed_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        r["n_citations"] = len(enriched.get("citations", []))
        r["n_resolved"] = sum(1 for c in enriched.get("citations", []) if c.get("resolution_status") == "ok")
        r["output"] = enriched
        fetcher.close()
        context["usc_cross_reference"] = enriched
        print(f"       [usc] enriched {r['n_citations']} cites, {r['n_resolved']} fully resolved")
    all_results.append(r)

    # 3-6. Independent agents.
    for label, cls in [
        ("pork_finder",            PorkFinder),
        ("conflict_spotter",       ConflictSpotter),
        ("fiscal_impact_estimator", FiscalImpactEstimator),
        ("stakeholder_tracer",     StakeholderTracer),
    ]:
        r = run_agent(label, cls(), chunk_text, chunk_id, context=context, title_marker=title_marker)
        all_results.append(r)
        if r.get("ok"):
            context[label] = r["output"]

    # 7. citation_validator (reasoner alias = same Instruct weights)
    r = run_agent("citation_validator", CitationValidator(), chunk_text, chunk_id, context=context)
    all_results.append(r)
    if r.get("ok"):
        context["citation_validator"] = r["output"]

    # 8. podcast_headlines_generator (needs summarizer output in context)
    r = run_agent("podcast_headlines_generator", PodcastHeadlinesGenerator(),
                  chunk_text, chunk_id, context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["podcast_headlines_generator"] = r["output"]

    # 9. headline_ranker (ranks headlines)
    r = run_agent("headline_ranker", HeadlineRanker(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["headline_ranker"] = r["output"]

    # 10. podcast_script_writer
    r = run_agent("podcast_script_writer", PodcastScriptWriter(),
                  chunk_text, chunk_id, context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["podcast_script_writer"] = r["output"]

    # 11. podcast_generator (longer-form podcast generation)
    r = run_agent("podcast_generator", PodcastGenerator(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["podcast_generator"] = r["output"]

    # 12. prompt_relay_author -- note: has **kwargs signature, no max_retries
    print(f"[RUN ] prompt_relay_author (spine, 6000 max_tok)")
    t0 = time.perf_counter()
    try:
        author = PromptRelayAuthor()
        result = author.run(chunk_text=chunk_text, chunk_id=chunk_id, context=context, title_marker=title_marker)
        elapsed = time.perf_counter() - t0
        if result.errors:
            print(f"       [ERR] {result.errors}")
            all_results.append({"agent": "prompt_relay_author", "ok": False, "elapsed_s": round(elapsed, 2),
                                "errors": result.errors, "raw_preview": result.raw_response[:600]})
        else:
            out = result.output if isinstance(result.output, dict) else {}
            print(f"       [OK ] {elapsed:.1f}s  prompt={result.prompt_tokens:,}  compl={result.completion_tokens:,}  keys={list(out.keys())[:5]}")
            all_results.append({"agent": "prompt_relay_author", "ok": True, "elapsed_s": round(elapsed, 2),
                                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                                "output": out})
            context["prompt_relay_author"] = out
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"       [EXC] {type(e).__name__}: {e}")
        all_results.append({"agent": "prompt_relay_author", "ok": False, "elapsed_s": round(elapsed, 2),
                            "exception": f"{type(e).__name__}: {e}"})

    # 13. slide_prompt_generator
    r = run_agent("slide_prompt_generator", SlidePromptGenerator(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["slide_prompt_generator"] = r["output"]

    # 14. wan_motion_prompt_generator
    r = run_agent("wan_motion_prompt_generator", WanMotionPromptGenerator(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    all_results.append(r)
    if r.get("ok"):
        context["wan_motion_prompt_generator"] = r["output"]

    # 15. youtube_metadata_generator
    r = run_agent("youtube_metadata_generator", YouTubeMetadataGenerator(), chunk_text, chunk_id,
                  context=context, title_marker=title_marker)
    all_results.append(r)

    # slide_critic skipped (needs vision model + rendered slide image we don't have for HR1)

    total = time.perf_counter() - t_start
    n_ok = sum(1 for r in all_results if r.get("ok"))
    n_err = sum(1 for r in all_results if r.get("ok") is False)

    print()
    print(f"=== SUMMARY: {n_ok}/{len(all_results)} ok, {n_err} errored, total {total:.1f}s")
    print(f"=== Per-agent timing:")
    for r in all_results:
        status = "OK " if r.get("ok") else "ERR"
        print(f"     [{status}] {r['elapsed_s']:>7.1f}s  {r['agent']}")

    report = {
        "test": "hr1_ch01_full_agent_suite",
        "spine": "qwen3:30b-a3b-instruct-2507-q4_K_M @ 64K ctx, KV q8_0",
        "reasoner": "spine (same)",
        "vision": "qwen3-vl:8b-instruct (not used in this run)",
        "chunk": {"id": chunk_id, "marker": ch01.marker, "tokens_cl100k": ch01.tokens,
                  "pages": [ch01.start_page, ch01.end_page]},
        "total_elapsed_s": round(total, 2),
        "n_ok": n_ok,
        "n_errored": n_err,
        "results": all_results,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ARTIFACT] {OUT}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
