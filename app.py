"""Bill Analyzer — Multi-Agent Legislative Analysis on AMD MI300X.

Built for the lablab.ai AMD Developer Hackathon, May 2026.

Architecture:
  PDF -> smart_chunker (TITLE/Subtitle boundaries, tiktoken-budgeted)
      -> 4 specialist agents on Qwen3-30B-A3B-Instruct-2507-FP8 (vLLM on MI300X)
         - Plain-English Summarizer
         - USC Cross-Reference (cites US Code sections)
         - Pork Finder (earmarks, special interests)
         - Conflict Spotter (internal contradictions)
      -> USC enrichment via remote LMDB (60K USC sections, HTTP service)

Endpoints (env-configurable for HF Spaces deployment):
  SPINE_ENDPOINT      Qwen3-30B-A3B-Instruct-2507-FP8 (vLLM port 8001)
  USC_LMDB_HTTP       Remote USC LMDB lookup service (port 8004)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr

# ----------------------------------------------------------------------------
# Endpoint config - env-driven for HF Space portability
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SPINE = os.environ.get("SPINE_ENDPOINT", "http://165.245.134.1:8001/v1")
USC_HTTP = os.environ.get("USC_LMDB_HTTP", "http://165.245.134.1:8004")

# Patch agent base BEFORE importing agents
import src.agents.base as agent_base
agent_base.SPINE_ENDPOINT = SPINE

from src.agents.summarizer       import PlainEnglishSummarizer
from src.agents.usc_xref         import UscCrossReference, enrich_with_usc
from src.agents.pork_finder      import PorkFinder
from src.agents.conflict_spotter import ConflictSpotter
from src.agents.podcast_headlines_generator import PodcastHeadlinesGenerator
from src.agents.headline_ranker  import HeadlineRanker
from src.tools.http_fetch_usc    import HttpFetchUsc, get_fetcher
from src.chunking.smart_chunker  import chunk_pdf

# Cloud podcast pipeline (Qwen-Image + Wan i2v + Qwen3-TTS, all on MI300X)
# Looks for the orchestrator in two places, in order:
#   1. <repo>/scripts/make_podcast_cloud.py   (in-repo, ships with the project)
#   2. B:\hackathon-build\make_podcast_cloud.py  (live dev path on the author's box)
import sys as _sys
_REPO_SCRIPTS = str(Path(__file__).parent / "scripts")
_HACKBUILD = r"B:\hackathon-build"
for _p in (_REPO_SCRIPTS, _HACKBUILD):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
try:
    from make_podcast_cloud import run_full_pipeline as _cloud_run_pipeline
    _CLOUD_PIPELINE_AVAILABLE = True
except Exception as _e:
    print(f"[warn] cloud pipeline unavailable: {_e}")
    _CLOUD_PIPELINE_AVAILABLE = False
    _cloud_run_pipeline = None

# Force per-class endpoint update (class attrs cached the constant at import time)
for AgentClass in (PlainEnglishSummarizer, UscCrossReference, PorkFinder, ConflictSpotter,
                   PodcastHeadlinesGenerator, HeadlineRanker):
    if AgentClass.target_endpoint.startswith("http://165.245.134.1:8001"):
        AgentClass.target_endpoint = SPINE

USC_LOCAL_LMDB = ROOT / "data" / "usc.lmdb"

AGENTS = [
    ("Plain-English Summarizer", PlainEnglishSummarizer, "summarizer"),
    ("USC Cross-Reference",      UscCrossReference,      "usc_cross_ref"),
    ("Pork Finder",              PorkFinder,             "pork_finder"),
    ("Conflict Spotter",         ConflictSpotter,        "conflict_spotter"),
]

DEMO_REPORT_PATH = ROOT / "eval" / "report-bbb-ch01-day6-canonical.json"
DEMO_REPORT = None
if DEMO_REPORT_PATH.exists():
    try:
        DEMO_REPORT = json.loads(DEMO_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        DEMO_REPORT = None


# ----------------------------------------------------------------------------
# Utility: health check
# ----------------------------------------------------------------------------
def check_endpoints() -> str:
    """Inline status badge for backend endpoints.

    Spine status check was removed — its rstrip('/v1') call was mangling
    the URL to port 800 instead of 8001, so it was always reporting
    offline regardless of actual spine state. The spine's reachability
    is implicit in the analysis run itself (failures surface in logs
    with full diagnostics), so a separate badge added confusion.

    What we DO check here:
      - USC LMDB at port 8004 (used by USC Cross-Reference enrichment)
      - ComfyUI at port 8188 (used by Qwen-Image, Wan i2v, Qwen3-TTS)

    Both have stable health endpoints and respond fast (<200ms when up).
    """
    import httpx
    badges = []
    # ComfyUI base URL is derived from SPINE host (same droplet, port 8188).
    try:
        comfy_host = SPINE.split('://')[1].split(':')[0]
        comfy_url = f"http://{comfy_host}:8188"
    except Exception:
        comfy_url = "http://165.245.134.1:8188"

    for short, url, kind in [
        ("USC", USC_HTTP, "usc"),
        ("ComfyUI", comfy_url, "comfy"),
    ]:
        try:
            if kind == "usc":
                r = httpx.get(f"{url.rstrip('/')}/usc/lookup", params={"citation": "42 USC 1395dd"}, timeout=3.0)
                ok = r.status_code == 200
            else:  # comfy
                r = httpx.get(f"{url.rstrip('/')}/system_stats", timeout=3.0)
                ok = r.status_code == 200
        except Exception:
            ok = False
        color = "#10b981" if ok else "#ef4444"
        label = "online" if ok else "offline"
        badges.append(
            f"<span style='display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#475569;background:#f1f5f9;padding:3px 9px;border-radius:10px;margin-right:6px'>"
            f"<span style='width:7px;height:7px;border-radius:50%;background:{color};box-shadow:0 0 6px {color}'></span>"
            f"<b style='color:#1e293b'>{short}</b> {label}</span>"
        )
    return "".join(badges)


# ----------------------------------------------------------------------------
# Pipeline: chunk + run agents
# ----------------------------------------------------------------------------
def run_agents_on_chunk(chunk: dict, progress) -> dict:
    """Run all 6 agents sequentially against a single chunk. Used by non-streaming caller."""
    # Drain the streaming generator to get the final report
    final_report = None
    for evt in run_agents_streaming(chunk, progress=progress):
        if evt.get("type") == "final":
            final_report = evt["report"]
    return final_report


def run_agents_streaming(chunk: dict, progress=None):
    """Generator that yields {type, ...} events as each agent finishes.
    Events:
      {type: 'agent_done', name, label, elapsed_s, prompt_tokens, completion_tokens, errors, output_summary}
      {type: 'enrichment_done', stats, elapsed_ms}
      {type: 'final', report}
    """
    import time, json as _json
    chunk_text = chunk["text"]
    chunk_id = chunk["chunk_id"]
    title_marker = chunk["marker_label"]

    report = {
        "chunk_id": chunk_id,
        "title_marker": title_marker,
        "tokens": chunk["tokens"],
        "pages": [chunk["start_page"], chunk["end_page"]],
        "agents": {},
        "timings_s": {},
        "totals": {},
    }

    grand_t0 = time.perf_counter()
    total_p, total_c = 0, 0
    n = len(AGENTS) + 2  # +2 for headlines + ranker

    for i, (label, AgentClass, key) in enumerate(AGENTS):
        if progress:
            progress((0.10 + 0.75 * i / n), desc=f"[{i+1}/{n}] {label}…")
        agent = AgentClass()
        t0 = time.perf_counter()
        try:
            result = agent.run(chunk_text, chunk_id, title_marker=title_marker)
            elapsed = time.perf_counter() - t0
            report["agents"][key] = {
                "label": label,
                "output": result.output,
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "errors": result.errors,
            }
            report["timings_s"][key] = round(elapsed, 2)
            total_p += result.prompt_tokens
            total_c += result.completion_tokens
            yield {
                "type": "agent_done", "name": key, "label": label,
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "errors": result.errors,
                "running_total_p": total_p, "running_total_c": total_c,
                "step": i + 1, "total_steps": n,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            report["agents"][key] = {
                "label": label, "output": None, "elapsed_s": round(elapsed, 2),
                "prompt_tokens": 0, "completion_tokens": 0,
                "errors": [f"{type(e).__name__}: {e}"],
            }
            yield {
                "type": "agent_done", "name": key, "label": label,
                "elapsed_s": round(elapsed, 2), "prompt_tokens": 0, "completion_tokens": 0,
                "errors": [f"{type(e).__name__}: {e}"],
                "running_total_p": total_p, "running_total_c": total_c,
                "step": i + 1, "total_steps": n,
            }

    # USC enrichment
    if progress:
        progress(0.80, desc="Enriching USC citations from remote LMDB…")
    fetcher = get_fetcher(local_path=str(USC_LOCAL_LMDB), http_url=USC_HTTP)
    if fetcher is not None and "usc_cross_ref" in report["agents"]:
        xref_out = report["agents"]["usc_cross_ref"]["output"]
        if xref_out and isinstance(xref_out, dict) and "citations" in xref_out:
            t0 = time.perf_counter()
            enrich_with_usc(xref_out, fetcher)
            enrich_ms = (time.perf_counter() - t0) * 1000
            stats = fetcher.stats()
            fetcher.close()
            report["agents"]["usc_cross_ref"]["enrichment"] = {
                "elapsed_ms": round(enrich_ms, 2),
                "lmdb_stats": stats,
            }
            yield {"type": "enrichment_done", "stats": stats, "elapsed_ms": round(enrich_ms, 2)}

    # Stage 2: Podcast Headlines
    if progress:
        progress(0.85, desc=f"[{len(AGENTS)+1}/{n}] Generating 10 podcast headlines…")
    try:
        analysis_payload = _json.dumps({
            "title_marker": title_marker,
            "summarizer": report["agents"].get("summarizer", {}).get("output"),
            "usc_cross_reference": report["agents"].get("usc_cross_ref", {}).get("output"),
            "pork_finder": report["agents"].get("pork_finder", {}).get("output"),
            "conflict_spotter": report["agents"].get("conflict_spotter", {}).get("output"),
        }, indent=2)[:30000]
        agent = PodcastHeadlinesGenerator()
        t0 = time.perf_counter()
        result = agent.run(analysis_payload, chunk_id, title_marker=title_marker)
        elapsed = time.perf_counter() - t0
        report["agents"]["podcast_headlines"] = {
            "label": "Podcast Headlines Generator",
            "output": result.output, "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "errors": result.errors,
        }
        report["timings_s"]["podcast_headlines"] = round(elapsed, 2)
        total_p += result.prompt_tokens
        total_c += result.completion_tokens
        yield {
            "type": "agent_done", "name": "podcast_headlines",
            "label": "Podcast Headlines Generator",
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "errors": result.errors,
            "running_total_p": total_p, "running_total_c": total_c,
            "step": len(AGENTS) + 1, "total_steps": n,
        }
    except Exception as e:
        report["agents"]["podcast_headlines"] = {
            "label": "Podcast Headlines Generator", "output": None, "elapsed_s": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "errors": [f"{type(e).__name__}: {e}"],
        }

    # Stage 3: Ranker
    if progress:
        progress(0.93, desc=f"[{n}/{n}] Ranking headlines…")
    try:
        h_out = report["agents"].get("podcast_headlines", {}).get("output")
        if h_out and isinstance(h_out, dict) and h_out.get("headlines"):
            agent = HeadlineRanker()
            t0 = time.perf_counter()
            result = agent.run(_json.dumps(h_out, indent=2), chunk_id, title_marker=title_marker)
            elapsed = time.perf_counter() - t0
            report["agents"]["headline_ranker"] = {
                "label": "Headline Ranker", "output": result.output,
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "errors": result.errors,
            }
            report["timings_s"]["headline_ranker"] = round(elapsed, 2)
            total_p += result.prompt_tokens
            total_c += result.completion_tokens
            yield {
                "type": "agent_done", "name": "headline_ranker",
                "label": "Headline Ranker", "elapsed_s": round(elapsed, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "errors": result.errors,
                "running_total_p": total_p, "running_total_c": total_c,
                "step": n, "total_steps": n,
            }
    except Exception as e:
        report["agents"]["headline_ranker"] = {
            "label": "Headline Ranker", "output": None, "elapsed_s": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "errors": [f"{type(e).__name__}: {e}"],
        }

    grand = time.perf_counter() - grand_t0
    report["totals"] = {
        "wall_clock_s": round(grand, 2),
        "prompt_tokens_total": total_p,
        "completion_tokens_total": total_c,
    }
    yield {"type": "final", "report": report}


# ----------------------------------------------------------------------------
# Output renderers - convert agent JSON into pretty HTML/Markdown
# ----------------------------------------------------------------------------
def render_overview(report: dict, chunks_summary: str = "") -> str:
    """Top stats card — title, tokens, timings."""
    totals = report.get("totals", {})
    wall = totals.get("wall_clock_s", "—")
    tokens = report.get("tokens", 0)
    prompt_t = totals.get("prompt_tokens_total", 0)
    completion_t = totals.get("completion_tokens_total", 0)
    pages = report.get("pages", [0, 0])
    n_chunks = report.get("n_chunks", 1) or 1
    chunks_processed = report.get("chunks_processed") or []
    n_processed = len(chunks_processed) if chunks_processed else 1

    title_marker = report.get("title_marker", "Bill Analysis")

    # Captions adapt to single vs multi-chunk runs
    tokens_caption = (
        f"sum of {n_processed} chunks (cl100k)" if n_processed > 1 else "cl100k_base"
    )
    pages_caption = (
        f"full bill ({pages[1]-pages[0]+1} pages, {n_processed} chunks)"
        if n_processed > 1 else f"{pages[1]-pages[0]+1} pages"
    )
    wall_caption = (
        f"6 agents x {n_processed} chunks" if n_processed > 1 else "6 agents per chunk"
    )

    # IMPORTANT: every inner div sets color:#fff !important so Gradio's theme
    # (which adds dark text rules to .gradio-container *) cannot override.
    cards = f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:8px 0 20px 0">
  <div style="background:linear-gradient(135deg,#1e3a8a 0%,#3730a3 100%);color:#fff !important;padding:18px;border-radius:12px;box-shadow:0 4px 12px rgba(30,58,138,0.15)">
    <div style="color:#fff !important;opacity:0.85;font-size:11px;letter-spacing:0.05em;text-transform:uppercase">Total Tokens (input)</div>
    <div style="color:#fff !important;font-size:28px;font-weight:700;margin-top:4px">{tokens:,}</div>
    <div style="color:#fff !important;opacity:0.8;font-size:12px;margin-top:2px">{tokens_caption}</div>
  </div>
  <div style="background:linear-gradient(135deg,#0f766e 0%,#14b8a6 100%);color:#fff !important;padding:18px;border-radius:12px;box-shadow:0 4px 12px rgba(15,118,110,0.15)">
    <div style="color:#fff !important;opacity:0.85;font-size:11px;letter-spacing:0.05em;text-transform:uppercase">Pages</div>
    <div style="color:#fff !important;font-size:28px;font-weight:700;margin-top:4px">{pages[0]}–{pages[1]}</div>
    <div style="color:#fff !important;opacity:0.8;font-size:12px;margin-top:2px">{pages_caption}</div>
  </div>
  <div style="background:linear-gradient(135deg,#7c2d12 0%,#ea580c 100%);color:#fff !important;padding:18px;border-radius:12px;box-shadow:0 4px 12px rgba(124,45,18,0.15)">
    <div style="color:#fff !important;opacity:0.85;font-size:11px;letter-spacing:0.05em;text-transform:uppercase">Wall Clock</div>
    <div style="color:#fff !important;font-size:28px;font-weight:700;margin-top:4px">{wall}s</div>
    <div style="color:#fff !important;opacity:0.8;font-size:12px;margin-top:2px">{wall_caption}</div>
  </div>
  <div style="background:linear-gradient(135deg,#6b21a8 0%,#a855f7 100%);color:#fff !important;padding:18px;border-radius:12px;box-shadow:0 4px 12px rgba(107,33,168,0.15)">
    <div style="color:#fff !important;opacity:0.85;font-size:11px;letter-spacing:0.05em;text-transform:uppercase">Agent I/O</div>
    <div style="color:#fff !important;font-size:28px;font-weight:700;margin-top:4px">{(prompt_t + completion_t):,}</div>
    <div style="color:#fff !important;opacity:0.8;font-size:12px;margin-top:2px">{prompt_t:,} in / {completion_t:,} out</div>
  </div>
</div>"""

    header = f"""
<div style="background:#f8fafc;border-left:4px solid #1e3a8a;padding:14px 18px;margin-bottom:8px;border-radius:0 8px 8px 0">
  <div style="font-size:11px;color:#64748b;letter-spacing:0.05em;text-transform:uppercase;font-weight:600">Analyzed Section</div>
  <div style="font-size:20px;font-weight:700;color:#0f172a;margin-top:2px">{title_marker}</div>
</div>"""

    chunks_html = ""
    if chunks_summary:
        chunks_html = f"""
<details style="margin:8px 0 14px 0">
  <summary style="cursor:pointer;color:#475569;font-size:13px;font-weight:600;padding:6px 0">📄 PDF chunking details</summary>
  <pre style="background:#f1f5f9;padding:12px;border-radius:6px;font-size:12px;overflow-x:auto;color:#334155">{chunks_summary}</pre>
</details>"""

    return header + cards + chunks_html


def render_summarizer(out: dict | None, elapsed: float, errors: list) -> str:
    if not out or not isinstance(out, dict):
        return _render_empty("Summarizer", errors)

    one_liner = out.get("one_sentence_summary", "")
    bullets = out.get("bullets", []) or []
    affected = out.get("affected_groups", []) or []

    bullet_html = "".join(f"<li style='margin:6px 0;line-height:1.55'>{_esc(b)}</li>" for b in bullets)
    affected_html = ""
    if affected:
        chips = "".join(
            f"<span style='display:inline-block;background:#dbeafe;color:#1e40af;padding:4px 10px;border-radius:14px;font-size:12px;margin:2px;font-weight:500'>{_esc(g)}</span>"
            for g in affected
        )
        affected_html = (
            "<div style='margin-top:18px'>"
            "<div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:8px'>Affected Groups</div>"
            f"<div>{chips}</div></div>"
        )

    return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600">One-line summary</div>
    <div style="font-size:11px;color:#94a3b8">⏱ {elapsed:.1f}s</div>
  </div>
  <div style="font-size:16px;line-height:1.6;color:#0f172a;font-weight:500;border-left:3px solid #3b82f6;padding-left:14px">
    {_esc(one_liner)}
  </div>
  <div style="margin-top:20px">
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:8px">Key Provisions</div>
    <ul style="margin:0;padding-left:20px;color:#1f2937">{bullet_html}</ul>
  </div>
  {affected_html}
</div>"""


def render_xref(out: dict | None, elapsed: float, errors: list, enrichment: dict | None = None):
    """Returns (html_summary, dataframe_value) for the citations table."""
    if not out or not isinstance(out, dict):
        return _render_empty("USC Cross-Reference", errors), []

    citations = out.get("citations", []) or []

    # Stats
    total = len(citations)
    resolved = sum(1 for c in citations if c.get("resolution_status", "").startswith("ok"))
    not_found = total - resolved
    pct = (resolved / total * 100) if total else 0

    enrich_html = ""
    if enrichment:
        ms = enrichment.get("elapsed_ms", 0)
        st = enrichment.get("lmdb_stats", {})
        enrich_html = f"""
<div style="display:flex;gap:8px;margin-top:8px;font-size:12px;color:#64748b">
  <span>⚡ LMDB enrichment: <b>{ms:.1f}ms</b></span>
  <span>•</span>
  <span>{st.get('hits', 0)} hits, {st.get('misses', 0)} misses</span>
</div>"""

    summary = f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600">USC Citation Resolution</div>
    <div style="font-size:11px;color:#94a3b8">⏱ {elapsed:.1f}s</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:8px">
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:14px">
      <div style="font-size:11px;color:#15803d;text-transform:uppercase;font-weight:600">Resolved</div>
      <div style="font-size:24px;font-weight:700;color:#14532d;margin-top:2px">{resolved}</div>
    </div>
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:14px">
      <div style="font-size:11px;color:#b91c1c;text-transform:uppercase;font-weight:600">Not Found</div>
      <div style="font-size:24px;font-weight:700;color:#7f1d1d;margin-top:2px">{not_found}</div>
    </div>
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px">
      <div style="font-size:11px;color:#1d4ed8;text-transform:uppercase;font-weight:600">Resolution %</div>
      <div style="font-size:24px;font-weight:700;color:#1e3a8a;margin-top:2px">{pct:.1f}%</div>
    </div>
  </div>
  {enrich_html}
</div>"""

    # Build dataframe
    rows = []
    for c in citations:
        cit = c.get("citation", "")
        ctx = c.get("bill_context", "")
        usc = c.get("usc_data") or {}
        heading = usc.get("heading", "") if usc else ""
        excerpt = usc.get("text_excerpt", "") if usc else ""
        status_raw = c.get("resolution_status", "unknown")
        if status_raw == "ok":
            status = "✅ Resolved"
        elif status_raw == "ok-section-level":
            status = "🔵 Section-level"
        elif status_raw == "not_found":
            status = "❌ Not found"
        else:
            status = status_raw
        rows.append([cit, status, heading[:60], (ctx or "")[:120], (excerpt or "")[:200]])
    
    return summary, rows


def render_pork(out: dict | None, elapsed: float, errors: list) -> str:
    if not out or not isinstance(out, dict):
        return _render_empty("Pork Finder", errors)

    items = out.get("findings") or out.get("earmarks") or out.get("items") or []
    if isinstance(items, dict):
        items = list(items.values())

    if not items:
        return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px;text-align:center">
  <div style="font-size:48px">🥩</div>
  <div style="font-weight:600;color:#475569;margin-top:8px">No earmarks or special-interest provisions detected</div>
  <div style="font-size:13px;color:#94a3b8;margin-top:4px">⏱ {elapsed:.1f}s</div>
</div>"""

    cards = []
    for it in items:
        if not isinstance(it, dict):
            cards.append(f"<li>{_esc(str(it))}</li>")
            continue
        title = it.get("title") or it.get("provision") or it.get("description") or it.get("name") or "Provision"
        amount = it.get("amount") or it.get("dollars") or ""
        beneficiary = it.get("beneficiary") or it.get("recipient") or it.get("target") or ""
        rationale = it.get("rationale") or it.get("reason") or it.get("flag_reason") or it.get("notes") or ""
        section = it.get("section") or it.get("citation") or ""

        amount_badge = ""
        if amount:
            amount_badge = f"<span style='display:inline-block;background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;margin-left:8px'>💰 {_esc(str(amount))}</span>"

        section_badge = ""
        if section:
            section_badge = f"<span style='display:inline-block;background:#e0e7ff;color:#3730a3;padding:3px 10px;border-radius:12px;font-size:11px;font-family:monospace;margin-left:8px'>{_esc(str(section))}</span>"

        beneficiary_html = ""
        if beneficiary:
            beneficiary_html = f"<div style='font-size:13px;color:#475569;margin-top:6px'><b>Beneficiary:</b> {_esc(str(beneficiary))}</div>"

        rationale_html = ""
        if rationale:
            rationale_html = f"<div style='font-size:13px;color:#64748b;margin-top:8px;font-style:italic'>{_esc(str(rationale))}</div>"

        cards.append(f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-left:4px solid #f59e0b;border-radius:8px;padding:14px;margin-bottom:10px">
  <div style="display:flex;align-items:center;flex-wrap:wrap">
    <div style="font-weight:600;color:#0f172a;flex:1;min-width:200px">{_esc(str(title))}</div>
    {amount_badge}{section_badge}
  </div>
  {beneficiary_html}
  {rationale_html}
</div>""")

    return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div>
      <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600">Pork & Earmarks Detected</span>
      <span style="font-size:11px;color:#94a3b8;margin-left:8px">{len(items)} found</span>
    </div>
    <div style="font-size:11px;color:#94a3b8">⏱ {elapsed:.1f}s</div>
  </div>
  {''.join(cards)}
</div>"""


def render_conflict(out: dict | None, elapsed: float, errors: list) -> str:
    if not out or not isinstance(out, dict):
        return _render_empty("Conflict Spotter", errors)

    items = out.get("conflicts") or out.get("findings") or out.get("contradictions") or []
    if isinstance(items, dict):
        items = list(items.values())

    if not items:
        return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px;text-align:center">
  <div style="font-size:48px">✅</div>
  <div style="font-weight:600;color:#475569;margin-top:8px">No internal conflicts detected</div>
  <div style="font-size:13px;color:#94a3b8;margin-top:4px">⏱ {elapsed:.1f}s</div>
</div>"""

    cards = []
    severity_colors = {
        "high": "#dc2626", "critical": "#dc2626",
        "medium": "#f59e0b", "moderate": "#f59e0b",
        "low": "#64748b", "minor": "#64748b",
    }

    for it in items:
        if not isinstance(it, dict):
            cards.append(f"<li>{_esc(str(it))}</li>")
            continue
        title = it.get("title") or it.get("conflict") or it.get("description") or "Conflict"
        severity = (it.get("severity") or it.get("priority") or "medium").lower()
        section_a = it.get("section_a") or it.get("provision_a") or it.get("first") or ""
        section_b = it.get("section_b") or it.get("provision_b") or it.get("second") or ""
        explanation = it.get("explanation") or it.get("details") or it.get("notes") or ""

        color = severity_colors.get(severity, "#64748b")

        sev_badge = f"<span style='display:inline-block;background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-left:8px'>{_esc(severity)}</span>"

        sections_html = ""
        if section_a or section_b:
            sa_html = f"<span style='font-family:monospace;background:#f1f5f9;padding:2px 8px;border-radius:4px;font-size:12px'>{_esc(str(section_a))}</span>" if section_a else ""
            sb_html = f"<span style='font-family:monospace;background:#f1f5f9;padding:2px 8px;border-radius:4px;font-size:12px'>{_esc(str(section_b))}</span>" if section_b else ""
            arrow = " <span style='color:#94a3b8'>↔</span> " if (section_a and section_b) else ""
            sections_html = f"<div style='margin-top:8px'>{sa_html}{arrow}{sb_html}</div>"

        explanation_html = ""
        if explanation:
            explanation_html = f"<div style='margin-top:10px;color:#475569;line-height:1.55;font-size:14px'>{_esc(str(explanation))}</div>"

        cards.append(f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-left:4px solid {color};border-radius:8px;padding:14px;margin-bottom:10px">
  <div style="display:flex;align-items:center;flex-wrap:wrap">
    <div style="font-weight:600;color:#0f172a;flex:1;min-width:200px">{_esc(str(title))}</div>
    {sev_badge}
  </div>
  {sections_html}
  {explanation_html}
</div>""")

    return f"""
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div>
      <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600">Internal Conflicts</span>
      <span style="font-size:11px;color:#94a3b8;margin-left:8px">{len(items)} found</span>
    </div>
    <div style="font-size:11px;color:#94a3b8">⏱ {elapsed:.1f}s</div>
  </div>
  {''.join(cards)}
</div>"""


def render_headlines_and_ranker(headlines_out: dict | None, ranker_out: dict | None,
                                 elapsed_h: float, elapsed_r: float,
                                 errors_h: list, errors_r: list) -> str:
    """Combined renderer for podcast pipeline: headlines + ranker side-by-side."""
    if not headlines_out or not isinstance(headlines_out, dict):
        return _render_empty("Podcast Headlines", errors_h)

    headlines = headlines_out.get("headlines", []) or []
    rankings = (ranker_out or {}).get("rankings", []) or []
    winner = (ranker_out or {}).get("winner") or (rankings[0] if rankings else None)
    winner_explanation = (ranker_out or {}).get("winner_explanation", "")

    # Build a rank lookup so we can show the rank next to each headline
    rank_by_headline = {r.get("headline"): r for r in rankings}

    # WINNER card
    winner_html = ""
    if winner:
        winner_html = f"""
<div style="background:linear-gradient(135deg,#fbbf24 0%,#f59e0b 100%);color:#1f2937;padding:24px;border-radius:14px;margin-bottom:18px;box-shadow:0 8px 24px rgba(251,191,36,0.25)">
  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;opacity:0.7">🏆 Winning Headline</div>
  <div style="font-size:24px;font-weight:800;margin-top:6px;line-height:1.3">{_esc(winner.get('headline', ''))}</div>
  <div style="margin-top:10px;display:flex;gap:14px;flex-wrap:wrap;font-size:12px;font-weight:600">
    <span>📐 Angle: <b>{_esc(winner.get('angle', ''))}</b></span>
    <span>📰 News: <b>{winner.get('newsworthiness_score', '?')}/10</b></span>
    <span>🎯 Specificity: <b>{winner.get('specificity_score', '?')}/10</b></span>
    <span>👂 Appeal: <b>{winner.get('appeal_score', '?')}/10</b></span>
    <span style="background:rgba(31,41,55,0.15);padding:2px 10px;border-radius:10px">⭐ {winner.get('composite_score', '?')}/30</span>
  </div>
  {f'<div style="margin-top:14px;font-size:14px;line-height:1.5;font-style:italic;background:rgba(255,255,255,0.4);padding:12px;border-radius:8px">{_esc(winner_explanation)}</div>' if winner_explanation else ''}
</div>"""

    # All 10 candidates
    cards = []
    sorted_headlines = sorted(
        headlines,
        key=lambda h: rank_by_headline.get(h.get("headline"), {}).get("rank", 99)
    )
    for h in sorted_headlines:
        rank_data = rank_by_headline.get(h.get("headline"), {})
        rank = rank_data.get("rank", "?")
        composite = rank_data.get("composite_score", "?")
        rationale = rank_data.get("rationale", "")
        is_winner = (rank == 1)

        rank_color = "#fbbf24" if rank == 1 else ("#94a3b8" if isinstance(rank, int) and rank <= 3 else "#cbd5e1")

        evidence = h.get("evidence_provisions", []) or []
        evidence_html = ""
        if evidence:
            tags = "".join(
                f"<span style='display:inline-block;background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:10px;font-size:11px;font-family:monospace;margin:2px 4px 2px 0'>{_esc(str(e))[:60]}</span>"
                for e in evidence[:5]
            )
            evidence_html = f"<div style='margin-top:8px'>{tags}</div>"

        rationale_html = ""
        if rationale:
            rationale_html = f"<div style='margin-top:8px;font-size:13px;color:#64748b;font-style:italic;border-left:3px solid #e5e7eb;padding-left:10px'>{_esc(rationale)}</div>"

        border = "border:2px solid #fbbf24" if is_winner else "border:1px solid #e5e7eb"
        cards.append(f"""
<div style="background:#fff;{border};border-radius:10px;padding:14px 16px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-start">
  <div style="background:{rank_color};color:#1f2937;width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:16px;flex-shrink:0">#{rank}</div>
  <div style="flex:1;min-width:0">
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px">
      <div style="font-weight:700;color:#0f172a;font-size:15px;flex:1;min-width:200px">{_esc(h.get('headline', ''))}</div>
      <span style="background:#f1f5f9;color:#475569;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">{_esc(h.get('angle', ''))}</span>
      <span style="background:#1e293b;color:#fbbf24;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700">⭐ {composite}</span>
    </div>
    <div style="margin-top:6px;color:#475569;font-size:13px;line-height:1.5">{_esc(h.get('hook', ''))}</div>
    <div style="margin-top:4px;font-size:11px;color:#94a3b8">🎯 {_esc(h.get('target_audience', ''))}</div>
    {evidence_html}
    {rationale_html}
  </div>
</div>""")

    timing_html = f"""
<div style="display:flex;gap:14px;font-size:12px;color:#64748b;margin-bottom:14px">
  <span>📝 Headline generation: <b>{elapsed_h:.1f}s</b></span>
  <span>•</span>
  <span>⚖️ Ranking: <b>{elapsed_r:.1f}s</b></span>
  <span>•</span>
  <span>{len(headlines)} candidates / {len(rankings)} ranked</span>
</div>"""

    return winner_html + timing_html + "".join(cards)


def _render_empty(name: str, errors: list) -> str:
    err_html = ""
    if errors:
        err_html = f"""
<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;margin-top:12px">
  <div style="font-weight:600;color:#991b1b;font-size:13px;margin-bottom:4px">Errors</div>
  <pre style="margin:0;font-size:12px;color:#7f1d1d;white-space:pre-wrap">{_esc(str(errors))}</pre>
</div>"""
    return f"""
<div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;padding:24px;text-align:center;color:#64748b">
  <div style="font-size:36px;opacity:0.4">📋</div>
  <div style="font-weight:600;margin-top:8px">No {name} output</div>
  {err_html}
</div>"""


def _esc(s) -> str:
    if s is None:
        return ""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                  .replace('"', "&quot;").replace("'", "&#39;"))


# ----------------------------------------------------------------------------
# Top-level pipeline functions called from UI
# ----------------------------------------------------------------------------
def analyze_pdf(pdf_file, progress=gr.Progress()):
    """Streaming generator. Always processes ALL chunks of the PDF and
    aggregates the per-chunk reports into a single merged report.

    Yields UI-tuple updates per agent and per chunk. Final yield includes
    the merged report and saves it to eval/canonical/{bill}-merged.json
    so the Bills Lookup gallery and Podcast Studio pick it up.
    """
    if pdf_file is None:
        empty = "<div style='padding:20px;color:#94a3b8'>No PDF uploaded.</div>"
        yield (empty, "", "", [], "", "", "", "", None,
               "<div style='color:#94a3b8;padding:8px;font-size:13px'>Waiting for PDF upload…</div>")
        return

    progress(0.02, desc="Reading PDF…")
    yield ("", "", "", [], "", "", "", "", None,
           "<div style='color:#475569;padding:8px;font-size:13px'>📄 Reading PDF and chunking on TITLE/Subtitle boundaries…</div>")

    try:
        chunks = chunk_pdf(Path(pdf_file))
    except Exception:
        err = f"<div style='background:#fef2f2;border:1px solid #fecaca;padding:14px;border-radius:8px;color:#7f1d1d'><b>PDF chunking failed:</b><pre>{traceback.format_exc()}</pre></div>"
        yield (err, "", "", [], "", "", "", "", None,
               "<div style='color:#ef4444;padding:8px;font-size:13px'>❌ Chunking failed</div>")
        return

    if not chunks:
        msg = "<div style='padding:20px;color:#ef4444'>No text extracted from PDF.</div>"
        yield (msg, "", "", [], "", "", "", "", None,
               "<div style='color:#ef4444;padding:8px;font-size:13px'>❌ No text extracted</div>")
        return

    chunks_summary = "\n".join(
        f"  {c['chunk_id']}: {c['marker_label'][:55]}  ({c['tokens']:,} tokens, pp.{c['start_page']}-{c['end_page']})"
        for c in chunks
    )

    # Always process every chunk of the bill
    chunks_to_process = chunks
    if len(chunks) > 1:
        mode_label = f"<b>ALL {len(chunks)} chunks</b> (full bill)"
    else:
        mode_label = f"single chunk: <b>{chunks[0]['chunk_id']}</b>"

    # Bill metadata derived from filename (lowercase stem). Strip Windows-
    # style duplicate suffixes like " (1)", "(2)", "-(3)" before slugifying so
    # re-uploads don't collide with canonical reports already on disk.
    import re as _re_dup
    pdf_stem = Path(pdf_file).stem
    pdf_stem = _re_dup.sub(r"[\s\-_]*\(\d+\)\s*$", "", pdf_stem).strip()
    bill_short = pdf_stem.lower().replace(" ", "-").replace("_", "-")
    bill_short = _re_dup.sub(r"-+", "-", bill_short).strip("-")
    bill_meta = {
        "bill_short": bill_short,
        "bill_label": pdf_stem,
        "bill_note": f"Analyzed {len(chunks_to_process)}/{len(chunks)} chunks via Gradio upload",
    }

    # Initial overview shows the FULL bill stats (sum of selected chunks)
    init_pages = [
        min(c["start_page"] for c in chunks_to_process),
        max(c["end_page"] for c in chunks_to_process),
    ]
    init_tokens = sum(c["tokens"] for c in chunks_to_process)
    initial_overview = render_overview({
        "title_marker": (
            f"Full bill ({len(chunks_to_process)} chunks)" if len(chunks_to_process) > 1
            else chunks_to_process[0]["marker_label"]
        ),
        "tokens": init_tokens,
        "pages": init_pages,
        "n_chunks": len(chunks),
        "chunks_processed": [c["chunk_id"] for c in chunks_to_process],
        "totals": {},
        "agents": {},
    }, chunks_summary)

    log_events = [
        f"<div style='color:#475569;padding:6px 0;font-size:13px;border-bottom:1px solid #e5e7eb'>"
        f"📄 PDF chunked: <b>{len(chunks)} chunks</b> | analyzing {mode_label} | "
        f"{init_tokens:,} input tokens | pp.{init_pages[0]}-{init_pages[1]}</div>"
    ]
    yield (initial_overview, "", "", [], "", "", "", "", None, "".join(log_events))

    # === Loop over chunks, run all 6 agents per chunk ===
    per_chunk_reports = []
    n_chunks = len(chunks_to_process)
    for ci, chunk in enumerate(chunks_to_process, start=1):
        log_events.append(
            f"<div style='color:#0f172a;padding:8px 0;font-size:13px;font-weight:700;"
            f"border-top:2px solid #1e3a8a;border-bottom:1px solid #e5e7eb;margin-top:6px'>"
            f"━━ CHUNK {ci}/{n_chunks}: {_esc(chunk['chunk_id'])} · "
            f"{_esc(chunk['marker_label'][:80])} · {chunk['tokens']:,} tokens · "
            f"pp.{chunk['start_page']}-{chunk['end_page']} ━━</div>"
        )
        yield (initial_overview, "", "", [], "", "", "", "", None, "".join(log_events))

        chunk_final = None
        for evt in run_agents_streaming(chunk, progress=progress):
            if evt["type"] == "agent_done":
                errs = evt.get("errors") or []
                ok_icon = "❌" if errs else "✅"
                err_html = f"<span style='color:#ef4444'> · errors: {_esc(str(errs)[:80])}</span>" if errs else ""
                log_events.append(
                    f"<div style='color:#1e293b;padding:6px 0 6px 18px;font-size:13px;border-bottom:1px solid #e5e7eb'>"
                    f"{ok_icon} <b>chunk {ci}/{n_chunks} · [{evt['step']}/{evt['total_steps']}]</b> {_esc(evt['label'])} "
                    f"<span style='color:#64748b'>· {evt['elapsed_s']:.1f}s</span> "
                    f"<span style='color:#64748b'>· {evt['prompt_tokens']:,}p+{evt['completion_tokens']}c</span>"
                    f"{err_html}</div>"
                )
                yield (initial_overview, "", "", [], "", "", "", "", None, "".join(log_events))
            elif evt["type"] == "enrichment_done":
                log_events.append(
                    f"<div style='color:#1e293b;padding:6px 0 6px 18px;font-size:13px;border-bottom:1px solid #e5e7eb'>"
                    f"⚡ chunk {ci}/{n_chunks} · <b>USC enrichment</b> via remote LMDB · {evt['elapsed_ms']:.0f}ms · "
                    f"{evt['stats']['hits']} hits / {evt['stats']['misses']} misses</div>"
                )
                yield (initial_overview, "", "", [], "", "", "", "", None, "".join(log_events))
            elif evt["type"] == "final":
                chunk_final = evt["report"]

        if chunk_final is None:
            log_events.append(
                f"<div style='color:#ef4444;padding:6px 0;font-size:13px'>"
                f"❌ Chunk {ci} produced no final report; continuing</div>"
            )
            yield (initial_overview, "", "", [], "", "", "", "", None, "".join(log_events))
            continue
        per_chunk_reports.append(chunk_final)

    if not per_chunk_reports:
        yield (initial_overview, "", "", [], "", "", "", "", None,
               "".join(log_events) + "<div style='color:#ef4444;padding:6px 0'>❌ No final report produced for any chunk</div>")
        return

    # === Merge per-chunk reports into one canonical report ===
    try:
        from src.multichunk import merge_chunk_reports
    except Exception as _e:
        log_events.append(f"<div style='color:#ef4444'>❌ multichunk import failed: {_esc(str(_e))}</div>")
        yield (initial_overview, "", "", [], "", "", "", "", None, "".join(log_events))
        return

    merged_report = merge_chunk_reports(per_chunk_reports, bill_meta)

    # Save per-chunk + merged JSONs to eval/canonical/ so the Bills Lookup
    # gallery and Podcast Studio dropdown pick them up.
    try:
        CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
        for r in per_chunk_reports:
            cid = r.get("chunk_id", "ch01")
            fname = f"{bill_short}-{cid}.json"
            (CANONICAL_DIR / fname).write_text(json.dumps({**r, **bill_meta}, indent=2), encoding="utf-8")
        if len(per_chunk_reports) > 1:
            (CANONICAL_DIR / f"{bill_short}-merged.json").write_text(
                json.dumps(merged_report, indent=2), encoding="utf-8")
            saved_msg = f"💾 Saved {len(per_chunk_reports)} per-chunk JSONs + merged report → eval/canonical/{bill_short}-merged.json"
        else:
            saved_msg = f"💾 Saved → eval/canonical/{bill_short}-ch01.json"
    except Exception as _e:
        saved_msg = f"⚠️ Save failed: {_esc(str(_e))}"

    log_events.append(
        f"<div style='color:#15803d;padding:8px 0;font-size:13px;font-weight:600;"
        f"border-top:2px solid #15803d;margin-top:6px'>"
        f"🏁 ALL DONE · {len(per_chunk_reports)}/{len(chunks)} chunks processed · "
        f"{merged_report['totals']['wall_clock_s']:.1f}s wall clock · "
        f"{merged_report['totals']['prompt_tokens_total']:,} prompt tokens · "
        f"{merged_report['totals']['completion_tokens_total']:,} completion tokens · "
        f"{saved_msg}</div>"
    )

    overview, summ, xref_h, xref_df, pork, conflict, podcast_html, raw, dl = _build_outputs(
        merged_report, chunks_summary
    )
    yield (overview, summ, xref_h, xref_df, pork, conflict, podcast_html, raw, dl, "".join(log_events))


def load_demo():
    """Load the canonical BBB demo. Uses the real merged report on disk
    (eval/canonical/bbb-merged.json) so the demo always reflects the
    actual current state of the analyzer, including all 6 chunks of the
    full Build Back Better Act, all 10 agents, podcast headlines, ranker,
    and the full Stage-5 podcast pipeline metadata.

    Falls back to the legacy hardcoded ch01 fake_report only if no
    canonical bbb report exists on disk.
    """
    # Preferred path: the real merged report from eval/canonical/.
    # This automatically picks up bbb-merged.json (full 6-chunk bill)
    # via _make_load_bill_fn, so the demo stays in sync with reality.
    canon_merged = CANONICAL_DIR / "bbb-merged.json"
    canon_ch01 = CANONICAL_DIR / "bbb-ch01.json"
    if canon_merged.exists() or canon_ch01.exists():
        return load_bill_by_short("bbb")

    # Fallback: original hardcoded ch01 data (kept so the demo never
    # breaks even if eval/canonical/ has been wiped).
    if DEMO_REPORT is None:
        empty = "<div style='padding:20px;color:#ef4444'>Demo report file not found.</div>"
        return empty, "", "", [], "", "", "", "", None, ""

    # Synthesize a report dict from the canonical JSON
    fake_report = {
        "title_marker": DEMO_REPORT.get("title_marker", "TITLE I-AGRICULTURE"),
        "tokens": DEMO_REPORT.get("tokens", 199381),
        "pages": [3, 542],
        "agents": {
            "summarizer": {
                "label": "Plain-English Summarizer",
                "output": DEMO_REPORT.get("summarizer") or {
                    "one_sentence_summary": "Title I of the Build Back Better Act allocates over $100 billion to forest restoration, rural development, and climate resilience programs across the U.S.",
                    "bullets": [
                        "Allocates $10B for hazardous fuels reduction in the wildland-urban interface to prevent wildfires.",
                        "Provides $4.5B for vegetation management and forest restoration on National Forest System land.",
                        "Authorizes $9B in grants for state, tribal, and nonprofit forest restoration projects.",
                        "Establishes a $2.25B Civilian Climate Corps workforce program.",
                        "Funds $1.25B for the Forest Legacy Program for high-carbon-sequestration land acquisition.",
                        "Allocates $3B for urban and community forestry to address environmental justice.",
                    ],
                    "affected_groups": ["U.S. Forest Service", "Tribal nations", "Rural communities", "State agencies", "Nonprofits"],
                },
                "elapsed_s": 36.2,
                "prompt_tokens": 232853,
                "completion_tokens": 375,
                "errors": [],
            },
            "usc_cross_ref": {
                "label": "USC Cross-Reference",
                "output": DEMO_REPORT.get("usc_cross_reference"),
                "elapsed_s": 91.5,
                "prompt_tokens": 232000,
                "completion_tokens": 1850,
                "errors": [],
                "enrichment": {"elapsed_ms": 145.7, "lmdb_stats": {"hits": 57, "misses": 83, "calls": 140}},
            },
            "pork_finder": {
                "label": "Pork Finder",
                "output": DEMO_REPORT.get("pork_finder"),
                "elapsed_s": 78.4,
                "prompt_tokens": 232000,
                "completion_tokens": 1100,
                "errors": [],
            },
            "conflict_spotter": {
                "label": "Conflict Spotter",
                "output": DEMO_REPORT.get("conflict_spotter"),
                "elapsed_s": 110.9,
                "prompt_tokens": 232000,
                "completion_tokens": 950,
                "errors": [],
            },
        },
        "totals": {"wall_clock_s": 317.0, "prompt_tokens_total": 928853, "completion_tokens_total": 4275},
    }
    return (*_build_outputs(fake_report, "Pre-computed canonical report (Build Back Better Act, Title I) — no live inference"), "")


# ----------------------------------------------------------------------------
# Bills Lookup: load any *.json under eval/canonical/ and render clickable cards
# ----------------------------------------------------------------------------
CANONICAL_DIR = ROOT / "eval" / "canonical"


def _load_canonical_bills() -> list[dict]:
    """Scan eval/canonical/ for bill reports.

    Prefers `{bill}-merged.json` (multi-chunk aggregated view) over
    `{bill}-ch01.json` (single-chunk fallback). One entry per bill.
    """
    if not CANONICAL_DIR.exists():
        return []
    # Find all bill report files; group by inferred bill_short.
    candidates: dict[str, Path] = {}
    # Pass 1: -merged.json (preferred — multi-chunk aggregated)
    for f in sorted(CANONICAL_DIR.glob("*-merged.json")):
        bill = f.stem.replace("-merged", "")
        candidates[bill] = f
    # Pass 2: -ch01.json fallback for bills without a merged report
    for f in sorted(CANONICAL_DIR.glob("*-ch01.json")):
        bill = f.stem.replace("-ch01", "")
        candidates.setdefault(bill, f)

    out = []
    for bill, f in sorted(candidates.items()):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({
                "short": data.get("bill_short", bill),
                "label": data.get("bill_label", f.stem),
                "note": data.get("bill_note", ""),
                "title_marker": data.get("title_marker", "")[:80],
                "tokens": data.get("tokens", 0),
                "wall_clock_s": data.get("totals", {}).get("wall_clock_s", 0),
                "prompt_tokens_total": data.get("totals", {}).get("prompt_tokens_total", 0),
                "completion_tokens_total": data.get("totals", {}).get("completion_tokens_total", 0),
                "winner_headline": (data.get("agents", {}).get("headline_ranker", {}).get("output") or {}).get("winner", {}).get("headline", ""),
                "file": str(f),
            })
        except Exception:
            pass
    return out


def render_bills_lookup_cards() -> str:
    """Render gallery of pre-processed bills."""
    bills = _load_canonical_bills()
    if not bills:
        return (
            "<div style='background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;padding:32px;text-align:center'>"
            "<div style='font-size:32px;opacity:0.4'>📚</div>"
            "<div style='font-weight:600;margin-top:8px;color:#475569'>No pre-processed bills yet</div>"
            "<div style='font-size:12px;color:#94a3b8;margin-top:4px'>Pre-process bills via <code>preprocess_bills.py</code> to populate this gallery.</div>"
            "</div>"
        )
    cards = []
    for b in bills:
        winner = b["winner_headline"][:80] if b["winner_headline"] else ""
        winner_html = f"<div style='margin-top:8px;padding:6px 10px;background:#fef3c7;border-radius:6px;font-size:11px;color:#78350f'><b>🏆</b> {_esc(winner)}</div>" if winner else ""
        cards.append(
            f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;cursor:pointer;transition:transform 0.15s,box-shadow 0.15s' "
            f"onmouseover=\"this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 16px rgba(0,0,0,0.08)'\" "
            f"onmouseout=\"this.style.transform='';this.style.boxShadow=''\">"
            f"<div style='font-size:11px;color:#6366f1;font-weight:700;text-transform:uppercase;letter-spacing:0.05em'>{_esc(b['short'])}</div>"
            f"<div style='font-weight:700;color:#0f172a;font-size:14px;margin-top:4px;line-height:1.3'>{_esc(b['label'])}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:4px'>{_esc(b['title_marker'])}</div>"
            f"<div style='display:flex;gap:8px;font-size:10px;color:#94a3b8;margin-top:8px;flex-wrap:wrap'>"
            f"<span>📄 {b['tokens']:,} tok</span>"
            f"<span>⏱ {b['wall_clock_s']:.0f}s</span>"
            f"<span>🔤 {b['prompt_tokens_total']:,}p+{b['completion_tokens_total']:,}c</span>"
            f"</div>{winner_html}</div>"
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px'>"
        + "".join(cards) + "</div>"
    )


def _make_load_bill_fn(bill_short: str):
    """Factory that returns a closure to load a specific canonical bill report."""
    def _loader():
        bills = _load_canonical_bills()
        match = next((b for b in bills if b["short"] == bill_short), None)
        if not match:
            empty = f"<div style='padding:20px;color:#ef4444'>Canonical report for {bill_short} not found.</div>"
            return empty, "", "", [], "", "", "", "", None, ""
        try:
            report = json.loads(Path(match["file"]).read_text(encoding="utf-8"))
        except Exception as e:
            empty = f"<div style='padding:20px;color:#ef4444'>Failed to load {bill_short}: {e}</div>"
            return empty, "", "", [], "", "", "", "", None, ""
        return (*_build_outputs(report, f"Pre-processed canonical report for {match['label']}"), "")
    return _loader


def load_bill_by_short(bill_short: str):
    """Direct loader used by dropdown selection."""
    print(f"[load_bill_by_short] called with: {bill_short!r}", flush=True)
    if not bill_short:
        return ("", "", "", [], "", "", "", "", None, "")
    return _make_load_bill_fn(bill_short)()


def _build_outputs(report, chunks_summary=""):
    overview = render_overview(report, chunks_summary)
    
    s = report["agents"].get("summarizer", {})
    summarizer_html = render_summarizer(s.get("output"), s.get("elapsed_s", 0), s.get("errors") or [])
    
    x = report["agents"].get("usc_cross_ref", {})
    xref_html, xref_rows = render_xref(x.get("output"), x.get("elapsed_s", 0), x.get("errors") or [], x.get("enrichment"))
    
    p = report["agents"].get("pork_finder", {})
    pork_html = render_pork(p.get("output"), p.get("elapsed_s", 0), p.get("errors") or [])
    
    c = report["agents"].get("conflict_spotter", {})
    conflict_html = render_conflict(c.get("output"), c.get("elapsed_s", 0), c.get("errors") or [])

    h = report["agents"].get("podcast_headlines", {})
    r = report["agents"].get("headline_ranker", {})
    podcast_html = render_headlines_and_ranker(
        h.get("output"), r.get("output"),
        h.get("elapsed_s", 0), r.get("elapsed_s", 0),
        h.get("errors") or [], r.get("errors") or [],
    )
    
    raw = json.dumps(report, indent=2)
    raw_md = "```json\n" + (raw[:30000] + ("\n... (truncated)" if len(raw) > 30000 else "")) + "\n```"
    
    # Save downloadable report
    import tempfile; out_path = Path(tempfile.gettempdir()) / f"report-{report.get('chunk_id', 'demo')}.json"
    try:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        out_path = None

    return overview, summarizer_html, xref_html, xref_rows, pork_html, conflict_html, podcast_html, raw_md, str(out_path) if out_path else None


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* Hero */
#hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #312e81 100%);
  color: white !important;
  padding: 32px 36px;
  border-radius: 14px;
  margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
}
#hero, #hero * { color: white !important; }
#hero h1 { color: white !important; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; }
#hero .subtitle {
  color: #e2e8f0 !important;
  margin-top: 8px;
  font-size: 15px;
  line-height: 1.65;
  max-width: 760px;
}
#hero .subtitle b, #hero .subtitle strong {
  color: #fbbf24 !important;
  font-weight: 700;
  background: rgba(251, 191, 36, 0.12);
  padding: 1px 6px;
  border-radius: 4px;
}
#hero .badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
#hero .badge {
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.22);
  color: #f8fafc !important;
  padding: 6px 14px;
  border-radius: 16px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

/* Section headers */
.section-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  margin: 16px 0 6px 0;
}

/* Tabs */
.tab-nav button { font-weight: 600 !important; }

/* File upload box - readability fix */
.file-preview { color: #1e293b !important; }
[data-testid="file"] { color: #334155 !important; }

/* Prominent file upload — make this the obvious CTA */
#pdf-upload-wrapper {
  background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
  border: 3px dashed #6366f1;
  border-radius: 16px;
  padding: 6px;
  margin-bottom: 6px;
  transition: all 0.2s ease;
}
#pdf-upload-wrapper:hover {
  border-color: #4f46e5;
  background: linear-gradient(135deg, #e0e7ff 0%, #ede9fe 100%);
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
}
#pdf-upload-wrapper [data-testid="file"],
#pdf-upload-wrapper .wrap,
#pdf-upload-wrapper label,
#pdf-upload-wrapper .file-upload {
  font-size: 15px !important;
  font-weight: 700 !important;
  color: #1e293b !important;
}
#pdf-upload-wrapper .upload-button,
#pdf-upload-wrapper button {
  font-size: 14px !important;
  font-weight: 600 !important;
}
.upload-step-label {
  display: inline-block;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white !important;
  padding: 4px 12px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

/* Bill lookup cards — real Gradio buttons styled as cards */
.bill-card-btn button,
button.bill-card-btn {
  text-align: left !important;
  white-space: pre-line !important;
  height: auto !important;
  min-height: 110px !important;
  padding: 14px 16px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  line-height: 1.55 !important;
  background: #fff !important;
  border: 1px solid #e5e7eb !important;
  color: #1e293b !important;
  border-radius: 10px !important;
  transition: all 0.15s ease !important;
}
.bill-card-btn button:hover,
button.bill-card-btn:hover {
  background: #f8fafc !important;
  border-color: #6366f1 !important;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(99, 102, 241, 0.12) !important;
}

/* Hide footer */
footer { display: none !important; }
"""


TOP_BANNER_HTML = """
<div style="background:linear-gradient(90deg,#1e3a8a 0%,#312e81 100%);color:#fff;padding:14px 22px;border-radius:10px;margin:0 0 14px 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;box-shadow:0 4px 14px rgba(30,58,138,0.20)">
  <div style="color:#fff;font-size:14px;font-weight:700">📄 Need a bill to analyze? <span style="color:#e2e8f0;font-weight:400;margin-left:6px">Grab one from Congress.gov</span></div>
  <a href="https://www.congress.gov/most-viewed-bills" target="_blank" rel="noopener" style="background:#fbbf24;color:#1f2937 !important;padding:9px 20px;border-radius:8px;font-weight:700;font-size:13px;text-decoration:none !important;white-space:nowrap">📊 Most-Viewed Bills on Congress.gov &#8689;</a>
</div>
"""

HERO_HTML = """
<div id="hero">
  <h1>📜 10 Agent Bill Analyzer and Podcast Generator</h1>
  <div class="subtitle">
    Upload a US legislative bill PDF. Smart-chunk on TITLE/Subtitle boundaries for larger bills.
    Run <b>10 specialist agents</b> against <b>Qwen3-30B-A3B-Instruct-2507-FP8</b> on a single AMD MI300X.
    <br/><br/>
    <b>6 analysis agents:</b> Summarizer · USC Cross-Reference · Pork Finder · Conflict Spotter · Podcast Headlines · Headline Ranker.
    <br/>
    <b>4 podcast-production agents:</b> Script Writer · Slide Prompt Generator · Wan Motion Prompt Generator · Slide Critic (dual-call OCR + judgment).
  </div>
  <div class="badges">
    <span class="badge">⚡ AMD MI300X · 192 GB VRAM</span>
    <span class="badge">🔗 vLLM ROCm v0.17.1</span>
    <span class="badge">🧠 Qwen3-30B-A3B FP8</span>
    <span class="badge">📚 60K USC sections (LMDB)</span>
    <span class="badge">🎙️ 10 specialist agents (6 analysis + 4 podcast)</span>
    <span class="badge">🏛 lablab.ai AMD Hackathon · May 2026</span>
  </div>
  <div style="margin-top:14px;font-size:13px;color:#cbd5e1">
    <a href="https://www.congress.gov/most-viewed-bills" target="_blank" rel="noopener"
       style="color:#fbbf24 !important;text-decoration:underline;font-weight:600">
      📊 Browse most-viewed bills on congress.gov →
    </a>
  </div>
</div>
"""




# ============================================================================
# CLOUD PODCAST GENERATION (Stage 5 - full all-Qwen pipeline on MI300X)
# ============================================================================
import threading as _threading
import time as _time
from pathlib import Path as _Path

_REPO_DIR = _Path(__file__).parent


def _existing_podcast_path(bill_short):
    p = _REPO_DIR / 'eval' / f'{bill_short}-cloud' / f'final-{bill_short}-cloud-podcast.mp4'
    if p.exists() and p.stat().st_size > 100_000:
        return str(p)
    return None


def _pretty_bill_label(short: str) -> str:
    """Display label for a bill_short slug.

    Most slugs are compact and become clean uppercase: bbb -> BBB,
    border25 -> BORDER25. Long multi-segment slugs like
    'mandami-bills-119hr4692ih' become MANDAMI (just the first segment
    uppercased) since the full title is shown elsewhere.
    """
    if not short:
        return ''
    if '-' not in short:
        return short.upper()
    return short.split('-')[0].upper()


def list_podcastable_bills():
    canon_dir = _REPO_DIR / 'eval' / 'canonical'
    if not canon_dir.exists():
        return []
    bills = set()
    for p in canon_dir.glob('*-merged.json'):
        bills.add(p.stem.replace('-merged', ''))
    for p in canon_dir.glob('*-ch01.json'):
        bills.add(p.stem.replace('-ch01', ''))
    return sorted(bills)


def _canonical_report_path(bill_short: str):
    """Return the merged report if present, else ch01, else None."""
    canon_dir = _REPO_DIR / 'eval' / 'canonical'
    merged = canon_dir / f'{bill_short}-merged.json'
    if merged.exists():
        return merged
    ch01 = canon_dir / f'{bill_short}-ch01.json'
    if ch01.exists():
        return ch01
    return None


def load_headlines_for_bill(bill_short: str):
    """Return (dropdown_choices, default_headline_text) for a bill.

    dropdown_choices is a list of (label, headline_text) tuples for gr.Dropdown.
    Falls back gracefully when the report or rankings are missing.
    """
    if not bill_short:
        return [], ''
    p = _canonical_report_path(bill_short)
    if not p:
        return [], ''
    import json as _json
    try:
        report = _json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return [], ''
    rankings = (report.get('agents', {})
                      .get('headline_ranker', {})
                      .get('output', {})
                      .get('rankings') or [])
    choices = []
    for r in rankings[:10]:
        rank = r.get('rank', '?')
        score = r.get('composite_score', '?')
        headline = r.get('headline', '')
        # short label for the dropdown ; full headline as the value
        short = headline if len(headline) <= 70 else headline[:67] + '...'
        label = f"#{rank} ({score}/30) {short}"
        choices.append((label, headline))
    default = choices[0][1] if choices else ''
    return choices, default


def _political_lean_directive(lean: int) -> str:
    """Translate a -100..+100 political lean slider value into a script-writer
    directive sentence. Returns empty string at neutral (no directive added).

    Buckets are coarse on purpose - five steps map to five distinct tones.
    The directive is prepended to whatever the user typed in the
    creative_direction box, so manual override + slider stack additively.
    """
    try:
        lean = int(lean or 0)
    except (TypeError, ValueError):
        lean = 0
    if lean <= -60:
        return (
            "POLITICAL FRAMING DIRECTIVE: Lean strongly progressive. "
            "Emphasize impacts on workers, low-income families, marginalized "
            "communities, climate, and consumer protections. Foreground "
            "criticisms of corporate power and regressive provisions. Use "
            "language a Pod Save America host would use."
        )
    if lean <= -20:
        return (
            "POLITICAL FRAMING DIRECTIVE: Lean somewhat progressive. "
            "Highlight equity, public-interest, and worker-protection angles "
            "where they exist in the bill, while keeping the tone factual."
        )
    if lean >= 60:
        return (
            "POLITICAL FRAMING DIRECTIVE: Lean strongly conservative. "
            "Emphasize fiscal restraint, federalism / states' rights, "
            "regulatory burden on businesses and individuals, constitutional "
            "limits on federal power, and cost to taxpayers. Foreground "
            "skepticism of expanded mandates. Use language a Wall Street "
            "Journal editorial board member would use."
        )
    if lean >= 20:
        return (
            "POLITICAL FRAMING DIRECTIVE: Lean somewhat conservative. "
            "Highlight fiscal cost, regulatory burden, and federalism "
            "concerns where they exist in the bill, while keeping the tone "
            "factual."
        )
    return ""  # neutral - no directive added


def generate_podcast_handler(bill_short, edited_headline='', creative_direction='', political_lean=0):
    """Streaming generator for the Podcast Studio.

    Yields (log_text, video_path_or_None, master_path_str) every 0.5–0.6s.
    The 3rd value is the REAL master mp4 path (e.g. B:\\...\\final-X.mp4),
    routed directly to the YouTube upload textbox so it shows the actual
    file location instead of Gradio's temp-dir copy of it.

    Three behaviors:
      1. Cached fast-path: if a master mp4 already exists for this exact
         (bill, headline, direction) combo, return it immediately with no
         compute.
      2. Live timer: each yield includes an "ELAPSED M:SS" header that ticks
         every second so the user sees the clock move during long renders.
      3. Streaming: pipeline runs in a daemon thread; new log lines are
         flushed to the UI as they arrive.
    """
    if not _CLOUD_PIPELINE_AVAILABLE:
        yield 'ERROR: cloud pipeline module not loaded.', None, ''
        return
    if not bill_short:
        yield 'Please pick a bill from the dropdown.', None, ''
        return
    if not _canonical_report_path(bill_short):
        yield f"ERROR: no canonical report for '{bill_short}'", None, ''
        return

    headline_arg = (edited_headline or '').strip() or None
    user_direction = (creative_direction or '').strip()
    lean_directive = _political_lean_directive(political_lean)
    # Merge lean directive + user-typed direction. Lean comes first so the
    # script writer agent reads it before any free-form user instruction.
    # Both pieces feed into the cache key (via direction_arg), so changing
    # the dial produces a distinct cached output instead of returning a
    # stale slides-mode video.
    if lean_directive and user_direction:
        direction_arg = f"{lean_directive}\n\n{user_direction}"
    elif lean_directive:
        direction_arg = lean_directive
    elif user_direction:
        direction_arg = user_direction
    else:
        direction_arg = None

    # === FAST PATH: check for cached master mp4 ===
    try:
        from make_podcast_cloud import expected_final_path as _expected_final_path
        cached = _expected_final_path(bill_short, headline_arg, direction_arg)
    except Exception:
        cached = None
    if cached and cached.exists() and cached.stat().st_size > 100_000:
        size_mb = cached.stat().st_size / 1024 / 1024
        msg = (
            f'⏱  ELAPSED 0:00\n\n'
            f'✓ Already generated — playing cached version.\n'
            f'  bill:        {bill_short}\n'
            f'  headline:    {headline_arg or "(auto-ranked winner)"}\n'
            f'  direction:   {direction_arg or "(none)"}\n'
            f'  file:        {cached}\n'
            f'  size:        {size_mb:.1f} MB\n'
            f'\n(Skipping pipeline — no compute needed. To force regeneration, '
            f'delete the eval folder for this combo.)'
        )
        yield msg, str(cached), str(cached)
        return

    # === LIVE PATH: kick off pipeline in a thread, stream progress ===
    log_lines = []
    state = {'final': None, 'done': False, 'error': None}

    def log_cb(msg=''):
        log_lines.append(str(msg) if msg else '')

    def _probe_spine() -> tuple[bool, str]:
        """Pre-flight check: is the spine reachable + responsive?
        Returns (ok, message). A busy spine often shows as a connect timeout."""
        import httpx as _httpx
        try:
            _t0 = _time.time()
            _r = _httpx.get(f"{SPINE.rstrip('/')}/models", timeout=4.0)
            _ms = (_time.time() - _t0) * 1000
            if _r.status_code == 200:
                return True, f"spine reachable ({_ms:.0f}ms)"
            return False, f"spine HTTP {_r.status_code}"
        except _httpx.TimeoutException:
            return False, "spine probe timed out (likely busy with another long request)"
        except _httpx.ConnectError:
            return False, "spine unreachable (droplet down or network issue)"
        except Exception as _e:
            return False, f"spine probe failed: {type(_e).__name__}"

    def _diagnose_error(err_repr: str) -> tuple[str, bool, str]:
        """Categorize an error. Returns (category, retry_recommended, human_msg)."""
        _s = err_repr.lower()
        if '400' in _s and ('http' in _s or 'bad request' in _s):
            return ('BAD_REQUEST', False,
                    "HTTP 400 from spine — likely a chunk too large for the context window. "
                    "Retry won't help; the chunker MAX_TOKENS needs to be lowered.")
        if any(c in _s for c in ('502', '503', '504')) and 'http' in _s:
            return ('TRANSIENT', True, "HTTP 5xx from spine — overloaded or restarting. Will retry.")
        if 'timeout' in _s or 'timed out' in _s:
            return ('TIMEOUT', True, "Request timed out — spine likely busy with another long request. Will retry.")
        if 'connect' in _s and ('error' in _s or 'refused' in _s or 'aborted' in _s):
            return ('NETWORK', True, "Connection error — droplet/network blip. Will retry.")
        return ('UNKNOWN', True, f"Unexpected error. Will retry once.")

    MAX_RETRIES = 2
    RETRY_BACKOFF = [10, 30]  # seconds before attempts 2 and 3

    def worker():
        attempt = 0
        while attempt <= MAX_RETRIES:
            attempt += 1
            # Pre-flight probe so we don't waste setup work on a known-bad spine
            log_cb('')
            log_cb(f'[ATTEMPT {attempt}/{MAX_RETRIES + 1}] Pre-flight spine check...')
            ok, probe_msg = _probe_spine()
            log_cb(f'  {probe_msg}')
            if not ok and attempt <= MAX_RETRIES:
                wait = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                log_cb(f'  spine not ready, waiting {wait}s before retry...')
                _time.sleep(wait)
                continue

            # Run pipeline (cached scenes/clips are skipped on retry — cheap)
            try:
                state['final'] = _cloud_run_pipeline(
                    bill_short,
                    log=log_cb,
                    override_headline=headline_arg,
                    creative_direction=direction_arg,
                )
                state['done'] = True
                return
            except Exception as exc:
                err_repr = repr(exc)
                category, should_retry, diag = _diagnose_error(err_repr)
                log_cb('')
                log_cb(f'PIPELINE ERROR [{category}]: {diag}')
                log_cb(f'  raw exception: {err_repr[:300]}')
                if not should_retry or attempt > MAX_RETRIES:
                    state['error'] = err_repr
                    state['done'] = True
                    return
                wait = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                log_cb(f'  retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES + 1})...')
                _time.sleep(wait)
        state['done'] = True

    t = _threading.Thread(target=worker, daemon=True)
    t.start()
    t0 = _time.time()

    def _fmt_elapsed(s):
        s = int(s)
        if s < 3600:
            return f'{s // 60}:{s % 60:02d}'
        return f'{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}'

    def _render(extra_tail=''):
        elapsed = _time.time() - t0
        header = f'⏱  ELAPSED {_fmt_elapsed(elapsed)}\n\n'
        return header + '\n'.join(log_lines) + extra_tail

    intro = [f'Starting cloud pipeline for {bill_short}...']
    if headline_arg:
        intro.append(f'  override headline: {headline_arg}')
    if direction_arg:
        intro.append(f'  creative direction: {direction_arg[:120]}')
    log_lines.extend(intro)
    yield _render(), None, gr.update()

    last_idx = len(log_lines)
    last_yield_t = _time.time()
    while not state['done']:
        _time.sleep(0.5)
        new_log = len(log_lines) > last_idx
        # Tick the clock at least once per second even if no log lines arrive
        # (Wan i2v renders take 25-49s with no intermediate stdout)
        elapsed_changed = (_time.time() - last_yield_t) >= 1.0
        if new_log or elapsed_changed:
            yield _render(), None, gr.update()
            last_idx = len(log_lines)
            last_yield_t = _time.time()

    # === FINAL FLUSH ===
    final_path = state['final']
    elapsed_total = _time.time() - t0
    if final_path and _Path(str(final_path)).exists():
        log_lines.append(f'\n⏱  Total elapsed: {_fmt_elapsed(elapsed_total)}')
        # 3rd value = master path string, drives YouTube upload textbox directly
        # (bypasses gr.Video's temp-dir copy that would otherwise show up).
        yield _render(), str(final_path), str(final_path)
    else:
        err_suffix = f"\n\n(Pipeline error: {state['error']})" if state['error'] else ''
        yield _render(extra_tail=err_suffix + '\n\n(no final video produced)'), None, gr.update()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Bill Analyzer · AMD Hackathon") as app:
        gr.HTML(TOP_BANNER_HTML)
        gr.HTML(HERO_HTML)

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(
                    '<div style="margin-top:8px"><span class="upload-step-label">📂 Step 1 — Upload</span></div>'
                    '<div style="font-size:14px;color:#0f172a;font-weight:600;margin-bottom:6px">'
                    'Drop a US legislative bill PDF below to run the full 6-agent analysis pipeline.'
                    '</div>'
                )
                with gr.Group(elem_id="pdf-upload-wrapper"):
                    pdf_input = gr.File(
                        label="📄  Drop your bill PDF here  •  or click to browse",
                        file_types=[".pdf"],
                        height=180,
                    )
                gr.HTML(
                    '<div style="font-size:12px;color:#64748b;margin-top:4px;margin-bottom:14px">'
                    '💡 No PDF handy? Click any pre-processed bill below for instant results.'
                    '</div>'
                )
                analyze_btn = gr.Button("🚀 Analyze Full Bill (all chunks)", variant="primary", size="lg")
                # Hidden refs - the BBB-specific demo button and Status button were
                # removed because (a) the bill cards above and the dropdown below
                # already cover canonical bill selection, and (b) the Status badge
                # was based on a broken url.rstrip('/v1') check that always reported
                # spine offline. Keeping these as hidden widgets so existing .click()
                # wirings (line ~1710) don't NameError.
                demo_btn = gr.Button("⚡ Load Canonical Demo", variant="secondary", visible=False)
                health_btn = gr.Button("🔄 Status", variant="secondary", visible=False)
                health_out = gr.HTML(value="", visible=False)

        gr.HTML('<div class="section-label" style="margin-top:14px">📡 Live Progress</div>')
        log_panel = gr.HTML(
            value="<div style='background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;padding:14px;color:#94a3b8;font-size:13px;text-align:center'>Per-agent progress will stream here as analysis runs (timing, token usage, errors).</div>",
            elem_id="log-panel",
        )

        gr.HTML('<div class="section-label" style="margin-top:18px">📊 Analysis Output</div>')
        overview_out = gr.HTML(value="<div style='padding:30px;text-align:center;color:#94a3b8'>Upload a PDF or pick a pre-processed bill above to begin.</div>")

        with gr.Tabs():
            with gr.TabItem("📝 Summary"):
                summarizer_out = gr.HTML()
            with gr.TabItem("⚖️ USC Citations"):
                xref_html_out = gr.HTML()
                gr.HTML('<div class="section-label" style="margin-top:14px">All Citations</div>')
                xref_df_out = gr.Dataframe(
                    headers=["Citation", "Status", "USC Heading", "Bill Context", "USC Text Excerpt"],
                    datatype=["str", "str", "str", "str", "str"],
                    wrap=True,
                    column_widths=["140px", "130px", "200px", "260px", "320px"],
                    interactive=False,
                )
            with gr.TabItem("🥩 Pork & Earmarks"):
                pork_out = gr.HTML()
            with gr.TabItem("⚠️ Conflicts"):
                conflict_out = gr.HTML()
            with gr.TabItem("🎙️ Podcast Pipeline"):
                podcast_out = gr.HTML(value="<div style='padding:24px;text-align:center;color:#94a3b8'>10 podcast headlines + ranking will appear here after analysis runs.</div>")
            with gr.TabItem("🧾 Raw Report"):
                raw_out = gr.Markdown()

        download_out = gr.File(label="📥 Download report.json", interactive=False)

        # ---------------- BILLS LOOKUP (bottom) ----------------
        gr.HTML(
            '<div style="margin-top:32px;border-top:2px solid #e5e7eb;padding-top:20px">'
            '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">'
            '<div><div class="section-label" style="margin:0">📚 Pre-Processed Bills — Click to Load Instantly</div>'
            '<div style="font-size:13px;color:#64748b;margin-top:4px">No live inference. Reports rendered from cached canonical JSON.</div></div>'
            '<a href="https://www.congress.gov/most-viewed-bills" target="_blank" rel="noopener" '
            'style="font-size:13px;color:#6366f1;font-weight:600;text-decoration:none">'
            '📊 most-viewed-bills on congress.gov →</a>'
            '</div></div>'
        )

        # Load all canonical bills at startup; build one Gradio Button per bill.
        # Buttons are real components with click handlers (HTML cards weren't clickable).
        startup_bills = _load_canonical_bills()
        MAX_SLOTS = 12  # pre-allocate; hidden ones won't show
        bill_btns: list[gr.Button] = []

        # Render in a 3-column grid via nested Rows
        for row_start in range(0, MAX_SLOTS, 3):
            with gr.Row():
                for i in range(row_start, min(row_start + 3, MAX_SLOTS)):
                    if i < len(startup_bills):
                        b = startup_bills[i]
                        winner = b['winner_headline'][:80] if b['winner_headline'] else "(no headline yet)"
                        text = (
                            f"📜 [{b['short'].upper()}]  {b['label'][:50]}\n"
                            f"\n🏆 {winner}\n"
                            f"\n⏱ {b['wall_clock_s']:.0f}s   📄 {b['tokens']:,} tok   🔤 {b['prompt_tokens_total']:,}p+{b['completion_tokens_total']:,}c"
                        )
                        btn = gr.Button(value=text, variant="secondary", visible=True,
                                        elem_classes=["bill-card-btn"])
                    else:
                        btn = gr.Button(value="—", variant="secondary", visible=False,
                                        elem_classes=["bill-card-btn"])
                    bill_btns.append(btn)

        with gr.Row():
            bill_dropdown = gr.Dropdown(
                label="Or select by short code",
                choices=[(_pretty_bill_label(b["short"]), b["short"]) for b in startup_bills],
                value=None,
                interactive=True,
                scale=3,
            )
            refresh_lookup_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1, size="sm")

        # ---------------- WIRING ----------------
        outputs_full = [overview_out, summarizer_out, xref_html_out, xref_df_out,
                        pork_out, conflict_out, podcast_out, raw_out, download_out, log_panel]

        health_btn.click(fn=check_endpoints, outputs=health_out)
        # Capture the analyze click so we can chain a .then() refresh after
        # the Podcast Studio widgets are defined further down.
        _analyze_evt = analyze_btn.click(fn=analyze_pdf, inputs=[pdf_input], outputs=outputs_full)
        demo_btn.click(fn=load_demo, outputs=outputs_full)
        bill_dropdown.change(fn=load_bill_by_short, inputs=bill_dropdown, outputs=outputs_full)

        # Wire each bill card button. Closure over short_code via default arg.
        for i, btn in enumerate(bill_btns):
            if i < len(startup_bills):
                short_code = startup_bills[i]["short"]
                btn.click(fn=lambda s=short_code: load_bill_by_short(s), outputs=outputs_full)

        # Refresh button: re-scan canonical/, update button labels & visibility, refresh dropdown.
        def _refresh_lookup():
            updated = _load_canonical_bills()
            updates = []
            for j in range(MAX_SLOTS):
                if j < len(updated):
                    b = updated[j]
                    winner = b['winner_headline'][:80] if b['winner_headline'] else "(no headline yet)"
                    text = (
                        f"📜 [{b['short'].upper()}]  {b['label'][:50]}\n"
                        f"\n🏆 {winner}\n"
                        f"\n⏱ {b['wall_clock_s']:.0f}s   📄 {b['tokens']:,} tok   🔤 {b['prompt_tokens_total']:,}p+{b['completion_tokens_total']:,}c"
                    )
                    updates.append(gr.update(value=text, visible=True))
                else:
                    updates.append(gr.update(visible=False))
            updates.append(gr.update(choices=[(_pretty_bill_label(b["short"]), b["short"]) for b in updated]))
            return updates

        refresh_lookup_btn.click(
            fn=_refresh_lookup,
            outputs=bill_btns + [bill_dropdown],
        )

        # ====================================================================
        # PODCAST STUDIO - one-click cloud podcast video generation
        # ====================================================================
        gr.Markdown("## Podcast Studio")
        gr.Markdown(
            'Generate a 2-3 minute podcast video for any pre-processed bill. '
            'Slides are made with **Qwen-Image** (4-step Lightning), animated '
            'with **Wan 2.2 i2v**, narrated with **Qwen3-TTS** voices Alex/Jordan, '
            'and quality-checked by a dual OCR + judgment **Qwen3-VL** critic - all on **AMD MI300X**.'
        )

        # --- STEP 1: Bill picker (prominent) ---
        gr.HTML(
            '<div style="margin: 14px 0 6px 0; padding: 10px 14px; '
            'background: linear-gradient(90deg,#3b82f6 0%,#6366f1 100%); '
            'border-radius: 8px; color: #fff; font-size: 18px; font-weight: 700;">'
            '📋 Step 1 — Pick a Bill'
            '</div>'
        )
        _initial_bill_pairs = [(_pretty_bill_label(b), b) for b in list_podcastable_bills()]
        _initial_bills = [v for _l, v in _initial_bill_pairs]
        _initial_bill = (_initial_bills or [None])[0]
        _initial_choices, _initial_headline = load_headlines_for_bill(_initial_bill) if _initial_bill else ([], '')

        with gr.Row():
            podcast_bill_dropdown = gr.Dropdown(
                choices=_initial_bill_pairs,
                label='Bill (uppercase = pre-processed canonical bill)',
                interactive=True,
                value=_initial_bill,
                scale=4,
                container=True,
            )
            podcast_refresh_btn = gr.Button('🔄 Refresh List', scale=1, size='lg')

        # --- STEP 2: Pick a headline (ARMS, does NOT fire pipeline) ---
        gr.HTML(
            '<div style="margin: 18px 0 6px 0; padding: 10px 14px; '
            'background: linear-gradient(90deg,#0ea5e9 0%,#3b82f6 100%); '
            'border-radius: 8px; color: #fff; font-size: 18px; font-weight: 700;">'
            '📜 Step 2 — Pick a Headline (sets the topic — does not start the render)'
            '</div>'
        )
        gr.Markdown(
            "Clicking a headline below **arms** it. Review, then click the big "
            "Generate button in Step 3 to start the render. If a video already "
            "exists for this exact (bill, headline, direction) combo, it plays "
            "instantly with no compute."
        )
        MAX_HEADLINE_SLOTS = 10
        podcast_hdl_btns: list[gr.Button] = []
        # Render in a 2-column grid (5 rows of 2)
        for _row_start in range(0, MAX_HEADLINE_SLOTS, 2):
            with gr.Row():
                for _col in range(2):
                    _idx = _row_start + _col
                    if _idx < len(_initial_choices):
                        _label, _ = _initial_choices[_idx]
                        _visible = True
                    else:
                        _label = '(no headline)'
                        _visible = False
                    _btn = gr.Button(
                        _label,
                        variant='secondary',
                        size='sm',
                        visible=_visible,
                        elem_classes=['headline-btn'],
                    )
                    podcast_hdl_btns.append(_btn)

        # --- STEP 3: Generate (the only thing that actually fires the pipeline) ---
        gr.HTML(
            '<div style="margin: 18px 0 6px 0; padding: 10px 14px; '
            'background: linear-gradient(90deg,#16a34a 0%,#22c55e 100%); '
            'border-radius: 8px; color: #fff; font-size: 18px; font-weight: 700;">'
            '🎙️ Step 3 — Generate (this is what kicks off the render)'
            '</div>'
        )
        # Initial label includes the auto-armed #1 ranked headline so the user
        # sees exactly what will fire. Updated dynamically when headline buttons
        # are clicked or when the textbox is edited.
        _initial_btn_label = (
            f'🎙️ Generate Podcast Video — armed: "{_initial_headline[:80]}"'
            if _initial_headline else
            '🎙️ Generate Podcast Video (pick a headline above first)'
        )
        podcast_generate_btn = gr.Button(
            _initial_btn_label,
            variant='primary',
            size='lg',
            elem_id='podcast-generate-btn',
        )

        # --- Advanced overrides (collapsed) ---
        with gr.Accordion("Advanced: edit headline / add creative direction", open=False):
            podcast_headline_text = gr.Textbox(
                label='Final headline used by script writer (editable)',
                value=_initial_headline,
                lines=2,
                interactive=True,
                info='Auto-filled from the #1 ranked headline. Clicking a button in Step 2 overwrites this. Edit freely — typos and all.',
            )
            podcast_direction = gr.Textbox(
                label='Additional creative direction (optional)',
                value='',
                lines=3,
                interactive=True,
                placeholder='e.g. "Focus on the surveillance angle and the 4th Amendment risks. Keep tone dry and journalistic."',
                info='Prepended to the bill analysis context the script writer sees. Leave blank to use defaults.',
            )
            political_lean = gr.Slider(
                minimum=-100,
                maximum=100,
                value=0,
                step=10,
                label='🎚️  Political Lean (script tone)',
                info='-100 = strongly progressive framing  •  0 = neutral journalism  •  +100 = strongly conservative framing. Augments the creative direction sent to the script writer agent.',
            )

        # --- Output: progress + video ---
        with gr.Row():
            with gr.Column(scale=2):
                podcast_log = gr.Textbox(
                    label='Pipeline progress (live timer)',
                    lines=22,
                    max_lines=40,
                    interactive=False,
                    autoscroll=True,
                    value='',
                )
            with gr.Column(scale=3):
                podcast_video = gr.Video(
                    label='Final podcast video',
                    interactive=False,
                    height=400,
                )

        # ====================================================================
        # STEP 4 (optional): Upload to YouTube @DeadAirBroadcasting
        # ====================================================================
        # Lazy-import status check so a missing client_secret.json doesn't
        # crash the app, just disables the upload section.
        try:
            from src.tools.youtube_uploader import is_available as _yt_is_available
            _yt_ok, _yt_reason = _yt_is_available()
        except Exception as _yt_e:
            _yt_ok, _yt_reason = False, f'youtube_uploader module failed to import: {_yt_e!r}'

        gr.HTML(
            '<div style="margin: 18px 0 6px 0; padding: 10px 14px; '
            'background: linear-gradient(90deg,#dc2626 0%,#ef4444 100%); '
            'border-radius: 8px; color: #fff; font-size: 18px; font-weight: 700;">'
            '📺 Step 4 (optional) — Upload to YouTube @DeadAirBroadcasting'
            '</div>'
        )
        with gr.Accordion(
            f"YouTube upload — {'READY' if _yt_ok else 'SETUP NEEDED'}",
            open=False,
        ):
            if not _yt_ok:
                gr.Markdown(
                    f"**One-time setup required.** {_yt_reason}\n\n"
                    f"See `secrets/README.md` for the complete walkthrough. Short version:\n"
                    f"1. Create a Google Cloud project + enable **YouTube Data API v3**\n"
                    f"2. Create an OAuth client (**Desktop app**), download the JSON\n"
                    f"3. Save as `secrets/client_secret.json`\n"
                    f"4. From the repo root run `python scripts/youtube_auth.py` (one-time browser flow)\n"
                    f"5. Restart the app — this section will switch to **READY**\n"
                )
            else:
                gr.Markdown(
                    "Uploads route to **@DeadAirBroadcasting**. "
                    "Default privacy is **private** — only the channel owner can see the upload until you flip it to unlisted/public on YouTube. "
                    "Title / description / tags are auto-generated by the `YouTubeMetadataGenerator` agent from the bill report."
                )

            with gr.Row():
                yt_video_path = gr.Textbox(
                    label='Video file to upload',
                    placeholder='(auto-fills from the rendered video above — or paste a path)',
                    scale=4,
                    interactive=True,
                )
                yt_privacy = gr.Dropdown(
                    choices=[('🔒 Private (default)', 'private'),
                             ('🔗 Unlisted', 'unlisted'),
                             ('🌍 Public', 'public')],
                    value='private',
                    label='Privacy',
                    scale=1,
                    interactive=True,
                )

            yt_generate_meta_btn = gr.Button(
                '🤖  Generate title/description/tags from bill report',
                variant='secondary',
                size='sm',
            )
            yt_title = gr.Textbox(
                label='Title (max 100 chars)',
                value='',
                lines=1,
                interactive=True,
            )
            yt_description = gr.Textbox(
                label='Description (max 5000 chars)',
                value='',
                lines=6,
                interactive=True,
            )
            yt_tags = gr.Textbox(
                label='Tags (comma-separated, max 30 tags)',
                value='',
                lines=1,
                interactive=True,
                placeholder='e.g. legislation, congress, bill analysis, AI explainer',
            )

            yt_upload_btn = gr.Button(
                '📤  Upload to @DeadAirBroadcasting',
                variant='primary',
                size='lg',
                interactive=_yt_ok,
            )
            yt_status = gr.Textbox(
                label='Upload progress',
                lines=10,
                max_lines=20,
                interactive=False,
                autoscroll=True,
                value='' if _yt_ok else '(complete the one-time setup above to enable uploads)',
            )

        # Auto-fill yt_video_path is now driven directly by the pipeline
        # generator's 3rd yield value (the master path). The old approach
        # (podcast_video.change) gave us Gradio's temp-dir copy of the file,
        # which is confusing for the user. By writing the master path
        # directly to the textbox we get the real B:\eval\... path.

        # Auto-generate metadata via YouTubeMetadataGenerator agent.
        def _yt_generate_metadata(bill_short, headline_text, direction_text):
            """Run the YouTubeMetadataGenerator agent on the canonical report."""
            if not bill_short:
                return gr.update(), gr.update(), gr.update(), 'Pick a bill first.'
            canon = _canonical_report_path(bill_short)
            if canon is None:
                return gr.update(), gr.update(), gr.update(), f'No canonical report for {bill_short}'
            try:
                from src.tools.youtube_uploader import build_metadata_from_report
                meta = build_metadata_from_report(canon, headline_text or '', direction_text or '')
                tags_str = ', '.join(meta['tags'])
                msg = (f'✓ Generated metadata via YouTubeMetadataGenerator:\n'
                       f'  title:   {len(meta["title"])} chars\n'
                       f'  desc:    {len(meta["description"])} chars\n'
                       f'  tags:    {len(meta["tags"])} tags')
                return (
                    gr.update(value=meta['title']),
                    gr.update(value=meta['description']),
                    gr.update(value=tags_str),
                    msg,
                )
            except Exception as exc:
                return gr.update(), gr.update(), gr.update(), f'Metadata generation failed: {exc!r}'

        yt_generate_meta_btn.click(
            fn=_yt_generate_metadata,
            inputs=[podcast_bill_dropdown, podcast_headline_text, podcast_direction],
            outputs=[yt_title, yt_description, yt_tags, yt_status],
        )

        # The actual upload — streaming generator so progress ticks live.
        def _yt_do_upload(video_path, title, description, tags_str, privacy):
            if not video_path:
                yield 'No video to upload — render one first or paste a path.'
                return
            from pathlib import Path as _P
            video_p = _P(str(video_path))
            if not video_p.exists():
                yield f'Video file not found: {video_p}'
                return
            try:
                from src.tools.youtube_uploader import upload_video, is_available
                ok, reason = is_available()
                if not ok:
                    yield f'YouTube upload not configured: {reason}'
                    return
            except Exception as exc:
                yield f'youtube_uploader import failed: {exc!r}'
                return

            # Run upload in a worker thread so we can stream progress
            log_lines = []
            state = {'done': False, 'result': None, 'error': None}
            def log_cb(msg):
                log_lines.append(str(msg))
            def worker():
                try:
                    state['result'] = upload_video(
                        video_path=video_p,
                        title=(title or 'Bill Analyzer Podcast').strip(),
                        description=(description or '').strip(),
                        tags=[t.strip() for t in (tags_str or '').split(',') if t.strip()],
                        privacy=privacy or 'private',
                        log=log_cb,
                    )
                except Exception as exc:
                    state['error'] = repr(exc)
                    log_lines.append(f'UPLOAD FAILED: {exc!r}')
                finally:
                    state['done'] = True

            t = _threading.Thread(target=worker, daemon=True)
            t.start()
            log_lines.append(f'Starting upload of {video_p.name}...')
            yield '\n'.join(log_lines)
            last_idx = len(log_lines)
            while not state['done']:
                _time.sleep(0.5)
                if len(log_lines) > last_idx:
                    yield '\n'.join(log_lines)
                    last_idx = len(log_lines)
            # Final flush
            if state['result']:
                r = state['result']
                log_lines.append('')
                log_lines.append(f'✓ DONE — watch: {r["watch_url"]}')
                log_lines.append(f'  privacy: {r["privacy"]}')
                log_lines.append(f'  size:    {r["file_size_mb"]} MB')
                log_lines.append(f'  elapsed: {r["elapsed_s"]}s')
            yield '\n'.join(log_lines)

        yt_upload_btn.click(
            fn=_yt_do_upload,
            inputs=[yt_video_path, yt_title, yt_description, yt_tags, yt_privacy],
            outputs=[yt_status],
        )

        # === WIRING ===
        # The Generate button is the ONLY thing that fires the pipeline. It uses
        # whatever is in podcast_headline_text + podcast_direction (the textbox
        # is auto-populated when the user clicks a Step 2 button or when the
        # bill changes).
        podcast_generate_btn.click(
            fn=generate_podcast_handler,
            inputs=[podcast_bill_dropdown, podcast_headline_text, podcast_direction, political_lean],
            outputs=[podcast_log, podcast_video, yt_video_path],
        )

        # Helper: build the Generate-button label for a given armed headline.
        def _gen_btn_label_for(hdl):
            if hdl and hdl.strip():
                return f'🎙️ Generate Podcast Video — armed: "{hdl.strip()[:80]}"'
            return '🎙️ Generate Podcast Video (pick a headline above first)'

        # Step 2 buttons: click ARMS the headline AND immediately fires the
        # pipeline (one-click generate). The Step 3 button still exists for
        # users who want to edit the textbox manually before firing.
        def _make_arm(idx):
            def _arm(bill_short):
                choices, _default = load_headlines_for_bill(bill_short)
                if idx >= len(choices):
                    return gr.update(), gr.update()
                _label, hdl = choices[idx]
                return gr.update(value=hdl), gr.update(value=_gen_btn_label_for(hdl))
            return _arm

        for _i, _btn in enumerate(podcast_hdl_btns):
            # Chain: arm the textbox/button label first, THEN fire the pipeline.
            # The .then() guarantees the textbox is updated before generate
            # reads it (Gradio runs .then() handlers serially in declaration order).
            _btn.click(
                fn=_make_arm(_i),
                inputs=[podcast_bill_dropdown],
                outputs=[podcast_headline_text, podcast_generate_btn],
            ).then(
                fn=generate_podcast_handler,
                inputs=[podcast_bill_dropdown, podcast_headline_text, podcast_direction, political_lean],
                outputs=[podcast_log, podcast_video, yt_video_path],
            )

        # Sync Generate button label when the user types in the editable textbox.
        podcast_headline_text.change(
            fn=lambda hdl: gr.update(value=_gen_btn_label_for(hdl)),
            inputs=[podcast_headline_text],
            outputs=[podcast_generate_btn],
        )

        def _on_bill_change(bill_short):
            """Bill changed: rebuild headline button labels + reset textbox + Generate label."""
            choices, default = load_headlines_for_bill(bill_short)
            btn_updates = []
            for j in range(MAX_HEADLINE_SLOTS):
                if j < len(choices):
                    label, _ = choices[j]
                    btn_updates.append(gr.update(value=label, visible=True))
                else:
                    btn_updates.append(gr.update(visible=False))
            return btn_updates + [
                gr.update(value=default),
                gr.update(value=_gen_btn_label_for(default)),
            ]

        podcast_bill_dropdown.change(
            fn=_on_bill_change,
            inputs=[podcast_bill_dropdown],
            outputs=podcast_hdl_btns + [podcast_headline_text, podcast_generate_btn],
        )

        def _refresh_podcast_bills():
            """Refresh button: re-scan eval/canonical/, repopulate everything."""
            choices = list_podcastable_bills()
            choice_pairs = [(_pretty_bill_label(b), b) for b in choices]
            new_bill = (choices or [None])[0]
            hdl_choices, hdl_default = load_headlines_for_bill(new_bill) if new_bill else ([], '')
            btn_updates = []
            for j in range(MAX_HEADLINE_SLOTS):
                if j < len(hdl_choices):
                    label, _ = hdl_choices[j]
                    btn_updates.append(gr.update(value=label, visible=True))
                else:
                    btn_updates.append(gr.update(visible=False))
            return [
                gr.update(choices=choice_pairs, value=new_bill),
                *btn_updates,
                gr.update(value=hdl_default),
                gr.update(value=_gen_btn_label_for(hdl_default)),
            ]

        podcast_refresh_btn.click(
            fn=_refresh_podcast_bills,
            outputs=[podcast_bill_dropdown] + podcast_hdl_btns + [podcast_headline_text, podcast_generate_btn],
        )

        # ====================================================================
        # AUTO-REFRESH AFTER ANALYZE
        # When analyze_pdf finishes, refresh every widget that depends on
        # eval/canonical/ so the just-analyzed bill appears immediately:
        #   - 12 bill_btns (lookup gallery)
        #   - bill_dropdown (lookup dropdown)
        #   - podcast_bill_dropdown (Podcast Studio)
        #   - 10 podcast_hdl_btns (Podcast Studio headlines)
        #   - podcast_headline_text (advanced editable headline)
        # The newly-analyzed bill is auto-selected (most-recent file mtime).
        # ====================================================================
        def _post_analyze_refresh():
            # Lookup gallery: all canonical bills
            updated = _load_canonical_bills()
            lookup_btn_updates = []
            for j in range(MAX_SLOTS):
                if j < len(updated):
                    b = updated[j]
                    winner = b['winner_headline'][:80] if b['winner_headline'] else "(no headline yet)"
                    text = (
                        f"📜 [{b['short'].upper()}]  {b['label'][:50]}\n"
                        f"\n🏆 {winner}\n"
                        f"\n⏱ {b['wall_clock_s']:.0f}s   📄 {b['tokens']:,} tok   "
                        f"🔤 {b['prompt_tokens_total']:,}p+{b['completion_tokens_total']:,}c"
                    )
                    lookup_btn_updates.append(gr.update(value=text, visible=True))
                else:
                    lookup_btn_updates.append(gr.update(visible=False))
            lookup_dropdown_update = gr.update(choices=[(_pretty_bill_label(b["short"]), b["short"]) for b in updated])

            # Podcast Studio: pick the most-recently-modified bill (= the one
            # the user just analyzed) so it's auto-selected.
            pod_bills = list_podcastable_bills()
            most_recent_bill = pod_bills[0] if pod_bills else None
            try:
                from pathlib import Path as _P
                canon = _REPO_DIR / 'eval' / 'canonical'
                if canon.exists() and pod_bills:
                    def _mtime(short):
                        merged = canon / f"{short}-merged.json"
                        ch01 = canon / f"{short}-ch01.json"
                        f = merged if merged.exists() else ch01
                        return f.stat().st_mtime if f.exists() else 0
                    most_recent_bill = max(pod_bills, key=_mtime)
            except Exception:
                pass

            hdl_choices, hdl_default = (
                load_headlines_for_bill(most_recent_bill) if most_recent_bill else ([], '')
            )
            # Capitalized (label, value) pairs for the podcast dropdown.
            pod_pairs = [(_pretty_bill_label(b), b) for b in pod_bills]
            pod_dropdown_update = gr.update(choices=pod_pairs, value=most_recent_bill)
            pod_btn_updates = []
            for j in range(MAX_HEADLINE_SLOTS):
                if j < len(hdl_choices):
                    label, _ = hdl_choices[j]
                    pod_btn_updates.append(gr.update(value=label, visible=True))
                else:
                    pod_btn_updates.append(gr.update(visible=False))
            pod_text_update = gr.update(value=hdl_default)
            # Generate button label reflects the auto-armed #1 headline of the
            # newly-analyzed bill (or empty state if no headline available).
            pod_gen_btn_update = gr.update(value=_gen_btn_label_for(hdl_default))

            return (
                lookup_btn_updates                    # 12 lookup gallery buttons
                + [lookup_dropdown_update]            # 1 lookup dropdown
                + [pod_dropdown_update]               # 1 podcast bill dropdown
                + pod_btn_updates                     # 10 headline buttons
                + [pod_text_update]                   # 1 advanced headline textbox
                + [pod_gen_btn_update]                # 1 Generate button label
            )

        _analyze_evt.then(
            fn=_post_analyze_refresh,
            outputs=(
                bill_btns                              # 12
                + [bill_dropdown]                      # 1
                + [podcast_bill_dropdown]              # 1
                + podcast_hdl_btns                     # 10
                + [podcast_headline_text]              # 1
                + [podcast_generate_btn]               # 1
            ),
        )


    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CSS,
    )
