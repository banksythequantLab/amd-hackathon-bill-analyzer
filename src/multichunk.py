"""Multi-chunk report merging.

When a bill has multiple chunks (e.g., BBB has 5), each chunk is analyzed by
the same 6 agents. This module merges N per-chunk reports into a single
canonical-shaped report so the existing renderers work without changes.

Single chunk: returns input unchanged (with bill_meta added).
N chunks: concatenates per-chunk outputs, sums totals, picks global winner.
"""
from __future__ import annotations
from typing import Any


def _merge_summarizer(per_outputs: list[tuple[str, str, dict]]) -> dict:
    """Each output: {one_sentence_summary, bullets, key_provisions}."""
    summaries = []
    bullets = []
    provisions = []
    for cid, tm, out in per_outputs:
        out = out or {}
        if out.get("one_sentence_summary"):
            summaries.append(f"[{cid} - {tm[:40]}] {out['one_sentence_summary']}")
        for b in (out.get("bullets") or []):
            bullets.append(f"[{cid}] {b}")
        for p in (out.get("key_provisions") or []):
            provisions.append({**p, "_chunk": cid})
    return {
        "chunk_id": "merged",
        "title_marker": f"Full bill ({len(per_outputs)} chunks merged)",
        "one_sentence_summary": " ".join(summaries),
        "bullets": bullets,
        "key_provisions": provisions,
    }


def _merge_list_field(per_outputs, field_name) -> dict:
    """For agents whose output has a single list field (citations/items/conflicts/headlines)."""
    items = []
    for cid, tm, out in per_outputs:
        for it in ((out or {}).get(field_name) or []):
            if isinstance(it, dict):
                items.append({**it, "_chunk": cid})
            else:
                items.append({"value": it, "_chunk": cid})
    return {
        "chunk_id": "merged",
        "title_marker": f"Full bill ({len(per_outputs)} chunks merged)",
        field_name: items,
    }


def _merge_headline_ranker(per_outputs) -> dict:
    """Concatenate rankings, sort by composite_score, pick global winner."""
    all_rankings = []
    for cid, tm, out in per_outputs:
        for r in ((out or {}).get("rankings") or []):
            all_rankings.append({**r, "_chunk": cid, "_chunk_section": tm[:60]})
    # Sort by composite score
    all_rankings.sort(key=lambda r: r.get("composite_score", 0), reverse=True)
    for i, r in enumerate(all_rankings):
        r["rank"] = i + 1
    return {
        "chunk_id": "merged",
        "title_marker": f"Full bill ({len(per_outputs)} chunks merged)",
        "rankings": all_rankings,
        "winner": all_rankings[0] if all_rankings else None,
        "winner_explanation": (
            f"Selected by composite score across {len(per_outputs)} chunks. "
            f"Winning headline came from {all_rankings[0].get('_chunk', '?')} "
            f"({all_rankings[0].get('_chunk_section', '?')})."
        ) if all_rankings else "",
    }


_AGENT_MERGE_FNS = {
    "summarizer": _merge_summarizer,
    "usc_cross_ref": lambda po: _merge_list_field(po, "citations"),
    "pork_finder": lambda po: _merge_list_field(po, "items"),
    "conflict_spotter": lambda po: _merge_list_field(po, "conflicts"),
    "podcast_headlines": lambda po: _merge_list_field(po, "headlines"),
    "headline_ranker": _merge_headline_ranker,
}


def merge_chunk_reports(per_chunk_reports: list[dict], bill_meta: dict) -> dict:
    """Merge N per-chunk reports into one canonical-shaped report.
    
    bill_meta: {bill_short, bill_label, bill_note (optional)}
    """
    if not per_chunk_reports:
        return {**bill_meta, "agents": {}, "totals": {}, "n_chunks": 0}

    if len(per_chunk_reports) == 1:
        out = dict(per_chunk_reports[0])
        out.update(bill_meta)
        out.setdefault("n_chunks", 1)
        return out

    first = per_chunk_reports[0]
    merged = {
        **bill_meta,
        "chunk_id": "merged",
        "n_chunks": len(per_chunk_reports),
        "chunks_processed": [r.get("chunk_id", "?") for r in per_chunk_reports],
        "title_marker": f"Full bill ({len(per_chunk_reports)} chunks merged)",
        "tokens": sum(r.get("tokens", 0) for r in per_chunk_reports),
        "pages": [
            min(r.get("pages", [0, 0])[0] for r in per_chunk_reports),
            max(r.get("pages", [0, 0])[1] for r in per_chunk_reports),
        ],
        "agents": {},
        "timings_s": {},
        "totals": {
            "wall_clock_s": round(sum(r.get("totals", {}).get("wall_clock_s", 0) for r in per_chunk_reports), 2),
            "prompt_tokens_total": sum(r.get("totals", {}).get("prompt_tokens_total", 0) for r in per_chunk_reports),
            "completion_tokens_total": sum(r.get("totals", {}).get("completion_tokens_total", 0) for r in per_chunk_reports),
        },
        "_per_chunk_summary": [
            {"chunk_id": r.get("chunk_id"), "title_marker": r.get("title_marker"),
             "tokens": r.get("tokens"), "pages": r.get("pages")}
            for r in per_chunk_reports
        ],
    }

    for agent_key, merge_fn in _AGENT_MERGE_FNS.items():
        per_outputs = []
        sum_p, sum_c, sum_t = 0, 0, 0.0
        all_errors = []
        label = None
        for r in per_chunk_reports:
            ad = (r.get("agents") or {}).get(agent_key)
            if not ad:
                continue
            label = label or ad.get("label", agent_key)
            sum_p += ad.get("prompt_tokens", 0)
            sum_c += ad.get("completion_tokens", 0)
            sum_t += ad.get("elapsed_s", 0)
            for e in (ad.get("errors") or []):
                all_errors.append(f"[{r.get('chunk_id', '?')}] {e}")
            if ad.get("output"):
                per_outputs.append((r.get("chunk_id", "?"), r.get("title_marker", ""), ad["output"]))

        if not per_outputs:
            continue

        merged["agents"][agent_key] = {
            "label": (label or agent_key) + f" (merged x{len(per_outputs)})",
            "output": merge_fn(per_outputs),
            "elapsed_s": round(sum_t, 2),
            "prompt_tokens": sum_p,
            "completion_tokens": sum_c,
            "errors": all_errors,
            "_per_chunk": [
                {"chunk_id": cid, "title_marker": tm, "output": o}
                for cid, tm, o in per_outputs
            ],
        }
        merged["timings_s"][agent_key] = round(sum_t, 2)

    return merged
