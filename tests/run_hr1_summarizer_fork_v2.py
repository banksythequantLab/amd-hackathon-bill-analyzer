"""
Run the Plain-English Summarizer over all HR1 chunks against the spine
alias (now backed by qwen3:30b-a3b-instruct-2507-q4_K_M, NO thinking).

This is the v2 of the fork test:
  - v1 (qwen3:30b-thinking-2507): ch04 JSON parse failed because
    thinking blocks ate the full 1500-token output budget. ch01
    truncated mid-JSON and lost affected_groups.
  - v2 (this script, instruct-2507): no thinking blocks at all.
  - Plus the chunker now splits TITLE VII at CHAPTER markers, so we
    have 7 chunks (all under 64K ctx) instead of 5 (one over).

Output: eval/hr1-summarizer-3090-fork-v2.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.summarizer import PlainEnglishSummarizer  # noqa: E402
from src.chunking.smart_chunker import (  # noqa: E402
    extract_text_with_pages,
    find_boundaries,
    pack_chunks,
)
import tiktoken  # noqa: E402


HR1_PDF  = ROOT / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"
BASELINE = ROOT / "eval" / "report-hr1-ch01.json"
OUT_FILE = ROOT / "eval" / "hr1-summarizer-3090-fork-v2.json"

MAX_INPUT_CL100K = 50_500  # leaves ~5K qwen-tokens headroom in 64K ctx


def load_baseline_summarizer() -> dict:
    with open(BASELINE, "r", encoding="utf-8") as f:
        data = json.load(f)
    s = data["agents"]["summarizer"]
    return {
        "chunk_id": data["chunk_id"],
        "title_marker": data["title_marker"],
        "chunk_tokens_cl100k": data["chunk_tokens_cl100k"],
        "summarizer_output": s.get("output", {}),
        "elapsed_s": s.get("elapsed_s"),
        "prompt_tokens": s.get("prompt_tokens"),
        "completion_tokens": s.get("completion_tokens"),
        "note": s.get("note"),
    }


def main() -> int:
    print(f"[INFO] chunking {HR1_PDF.name} at --max-tokens {MAX_INPUT_CL100K}")
    text, page_starts = extract_text_with_pages(HR1_PDF)
    enc = tiktoken.get_encoding("cl100k_base")
    boundaries = find_boundaries(text, page_starts)
    chunks = pack_chunks(text, boundaries, page_starts, MAX_INPUT_CL100K, enc)
    print(f"[INFO] {len(chunks)} chunks")
    for c in chunks:
        print(f"   {c.chunk_id} pp.{c.start_page}-{c.end_page} "
              f"tok={c.tokens:>7,} | {c.marker[:55]}")
    print()

    baseline = load_baseline_summarizer()
    print("[INFO] AMD baseline (HR1 ch01 TITLE I, full 206K tokens, FP8 spine):")
    print(f"   prompt_tokens:     {baseline['prompt_tokens']:,}")
    print(f"   completion_tokens: {baseline['completion_tokens']}")
    print(f"   elapsed_s:         {baseline['elapsed_s']}")
    print(f"   bullets:           {len(baseline['summarizer_output'].get('bullets', []))}")
    print(f"   groups:            {len(baseline['summarizer_output'].get('affected_groups', []))}")
    print()

    agent = PlainEnglishSummarizer()
    results = []

    for c in chunks:
        if c.tokens > MAX_INPUT_CL100K + 2_000:
            print(f"[SKIP] {c.chunk_id} {c.marker[:40]} -- {c.tokens:,} cl100k "
                  f"tokens still over 64K ctx (post-chunker fix)")
            results.append({
                "chunk_id": c.chunk_id,
                "marker": c.marker,
                "marker_label": c.marker_label,
                "chunk_tokens_cl100k": c.tokens,
                "skipped": True,
                "skip_reason": f"chunk {c.tokens:,} cl100k tokens still over budget",
            })
            continue

        print(f"[RUN ] {c.chunk_id} {c.marker[:45]} -- {c.tokens:,} tok ...")
        t0 = time.perf_counter()
        result = agent.run(
            chunk_text=c.text,
            chunk_id=c.chunk_id,
            title_marker=c.marker_label,
            max_retries=1,
        )
        elapsed = time.perf_counter() - t0

        if result.errors:
            print(f"   [ERR] {result.errors}")
            results.append({
                "chunk_id": c.chunk_id,
                "marker": c.marker,
                "marker_label": c.marker_label,
                "chunk_tokens_cl100k": c.tokens,
                "ok": False,
                "errors": result.errors,
                "raw_preview": result.raw_response[:800],
            })
            continue

        out = result.output if isinstance(result.output, dict) else {}
        n_bullets = len(out.get("bullets", []))
        n_groups  = len(out.get("affected_groups", []))
        print(f"   [OK ] elapsed={elapsed:.1f}s prompt={result.prompt_tokens:,} "
              f"compl={result.completion_tokens:,} bullets={n_bullets} groups={n_groups}")
        results.append({
            "chunk_id": c.chunk_id,
            "marker": c.marker,
            "marker_label": c.marker_label,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "chunk_tokens_cl100k": c.tokens,
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "output": out,
        })

    n_ok   = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_err  = sum(1 for r in results if r.get("ok") is False)
    print()
    print(f"[SUMMARY] ok={n_ok}  skipped={n_skip}  errored={n_err}  total={len(results)}")

    report = {
        "test": "hr1_summarizer_3090_fork_v2",
        "spine": {
            "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
            "alias": "spine",
            "num_ctx": 65536,
            "kv_cache_type": "q8_0",
            "endpoint": "http://127.0.0.1:11434/v1",
            "thinking": False,
        },
        "max_input_cl100k_per_chunk": MAX_INPUT_CL100K,
        "amd_baseline_hr1_ch01": baseline,
        "fork_results": results,
    }
    OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ARTIFACT] {OUT_FILE}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
