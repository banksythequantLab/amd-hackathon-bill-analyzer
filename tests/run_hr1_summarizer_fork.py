"""
Run the Plain-English Summarizer over all HR1 chunks produced by the 50K
chunker, against the spine alias (qwen3:30b @ 64K ctx, KV q8_0).

Compares per-chunk token usage + elapsed time against the AMD baseline
recorded in eval/report-hr1-ch01.json (which used the full 262K-context
FP8 spine on the canonical infra).

Skips chunks that exceed our 64K ctx budget; logs the skip with the
reason rather than truncating.

Output: eval/hr1-summarizer-3090-fork.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.summarizer import PlainEnglishSummarizer  # noqa: E402


CHUNKS_FILE = ROOT / "eval" / "hr1-chunks-50k.json"
HR1_PDF     = ROOT / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"
BASELINE    = ROOT / "eval" / "report-hr1-ch01.json"
OUT_FILE    = ROOT / "eval" / "hr1-summarizer-3090-fork.json"

# Token budgets. Spine = qwen3:30b @ num_ctx=65536, KV q8_0.
# Reserve ~3,000 qwen tokens for system + user-prompt wrapper + output buffer.
# cl100k -> qwen inflation ~16.5% on legal text. So a chunk of N cl100k
# tokens consumes ~N * 1.165 qwen tokens for the chunk_text injection
# alone. Cap input chunk_text at 50K cl100k = ~58K qwen, leaving ~5K
# qwen headroom for the rest of the prompt + output.
MAX_INPUT_CL100K = 50_500


def load_baseline_summarizer() -> dict:
    """Extract just the summarizer-agent slice from the AMD baseline."""
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
    if not CHUNKS_FILE.exists():
        print(f"[FAIL] chunks file missing: {CHUNKS_FILE}")
        return 1

    # Re-chunk with text (the --summary-only run only wrote previews).
    # Easier: just re-call the chunker programmatically to get full text.
    from src.chunking.smart_chunker import (
        extract_text_with_pages,
        find_boundaries,
        pack_chunks,
    )
    import tiktoken

    print(f"[INFO] re-chunking {HR1_PDF.name} with --max-tokens {MAX_INPUT_CL100K}")
    text, page_starts = extract_text_with_pages(HR1_PDF)
    enc = tiktoken.get_encoding("cl100k_base")
    boundaries = find_boundaries(text, page_starts)
    chunks = pack_chunks(text, boundaries, page_starts, MAX_INPUT_CL100K, enc)
    print(f"[INFO] got {len(chunks)} chunks")
    for c in chunks:
        print(f"   {c.chunk_id} {c.marker:20s} pp.{c.start_page}-{c.end_page} "
              f"tok={c.tokens:>7,}")
    print()

    baseline = load_baseline_summarizer()
    print(f"[INFO] AMD baseline summarizer:")
    print(f"   chunk_tokens_cl100k: {baseline['chunk_tokens_cl100k']:,}")
    print(f"   prompt_tokens:       {baseline['prompt_tokens']:,}")
    print(f"   completion_tokens:   {baseline['completion_tokens']}")
    print(f"   elapsed_s:           {baseline['elapsed_s']}")
    print(f"   bullets:             {len(baseline['summarizer_output'].get('bullets', []))}")
    print()

    agent = PlainEnglishSummarizer()  # spine + 1500 max output tokens
    results = []

    for c in chunks:
        # Skip oversized chunks (input would push us over 64K ctx).
        if c.tokens > MAX_INPUT_CL100K + 2_000:  # small slack
            print(f"[SKIP] {c.chunk_id} {c.marker} -- {c.tokens:,} cl100k "
                  f"tokens exceeds our 64K-ctx budget (~{MAX_INPUT_CL100K:,} cl100k)")
            results.append({
                "chunk_id": c.chunk_id,
                "marker": c.marker,
                "marker_label": c.marker_label,
                "start_page": c.start_page,
                "end_page": c.end_page,
                "chunk_tokens_cl100k": c.tokens,
                "skipped": True,
                "skip_reason": f"chunk {c.tokens:,} cl100k tokens exceeds 64K ctx budget",
            })
            continue

        print(f"[RUN ] {c.chunk_id} {c.marker} -- {c.tokens:,} cl100k tokens ...")
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
                "start_page": c.start_page,
                "end_page": c.end_page,
                "chunk_tokens_cl100k": c.tokens,
                "ok": False,
                "errors": result.errors,
                "raw_preview": result.raw_response[:400],
            })
            continue

        out = result.output if isinstance(result.output, dict) else {}
        n_bullets = len(out.get("bullets", []))
        n_groups = len(out.get("affected_groups", []))
        print(f"   [OK ] elapsed={elapsed:.1f}s  "
              f"prompt={result.prompt_tokens:,}  "
              f"completion={result.completion_tokens:,}  "
              f"bullets={n_bullets}  groups={n_groups}")
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

    n_ok = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_err = sum(1 for r in results if r.get("ok") is False)
    print()
    print(f"[SUMMARY] {n_ok} ok, {n_skip} skipped, {n_err} errored, "
          f"total {len(results)}")

    report = {
        "test": "hr1_summarizer_3090_fork",
        "spine": {
            "model": "qwen3:30b (Q4_K_M)",
            "alias": "spine",
            "num_ctx": 65536,
            "kv_cache_type": "q8_0",
            "endpoint": "http://127.0.0.1:11434/v1",
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
