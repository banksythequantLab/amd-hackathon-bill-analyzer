"""
APC Benchmark — proves vLLM Automatic Prefix Caching delivers TTFT speedup
on ROCm/MI300X.

The test:
  1. Build a large shared prefix (~100K tokens of bill text)
  2. Send Request A: shared prefix + Question 1
  3. Send Request B: same shared prefix + Question 2
  4. APC should detect the shared prefix in Request B and skip prefill.
     Expected: B's TTFT is dramatically faster than A's.

This is the headline measurement for BiP Post #1. If speedup < 3x, the
escape valve (docs/escape-valves.md, Day 1) fires and we evaluate SGLang.

Usage on the cloud instance:
  python infra/apc_benchmark.py \\
    --bill /scratch/bills/build_back_better_2021_hr5376.pdf \\
    --endpoint http://localhost:8001/v1 \\
    --model spine \\
    --target-prefix-tokens 100000

Outputs JSON to stdout:
  {
    "request_a": {"tokens_in": ..., "ttft_ms": ..., "total_ms": ...},
    "request_b": {"tokens_in": ..., "ttft_ms": ..., "total_ms": ...},
    "speedup_ttft": float,
    "passes_3x_floor": bool,
    "passes_5x_target": bool
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def extract_pdf_text(pdf_path: Path) -> str:
    """Pull plain text from the bill PDF for use as a shared prefix."""
    import pdfplumber
    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            chunks.append(t)
    return "\n\n".join(chunks)


def trim_to_token_target(text: str, target_tokens: int, model: str, endpoint: str) -> str:
    """Trim text to approximately target_tokens by binary-searching character length.

    Uses the vLLM tokenizer endpoint for accuracy rather than guessing 4 chars/token.
    """
    client = httpx.Client(timeout=30.0)

    # vLLM's /tokenize lives at the server root, not under /v1
    base = endpoint.rsplit("/v1", 1)[0]

    def count_tokens(s: str) -> int:
        r = client.post(
            f"{base}/tokenize",
            json={"model": model, "prompt": s},
        )
        r.raise_for_status()
        return r.json()["count"]

    # Initial estimate at ~4 chars/token
    if len(text) // 4 < target_tokens:
        return text

    lo, hi = target_tokens * 3, min(len(text), target_tokens * 6)
    best = text[:lo]
    for _ in range(8):  # bounded binary search
        mid = (lo + hi) // 2
        sample = text[:mid]
        n = count_tokens(sample)
        if n < target_tokens:
            lo = mid
            best = sample
        else:
            hi = mid
    return best


def run_request(endpoint: str, model: str, prompt: str, max_tokens: int = 50) -> dict:
    """Run a streaming request and capture TTFT + total time."""
    client = httpx.Client(timeout=600.0)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    t_start = time.perf_counter()
    t_first_token: float | None = None
    n_chunks = 0
    output_text = ""

    with client.stream("POST", f"{endpoint}/completions", json=payload) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = j.get("choices", [{}])[0]
            text_piece = choice.get("text", "")
            if text_piece:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                output_text += text_piece
                n_chunks += 1

    t_end = time.perf_counter()

    return {
        "ttft_ms": round((t_first_token - t_start) * 1000, 1) if t_first_token else None,
        "total_ms": round((t_end - t_start) * 1000, 1),
        "output_chars": len(output_text),
        "stream_chunks": n_chunks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bill", type=Path, required=True, help="PDF to use as shared prefix source")
    ap.add_argument("--endpoint", default="http://localhost:8001/v1", help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--model", default="spine", help="Served model name (--served-model-name on vllm serve)")
    ap.add_argument("--target-prefix-tokens", type=int, default=100_000, help="Approximate prefix length in tokens")
    ap.add_argument("--question-a", default="Summarize the first major Title of the bill in two sentences.")
    ap.add_argument("--question-b", default="List three named entities mentioned in the bill text above.")
    args = ap.parse_args()

    print(f"[apc_benchmark] Loading bill text from {args.bill} ...", file=sys.stderr)
    text = extract_pdf_text(args.bill)
    print(f"[apc_benchmark] Raw text length: {len(text):,} chars", file=sys.stderr)

    print(f"[apc_benchmark] Trimming to ~{args.target_prefix_tokens:,} tokens ...", file=sys.stderr)
    prefix = trim_to_token_target(text, args.target_prefix_tokens, args.model, args.endpoint)

    # Token count for the actual prefix used
    client = httpx.Client(timeout=30.0)
    base = args.endpoint.rsplit("/v1", 1)[0]
    r = client.post(f"{base}/tokenize", json={"model": args.model, "prompt": prefix})
    r.raise_for_status()
    actual_prefix_tokens = r.json()["count"]
    print(f"[apc_benchmark] Actual prefix tokens: {actual_prefix_tokens:,}", file=sys.stderr)

    prompt_a = prefix + "\n\n---\n\nQuestion: " + args.question_a + "\n\nAnswer:"
    prompt_b = prefix + "\n\n---\n\nQuestion: " + args.question_b + "\n\nAnswer:"

    print(f"[apc_benchmark] Running Request A (cold prefix)...", file=sys.stderr)
    res_a = run_request(args.endpoint, args.model, prompt_a)
    print(f"[apc_benchmark]   TTFT: {res_a['ttft_ms']} ms", file=sys.stderr)

    print(f"[apc_benchmark] Running Request B (warm prefix — APC should hit)...", file=sys.stderr)
    res_b = run_request(args.endpoint, args.model, prompt_b)
    print(f"[apc_benchmark]   TTFT: {res_b['ttft_ms']} ms", file=sys.stderr)

    speedup = res_a["ttft_ms"] / res_b["ttft_ms"] if res_b["ttft_ms"] else None
    out = {
        "model": args.model,
        "endpoint": args.endpoint,
        "prefix_tokens": actual_prefix_tokens,
        "request_a_cold":  {**res_a, "question": args.question_a},
        "request_b_warm":  {**res_b, "question": args.question_b},
        "speedup_ttft":    round(speedup, 2) if speedup else None,
        "passes_3x_floor": speedup is not None and speedup >= 3.0,
        "passes_5x_target": speedup is not None and speedup >= 5.0,
    }
    print(json.dumps(out, indent=2))

    if out["passes_3x_floor"]:
        print(f"\n[apc_benchmark] PASS — APC delivered {out['speedup_ttft']}x speedup", file=sys.stderr)
        return 0
    print(f"\n[apc_benchmark] FAIL — APC speedup {out['speedup_ttft']}x is below 3x floor", file=sys.stderr)
    print(f"[apc_benchmark] Trigger SGLang fallback evaluation (see docs/escape-valves.md)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
