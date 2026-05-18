"""
3090-fork wiring smoke test for the Plain English Summarizer via Ollama
OpenAI-compatible endpoint, using the spine alias (qwen3:30b).

GOAL: Prove that base.py's call_llm round-trips correctly through
http://127.0.0.1:11434/v1/chat/completions, that JSON extraction works,
and that the Pydantic schema validates the result.

ComfyUI must be down -- spine (qwen3:30b) needs ~17 GB VRAM + KV cache.

Quality is NOT validated against canonical here; this only confirms
wiring. A second test runs the same agent against full chunk[0] to
compare token usage and bullets against eval/canonical/bbb-ch01.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pdfplumber  # noqa: E402

from src.agents.summarizer import PlainEnglishSummarizer  # noqa: E402


HR1 = ROOT / "tests" / "fixtures" / "one_big_beautiful_bill_2025_hr1.pdf"


def extract_first_section_text(pdf_path: Path, max_chars: int = 4000) -> str:
    """Pull the first ~max_chars of legible text from page 1+ of the PDF.

    Skip the table-of-contents-ish prefix; jump to the first real
    section header so the summarizer has substantive text to chew on.
    """
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:8]:  # cap at first 8 pages
            txt = page.extract_text() or ""
            out.append(txt)
            if sum(len(t) for t in out) >= max_chars * 3:
                break
    full = "\n".join(out)
    for marker in ("SEC. 10001", "Sec. 10001", "TITLE I", "DIVISION A"):
        idx = full.find(marker)
        if idx >= 0:
            return full[idx : idx + max_chars]
    return full[:max_chars]


def main() -> int:
    if not HR1.exists():
        print(f"[FAIL] HR1 fixture missing: {HR1}")
        return 2

    chunk_text = extract_first_section_text(HR1, max_chars=3500)
    print(f"[INFO] chunk_text length: {len(chunk_text)} chars")
    print(f"[INFO] preview: {chunk_text[:200]!r}")
    print()

    agent = PlainEnglishSummarizer()
    # spine alias -> qwen3:30b (Q4_K_M MoE, ~17 GB resident, ~3B active).
    # ComfyUI must be down for this to load. target_model defaults to
    # "spine" already; set explicitly for clarity.
    agent.target_model = "spine"
    # Keep default max_tokens=1500 from PlainEnglishSummarizer.

    print(f"[INFO] endpoint={agent.target_endpoint} model={agent.target_model}")
    t0 = time.perf_counter()
    result = agent.run(
        chunk_text=chunk_text,
        chunk_id="hr1-smoke-ch01-first3500",
        title_marker="TITLE I (wiring-test slice)",
        max_retries=1,
    )
    elapsed = time.perf_counter() - t0
    print(f"[INFO] returned in {elapsed:.1f}s")
    print()

    if result.errors:
        print(f"[FAIL] errors: {result.errors}")
        print(f"[INFO] raw_response (first 1500 chars):\n{result.raw_response[:1500]}")
        return 1

    print(f"[OK]   agent={result.agent_name}")
    print(f"[OK]   chunk_id={result.chunk_id}")
    print(f"[OK]   elapsed_ms={result.elapsed_ms:.0f}")
    print(f"[OK]   prompt_tokens={result.prompt_tokens}")
    print(f"[OK]   completion_tokens={result.completion_tokens}")
    print(f"[OK]   total_tokens={result.total_tokens}")
    print()

    out = result.output
    if isinstance(out, dict):
        print(f"[OK]   one_sentence_summary: {out.get('one_sentence_summary', '')!r}")
        bullets = out.get("bullets", [])
        print(f"[OK]   bullet count: {len(bullets)}")
        for i, b in enumerate(bullets[:10], 1):
            print(f"         {i}. {b}")
        affected = out.get("affected_groups", [])
        print(f"[OK]   affected_groups: {affected}")
    else:
        print(f"[WARN] output is not a dict: {type(out)}")
        print(f"       value: {out!r}")

    # Persist for downstream comparison/debugging.
    artefact = ROOT / "eval" / "smoke-summarizer-wiring-qwen30b.json"
    artefact.parent.mkdir(parents=True, exist_ok=True)
    with open(artefact, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test": "ollama_openai_compat_wiring",
                "model": agent.target_model,
                "alias_target": "qwen3:30b",
                "endpoint": agent.target_endpoint,
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "output": out,
                "raw_response_preview": result.raw_response[:600],
            },
            f,
            indent=2,
        )
    print(f"\n[ARTIFACT] {artefact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
