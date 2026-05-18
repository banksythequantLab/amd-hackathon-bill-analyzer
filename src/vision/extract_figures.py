"""
Vision pipeline smoke: render BBB-2021 page 2222 (a tax bracket schedule)
as an image, send to Qwen3-VL, request structured JSON of the bracket data.

The model lives on the cloud instance. We:
  1. Render page locally (pdfplumber + Pillow) at 200 DPI for clarity
  2. Save as PNG
  3. Base64-encode and POST to vision endpoint as image_url
  4. Parse the response, save both the raw and (if JSON) structured output
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import pdfplumber
import httpx
from PIL import Image


def render_page(pdf_path: Path, page_num: int, dpi: int = 200) -> Image.Image:
    """1-based page_num. Returns Pillow image."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        # pdfplumber's to_image uses pdf2image internally which uses poppler.
        # Falling back: use page.to_image(resolution=dpi) which works on Windows
        # if poppler is installed. If not, we'll fail loudly.
        pi = page.to_image(resolution=dpi)
        return pi.original  # PIL.Image


def encode_image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def call_vision(endpoint: str, model: str, b64_png: str, prompt: str, max_tokens: int = 8000) -> dict:
    """OpenAI-style chat-completions call with image_url input."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Qwen3-VL-Thinking emits a long reasoning chain. Keep enable_thinking=false
        # for production extraction speed; we want the answer, not the reasoning trace.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{endpoint}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


def extract_json(content: str) -> dict | None:
    """Pull a JSON object out of the model's text response.

    Strategy:
      1. Try fenced ```json``` block
      2. Try LAST {...} substring (Thinking models put final answer at end)
      3. Try first {...} (fallback)
    """
    # Try fenced code block first (last one wins, since model may show schema first)
    fences = list(re.finditer(r"```(?:json)?\s*(\{.+?\})\s*```", content, re.DOTALL))
    if fences:
        try:
            return json.loads(fences[-1].group(1))
        except json.JSONDecodeError:
            pass

    # Find the LAST balanced top-level JSON object in the text. We scan
    # backward from end, looking for the last `}` and matching it to its `{`.
    last_close = content.rfind("}")
    while last_close > 0:
        depth = 0
        start = -1
        for i in range(last_close, -1, -1):
            ch = content[i]
            if ch == "}":
                depth += 1
                if start == -1:
                    start = i
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    candidate = content[i:last_close + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        last_close = content.rfind("}", 0, last_close)
    return None


def main() -> int:
    # 3090 FORK: --pdf default derived from script location (was hardcoded
    # to B:\amd-hackathon-bill-analyzer\..., old fork). parents[2] = repo root.
    _repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=_repo / "tests" / "fixtures" / "build_back_better_2021_hr5376.pdf")
    ap.add_argument("--page", type=int, default=2222)
    # 3090 FORK: was --endpoint http://165.245.134.1:8002 on the AMD vision
    # server. Now defaults to local Ollama OpenAI-compat; will 404 until
    # `ollama pull` + `cp` creates the 'vision' alias (TODO #4).
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="vision")
    ap.add_argument("--out-dir", type=Path, default=Path(r"B:\hackathon-build\vision-smoke"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vision-smoke] Rendering {args.pdf.name} page {args.page} ...")
    img = render_page(args.pdf, args.page)
    png_path = args.out_dir / f"page-{args.page}.png"
    img.save(png_path)
    print(f"[vision-smoke] Saved render: {png_path} ({png_path.stat().st_size / 1024:.1f} KB)")

    b64 = encode_image_b64(img)
    print(f"[vision-smoke] Image base64 size: {len(b64):,} chars")

    prompt = """Extract first tax bracket category visible. JSON only:
{"type":"tax_bracket","applies_to":"...","rows":[{"lower_bound_usd":N,"upper_bound_usd":N,"tax_rate_pct":N,"notes":"..."}]}"""

    print(f"[vision-smoke] Calling vision model at {args.endpoint} ...")
    response = call_vision(args.endpoint, args.model, b64, prompt)

    raw_path = args.out_dir / f"response-page-{args.page}.json"
    raw_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    print(f"[vision-smoke] Raw response saved: {raw_path}")

    content = response["choices"][0]["message"]["content"]
    print(f"\n=== Raw model response ({len(content)} chars) ===")
    print(content[:2000])
    if len(content) > 2000:
        print(f"... [{len(content) - 2000} more chars truncated]")

    parsed = extract_json(content)
    if parsed:
        out_json = args.out_dir / f"extracted-page-{args.page}.json"
        out_json.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print(f"\n=== Parsed JSON ===")
        print(json.dumps(parsed, indent=2)[:2000])
        print(f"\n[vision-smoke] PARSED OK -> {out_json}")
    else:
        print(f"\n[vision-smoke] WARN: response was not parseable as JSON.")
        return 2

    usage = response.get("usage", {})
    print(f"\n=== Usage ===")
    print(f"  prompt_tokens:     {usage.get('prompt_tokens')}")
    print(f"  completion_tokens: {usage.get('completion_tokens')}")
    print(f"  total_tokens:      {usage.get('total_tokens')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
