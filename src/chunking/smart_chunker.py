"""
Smart Chunker for legislative bills.

Splits a bill PDF into chunks of <= MAX_TOKENS, preferring TITLE / Subtitle /
DIVISION boundaries to keep statutory cross-references intact within a chunk.

Output structure:
    [
        {
            "chunk_id": "ch01",
            "marker": "TITLE I",
            "marker_label": "TITLE I-AGRICULTURE",
            "start_page": 1,
            "end_page": 412,
            "tokens": 240118,
            "char_count": 982344,
            "text": "...full text..."
        },
        ...
    ]

Usage:
    python smart_chunker.py <bill.pdf> [--max-tokens 250000] [--out chunks.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pdfplumber
import tiktoken


# Boundary patterns, ordered by preference for splitting.
# Highest preference (DIVISION) -> lowest (TITLE).
# Subtitle is the fallback when a Title is too big.
BOUNDARY_PATTERNS = [
    ("DIVISION",  re.compile(r"^\s*(DIVISION\s+([A-Z]|[IVXLCDM]+))(?:\b|$)", re.MULTILINE)),
    ("TITLE",     re.compile(r"^\s*(TITLE\s+([IVXLCDM]+|[0-9]+))(?:\b|$)",   re.MULTILINE | re.IGNORECASE)),
    ("Subtitle",  re.compile(r"^\s*(Subtitle\s+([A-Z]|[0-9]+))(?:\b|$)",     re.MULTILINE)),
]

# Default token budget per chunk. Empirically calibrated against the spine
# (Qwen3-30B-A3B-Instruct-2507-FP8) tokenizer: legislative English inflates
# from cl100k_base to Qwen at ~16.5% (measured on BBB-2021 chunk: 234,379
# cl100k -> 272,946 Qwen). With a 262,144-token spine context and a 1500-
# token output budget, the safe input ceiling is ~223,800 cl100k tokens.
# We set 220K to leave additional safety margin for prompt-wrapper overhead.
MAX_TOKENS_DEFAULT = 220_000


@dataclass
class Boundary:
    """A structural break point in the bill text."""
    label: str          # 'DIVISION', 'TITLE', 'Subtitle'
    marker: str         # e.g. 'TITLE I'
    full_line: str      # full first-line context, e.g. 'TITLE I-AGRICULTURE'
    char_offset: int    # offset into the full text
    page: int           # 1-based page number


@dataclass
class Chunk:
    chunk_id: str
    marker: str
    marker_label: str
    start_page: int
    end_page: int
    tokens: int
    char_count: int
    text: str

    def to_summary(self) -> dict:
        d = asdict(self)
        d["text_preview"] = d.pop("text")[:200] + "..."
        return d


def extract_text_with_pages(pdf_path: Path) -> tuple[str, list[int]]:
    """Returns (full_text, page_starts) where page_starts[i] is the char
    offset in full_text where page i+1 begins."""
    full_text = []
    page_starts = []
    cursor = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_starts.append(cursor)
            t = page.extract_text() or ""
            t += "\n\n"
            full_text.append(t)
            cursor += len(t)
    return "".join(full_text), page_starts


def find_boundaries(text: str, page_starts: list[int]) -> list[Boundary]:
    """Find every DIVISION/TITLE/Subtitle marker in the text."""
    boundaries: list[Boundary] = []
    for label, regex in BOUNDARY_PATTERNS:
        for m in regex.finditer(text):
            full_line_end = text.find("\n", m.end())
            full_line = text[m.start():full_line_end if full_line_end > 0 else m.end() + 80].strip()
            full_line = full_line.replace("\n", " ")
            boundaries.append(Boundary(
                label=label,
                marker=m.group(1).strip(),
                full_line=full_line,
                char_offset=m.start(),
                page=page_for_offset(m.start(), page_starts),
            ))

    # Dedupe (a "Subtitle" inside a "TITLE" line shouldn't double-count if they coincide)
    boundaries.sort(key=lambda b: b.char_offset)
    deduped: list[Boundary] = []
    for b in boundaries:
        if deduped and abs(deduped[-1].char_offset - b.char_offset) < 20 and deduped[-1].label != b.label:
            # keep the higher-precedence label
            order = {"DIVISION": 0, "TITLE": 1, "Subtitle": 2}
            if order[b.label] < order[deduped[-1].label]:
                deduped[-1] = b
            continue
        deduped.append(b)
    return deduped


def page_for_offset(offset: int, page_starts: list[int]) -> int:
    """1-based page index for a char offset."""
    lo, hi = 0, len(page_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if page_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(text, disallowed_special=()))


def pack_chunks(text: str, boundaries: list[Boundary], page_starts: list[int],
                max_tokens: int, encoder) -> list[Chunk]:
    """Greedy left-to-right packing.

    Strategy:
      1. Walk the boundaries, prefer the highest-precedence labels first.
      2. Build a chunk by accumulating boundary segments until adding the next
         segment would push the chunk over max_tokens.
      3. If a single segment between boundaries already exceeds max_tokens,
         we sub-split it on Subtitle. If even that fails, the segment is
         emitted whole (pathological — rare for real bills) and a warning is
         printed.
    """
    if not boundaries:
        # No structural markers; emit one giant chunk
        return [Chunk(
            chunk_id="ch01", marker="(no marker)", marker_label="(unstructured bill)",
            start_page=1, end_page=len(page_starts),
            tokens=count_tokens(text, encoder), char_count=len(text), text=text,
        )]

    # We segment the text by the boundaries — each segment runs from a boundary
    # to the next boundary (or EOF).
    segments: list[tuple[Boundary, str]] = []
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1].char_offset if i + 1 < len(boundaries) else len(text)
        seg_text = text[b.char_offset:end]
        segments.append((b, seg_text))

    # Greedy pack
    chunks: list[Chunk] = []
    current_segs: list[tuple[Boundary, str]] = []
    current_tokens = 0

    def emit() -> None:
        if not current_segs:
            return
        first_b = current_segs[0][0]
        last_b = current_segs[-1][0]
        last_text = current_segs[-1][1]
        chunk_text = "".join(s[1] for s in current_segs)
        chunk_id = f"ch{len(chunks) + 1:02d}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            marker=first_b.marker if first_b.label != "Subtitle" else f"{first_b.marker} (subtitle-cut)",
            marker_label=first_b.full_line,
            start_page=first_b.page,
            end_page=page_for_offset(last_b.char_offset + len(last_text) - 1, page_starts),
            tokens=current_tokens,
            char_count=len(chunk_text),
            text=chunk_text,
        ))

    for b, seg in segments:
        seg_tokens = count_tokens(seg, encoder)

        # Single segment too large to fit alone? Sub-split on Subtitle (already done
        # by find_boundaries since Subtitle is in the boundary list). If it's STILL
        # too big, emit anyway with a warning.
        if seg_tokens > max_tokens:
            if current_segs:
                emit()
                current_segs = []
                current_tokens = 0
            print(f"   ! Single segment {b.marker} is {seg_tokens:,} tokens "
                  f"(over max {max_tokens:,}); emitting as oversized chunk.",
                  file=sys.stderr)
            current_segs = [(b, seg)]
            current_tokens = seg_tokens
            emit()
            current_segs = []
            current_tokens = 0
            continue

        if current_tokens + seg_tokens > max_tokens and current_segs:
            emit()
            current_segs = []
            current_tokens = 0

        current_segs.append((b, seg))
        current_tokens += seg_tokens

    emit()
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path, help="Bill PDF path")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT,
                    help=f"cl100k tokens per chunk; default {MAX_TOKENS_DEFAULT:,} "
                         f"calibrated for Qwen-tokenizer 16.5% inflation on legal text")
    ap.add_argument("--out", type=Path, help="Optional path to write JSON")
    ap.add_argument("--summary-only", action="store_true",
                    help="Print only chunk summaries to stdout, don't write text")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: {args.pdf} not found", file=sys.stderr)
        return 1

    print(f"[chunker] Reading {args.pdf} ...", file=sys.stderr)
    text, page_starts = extract_text_with_pages(args.pdf)
    print(f"[chunker]   pages: {len(page_starts):,}", file=sys.stderr)
    print(f"[chunker]   chars: {len(text):,}", file=sys.stderr)

    print(f"[chunker] Loading tokenizer (cl100k_base, GPT-4 family — close enough "
          f"for budgeting) ...", file=sys.stderr)
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = count_tokens(text, enc)
    print(f"[chunker]   tokens: {total_tokens:,}", file=sys.stderr)

    print(f"[chunker] Finding structural boundaries ...", file=sys.stderr)
    boundaries = find_boundaries(text, page_starts)
    by_label = {}
    for b in boundaries:
        by_label.setdefault(b.label, 0)
        by_label[b.label] += 1
    for label, cnt in by_label.items():
        print(f"   {label:10s}: {cnt} boundaries", file=sys.stderr)

    print(f"[chunker] Packing into chunks (max {args.max_tokens:,} tokens) ...",
          file=sys.stderr)
    chunks = pack_chunks(text, boundaries, page_starts, args.max_tokens, enc)

    print(f"\n=== {len(chunks)} chunks ===", file=sys.stderr)
    for c in chunks:
        print(f"  {c.chunk_id} {c.marker:25s} pp.{c.start_page:>4}-{c.end_page:<4} "
              f"{c.tokens:>7,} tokens  ({c.char_count:>9,} chars)  "
              f"{c.marker_label[:60]}", file=sys.stderr)

    if args.out:
        out_data = [asdict(c) for c in chunks]
        if args.summary_only:
            for d in out_data:
                d["text_preview"] = d.pop("text")[:200] + "..."
        args.out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\n[chunker] Wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
