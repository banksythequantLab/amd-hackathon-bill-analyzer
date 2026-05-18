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
# Highest preference (DIVISION) -> lowest (Subtitle, CHAPTER, SEC.).
#
# TITLE is intentionally case-sensitive (UPPER only) and Roman-only.
# A line that starts "title VII of the Tariff Act of 1930" or "title 42,
# Code of Federal Regulations" is a *reference* to an external statute,
# not a bill structural marker -- the case-sensitive match avoids those.
# Similarly we drop the `[0-9]+` arm (no real bill numbers TITLEs in
# Arabic in our corpus) to prevent "TITLE 18" / "title 42" false hits.
#
# CHAPTER and SEC. are added as finer-grained fallbacks for monster
# segments like HR1 TITLE VII Finance (93K cl100k tokens, no inner
# Subtitle within the first half). They're only consulted by the
# recursive sub-splitter in pack_chunks; they're NOT in the primary
# greedy pack because that would over-fragment normal bills.
BOUNDARY_PATTERNS = [
    ("DIVISION",  re.compile(r"^\s*(DIVISION\s+([A-Z]|[IVXLCDM]+))(?:\b|$)", re.MULTILINE)),
    ("TITLE",     re.compile(r"^\s*(TITLE\s+([IVXLCDM]+))(?:\b|$)",          re.MULTILINE)),
    ("Subtitle",  re.compile(r"^\s*(Subtitle\s+([A-Z]|[0-9]+))(?:\b|$)",     re.MULTILINE)),
]

# Sub-splitter patterns -- consulted only when a single primary segment
# is too large to fit in a chunk on its own.
SUBSPLIT_PATTERNS = [
    ("CHAPTER",   re.compile(r"^\s*(CHAPTER\s+(\d+|[IVXLCDM]+))\b",          re.MULTILINE)),
    ("Subchapter",re.compile(r"^\s*(Subchapter\s+[A-Z])\b",                  re.MULTILINE)),
    ("PART",      re.compile(r"^\s*(PART\s+[IVXLCDM]+)\b",                   re.MULTILINE)),
    ("SEC",       re.compile(r"^\s*(SEC\.\s+\d+)\b",                         re.MULTILINE)),
]

# Default token budget per chunk. Empirically calibrated against the spine
# (Qwen3-30B-A3B-Instruct-2507-FP8) tokenizer: legislative English inflates
# from cl100k_base to Qwen at ~16.5% (measured on BBB-2021 chunk: 234,379
# cl100k -> 272,946 Qwen). With a 262,144-token spine context, an 8000-token
# output budget (USC Cross-Reference produces large citation lists), and
# ~500 tokens of system-prompt overhead, the safe input ceiling is:
#   (262,144 - 8,000 - 500) / 1.165 = ~217,200 cl100k tokens
# We set 200K to leave ~17K cl100k tokens (~20K Qwen tokens) of safety margin
# for prompt-wrapper variance and tool-use overhead. (Previous setting of
# 220K caused USC Cross-Reference HTTP 400 failures on 218K-token BBB chunks.)
MAX_TOKENS_DEFAULT = 200_000


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

        # Single segment too large to fit alone? Try the recursive
        # sub-splitter on CHAPTER/Subchapter/PART/SEC markers inside the
        # segment. This handles HR1 TITLE VII (93K cl100k tokens, no
        # inner Subtitle but 7 CHAPTER markers and 107 SEC. markers).
        if seg_tokens > max_tokens:
            if current_segs:
                emit()
                current_segs = []
                current_tokens = 0
            sub_chunks = _subsplit_oversized(b, seg, page_starts, max_tokens, encoder)
            if len(sub_chunks) > 1:
                # We managed to split it. Emit each sub-chunk as its own chunk.
                for sb_label, sb_text, sb_first_b in sub_chunks:
                    chunks.append(Chunk(
                        chunk_id=f"ch{len(chunks) + 1:02d}",
                        marker=f"{b.marker} {sb_label}",
                        marker_label=f"{b.full_line} -> {sb_label}",
                        start_page=sb_first_b.get("page", b.page),
                        end_page=page_for_offset(
                            sb_first_b.get("end_offset", b.char_offset + len(sb_text) - 1),
                            page_starts,
                        ),
                        tokens=count_tokens(sb_text, encoder),
                        char_count=len(sb_text),
                        text=sb_text,
                    ))
                continue
            # Could not split further -- emit whole with a warning (rare
            # pathological case; the orchestrator will refuse this chunk).
            print(f"   ! Single segment {b.marker} is {seg_tokens:,} tokens "
                  f"(over max {max_tokens:,}); no sub-split found, emitting whole.",
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


def _subsplit_oversized(parent_b: Boundary, seg_text: str, page_starts: list[int],
                        max_tokens: int, encoder) -> list[tuple[str, str, dict]]:
    """Recursively split an oversized parent segment.

    Try SUBSPLIT_PATTERNS in order of preference. Return list of
    (label, text, meta) tuples where meta has 'page' and 'end_offset'
    keyed to absolute offsets in the parent text.

    Returns the original segment as a single-element list if no
    sub-split markers are found inside it.
    """
    # parent_b.char_offset is where seg_text starts in the FULL bill text;
    # all sub-split match positions are relative to seg_text. We translate
    # back to absolute positions via parent_b.char_offset + local_offset.
    parent_offset = parent_b.char_offset

    for label, regex in SUBSPLIT_PATTERNS:
        matches = list(regex.finditer(seg_text))
        # Need at least 2 matches to split meaningfully; the first match
        # is usually the same point as the parent boundary (just deeper).
        if len(matches) < 2:
            continue

        # Convert matches to (label_str, local_offset) pairs.
        cuts: list[tuple[str, int]] = []
        for m in matches:
            line_end = seg_text.find("\n", m.end())
            line = seg_text[m.start():line_end if line_end > 0 else m.end() + 80].strip()
            cuts.append((line.replace("\n", " ")[:60], m.start()))

        # Greedy-pack the sub-segments using token budget.
        results: list[tuple[str, str, dict]] = []
        pack_text: list[str] = []
        pack_tokens = 0
        pack_first_label = ""
        pack_first_local_offset = 0

        for i, (sub_label, sub_local_offset) in enumerate(cuts):
            sub_end = cuts[i + 1][1] if i + 1 < len(cuts) else len(seg_text)
            sub_text = seg_text[sub_local_offset:sub_end]
            sub_tokens = count_tokens(sub_text, encoder)

            # If this single sub-segment itself overflows, emit whatever's
            # packed and then emit the sub-segment as its own chunk (still
            # may be too big -- caller will handle if so).
            if sub_tokens > max_tokens:
                if pack_text:
                    results.append((
                        pack_first_label,
                        "".join(pack_text),
                        {"page": page_for_offset(parent_offset + pack_first_local_offset, page_starts),
                         "end_offset": parent_offset + sub_local_offset - 1},
                    ))
                    pack_text = []
                    pack_tokens = 0
                results.append((
                    sub_label,
                    sub_text,
                    {"page": page_for_offset(parent_offset + sub_local_offset, page_starts),
                     "end_offset": parent_offset + sub_end - 1},
                ))
                continue

            if pack_tokens + sub_tokens > max_tokens and pack_text:
                results.append((
                    pack_first_label,
                    "".join(pack_text),
                    {"page": page_for_offset(parent_offset + pack_first_local_offset, page_starts),
                     "end_offset": parent_offset + sub_local_offset - 1},
                ))
                pack_text = []
                pack_tokens = 0

            if not pack_text:
                pack_first_label = sub_label
                pack_first_local_offset = sub_local_offset

            pack_text.append(sub_text)
            pack_tokens += sub_tokens

        if pack_text:
            results.append((
                pack_first_label,
                "".join(pack_text),
                {"page": page_for_offset(parent_offset + pack_first_local_offset, page_starts),
                 "end_offset": parent_offset + len(seg_text) - 1},
            ))

        # Only return if we actually managed multiple chunks.
        if len(results) > 1:
            return results

    # No sub-pattern produced a useful split.
    return [(parent_b.marker, seg_text, {"page": parent_b.page,
                                          "end_offset": parent_offset + len(seg_text) - 1})]


def chunk_pdf(pdf_path: Path, max_tokens: int = MAX_TOKENS_DEFAULT) -> list[dict]:
    """Convenience function: chunk a PDF and return a list of dicts.
    
    Each dict has keys: chunk_id, marker, marker_label, start_page, end_page,
    tokens, char_count, text.
    """
    text, page_starts = extract_text_with_pages(pdf_path)
    enc = tiktoken.get_encoding("cl100k_base")
    boundaries = find_boundaries(text, page_starts)
    chunks = pack_chunks(text, boundaries, page_starts, max_tokens, enc)
    return [asdict(c) for c in chunks]


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
