"""
USC Corpus Builder
==================

Builds an LMDB key-value store from the bulk USC USLM XML release.

Source : https://uscode.house.gov/download/download.shtml
Schema : USLM 1.0 (http://xml.house.gov/schemas/uslm/1.0)

Why LMDB
--------
Our access pattern is *exact-key lookup* of legal citations like "26:401(k)".
Memory-mapped LMDB delivers ~1 microsecond hot-path lookups vs SQLite's ~1 ms,
and read-only mode is lock-free across multiple agent processes.

Key format
----------
  "{title}:{section}"   normalized lowercase, e.g. "26:401(k)"
  Title can be "5a", "11a", "18a", "28a", "50a" for appendix titles.

Value format (orjson-serialized)
--------------------------------
  {
    "title":         "26",
    "section":       "401(k)",
    "heading":       "Cash or deferred arrangements",
    "text":          "...full plain-text statutory body...",
    "sourceCredit":  "(Pub. L. 95-600, ...)",
    "cross_refs":    ["/us/usc/t26/s402", ...],
    "source_url":    "https://uscode.house.gov/view.xhtml?req=...",
    "release_point": "119-36"
  }

Usage
-----
  python infra/usc_corpus_build.py \\
    --xml-dir   D:/hackathon-build/xml \\
    --lmdb-path D:/hackathon-build/usc.lmdb \\
    --release   119-36
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import lmdb
import orjson
from lxml import etree
from tqdm import tqdm

USLM_NS = "{http://xml.house.gov/schemas/uslm/1.0}"
SECTION_TAG = f"{USLM_NS}section"
NUM_TAG = f"{USLM_NS}num"
HEADING_TAG = f"{USLM_NS}heading"
CONTENT_TAG = f"{USLM_NS}content"
SOURCE_CREDIT_TAG = f"{USLM_NS}sourceCredit"
REF_TAG = f"{USLM_NS}ref"

# Identifier paths look like /us/usc/t26/s401(k) or /us/usc/t5A/s101
ID_RE = re.compile(r"^/us/usc/t([0-9]+[a-zA-Z]?)/s(.+)$")


def normalize_citation_key(title: str, section: str) -> str:
    """Build the LMDB lookup key from title + section.

    Lowercase, strip any whitespace. Section keeps its native form
    (e.g. 401(k)) so cross-refs from agents match without translation.
    """
    return f"{title.lower()}:{section.lower().strip()}"


def extract_text(elem: etree._Element) -> str:
    """Flatten an element's text, recursively, with whitespace collapsed."""
    parts = list(elem.itertext())
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def parse_section(section_el: etree._Element, release_point: str) -> tuple[str, bytes] | None:
    """Convert a USLM <section> to an (lmdb_key, orjson_value) tuple.

    Returns None if the section lacks a parseable identifier.
    """
    identifier = section_el.get("identifier", "")
    m = ID_RE.match(identifier)
    if not m:
        return None
    title, section_num = m.group(1), m.group(2)

    # Heading — sometimes absent on placeholder/repealed sections.
    heading_el = section_el.find(HEADING_TAG)
    heading = extract_text(heading_el) if heading_el is not None else ""

    # Body text from <content>; falls back to whole section text if absent.
    content_el = section_el.find(CONTENT_TAG)
    text = extract_text(content_el) if content_el is not None else extract_text(section_el)

    # Amendment history.
    src_el = section_el.find(SOURCE_CREDIT_TAG)
    source_credit = extract_text(src_el) if src_el is not None else ""

    # Outbound cross-references.
    refs: list[str] = []
    seen = set()
    for ref_el in section_el.iter(REF_TAG):
        href = ref_el.get("href", "")
        if href.startswith("/us/usc/") and href not in seen:
            refs.append(href)
            seen.add(href)

    # Constructable URL on the OLRC website.
    source_url = (
        f"https://uscode.house.gov/view.xhtml?"
        f"req=granuleid:USC-prelim-title{title}-section{section_num}"
        f"&num=0&edition=prelim"
    )

    record = {
        "title": title,
        "section": section_num,
        "heading": heading,
        "text": text,
        "sourceCredit": source_credit,
        "cross_refs": refs,
        "source_url": source_url,
        "release_point": release_point,
    }
    key = normalize_citation_key(title, section_num)
    return key, orjson.dumps(record)


def iter_sections(xml_path: Path):
    """Stream <section> elements from a USLM title file.

    Uses iterparse to keep memory bounded on large titles like 26 USC (~92 MB).
    """
    context = etree.iterparse(str(xml_path), events=("end",), tag=SECTION_TAG)
    for _, elem in context:
        yield elem
        # Free memory: clear processed element + ancestors' tail data.
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


def build(xml_dir: Path, lmdb_path: Path, release_point: str, map_size_gb: int = 4) -> dict:
    """Build the LMDB. Returns a stats dict."""
    xml_files = sorted(xml_dir.glob("usc*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No usc*.xml files found under {xml_dir}")

    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_gb * 1024**3,
        subdir=True,
        max_dbs=1,
        sync=False,         # batch syncs; we'll do one at the end
        writemap=True,      # faster bulk insert on Windows
        map_async=True,
    )

    stats = {
        "files": len(xml_files),
        "sections_indexed": 0,
        "sections_skipped": 0,
        "elapsed_seconds": 0.0,
        "per_title": {},
    }
    t_start = time.time()

    for xml_path in tqdm(xml_files, desc="Titles", unit="title"):
        title_indexed = 0
        title_skipped = 0
        with env.begin(write=True) as txn:
            for section_el in iter_sections(xml_path):
                parsed = parse_section(section_el, release_point)
                if parsed is None:
                    title_skipped += 1
                    continue
                key, value = parsed
                txn.put(key.encode("utf-8"), value, overwrite=True)
                title_indexed += 1
        stats["per_title"][xml_path.stem] = {
            "indexed": title_indexed,
            "skipped": title_skipped,
        }
        stats["sections_indexed"] += title_indexed
        stats["sections_skipped"] += title_skipped

    env.sync()
    env.close()

    stats["elapsed_seconds"] = round(time.time() - t_start, 1)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml-dir", type=Path, required=True, help="Directory of usc*.xml files")
    p.add_argument("--lmdb-path", type=Path, required=True, help="Output LMDB directory")
    p.add_argument("--release", default="unknown", help="Release point label (e.g. 119-36)")
    p.add_argument("--map-size-gb", type=int, default=4, help="LMDB map size in GB")
    args = p.parse_args()

    print(f"[usc_corpus_build] xml_dir   = {args.xml_dir}")
    print(f"[usc_corpus_build] lmdb_path = {args.lmdb_path}")
    print(f"[usc_corpus_build] release   = {args.release}")
    print()

    stats = build(args.xml_dir, args.lmdb_path, args.release, args.map_size_gb)

    print()
    print(f"=== Build complete in {stats['elapsed_seconds']}s ===")
    print(f"Files processed   : {stats['files']}")
    print(f"Sections indexed  : {stats['sections_indexed']:,}")
    print(f"Sections skipped  : {stats['sections_skipped']:,}")
    print()
    print("--- Per-title breakdown (top 10 by section count) ---")
    top = sorted(stats["per_title"].items(), key=lambda kv: -kv[1]["indexed"])[:10]
    for stem, s in top:
        print(f"  {stem:12s}  {s['indexed']:>7,} indexed  {s['skipped']:>4} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
