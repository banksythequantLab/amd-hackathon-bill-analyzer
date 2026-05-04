"""
fetch_usc — primary tool for USC cross-reference and citation validator agents.

Reads from the LMDB built by infra/usc_corpus_build.py.

Usage:
    from src.tools.fetch_usc import FetchUsc
    fetcher = FetchUsc("/path/to/usc.lmdb")
    record = fetcher("26:401")
    # record = {"title": "26", "section": "401", "heading": "...", "text": "...", ...}
    fetcher.close()
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import lmdb
import orjson


# Citation patterns the agents may produce, in order of preference:
#  '26:401(k)'  - our normalized form
#  '26 USC 401' / '26 U.S.C. 401' - human-readable
#  'Section 401 of Title 26' / '26 U.S.C. § 401'
CITE_RE_NORMALIZED = re.compile(r"^(?P<title>[0-9]+[a-zA-Z]?)\s*[:\s]\s*(?P<section>[0-9A-Za-z\-\(\)\.]+)$")
CITE_RE_USC        = re.compile(r"(?P<title>[0-9]+[a-zA-Z]?)\s*U\.?\s*S\.?\s*C\.?\s*(?:§|sec\.?|section)?\s*(?P<section>[0-9A-Za-z\-\(\)\.]+)", re.IGNORECASE)


class FetchUsc:
    """Reusable wrapper around the USC LMDB."""

    def __init__(self, lmdb_path: str | Path):
        path = Path(lmdb_path)
        if not path.exists():
            raise FileNotFoundError(f"USC LMDB not found at {path}")
        self.env = lmdb.open(str(path), readonly=True, lock=False, subdir=True)
        self._stats = {"hits": 0, "misses": 0, "calls": 0}

    @staticmethod
    def normalize(citation: str) -> Optional[str]:
        """Normalize a free-form citation into the LMDB key form 'title:section'."""
        c = citation.strip()
        if not c:
            return None

        # Already normalized?
        m = CITE_RE_NORMALIZED.match(c)
        if m:
            return f"{m.group('title').lower()}:{m.group('section').lower()}"

        # USC-style?
        m = CITE_RE_USC.search(c)
        if m:
            return f"{m.group('title').lower()}:{m.group('section').lower()}"

        return None

    def __call__(self, citation: str, *, max_text_len: int = 8000) -> Optional[dict]:
        """Fetch a USC section. Returns None if not found.

        max_text_len bounds the returned `text` field — for many statutes
        the full text is 100K+ chars which would blow up the agent prompt
        budget. 8K chars (~2K tokens) is plenty for citation validation.
        """
        self._stats["calls"] += 1
        key = self.normalize(citation)
        if key is None:
            self._stats["misses"] += 1
            return None

        with self.env.begin() as txn:
            raw = txn.get(key.encode("utf-8"))

        if raw is None:
            self._stats["misses"] += 1
            return None

        record = orjson.loads(raw)
        if max_text_len and len(record.get("text", "")) > max_text_len:
            record["text"] = record["text"][:max_text_len] + "...[truncated]"
            record["_truncated"] = True
        self._stats["hits"] += 1
        return record

    def stats(self) -> dict:
        return dict(self._stats)

    def close(self) -> None:
        self.env.close()
