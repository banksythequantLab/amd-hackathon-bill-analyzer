"""HTTP-mode USC fetcher.

Drop-in replacement for FetchUsc when the LMDB lives on a remote service.
Same callable interface so enrich_with_usc works unchanged.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx


class HttpFetchUsc:
    """Calls a remote USC LMDB service over HTTP. Same interface as FetchUsc."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, headers={"Connection": "close"})
        self._stats = {"hits": 0, "misses": 0, "calls": 0, "errors": 0}

    def __call__(self, citation: str, *, max_text_len: int = 8000) -> Optional[dict]:
        self._stats["calls"] += 1
        try:
            r = self.client.get(
                f"{self.base_url}/usc/lookup",
                params={"citation": citation},
            )
            if r.status_code == 404:
                self._stats["misses"] += 1
                return None
            r.raise_for_status()
            record = r.json()
            if max_text_len and len(record.get("text", "")) > max_text_len:
                record["text"] = record["text"][:max_text_len] + "...[truncated]"
                record["_truncated"] = True
            self._stats["hits"] += 1
            return record
        except httpx.HTTPError as e:
            self._stats["errors"] += 1
            self._stats["misses"] += 1
            return None

    def stats(self) -> dict:
        return dict(self._stats)

    def close(self) -> None:
        self.client.close()


def get_fetcher(local_path: str | None = None, http_url: str | None = None):
    """Returns either a local FetchUsc or HttpFetchUsc depending on what's available.

    Priority:
      1. If http_url given, use HTTP (preferred for HF Space deployment)
      2. If local_path exists as LMDB, use local
      3. Else None (graceful degrade — citations still listed but not enriched)
    """
    from pathlib import Path
    
    if http_url:
        return HttpFetchUsc(http_url)
    
    if local_path:
        p = Path(local_path)
        if p.exists() and (p / "data.mdb").exists():
            from src.tools.fetch_usc import FetchUsc
            return FetchUsc(p)
    
    return None
