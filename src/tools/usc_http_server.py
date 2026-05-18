"""HTTP server fronting the local USC LMDB on port 8004.

This is the 3090-fork's local equivalent of the AMD droplet's USC service.
The Gradio app (app.py) and citation_validator both expect:

    GET /usc/lookup?citation=<cite>   -> 200 JSON | 404 if not found

JSON shape matches src.tools.fetch_usc.FetchUsc output:
    {title, section, heading, text, sourceCredit, cross_refs, source_url, release_point}

Why stdlib http.server instead of FastAPI/Flask:
  - Zero new dependencies (matches the existing requirements.txt)
  - The endpoint is dead-simple: one route, one query param
  - 60K LMDB entries, sub-millisecond lookups -- the bottleneck is never the framework
  - Single-process is fine; LMDB is read-only and lock-free across readers anyway

Usage:
    python -m src.tools.usc_http_server                # uses defaults
    python -m src.tools.usc_http_server --port 8004
    python -m src.tools.usc_http_server --lmdb /path/to/usc.lmdb --port 8004 --host 0.0.0.0

Health endpoints:
    GET /                   -> 200, text/plain banner
    GET /health             -> 200, JSON {"ok": true, "entries": N}
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Path setup so we can run as `python src/tools/usc_http_server.py`
# OR `python -m src.tools.usc_http_server`.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.tools.fetch_usc import FetchUsc


# Fetcher is opened once at server-start and shared across requests.
# Set in main() before serve_forever().
_FETCHER: FetchUsc | None = None
_LMDB_ENTRIES: int = 0


class UscLookupHandler(BaseHTTPRequestHandler):
    """One handler instance per request. _FETCHER is module-global."""

    server_version = "UscLmdb/1.0"

    # Override the noisy default that prints every hit to stderr; we use
    # our own short access log so the format stays compact.
    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(f"[usc] {self.address_string()} {fmt % args}\n")
        sys.stdout.flush()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if _FETCHER is None:
            self._send_json(503, {"error": "fetcher not initialized"})
            return

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_text(
                200,
                "USC LMDB HTTP server (3090 fork)\n"
                f"entries: {_LMDB_ENTRIES:,}\n"
                "endpoints: GET /usc/lookup?citation=<cite>, GET /health\n",
            )
            return

        if path == "/health":
            self._send_json(200, {"ok": True, "entries": _LMDB_ENTRIES})
            return

        if path == "/usc/lookup":
            qs = parse_qs(parsed.query)
            cites = qs.get("citation", [])
            if not cites:
                self._send_json(400, {"error": "missing citation query param"})
                return
            citation = cites[0]
            try:
                record = _FETCHER(citation)
            except Exception as e:
                self._send_json(500, {
                    "error": f"{type(e).__name__}: {e}",
                    "citation": citation,
                })
                return
            if record is None:
                self._send_json(404, {
                    "error": "not found",
                    "citation": citation,
                })
                return
            self._send_json(200, record)
            return

        self._send_json(404, {"error": f"unknown path: {path}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lmdb",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "usc.lmdb",
        help="Path to USC LMDB directory (default: <repo>/data/usc.lmdb)",
    )
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind host (default: 127.0.0.1, set 0.0.0.0 to expose)")
    ap.add_argument("--port", type=int, default=8004,
                    help="Bind port (default: 8004 to match app.py USC_HTTP default)")
    args = ap.parse_args()

    if not args.lmdb.exists():
        print(f"[FAIL] LMDB not found at {args.lmdb}", file=sys.stderr)
        return 2

    global _FETCHER, _LMDB_ENTRIES
    print(f"[usc] opening {args.lmdb} ...")
    _FETCHER = FetchUsc(str(args.lmdb))

    # Count entries once at startup so /health is O(1). Reuse the
    # fetcher's already-open env -- lmdb refuses a second open of the
    # same environment within one process.
    with _FETCHER.env.begin() as txn:
        _LMDB_ENTRIES = txn.stat()["entries"]
    print(f"[usc] LMDB loaded: {_LMDB_ENTRIES:,} entries")

    server = ThreadingHTTPServer((args.host, args.port), UscLookupHandler)
    print(f"[usc] listening on http://{args.host}:{args.port}")
    print(f"[usc] try: curl 'http://{args.host}:{args.port}/usc/lookup?citation=42+USC+1395dd'")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[usc] shutting down")
    finally:
        server.server_close()
        if _FETCHER is not None:
            _FETCHER.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
