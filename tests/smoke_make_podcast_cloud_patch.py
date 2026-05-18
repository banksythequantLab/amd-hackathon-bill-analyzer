"""Smoke-verify the make_podcast_cloud.py patch.

Ensures the module imports without errors and the COMFY constant
now points at the local ComfyUI instance.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ast
src = (REPO / "scripts" / "make_podcast_cloud.py").read_text(encoding="utf-8")
ast.parse(src)
print("[OK] syntax: parses cleanly")

# Import the module to confirm the constant is what we set.
from scripts import make_podcast_cloud as m

assert m.COMFY == "http://127.0.0.1:8188", f"COMFY is {m.COMFY!r}"
print(f"[OK] COMFY = {m.COMFY!r}")

# Also sanity-check that no other hard-coded AMD-cluster IP slipped through.
if "165.245.134.1" in src:
    raise SystemExit("[FAIL] file still contains 165.245.134.1 somewhere")
print("[OK] no residual 165.245.134.1 references")
print("[OK] patch verified")
