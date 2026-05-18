"""Verify all 3090-fork repo-path patches.

Confirms:
  - Every patched file parses cleanly as Python.
  - No live .py file references B:\\amd-hackathon-bill-analyzer\\ (without
    the -3090 suffix) except in comments labeled "was" or "3090 FORK".

Class D files (docs/, eval/canonical/archive*, _master.txt frozen artifacts,
public URLs in build_deck_pdf.py) are out of scope and untouched.
"""
from __future__ import annotations
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PATCHED = [
    "comfy/edit_references.py",
    "comfy/render_references.py",
    "comfy/render_ltx_batch.py",
    "comfy/render_ltx_smoke.py",
    "comfy/run_scene_critic.py",
    "eval/submission/dump_lines.py",
    "eval/submission/dump_top.py",
    "eval/submission/dump_wf.py",
    "scripts/infinitetalk_pipeline.py",
    "src/orchestrator/run_chunk.py",
    "src/vision/extract_figures.py",
    "tests/run_one_agent.py",
    "tests/run_one_agent_remote.py",
    "tests/smoke_four_agents.py",
    "tests/smoke_first_two_agents.py",
]

# Files allowed to mention the old path (in their code, not comments):
ALLOWED_RESIDUAL = {
    "eval/submission/build_deck_pdf.py",      # public GitHub/HF URLs (different thing)
    "scripts/make_podcast_cloud.py",           # has "Was a hardcoded ..." breadcrumb comment
    "tests/verify_repo_path_patches.py",       # this verifier
}

# Pattern matches B:\amd-hackathon-bill-analyzer\ NOT followed by -3090.
old_path_re = re.compile(r"B:[\\/]amd-hackathon-bill-analyzer(?!-3090)", re.IGNORECASE)

# ---------------- PHASE 1: syntax check ----------------
print("=" * 70)
print("PHASE 1: syntax check every patched file")
print("=" * 70)
syntax_failures = []
for rel in PATCHED:
    path = REPO / rel
    if not path.exists():
        print(f"  [MISS] {rel}")
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        print(f"  [OK ] {rel}")
    except SyntaxError as e:
        print(f"  [FAIL] {rel}: {e}")
        syntax_failures.append(rel)

# ---------------- PHASE 2: scan repo .py for old path ----------------
print()
print("=" * 70)
print("PHASE 2: scan all repo .py for residual old-fork paths")
print("=" * 70)

skip_dir_parts = {".git", "__pycache__", "node_modules", ".venv", "venv"}
unexpected = []   # bare code references
expected = []     # comment breadcrumbs and allow-listed files

for path in REPO.rglob("*.py"):
    if any(part in skip_dir_parts for part in path.parts):
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    rel = path.relative_to(REPO).as_posix()
    for i, line in enumerate(text.splitlines(), start=1):
        if old_path_re.search(line):
            is_allowed_file = rel in ALLOWED_RESIDUAL
            stripped = line.lstrip()
            # Comment context = either the line starts with # OR an inline
            # trailing # comment contains the old path. We split on # and
            # see whether the old path appears only on the right side.
            code_part, sep, comment_part = line.partition("#")
            is_pure_comment_line = stripped.startswith("#") or '"""' in line or "'''" in line
            is_inline_comment_only = sep and old_path_re.search(comment_part) and not old_path_re.search(code_part)
            is_breadcrumb_text = (
                "was" in line.lower() or "3090 FORK" in line or "old fork" in line.lower()
            )
            if is_allowed_file or ((is_pure_comment_line or is_inline_comment_only) and is_breadcrumb_text):
                expected.append((rel, i, line.strip()[:100]))
            else:
                unexpected.append((rel, i, line.strip()[:100]))

print(f"  Expected residuals (breadcrumbs + allow-listed): {len(expected)}")
for rel, ln, line in expected[:25]:
    print(f"    {rel}:{ln}  {line}")
if len(expected) > 25:
    print(f"    ... and {len(expected) - 25} more")

print()
print(f"  Unexpected residuals (should be ZERO): {len(unexpected)}")
for rel, ln, line in unexpected:
    print(f"    {rel}:{ln}  {line}")

# ---------------- VERDICT ----------------
print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
ok = not syntax_failures and not unexpected
print(f"  Syntax failures:           {len(syntax_failures)}")
print(f"  Unexpected path residuals: {len(unexpected)}")
print(f"  REPO-PATH SWEEP: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
