"""Verify all 3090-fork URL patches.

Re-scans the repo for `165.245.134.1` references and asserts only Class D
historical files retain them. Also AST-parses every file we patched to
catch any syntax breakage.
"""
from __future__ import annotations
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files we patched in this sweep (Class A, B, C):
PATCHED = [
    # Class A: ComfyUI :8188 endpoints
    "scripts/make_podcast_cloud.py",
    "comfy/edit_references.py",
    "comfy/render_ltx_batch.py",
    "comfy/render_ltx_smoke.py",
    "comfy/render_references.py",
    "comfy/run_scene_critic.py",
    "eval/submission/ping_spine.py",
    "eval/submission/submit_banksy_doodle.py",
    "eval/submission/submit_slides.py",
    "eval/submission/submit_team_covers.py",
    # Class B: Spine :8001 endpoints
    "app.py",
    "tests/run_one_agent.py",
    "tests/run_one_agent_remote.py",
    # Class C: Vision :8002 endpoints
    "src/agents/scene_critic.py",
    "src/vision/extract_figures.py",
]

# Files that legitimately retain the AMD IP (historical / SSH-only / test guards).
# Any line in a file NOT in this set that still mentions 165.245.134.1
# must be a comment breadcrumb ("# was 165...") to be acceptable.
ALLOWED_RESIDUAL = {
    "docs/day3-runbook.md",                                  # historical narrative
    "eval/canonical/archive-5chunk-bbb/bbb-ch02.json",       # frozen baseline log
    "eval/canonical/archive-5chunk-bbb/bbb-merged.json",     # frozen baseline log
    "tests/run_one_agent_remote.py",                          # SSH-tunnels to AMD by design
    "tests/smoke_make_podcast_cloud_patch.py",                # asserts IP absence
    "tests/verify_amd_url_patches.py",                        # this verifier itself
}

print("=" * 70)
print("PHASE 1: syntax check every patched file")
print("=" * 70)
syntax_failures = []
for rel in PATCHED:
    path = REPO / rel
    if not path.exists():
        print(f"  [MISS] {rel} — file not found, skipping")
        continue
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            print(f"  [OK ] syntax: {rel}")
        except SyntaxError as e:
            print(f"  [FAIL] syntax: {rel}: {e}")
            syntax_failures.append(rel)
    else:
        print(f"  [skip non-python] {rel}")

print()
print("=" * 70)
print("PHASE 2: scan repo for residual 165.245.134.1 references")
print("=" * 70)

ip_pattern = re.compile(r"165\.245\.134\.1")
unexpected: list[tuple[Path, int, str]] = []
expected: list[tuple[Path, int, str]] = []

scan_exts = {".py", ".json", ".md", ".txt", ".toml", ".cfg", ".yml", ".yaml"}
skip_dir_parts = {".git", "__pycache__", "node_modules", ".venv", "venv"}

for path in REPO.rglob("*"):
    if not path.is_file():
        continue
    if any(part in skip_dir_parts for part in path.parts):
        continue
    if path.suffix.lower() not in scan_exts:
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), start=1):
        if ip_pattern.search(line):
            rel = path.relative_to(REPO).as_posix()
            is_in_allowed_set = rel in ALLOWED_RESIDUAL
            is_comment_breadcrumb = (
                line.lstrip().startswith("#")
                or "was 165" in line.lower()
                or "was http://165" in line.lower()
                or "AMD" in line
            )
            if is_in_allowed_set or is_comment_breadcrumb:
                expected.append((rel, i, line.strip()[:90]))
            else:
                unexpected.append((rel, i, line.strip()[:90]))

print(f"  Expected residuals (historical, breadcrumbs): {len(expected)}")
for rel, ln, line in expected[:20]:
    print(f"    {rel}:{ln}  {line}")
if len(expected) > 20:
    print(f"    ... and {len(expected) - 20} more")

print()
print(f"  Unexpected residuals (should be ZERO): {len(unexpected)}")
for rel, ln, line in unexpected:
    print(f"    {rel}:{ln}  {line}")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
ok = not syntax_failures and not unexpected
print(f"  Syntax failures:        {len(syntax_failures)}")
print(f"  Unexpected IP residuals: {len(unexpected)}")
print(f"  PATCH SWEEP: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
