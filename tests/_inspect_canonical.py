"""Inspect the actual output shape of each canonical agent."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
baseline = json.loads((REPO / "eval" / "report-hr1-ch01.json").read_text(encoding="utf-8"))
agents = baseline.get("agents", {})
for name, payload in agents.items():
    print(f"=== {name} ===")
    out = payload.get("output")
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, list):
                summary = f"list[{len(v)}]" + (f"  e.g. {v[0]}" if v and not isinstance(v[0], dict) else f"  e.g. {list(v[0].keys()) if v and isinstance(v[0], dict) else ''}")
                print(f"   {k}: {summary[:140]}")
            elif isinstance(v, dict):
                print(f"   {k}: dict({list(v.keys())[:6]})")
            else:
                s = str(v).replace('\n', ' ')[:90]
                print(f"   {k}: {s}")
    else:
        print(f"   (output is not a dict: {type(out).__name__})")
    note = payload.get("note")
    if note:
        print(f"   note: {note[:120]}")
    print()
