"""List which agents have canonical baseline entries in eval/report-hr1-ch01.json."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
baseline = json.loads((REPO / "eval" / "report-hr1-ch01.json").read_text(encoding="utf-8"))
agents = baseline.get("agents", {})
print(f"Canonical AMD baseline has {len(agents)} agents for HR1 ch01:")
for name, payload in agents.items():
    n_keys = len(payload.get("output", {})) if isinstance(payload.get("output"), dict) else "?"
    elapsed = payload.get("elapsed_s")
    prompt = payload.get("prompt_tokens")
    compl = payload.get("completion_tokens")
    elapsed_s = "—" if elapsed is None else f"{elapsed}"
    prompt_s = "—" if prompt is None else f"{prompt}"
    compl_s = "—" if compl is None else f"{compl}"
    print(f"  {name:<32}  prompt={prompt_s:>8}  compl={compl_s:>6}  elapsed={elapsed_s:>7}  out_keys={n_keys}")
print()
print("Note: this is just the AGENTS dict; other top-level fields:")
print(" ", sorted(set(baseline.keys()) - {"agents"}))
