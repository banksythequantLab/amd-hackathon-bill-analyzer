"""Compare 3090-fork SceneCritic results vs AMD canonical baseline."""
import json
from pathlib import Path

FORK = Path(r"B:\amd-hackathon-bill-analyzer-3090\eval\scene-critic-bbb-ch01-day6.json")
AMD  = Path(r"B:\amd-hackathon-bill-analyzer\eval\scene-critic-bbb-ch01-day6.json")

fork = json.loads(FORK.read_text(encoding="utf-8"))
amd  = json.loads(AMD.read_text(encoding="utf-8"))

print(f"{'':<22} {'FORK':<28}  {'AMD':<28}")
print(f"{'master':<22} {fork.get('master_image','?'):<28}  {amd.get('master_image','?'):<28}")
print(f"{'total elapsed s':<22} {fork.get('elapsed_s','?'):<28}  {amd.get('elapsed_s','?'):<28}")
print(f"{'summary':<22} {str(fork.get('summary')):<28}  {str(amd.get('summary')):<28}")
print()

# Index by scene_id
fork_by_id = {r["scene_id"]: r for r in fork.get("results", [])}
amd_by_id  = {r["scene_id"]: r for r in amd.get("results",  [])}

print(f"{'scene_id':<10} {'FORK verdict':<14} {'AMD verdict':<14} {'agree?':<8} {'FORK conf':<10} {'AMD conf':<10}")
print("-" * 76)
agree = 0
total = 0
for sid in sorted(set(fork_by_id) | set(amd_by_id)):
    fv = fork_by_id.get(sid, {}).get("verdict", "—")
    av = amd_by_id.get(sid, {}).get("verdict", "—")
    fc = fork_by_id.get(sid, {}).get("confidence", "—")
    ac = amd_by_id.get(sid, {}).get("confidence", "—")
    a  = "yes" if fv == av else "no"
    if fv == av:
        agree += 1
    total += 1
    marker = "" if fv == av else "  ***"
    print(f"{sid:<10} {fv:<14} {av:<14} {a:<8} {fc:<10} {ac:<10}{marker}")
print()
print(f"agreement: {agree}/{total} ({100*agree/total:.0f}%)")
