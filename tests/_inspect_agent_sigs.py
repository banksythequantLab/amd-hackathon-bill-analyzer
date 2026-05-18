"""Inspect signatures of all agent .run() methods so we know how to wire them."""
import ast
import importlib
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

AGENT_FILES = [
    "citation_validator",
    "conflict_spotter",
    "fiscal_impact_estimator",
    "headline_ranker",
    "podcast_generator",
    "podcast_headlines_generator",
    "podcast_script_writer",
    "pork_finder",
    "prompt_relay_author",
    "slide_critic",
    "slide_prompt_generator",
    "stakeholder_tracer",
    "wan_motion_prompt_generator",
    "youtube_metadata_generator",
]

for mod_name in AGENT_FILES:
    try:
        mod = importlib.import_module(f"src.agents.{mod_name}")
    except Exception as e:
        print(f"{mod_name}: IMPORT FAIL: {e}")
        continue

    # Find the Agent class - heuristic: class whose name doesn't start with _
    # and which has a .run method.
    found = None
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != mod.__name__:
            continue
        if hasattr(cls, "run"):
            found = (name, cls)
            break

    if not found:
        print(f"{mod_name}: no Agent class with .run() found")
        continue

    name, cls = found
    sig = inspect.signature(cls.run)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    endpoint = getattr(cls, "target_endpoint", "—")
    model = getattr(cls, "target_model", "—")
    max_tok = getattr(cls, "max_tokens", "—")
    print(f"{mod_name:<30} class={name:<28} model={model:<10} max_tok={max_tok}")
    for p in params:
        default = "" if p.default is inspect.Parameter.empty else f"={p.default!r}"
        ann = p.annotation if p.annotation is not inspect.Parameter.empty else ""
        print(f"   {p.name}: {ann} {default}")
    print()
