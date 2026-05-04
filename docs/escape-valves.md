# Day-by-Day Escape Valves — Scope Locked May 4, 2026

**Project:** AMD Hackathon Bill Analyzer
**Scope:** 14 agents + Definitions congruity enrichment
**Working days:** 6 (Mon May 4 — Fri May 8 build, Sat May 9 polish, Sat May 10 submit AM)
**Travel:** Derek flies CA Saturday, demo + submit must complete by Friday EOD

These are the **automatic cut decisions** that fire when a daily acceptance gate fails. They exist in writing, in the repo, before Day 1 burns. The cut decisions are not "should we cut?" — they are "the gate failed, so we cut."

---

## Daily acceptance gates and triggers

### Mon May 4 — Day 1 EOD gate

**Must have, no exceptions:**
- vLLM serving Qwen3.6-35B-A3B with `enable_prefix_caching=True` confirmed via TTFT delta
- APC speedup ≥ 3x (target 5–10x) on a 100K-token shared prefix smoke test
- USC LMDB uploaded to `/scratch/usc.lmdb` on the instance, `fetch_usc("26:401")` returns in <50 ms
- BiP Post #1 published with real APC numbers

**If APC speedup < 3x → trigger SGLang fallback evaluation tonight.**
SGLang ROCm image is one click away in the AMD UI. Burn 1 hour max comparing — if SGLang doesn't beat 3x either, the architecture itself is wrong and we need a different demo angle, not a different engine.

**If model weights aren't loadable in BF16 by midnight ET →**
roll back to Qwen2.5-32B + Qwen2.5-VL-7B as the proven-stable spine and vision. Less marketing-friendly story, but the architecture survives.

---

### Tue May 5 — Day 2 EOD gate

**Must have:**
- Vision pipeline: Qwen3-VL-8B reads BBB-2021 financial figures into structured JSON in <3 min
- Smart chunker: BBB-2021 → 4 chunks of ≤250K tokens, no mid-Title splits
- Telemetry harness logging timing + VRAM per node

**If selective figure extraction is buggy → cut to "extract all figures" naive mode.** Slower per run but unblocks Days 3-5. Re-add selective extraction Friday morning if time.

**If Title-boundary chunker has edge cases that don't fit the 3 demo bills →** hard-code per-bill chunk boundaries in `tests/fixtures/chunk_overrides.yaml` and ship. Hackathon, not a product.

---

### Wed May 6 — Day 3 EOD gate

**Must have:**
- 5 of the 14 agents implemented and unit-passing on a real BBB-2021 chunk:
  - **Plain-English Summarizer** (#2)
  - **USC Cross-Reference** (#4)
  - **Citation Validator** (#5)
  - **Pork Finder per-chunk** (#9)
  - **Final Report Synthesizer** (#14, even if minimal)
- Error-recovery wrapper (`tools/retry.py`) confirmed working via injected `fetch_usc` failure
- APC reuse measurable across these 5 spine agents on the same chunk

**If <4 agents passing by Wed EOD → AUTOMATIC CUT TO MVP-7 ROSTER.**
Drop in this priority order:
1. Definitions congruity enrichment (FIRST cut)
2. Risk Flagger (#11)
3. Constitutional/Preemption (#12)
4. Effective Date Tracker (#7)
5. Regulatory Authority Mapper (#8)
6. Definitions Extractor (#1)
7. Conflict Spotter (#3) — last to cut, important to demo

This is non-negotiable. By Wednesday night we know whether the agent-factory approach is producing or stalling. If it's stalling, cutting on Wednesday saves 2 days of rework on Friday.

---

### Thu May 7 — Day 4 EOD gate

**Must have:**
- 11 of the 14 agents working end-to-end on a single chunk (10 from Day 3 + ≥6 new today)
- All per-chunk agents (#1-9) integrated into the LangGraph DAG
- Pork Finder catches ≥2 distinct pork-types in BBB-2021 with named beneficiary identification
- Definitions Extractor (#1) emitting structured term definitions

**If <11 agents working by Thursday EOD → CUT DEFINITIONS CONGRUITY ENRICHMENT.**
The base 14 agents are the demo's structural integrity. Definitions congruity is the cherry on top. Without it: still a strong demo. Without working synthesis agents (#10-14): broken demo.

**If pork findings are weak or unconvincing →** swap demo bill #1 from BBB-2021 to FY24 NDAA which has historically been earmark-heavy. Don't show a Pork Finder slide with weak findings — that kills the demo's biggest emotional beat.

---

### Fri May 8 — Day 5+6 combined gate (the big one)

**By Fri 12pm ET:**
- All 14 agents implemented (or cleanly cut per Wed/Thu triggers)
- Full DAG runs end-to-end against BBB-2021 at BF16
- Definitions congruity working for ≥3 example terms (or formally cut)

**By Fri 4pm ET:**
- BF16 vs FP8 dual run complete on BBB-2021
- Agreement rate ≥ 90% (relaxed from 95% — single-day cut, not unreasonable)
- Pork findings rubric scored on at least 5 findings

**By Fri 8pm ET:**
- Dashboard UI working with click-to-bill-page citations
- Demo video recorded (sped-up screen capture, 60-90 sec)
- Slide deck drafted (5-7 slides max)
- BiP Post #4 published with demo clip

**Friday hard-cuts that fire automatically:**

| Time | If still doing | Action |
|---|---|---|
| Fri 12pm | Writing new agent code | STOP. Whatever's in main is the demo. |
| Fri 2pm | Tuning chunk size or precision | STOP. Use whatever produced the cleanest BF16 run. |
| Fri 4pm | Adding new findings to rubric | STOP. Score what's there, write the slide. |
| Fri 6pm | Iterating on UI design | STOP. Ugly-but-functional > pretty-but-broken. |
| Fri 9pm | Anything not directly demo-related | STOP. Sleep. Submit Saturday morning. |

**If demo isn't recordable by Fri 9pm ET → record what works on Friday + submit incomplete with honest README about what's missing.** Hackathon judges respect honesty more than hidden brokenness.

---

### Sat May 9 — pre-flight day

**0700-1000 ET only:**
- Final demo run, capture fresh metrics
- Submit on lablab.ai
- Submit Build-in-Public artifacts
- BiP Post #5 wrap

**1000 ET hard stop on all work.** Pack, fly to CA. The submission is in.

---

## Three things that don't get cut, ever

These are the structural integrity of the demo. If any of these break, we re-architect, we don't ship without them:

1. **Three Qwen models concurrent on one MI300X.** This is THE hook. No demo without this.
2. **APC delivering measurable speedup across ≥4 agents on a shared chunk.** This is what justifies "needs 192GB."
3. **USC cross-reference grounding.** Findings without statutory grounding are just LLM hallucination with extra steps.

If any of these is broken on Friday morning, the path forward is "fix this OR don't ship," not "ship without it."

---

## What the Definitions congruity enrichment actually adds

This is the spec, locked, so it can't grow during the build:

**Tool:** `fetch_term_definitions(term: str) -> list[Definition]`
- Queries LMDB with FTS-style search for `"{term} means"` and `"{term} has the meaning"` patterns
- Returns `[{title, section, definition_text, source_url}]`
- Cap at 10 results, sorted by Title (gives Title 26 / Title 42 priority)

**Agent integration:** Definitions Extractor (#1) emits its findings, then for each defined term in the bill, calls `fetch_term_definitions` and adds a `congruity` field:
```python
{
    "term": "qualified individual",
    "bill_definition": "...",
    "bill_location": "Title II, §201(a)",
    "congruity": {
        "status": "novel" | "consistent" | "conflicting",
        "existing_definitions": [...],  # from fetch_term_definitions
        "conflicts": [
            "26 USC §401(c)(2) defines 'qualified individual' as ...",
            "42 USC §3796b(7) defines 'qualified individual' as ..."
        ],
        "summary": "Bill creates a NEW definition that conflicts with..."
    }
}
```

**Demo punchline:** *"This bill quietly redefines 'qualified individual' in a way that conflicts with how Title 26 has used it since 1986. The Definitions Extractor caught it. No human reviewer would have."*

**If this isn't producing visibly compelling examples by Friday noon → cut it cleanly, don't kludge it.**
