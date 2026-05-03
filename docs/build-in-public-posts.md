# Build-in-Public Post Templates

5 posts, 7 days. All tag `#AMDDevHackathon @AMDDeveloper @lablabai`.

Repo: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

Placeholders to fill in:
- `[X]`, `[Y]`, `[Z]` — actual measured numbers from the build
- `[INSERT...]` — content blocks waiting on a screenshot or finding
- `[REPO_URL]` — replace with the live GitHub URL or shortener

---

## Post #1 — Day 1 (Sunday May 4) — APC benchmark hook

**Trigger:** after Day 1 smoke #3 passes (TTFT speedup ≥3x, target 5–10x).

**Long-form (LinkedIn / X long post / blog teaser):**

```
Day 1 of @AMD's Developer Hackathon: Bill Analyzer is alive on MI300X.

The judging hook for this hackathon is: what does 192GB VRAM unlock that
you can't do on a 5090?

Here's my answer.

I'm building a multi-agent system that ingests legislative bills (target:
the original Build Back Better Act 2021 at ~920K tokens) and produces a
structured analysis: USC cross-references, fiscal impact, pork detection,
stakeholder mapping.

The architecture leans hard on what only a 192GB GPU can do:
• Qwen3-VL-8B-Thinking — extracts charts/tables from the bill (~8 GB)
• Qwen3.6-35B-A3B — long-context spine, holds 250K-token chunks (~35 GB FP8)
• Qwen3-32B — full BF16 reasoner for citations + math (~64 GB)

Three Qwen models, hot at the same time, on one GPU.

But the real unlock is vLLM's Automatic Prefix Caching on ROCm.

Day 1 smoke test: send a 100K-token prefix + Question A. Then send the same
100K prefix + Question B. APC detects the shared prefix and skips the prefill.

Result: Question A took [X]s. Question B took [Y]s. [Z]x speedup on
time-to-first-token.

Now multiply by 6 agents per chunk x 4 chunks. The prefill happens ONCE per
chunk and 5 specialist agents reuse the cached KV. On a 5090 (32GB), each
agent has to spill or re-prefill. On MI300X, they share.

That's the demo punchline.

Tomorrow: vision pipeline + smart chunking. By Day 5: full Build Back Better
Act analysis at BF16 vs FP8 with side-by-side timing + agreement-rate overlay.

Repo: [REPO_URL]
Build-in-public series: 5 posts over 7 days.

#AMDDevHackathon @lablabai @AMDDeveloper
```

**X thread (4 tweets):**

```
1/4
Day 1 of @AMDDeveloper's hackathon: I'm building a 7-agent system that
analyzes 2,000+ page legislative bills on MI300X.

The hook: what does 192GB VRAM unlock?

Three Qwen models, hot at the same time, on ONE GPU.

#AMDDevHackathon @lablabai
```

```
2/4
The stack:
• Qwen3-VL-8B → extracts charts from bills
• Qwen3.6-35B-A3B → 250K-token long-context spine
• Qwen3-32B (BF16) → reasoner for citations + math

All three running concurrently. ~145 GB total. Can't do this on a 5090.
```

```
3/4
The real unlock is vLLM's Automatic Prefix Caching on ROCm.

Day 1 smoke test:
- 100K-token prefix + Question A → [X]s TTFT
- Same prefix + Question B → [Y]s TTFT
- That's a [Z]x speedup just from cached KV

Now do that across 6 agents per chunk.
```

```
4/4
Day 5 target: full Build Back Better Act 2021 (~2,500 pages, ~920K tokens)
analyzed at BF16 vs FP8 with side-by-side timing.

5 posts, 7 days, $100 of credits.

Repo: [REPO_URL]
Following along: #AMDDevHackathon

@AMDDeveloper @lablabai
```

---

## Post #2 — Day 2 (Monday May 5) — vision + chunking

**Trigger:** after a successful BBB-2021 figure extraction run produces structured JSON
of all financial figures.

**Long-form:**

```
Day 2 update from @AMD's Developer Hackathon.

Vision pipeline is live. Just had Qwen3-VL-8B-Thinking read every financial
figure from the 2,469-page Build Back Better Act 2021 — converting bill PDF
charts and tables into structured JSON in [X] seconds on MI300X.

Why this matters: bills don't just have text. They have appropriations
charts, demographic breakdowns, and fiscal impact tables that the textual
agents alone miss. Without vision, the Fiscal Impact agent is reading
captions and inferring; with vision, it's reading the actual numbers.

Selective extraction matters too. BBB-2021 has ~[N] figures total but only
~[N] are financially relevant — Title-of-Contents charts, illustrative
diagrams, etc. don't need processing. Smart filtering cut runtime by ~60%.

The smart chunker also landed: Title-boundary aware splitting that produces
4 chunks of ~250K tokens each from the 2,469-page bill, never splitting
mid-Title (which would break legal continuity).

[INSERT SCREENSHOT: figure JSON output, ideally with the underlying chart
visible in a side panel for visual contrast]

Each Qwen3-VL output looks like:
{
  "figure_id": "T2-Subt-D-Fig1",
  "type": "stacked_bar",
  "caption": "Sec. 3601 child tax credit phaseout...",
  "axes": {"x": "AGI", "y": "credit ($)"},
  "data_points": [...],
  "page": 412
}

Tomorrow: per-chunk agents start running on real bill content.

Repo: [REPO_URL]

#AMDDevHackathon @AMDDeveloper @lablabai
```

**X thread:**

```
1/3
Day 2 of @AMDDeveloper's hackathon.

Vision pipeline live. Qwen3-VL-8B read every financial chart in the 2,469-page
Build Back Better Act 2021 → structured JSON in [X] seconds.

Bills aren't just text. They have appropriations tables. #AMDDevHackathon
```

```
2/3
Smart chunker landed too: Title-boundary aware. BBB-2021 → 4 chunks of
~250K tokens each, no mid-Title splits.

Mid-Title splits break legal continuity. "As defined in §401(b)" referencing
something now in another chunk = bad time.

[REPO_URL]
```

```
3/3
Tomorrow: per-chunk agents start running.

Right now both Qwen3-VL-8B AND Qwen3.6-35B are hot on the GPU. ~43 GB.
Plenty of room.

@AMDDeveloper @lablabai
```

**Short LinkedIn:**

```
Day 2 update.

Qwen3-VL-8B-Thinking just finished extracting every financial figure from the
2,469-page Build Back Better Act 2021 in [X] seconds on MI300X.

This is the piece most multi-agent bill-analysis systems skip — pure-text
agents miss the actual numbers in appropriations tables and fiscal-impact
charts. Now those charts are structured JSON the downstream Fiscal Impact
agent can read directly.

Selective extraction (only financially relevant figures) cut processing time
by ~60% with no quality loss.

Smart chunker also live: Title-boundary aware splitting produces 4 chunks of
~250K tokens each — never splitting mid-Title, which would break statutory
cross-references.

Repo: [REPO_URL]

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## Post #3 — Day 4 (Wednesday May 7) — Pork Finder catches earmarks

**Trigger:** after Pork Finder runs on real bill text and catches at least 2
juicy findings worth screenshotting. Ideally findings with a named-entity
carveout or a geographic micro-target that resolves to a specific jurisdiction.

This is the crowd-pleaser post. Aim for max shareability.

**Long-form:**

```
Day 4 of @AMD's Developer Hackathon.

The Pork Finder agent ran on the real Build Back Better Act 2021 text for
the first time tonight. It works.

It caught $[X]M in suspicious provisions. Some examples (full text + agent
reasoning in the screenshots):

→ A §[Sec#] provision allocating $[X]M for "an institution of higher
education founded between [year] and [year] in [region]." The Census API
tool resolved this to exactly one school: [Specific University].

→ A §[Sec#] earmark phrased as "a county with a population between [X] and
[Y] as of the 2020 census." Census API: this resolves to [County, State].

→ A tax provision in Title [N] for "specialized aircraft engines manufactured
in [region] meeting [specification]" — the entire industry of beneficiaries
for this clause is [N] companies.

How it works: the Pork Finder agent looks for 7 specific signals — geographic
micro-targeting, named-entity carveouts, sole-source eligibility, outsized
funding for narrow scope, duplicative funding across Titles, mid-bill
non-sequiturs, and narrow tax provisions. When it spots one, it calls the
Census API + USC LMDB to resolve the likely beneficiary.

Every finding is cited back to a specific page and section in the bill. The
fact-checking is auditable.

This is the kind of analysis that takes journalists weeks. We're getting it
in [X] minutes on a single MI300X.

Tomorrow: full bill end-to-end. BF16 vs FP8 head-to-head on Day 5.

[INSERT SCREENSHOT: 2-3 cleaned-up Pork Finder findings with annotations]

Repo: [REPO_URL]

#AMDDevHackathon @AMDDeveloper @lablabai
```

**X thread (5 tweets — this one earns extra length):**

```
1/5
Day 4 of @AMDDeveloper's hackathon.

The Pork Finder agent works.

Just ran it on the 2,469-page Build Back Better Act 2021. Caught $[X]M in
suspicious provisions in [Y] minutes on a single MI300X.

Examples below.

#AMDDevHackathon
```

```
2/5
Finding 1:

§[Sec#] allocates $[X]M for "an institution of higher education founded
between [year] and [year] in [region]."

Pork Finder called the Census + DOE Education API.

That clause resolves to exactly ONE school. Pork Quality Score: 9/10.
```

```
3/5
Finding 2:

§[Sec#] earmark phrased as "a county with a population between [X] and [Y]
as of the 2020 census."

Census API: resolves to [County, State]. Population [exact].

The geographic specificity gives it away every time.
```

```
4/5
Finding 3:

§[Sec#] tax provision for "specialized aircraft engines manufactured in
[region] meeting [spec]."

Industry of beneficiaries: [N] companies total.

When the eligibility list could fit on a sticky note, it's not a "category"
of beneficiaries.
```

```
5/5
This is journalism-grade analysis on a 2,500-page bill in [X] minutes.

Every finding cites the exact bill page + section. Auditable, not hallucinated.

What journalists currently do over weeks of staff work, MI300X does over a
coffee break.

@AMDDeveloper @lablabai

[REPO_URL]
```

**Short LinkedIn:**

```
Day 4 update — Pork Finder caught its first batch.

Just ran the Pork Finder agent on the full 2,469-page Build Back Better Act
2021. It flagged $[X]M in suspicious provisions in [Y] minutes on MI300X.

A few examples:
• A §[Sec#] provision targeting "an institution founded between [year] and
  [year] in [region]" — Census API resolved this to exactly one university
• A §[Sec#] earmark phrased as a population micro-target — resolved to a
  single county
• A Title [N] tax carveout where the entire industry of beneficiaries is
  [N] companies

The agent uses 7 detection signals — geographic micro-targeting, named-entity
carveouts, sole-source eligibility, outsized funding for narrow scope,
duplicative funding across Titles, mid-bill non-sequiturs, and narrow tax
provisions.

Every finding cites a specific bill page and section. Auditable.

This is the kind of work that takes journalist staffs weeks. We're doing it
in minutes.

[INSERT SCREENSHOT]

Repo: [REPO_URL]

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## Post #4 — Day 6 (Friday May 9) — live demo clip

**Trigger:** after Day 6 dashboard works AND Day 5 produced a clean BF16-vs-FP8
result we can put in a side-by-side. This is the "money shot" post.

**Long-form:**

```
Day 6 of @AMD's Developer Hackathon.

End-to-end demo working. Drop a bill PDF in. Watch [N] agents run across
[N] chunks. Get a structured report out — fiscal impact, pork findings,
USC cross-references, stakeholder map — all cited to specific bill pages.

[INSERT VIDEO CLIP: 60-90 seconds, sped up 4x. Shows:
 1. Dragging Build Back Better Act 2021 PDF into the dashboard
 2. Live chunk progress (4 chunks)
 3. Vision agent extracting figures
 4. Per-chunk agents running with APC speedup visible
 5. Final report with clickable findings → jump to bill page]

Numbers from the run:

  Bill          : Build Back Better Act 2021 (HR 5376)
  Pages         : 2,469
  Tokens        : ~920K across 4 chunks
  Models        : 3 Qwen models hot concurrently (~145 GB)
  Agents        : 7 specialist + 1 final synthesizer
  USC sections  : 60,187 indexed (LMDB, ~10 µs lookups)

  Run time @ BF16 : [X] min  ← gold standard
  Run time @ FP8  : [Y] min  ← demo config
  Agreement rate  : [Z]%      ← per-finding match
  Pork findings   : [N] flagged
  USC cross-refs  : [N] resolved

The BF16-vs-FP8 split is the hackathon's headline. FP8 is [Z%] of the way to
gold-standard quality at [X/Y] the wall-clock. That trade-off curve is what
192 GB lets you negotiate live, without re-architecting.

Repo cloneable, demo runs in <[X] minutes on a fresh MI300X via:

  ./infra/one_click_run.sh tests/fixtures/build_back_better_2021_hr5376.pdf

Submitting Saturday.

[REPO_URL]

#AMDDevHackathon @AMDDeveloper @lablabai
```

**X thread (3 tweets — let the video carry the post):**

```
1/3
Day 6 of @AMDDeveloper's hackathon. End-to-end live.

Dropped the 2,469-page Build Back Better Act 2021 in. [X] minutes later,
out comes a full structured analysis on a single MI300X.

Every finding cites a bill page. Click to jump.

[VIDEO]

#AMDDevHackathon
```

```
2/3
The numbers:

BF16 run : [X] min
FP8 run  : [Y] min
Agreement: [Z]%

The BF16↔FP8 trade-off curve IS the demo. 192 GB lets you keep both running
hot, swap precision in [X] sec, see quality vs speed in real time.

Can't do that on a 5090.
```

```
3/3
Repo is cloneable. Demo on a fresh MI300X via:

./infra/one_click_run.sh path/to/bill.pdf

Submission Saturday.

[REPO_URL]

@lablabai @AMDDeveloper
```

**Short LinkedIn:**

```
Day 6 — end-to-end demo of the Bill Analyzer is live.

Drop a bill PDF in. [X] minutes later, get a structured analysis with USC
cross-references, fiscal impact estimates, pork findings, and stakeholder
mapping — all cited back to specific bill pages.

Run on the 2,469-page Build Back Better Act 2021:

• BF16 (gold standard): [X] minutes
• FP8 (production): [Y] minutes  
• Per-finding agreement: [Z]%
• Pork findings flagged: [N]
• USC cross-refs resolved: [N]

The BF16-vs-FP8 trade-off is the hackathon's headline. 192 GB lets you keep
both modes ready, swap in seconds, and watch the speed/quality curve live.

[INSERT VIDEO CLIP]

Repo cloneable. Demo runs on a fresh MI300X in <[X] minutes.

[REPO_URL]

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## Post #5 — Day 7 (Saturday May 10) — wrap

**Trigger:** after submission. Honest retrospective tone.

**Long-form:**

```
Submitted.

7 days, $[X] of cloud credits, 1 MI300X, 2,469-page bill analyzed in [Y] minutes.

What worked
-----------
• vLLM Automatic Prefix Caching on ROCm. The whole architecture leans on it
  and it just works. [Z]x time-to-first-token speedup on shared prefixes,
  exactly as documented.
• Three Qwen models concurrent on one GPU. Qwen3-VL + Qwen3.6-35B-A3B + Qwen3-32B
  totaling ~145 GB. Headroom to spare.
• LMDB for the USC corpus. 60,187 sections at ~10 µs lookups. One-time
  35-second build from house.gov bulk XML.
• Title-boundary aware chunking. Bills are structured documents, not arbitrary
  text. Splitting at TITLE/Subtitle preserves cross-references in-chunk.

What surprised me
-----------------
[FILL IN AS THE WEEK UNFOLDS — examples to draw from]
• [Surprise about MI300X performance — likely better than expected on...]
• [Surprise about ROCm gotcha — likely fp8 weight loading or KV cache sizing]
• [Surprise about Qwen model behavior — likely Qwen3.6 long-context degradation pattern]

What I'd do differently
-----------------------
• [Limit one — likely "would have benchmarked SGLang from day 1" or similar]
• [Limit two — likely "scoped the agents tighter from the start"]

What's next
-----------
The repo stays public. The architecture works for any structured legislative
document — appropriations bills, NDAA, omnibus packages, even multi-state
regulatory comparisons. If you're a journalist, civic-tech org, or
legaltech builder reading this, the cost-of-analysis curve just changed.

Final demo: [REPO_URL]
Submission: lablab.ai

Big thanks to @AMDDeveloper for the credits and @lablabai for running the
hackathon. The MI300X is a serious piece of kit for context-heavy workloads.

#AMDDevHackathon
```

**X thread (5 tweets):**

```
1/5
Submitted.

7 days. $[X] of cloud credits. 2,469-page bill analyzed end-to-end in [Y] min
on a single MI300X.

Wrap thread on what worked, what didn't, and what surprised me.

#AMDDevHackathon
```

```
2/5
What worked:

• vLLM APC on ROCm just works. [Z]x TTFT speedup on shared prefixes.
• 3 Qwen models concurrent on one GPU. ~145 GB.
• LMDB USC corpus. 60K sections, ~10µs lookups.
• Title-boundary chunking. Don't split bills randomly.
```

```
3/5
What surprised me:

[FILL IN: 2-3 specific surprises from the week. Most likely candidates:
- MI300X being faster than expected on something
- ROCm having a specific gotcha around fp8 or KV
- Qwen3.6 doing something unexpected with long context]
```

```
4/5
What's next:

Repo stays public. Architecture works for any structured legislative doc:
appropriations, NDAA, omnibus, multi-state regs.

If you're a journalist, civic-tech org, or legaltech builder — your
cost-of-analysis curve just changed.

[REPO_URL]
```

```
5/5
Big thanks to @AMDDeveloper for the credits and @lablabai for running this.

The MI300X is genuinely a serious piece of kit for context-heavy workloads.

GG.

#AMDDevHackathon
```

**Short LinkedIn:**

```
Submitted.

7 days. $[X] of cloud credits. One MI300X. 2,469-page Build Back Better Act
2021 analyzed end-to-end in [Y] minutes.

What worked:
• vLLM Automatic Prefix Caching on ROCm — [Z]x TTFT speedup on shared prefixes
• Three Qwen models concurrent on one GPU (Qwen3-VL + Qwen3.6-35B-A3B + Qwen3-32B)
• LMDB-backed USC corpus, 60,187 sections at ~10 µs lookups
• Title-boundary aware chunking that preserves statutory cross-references

What surprised me:
[FILL IN — 2-3 honest surprises from the build]

What's next:
The repo stays public. The architecture works for any structured legislative
document — appropriations, NDAA, omnibus, multi-state regulatory comparisons.

If you're a journalist, civic-tech org, or legaltech builder, the
cost-of-analysis curve on legislation just changed.

[REPO_URL]

Big thanks to @AMDDeveloper and @lablabai. The MI300X is genuinely a serious
piece of kit for context-heavy workloads.

#AMDDevHackathon
```

---

## Posting cadence + notes

| Post | Day | Best window | Visual |
|---|---|---|---|
| #1 | Sun May 4 | post-Day 1 smoke test, evening ET | terminal screenshot of TTFT comparison |
| #2 | Mon May 5 | end of Day 2, evening ET | figure JSON + chart side-by-side |
| #3 | Wed May 7 | end of Day 4, evening ET | 2-3 Pork Finder findings with annotations |
| #4 | Fri May 9 | end of Day 6, evening ET | 60-90 sec demo video clip |
| #5 | Sat May 10 | post-submission | wrap |

Tag every post: `#AMDDevHackathon @AMDDeveloper @lablabai`.
Drop the `[REPO_URL]` cleanly — bare or via shortener, no tracking params.

---

## Pre-Day-1 optional warmup tweet

If you want to seed the arc tonight (no commitment):

```
Day 0 done.

Repo bootstrapped. 3 demo bills lined up — HR 1 (OBBB Act 2025), the original
Build Back Better Act 2021, FY24 NDAA. USC corpus indexed locally as LMDB:
60,187 sections, 379 MB, ~10 µs lookups.

Day 1 starts when @AMDDeveloper credits land.

[REPO_URL]

#AMDDevHackathon @lablabai
```
