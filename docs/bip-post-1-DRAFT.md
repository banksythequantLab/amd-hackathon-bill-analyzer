# BiP Post #1 — Day 1 Real Numbers (May 4, 2026)

**Status:** Ready to publish.

**Repo:** https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

**Suggested visual:** Two-image carousel
1. The APC benchmark JSON (the 94.15x line is the showpiece)
2. `docker ps` + `rocm-smi` showing 3 vllm containers + 176GB VRAM in use

---

## Long-form (X long post / LinkedIn / blog teaser)

```
Day 1 of @AMDDeveloper's Developer Hackathon. Bill Analyzer is alive on MI300X.

The judging hook for this hackathon is: what does 192GB VRAM unlock that you
can't do on a 5090?

Here's my Day 1 answer.

I'm building a multi-agent system that ingests legislative bills (target: the
2,469-page Build Back Better Act 2021, ~920K tokens) and produces a structured
analysis: USC cross-references, fiscal impact, pork detection, stakeholder
mapping, citation validation.

The architecture leans hard on what only 192 GB unlocks:
   Qwen3-VL-8B-Thinking-FP8       — extracts charts/tables from the bill
   Qwen3-30B-A3B-Instruct-2507-FP8 — long-context spine, holds 250K-token chunks
   Qwen3-32B-FP8                   — full reasoner for citations + math

Three Qwen models, hot at the same time, on ONE GPU. 176 GB of 192 GB used.

But the real unlock is vLLM's Automatic Prefix Caching on ROCm.

Day 1 smoke test: 99,727-token shared prefix from the OBBB Act. Send Question A.
Then the same prefix + Question B. APC detects the shared prefix and skips the
prefill on Request B.

Numbers:

  Request A (cold prefix) TTFT : 51,046 ms
  Request B (warm prefix) TTFT :    542 ms
  ----------------------------------
  Speedup (TTFT)               :  94.15x

That's not a typo. NINETY-FOUR-X time-to-first-token speedup on the same prompt
prefix because vLLM's APC kept the KV cache hot.

Now multiply that by 6 specialist agents per chunk x 4 chunks. The expensive
prefill happens ONCE per chunk and 5 specialist agents reuse the cached KV.
On a 5090 (32 GB) every agent has to spill or re-prefill. On MI300X they share.

That's the demo punchline.

What I shipped today:
   - vLLM serving all 3 Qwen models concurrently on one MI300X
   - APC benchmark passing at 94.15x (target was 5x)
   - USC corpus indexed locally as LMDB: 60,187 sections, 379 MB,
     ~10us hot lookups (will be the fetch_usc tool the agents call)
   - Three demo bills loaded: HR 1 (OBBB Act 2025), original BBB Act 2021,
     FY24 NDAA

Tomorrow: vision pipeline + smart chunking on the full BBB-2021 PDF.
By Day 5: full bill end-to-end at BF16 vs FP8 with side-by-side timing +
agreement-rate overlay.

Repo: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer
Build-in-public: 5 posts over 7 days, this is #1.

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## X thread (5 tweets)

**Tweet 1 of 5:**
```
Day 1 of @AMDDeveloper's Developer Hackathon.

I'm building a 14-agent system that analyzes 2,400+ page legislative bills
on a single MI300X.

The hook: what does 192GB VRAM unlock?

Three Qwen models, hot at the same time, on ONE GPU. Numbers below.

#AMDDevHackathon
```

**Tweet 2 of 5:**
```
The stack:

  Qwen3-VL-8B-Thinking-FP8       extracts charts/tables from bills
  Qwen3-30B-A3B-Instruct-2507-FP8 250K-token long-context spine
  Qwen3-32B-FP8                   full reasoner for citations + math

All three running concurrently. 176 GB of 192 GB VRAM used.
Can't do this on a 5090.
```

**Tweet 3 of 5:**
```
The real unlock is vLLM's Automatic Prefix Caching on ROCm.

99,727-token shared prefix from the OBBB Act. Same prefix, two questions:

  Cold TTFT : 51,046 ms
  Warm TTFT :    542 ms

= 94.15x time-to-first-token speedup.

That's not a typo.
```

**Tweet 4 of 5:**
```
Why this matters: with 6 specialist agents per bill chunk, the expensive
prefill happens ONCE per chunk. Every additional agent reuses the cached KV.

On a 5090 each agent has to spill or re-prefill.
On MI300X they share.

That's the demo punchline.
```

**Tweet 5 of 5:**
```
Day 5 target: full Build Back Better Act 2021 (2,469 pages, ~920K tokens)
analyzed in <8 minutes at FP8.

5 posts over 7 days. $100 of cloud credits. 1 GPU.

Repo: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

@AMDDeveloper @lablabai #AMDDevHackathon
```

---

## Short LinkedIn

```
Day 1 update from the AMD Developer Hackathon.

Built and benchmarked a multi-agent system for analyzing 2,400+ page
legislative bills on a single MI300X.

THE BIG NUMBER: vLLM's Automatic Prefix Caching on ROCm delivered a
94.15x time-to-first-token speedup on a 99,727-token shared prefix.

  Cold TTFT : 51,046 ms
  Warm TTFT :    542 ms

This isn't theoretical. Every specialist agent that runs against the same
bill chunk reuses the cached KV from the first agent's prefill. On a 32 GB
GPU each agent re-pays that cost. On 192 GB they share.

What's running on the GPU right now, simultaneously:
   Qwen3-VL-8B-Thinking-FP8       (vision)
   Qwen3-30B-A3B-Instruct-2507-FP8 (spine, 250K ctx)
   Qwen3-32B-FP8                   (reasoner)

Combined 176 GB / 192 GB VRAM.

Tomorrow: vision pipeline runs on real bill PDFs. By Day 5: full Build Back
Better Act 2021 end-to-end at BF16 vs FP8 with agreement-rate overlay.

Repo: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## Posting checklist

- [ ] Take screenshot of `eval/apc-benchmark-hr1.json` (look pretty in editor)
- [ ] Take screenshot of `docker ps` output showing 3 vllm containers
- [ ] Optional: terminal recording / asciinema of the apc_benchmark.py run
- [ ] Post on X at peak engagement window (12-3pm ET)
- [ ] Cross-post on LinkedIn (long-form version) within 30 min
- [ ] Tag @AMDDeveloper @lablabai #AMDDevHackathon
- [ ] Reply to your own first tweet with the long-form thread
- [ ] Pin the tweet for the duration of the hackathon
