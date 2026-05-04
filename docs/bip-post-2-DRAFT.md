# BiP Post #2 — Day 2 Real Numbers (May 4, 2026 evening)

**Status:** Ready to publish.
**Landing page:** https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/
  (also wire up `nota.lawyer/bills` → same content via Cloudflare when convenient)

**Suggested visual stack:**
1. The metric tiles from the landing page (94.15× / 14.8× / 68.5% / 116)
2. A screenshot of the Pork Finder JSON output OR the USC citations grid

---

## Long-form (X long post / LinkedIn post / blog teaser)

```
Day 2 of @AMDDeveloper's Developer Hackathon. The architectural thesis works.

Yesterday I posted a 94.15x time-to-first-token speedup on AMD MI300X
from vLLM's Automatic Prefix Caching on a 99,727-token shared prefix.

Today I tested whether that translates into REAL multi-agent flow.

Setup:
   3 Qwen models hot at the same time on one MI300X (192 GB VRAM):
       Qwen3-VL-8B-Thinking-FP8           (vision, port 8002)
       Qwen3-30B-A3B-Instruct-2507-FP8    (long-context spine, port 8001)
       Qwen3-32B-FP8                      (full reasoner, port 8003)
   Combined VRAM in use: 176 / 192 GB

   Bill: HR 5376, the original Build Back Better Act 2021.
   2,468 pages. 927,623 tokens. Chunked title-boundary aware
   into 5 chunks of <=220K cl100k tokens each.

I ran 4 specialist agents against the same chunk back-to-back:

   Agent                       elapsed    APC state
   ----                        -------    ---------
   Plain-English Summarizer    316.4 s    cold prefill (baseline)
   Plain-English Summarizer     21.4 s    APC warm rerun  (14.8x speedup)
   USC Cross-Reference          ~95 s     APC partial
   Pork Finder                  92.4 s    APC partial
   Conflict Spotter            270.6 s    APC partial

Sustained 'Prefix cache hit rate: 68.5%' on the spine across all four agents.

THIS is the punchline. On a 32 GB GPU each agent has to spill or
re-prefill the bill text. On 192 GB they share. With 14 specialist
agents per chunk planned, that's the difference between 14 prefills
and 1 prefill.

What the agents actually FOUND in chunk 1 (TITLE I-AGRICULTURE):

   - 116 USC citations identified, 76.7% resolved against a local
     USC LMDB (60,187 sections, 379 MB, ~10 us hot lookups). Each
     enriched with the actual current statute heading + text excerpt
     so the LLM never speculates about what cited statutes say.

   - 27 pork/earmark candidates flagged with structural patterns:
     named-recipient, sole-source, geo-specific, oddly-specific-amount.

   - Plain-English summary correctly identifying $10 B for hazardous
     fuels, $4.5 B for vegetation management, $2.25 B Civilian Climate
     Corps, $9 B non-federal grants. All numbers verified against the
     source bill.

   - Vision pipeline (port 8002) reads page 2222 of the bill and emits
     structured JSON for the joint-filer tax bracket schedule in 11.4 sec.
     Numbers verified exact against source.

Cloud burn so far: ~$15 of $100 credits.

Full architecture, real JSON outputs, and the truncation-recovery code
that captured 116 citations from a max_tokens-cut response:
   https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/

Source: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

Submission deadline: Saturday May 10. Five days, ten more agents,
one orchestrator, one finished demo to go.

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## X thread (5 tweets)

**1/5**
```
Day 2, @AMDDeveloper Hackathon. The architectural thesis works.

Same 232K-token bill chunk. Plain-English Summarizer agent.

Cold call:  316.4 s
Warm rerun:  21.4 s

= 14.8x wall-clock speedup from vLLM Automatic Prefix Caching on AMD MI300X.

Code didn't change. The cache did its job.
```

**2/5**
```
Why this matters:

I'm building a 14-agent system for analyzing 2,400-page bills.
Every agent needs the same bill chunk in its context.

On a 32 GB GPU each agent re-prefills.
On 192 GB MI300X they share via APC.

That's the difference between 14 prefills and 1.
```

**3/5**
```
Tested it for real. 4 agents, same chunk, back-to-back:

Summarizer cold       316.4 s
Summarizer warm        21.4 s   <-- 14.8x
USC Cross-Reference    ~95 s
Pork Finder            92.4 s
Conflict Spotter      270.6 s

Sustained 'Prefix cache hit rate: 68.5%' on spine across all four.
```

**4/5**
```
What the agents found in chunk 1 (Build Back Better Act, TITLE I):

  116 USC citations identified, 76.7% resolved against
  a local USC LMDB (60,187 sections, ~10 us lookups)

  27 pork/earmark candidates with structural patterns
  
  $10 B / $4.5 B / $2.25 B forest restoration funding
  flagged + cited with section numbers

All grounded. No hallucinated citations.
```

**5/5**
```
Full landing page with metrics, real JSON outputs, and links:
https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/

Source: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

Day 1 was the speedup. Day 2 was the proof it composes.
Days 3-7: 10 more agents + orchestrator + full bill end-to-end.

@AMDDeveloper @lablabai #AMDDevHackathon
```

---

## Short LinkedIn

```
Day 2 update from the AMD Developer Hackathon.

Yesterday: 94.15x time-to-first-token speedup from APC on AMD MI300X.
Today: tested whether that translates into real multi-agent flow.

The setup: a Plain-English Summarizer agent reading a 232,853-token chunk
of the original Build Back Better Act 2021 (2,468 pages total).

Cold call: 316.4 seconds.
Same agent, same chunk, second call (APC warm): 21.4 seconds.

14.8x wall-clock speedup. No code changes. Just APC reusing the KV cache.

I then ran 3 more specialist agents against the same chunk:
   * USC Cross-Reference (identified 116 USC citations, 76.7% resolved
     against a local 60,187-section LMDB)
   * Pork Finder (27 earmark candidates flagged by structural pattern)
   * Conflict Spotter (correctly returned [] - no internal conflicts)

Sustained "Prefix cache hit rate: 68.5%" on the spine across all four
agents. This is the architectural thesis: 14 specialist agents amortizing
across one chunk prefill instead of paying 14 times.

On a 32 GB GPU each agent has to spill or re-prefill the bill text.
On 192 GB MI300X they share. That's the unlock.

Landing page:
https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/

Source:
https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

#AMDDevHackathon @AMDDeveloper @lablabai
```

---

## Wiring nota.lawyer/bills (manual, do whenever)

Two paths to point `nota.lawyer/bills` at the GH Pages content:

**Option A — Cloudflare Worker (~5 min, recommended):**
1. Cloudflare dashboard → nota.lawyer → Workers & Pages → Create Worker
2. Worker code: just fetch and rewrite from GH Pages
   ```js
   export default {
     async fetch(req, env) {
       const url = new URL(req.url);
       const target = "https://banksythequantlab.github.io/amd-hackathon-bill-analyzer" + url.pathname.replace(/^\/bills/, "");
       return fetch(target, { headers: req.headers });
     }
   }
   ```
3. Workers → Triggers → Add custom domain: `nota.lawyer/bills/*`

**Option B — DNS CNAME + GitHub Pages custom domain (longer-lived):**
1. GitHub repo → Settings → Pages → Custom domain: `bills.nota.lawyer`
2. Cloudflare DNS: CNAME `bills` → `banksythequantlab.github.io` (proxied=on, SSL=Full)
3. Then `bills.nota.lawyer` serves the page directly. (URL is slightly different
   shape from `nota.lawyer/bills` but cleaner.)

Either option, the GH Pages URL is the source of truth and works immediately.

---

## Posting checklist

- [ ] Confirm GH Pages is enabled on the repo (Settings → Pages → source `main` branch, `/docs` folder)
- [ ] Verify `https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/` loads after the next push
- [ ] Take screenshots:
  - The 4 metric tiles at the top (94.15× / 14.8× / 68.5% / 116)
  - The runs table showing cold-vs-warm
  - The USC citations grid
- [ ] Post on X at peak window (12-3 PM ET) Tuesday May 5
- [ ] Cross-post LinkedIn within 30 min
- [ ] Tag `@AMDDeveloper @lablabai #AMDDevHackathon`
- [ ] Pin the X tweet
- [ ] (Optional, separately) Wire nota.lawyer/bills via Cloudflare Worker
