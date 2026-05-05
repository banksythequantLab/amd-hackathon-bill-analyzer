# BiP #2 Post (FINAL) — Day 2, May 4, 2026

URL placeholder filled in with the actual live GitHub Pages URL:
https://bills.nota.lawyer/

Repo: https://github.com/banksythequantLab/amd-hackathon-bill-analyzer
Combined report (raw JSON): https://raw.githubusercontent.com/banksythequantLab/amd-hackathon-bill-analyzer/main/eval/report-bbb-ch01.json

---

## SHORT LINKEDIN (~250 words) — ready to paste

Day 2 update from the AMD Developer Hackathon: the architectural thesis is now proven end-to-end on a real legislative bill.

I ran four specialist agents — Plain-English Summarizer, USC Cross-Reference, Pork Finder, Conflict Spotter — against a single 199,381-token chunk of the Build Back Better Act 2021 (TITLE I, Agriculture, pp. 3-542). All four ran on a single AMD MI300X GPU, sharing one cached prefix via vLLM's Automatic Prefix Caching.

The numbers:

  Plain-English Summarizer (cold)   316.4 sec
  Plain-English Summarizer (APC-warm rerun)  21.4 sec
  --> 14.8x speedup from APC reuse on the identical 232,853-token chunk

  USC Cross-Reference        116 citations identified, 23 of 30 resolved
                             against a local LMDB index of US Code (76.7%
                             hit rate), each enriched with the actual
                             statute heading and text excerpt
  Pork Finder                27 line items extracted with structured
                             metadata (amount, recipient, bill section)
  Conflict Spotter           returned [] — correctly. The model declined
                             to invent contradictions where none existed.

  vLLM prefix cache hit rate: 66.6% to 70.6% sustained across all four agents

The point: 4 agents x ~232K-token chunk = 932K total prompt tokens, but only the first agent paid the full prefill cost. The other three rode the cache. On a 32GB consumer GPU each agent would re-pay that prefill or spill to CPU. On a 192GB MI300X they share.

Live demo + numbers: https://bills.nota.lawyer/
Repo (MIT): https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

#AMDDevHackathon @AMDDeveloper @lablabai

---

## POSTING NOTES

- Best time to post: 9-11am ET tomorrow (Tue 5/5) for max LinkedIn reach
- Visual to attach: screenshot of the landing-page hero (the four hero tiles
  with 14.8x, 116, 27, 0 numbers) — the page renders well at desktop size.
  Open https://bills.nota.lawyer/ and
  screenshot the top fold.
- After posting on LinkedIn, repost to X with the X-thread version from
  bip-post-2-DRAFT.md (5 tweets) for cross-platform reach.
- nota.lawyer wiring is a follow-up step whenever you have a sec — the page
  is shareable from the GH Pages URL right now.

