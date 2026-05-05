# BiP #3 -- Day 3+4 (FINAL)

## Source data
- `eval/report-bbb-ch01-day3.json` -- Day 3 fresh-start run on BBB ch01
- `eval/citation-validator-bbb-ch01-FULL.json` -- Day 4 full Citation Validator audit
- `eval/citation-validator-smoke5-bbb-ch01.json` -- Day 3 smoke run (5 citations)
- Pork Finder rewrite: commit `d00fa13`
- Citation Validator + enrich pipeline: commits `b328605` -> `5576cb9`

## SHORT LINKEDIN (~300 words) -- ready to post

Day 3-4 of the AMD Developer Hackathon. I broke an agent on purpose and shipped a 5th one.

**The bug fix.** On Day 2, my Pork Finder agent flagged 27 line items in a 199K-token chunk of the Build Back Better Act 2021 (Title I, Agriculture). Every single one was tagged `pattern: named-recipient` with `recipient: null` -- a structurally invalid combination. The model was over-classifying ordinary appropriations as earmarks.

Day 3 fix: reframed the prompt to lead with "most line items are NOT pork," added worked examples of each pattern with flag-vs-non-flag pairs, and added a Pydantic cross-validator that rejects `named-recipient` with empty `recipient`. Day 4 result on the same chunk: **0 items**, with a note "no pork-signature line items found in this chunk."

That's not a regression. That's the correct answer. Title I is class-of-recipient program funding, not earmarks. Day 2 was producing false positives; the fixed agent doesn't.

**The new agent.** Built a Citation Validator that runs on the reasoner endpoint (Qwen3-32B, 32K context), per-citation, in parallel with the chunk-level agents on the spine endpoint. It reads each USC citation the cross-reference agent identified, fetches the actual statutory text from a local LMDB index of US Code (60K+ sections, 174 microsecond lookups), and asks: does the bill's claimed intent match what the statute actually says?

Real findings on a 6-citation audit, all returned with high confidence:

  16 U.S.C. 6542(d)(1)  WRONG-SECTION    "Subsection (d)(1) does not exist;
                                          definitions are in (a)(1)"
  16 U.S.C. 6543(a)(3)  INTENT-MISMATCH  Bill labeled this as "definitions"
                                          but the section is the Watershed
                                          Condition Framework
  16 U.S.C. 7125        INTENT-MISMATCH  Bill labeled "definitions" but the
                                          section is the operational
                                          framework for Resource Advisory
                                          Committees
  16 U.S.C. 7303        INTENT-MISMATCH  Bill labeled "definitions" but
                                          establishes the Collaborative
                                          Forest Landscape Restoration Program

This is the kind of catch a paralegal would flag during a citation review pass. Five agents now running across two endpoint sizes on one MI300X via vLLM Automatic Prefix Caching. Works.

Live demo + reports: https://bills.nota.lawyer/
Repo (MIT): https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

#AMDDevHackathon @AMDDeveloper @lablabai

## POSTING NOTES
- Best time: 9-11am ET (Tue 5/6 morning if not posted tonight)
- Visual options:
  1. Screenshot of the bills.nota.lawyer top fold (re-use the BiP #2 hero tiles)
  2. JSON snippet of the wrong-section finding (reads like a bug report against the bill)
  3. The 27-vs-0 Pork Finder contrast (strong if you want the bug-fix angle)
- Cross-post the X thread version after LinkedIn lands