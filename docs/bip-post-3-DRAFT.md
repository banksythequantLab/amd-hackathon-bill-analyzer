# BiP #3 — Day 3 (DRAFT)

## Source data
- `eval/report-bbb-ch01-day3.json` — Day 3 fresh run on BBB ch01
- `eval/report-bbb-ch01.json` — Day 2 reference run
- Pork Finder commit: `d00fa13` — prompt rewrite + Pydantic cross-validation

## SHORT LINKEDIN (~280 words)

Day 3 of the AMD Developer Hackathon: I broke a working agent on purpose and fixed it. The diff is illustrative.

On Day 2 my Pork Finder agent ran against a 199K-token chunk of the Build Back Better Act 2021 (TITLE I, Agriculture) and found 27 line items it called "earmarks." Looking at the output, every single one was tagged `pattern: named-recipient` with `recipient: null` — a structurally invalid combination. The model had been over-classifying ordinary class-of-recipient appropriations like "$10B for hazardous fuels reduction" as if they were specific entity carveouts.

Three changes:

1. **Reframed the prompt.** The old version led with "earmark signatures include named-recipient, sole-source, geo-specific..." and the model picked the most prominent label. The rewrite leads with "most line items are NOT pork; empty items[] is the right answer for typical chunks."

2. **Added worked examples for each pattern.** Every label now has both a flag-example AND a non-flag-example, drawn from realistic bill language. The model now learns the boundary, not just the keyword.

3. **Added a Pydantic cross-validator** that rejects `pattern == "named-recipient"` when `recipient` is null. The agent's retry loop catches the model's own inconsistencies before they reach the report.

Day 3 result on the same chunk: **0 items, with the note "no pork-signature line items found in this chunk."**

That's not a regression. That's the correct answer. BBB Title I is class-of-recipient program funding, not earmarks. Day 2 was producing false positives; Day 3 doesn't.

This is the prosaic side of agent engineering. Most of the work is preventing your agents from being too eager to please you.

Numbers + repo: https://bills.nota.lawyer/
Repo (MIT): https://github.com/banksythequantLab/amd-hackathon-bill-analyzer

#AMDDevHackathon @AMDDeveloper @lablabai

## ALSO NOTABLE FROM DAY 3
- Cold-prefill summarizer: 282s (the baseline that the Day 2 21.4s warm rerun was 13x faster than)
- USC Cross-Reference: 142 citations recovered from a truncated response (Day 2 had 116; better truncation recovery)
- Linux test runner shipped: agents now run on the cloud instance directly, eliminated the Windows httpx hang class

## POSTING NOTES
- Best time: 9-11am ET (so Tue 5/5 morning if not posted tonight)
- Visual: screenshot of the Day 2 vs Day 3 pork output (the 27 vs 0 contrast is the punchline)
- This is genuinely good post material because it's about engineering rigor, not just numbers go up