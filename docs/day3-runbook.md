# Day 3 Runbook

Date target: Day 3 of the AMD Developer Hackathon.
Cloud cost so far: ~$15-18 of $100 credits.
Hackathon ends: Sat May 10, 2026.

## What state we're in

Day 1 + Day 2 deliverables are shipped. The architectural thesis (4 agents
on one MI300X via APC, 14.8x speedup) is proven on BBB-2021 ch01 with the
combined report at eval/report-bbb-ch01.json. BiP #2 is live at
https://bills.nota.lawyer/.

Two known issues going into Day 3:

1. **Windows-side Python hang.** Recurring failure mode where spine returns
   200 OK but Python on Windows never returns to the script. Diagnosed in
   commit 32bf7e0. Fix prepared: tests/run_one_agent_remote.py (commit
   4aec560) runs agents on the Linux instance directly. Validates Day 3.

2. **Pork Finder over-classification.** Day 2 BBB ch01 produced 27 items
   all tagged 'named-recipient' with recipient=null, when most were ordinary
   class-of-recipient appropriations that the prompt's own rules said to
   skip. Fix: prompt rewrite + Pydantic cross-validation in commit d00fa13.
   Validates Day 3.

## First-action workflow (paste-and-go)

### Step 1: Power on the AMD droplet via the AMD UI

(I cannot do this for you; it's the AMD console.)

The droplet was at IP 165.245.134.1. If a different IP is assigned on
power-up, update INSTANCE_HOST in tests/run_one_agent_remote.py.

### Step 2: Restart the three Qwen containers

SSH in and re-run the Day 2 launcher:

  ssh -i ~/.ssh/amd_hackathon root@<IP>
  bash /root/predl-launch.sh   # or whichever launch script worked last

Wait until docker ps shows all three healthy. The HF cache should still
be populated (~71GB), so weight downloads are skipped — total launch
should be under 5 minutes.

### Step 3: One-time Day 3 setup on the instance

  cd /root/repo
  git pull --ff-only      # picks up the Day 3 prep commits
  python3.12 -m venv .venv
  ./.venv/bin/pip install -q pdfplumber lmdb orjson httpx pydantic tiktoken

This creates the venv that run_one_agent_remote.py expects.

### Step 4: Run the validation pass on BBB ch01 (the chunk Day 2 used)

From Vesper or wherever:

  cd B:\amd-hackathon-bill-analyzer
  python tests\run_one_agent_remote.py --agent summarizer --bill bbb --chunk-id ch01
  python tests\run_one_agent_remote.py --agent xref       --bill bbb --chunk-id ch01
  python tests\run_one_agent_remote.py --agent pork       --bill bbb --chunk-id ch01
  python tests\run_one_agent_remote.py --agent conflict   --bill bbb --chunk-id ch01

Expected: each completes cleanly. Pork should produce dramatically fewer
items than Day 2's 27 (likely 0-5), with all real flags having a non-empty
recipient or geo-specific marker.

### Step 5: Run on HR1 ch01 to capture the second-bill data point

  python tests\run_one_agent_remote.py --agent summarizer --bill hr1  --chunk-id ch01
  python tests\run_one_agent_remote.py --agent xref       --bill hr1  --chunk-id ch01
  python tests\run_one_agent_remote.py --agent pork       --bill hr1  --chunk-id ch01
  python tests\run_one_agent_remote.py --agent conflict   --bill hr1  --chunk-id ch01

Same pattern. Compare pork outputs between bills - if one has flags and
the other doesn't, that's a real signal.

### Step 6: Assemble combined report-hr1-ch01.json

Mirror the BBB approach: B:\hackathon-build\assemble-report.py walks the
agent-smoke directory and builds a combined report. Will need a tweak to
read HR1 outputs.

## Day 3 stretch goals (in priority order)

1. **Citation Validator agent** (#5 in the agent roster). Per-citation,
   runs on reasoner endpoint (32K context is plenty for a single citation
   + context). Validates that each USC citation in the xref output is
   correctly cited (right title, right section, language matches).

2. **Fiscal Impact Estimator** (#6). Per-chunk, spine endpoint. Sums
   appropriations and categorizes by domain.

3. **Effective Date Tracker** (#7). Per-chunk. Finds and structures all
   effective-date provisions.

4. **Final Synthesizer** (minimal version). Combines outputs of agents 2-7
   into a coherent paragraph-level report for the chunk.

If steps 1-6 above land cleanly with time to spare, ship one stretch goal
and post BiP #3.

## Day 3 budget

- Cloud time budget: 4-6 hours x $1.99/hr = $8-12
- Total spend by EOD Day 3 should be under $30
- Credits remaining for Days 4-7: $70+

## What NOT to do Day 3

- Do NOT try to debug the Windows-side Python hang. The Linux test runner
  removes it from the path entirely. Spend the budget on agents, not debug.
- Do NOT try to scale to all 14 agents in one day. Steady progress is
  better than half-finished agents.
- Do NOT post BiP #3 unless there's a real new result (not just iteration
  on Day 2's content). The judges follow the build-in-public posts —
  empty BiPs hurt more than they help.

## Reference

  Repo:    https://github.com/banksythequantLab/amd-hackathon-bill-analyzer
  Demo:    https://bills.nota.lawyer/
  BBB report:  eval/report-bbb-ch01.json
  HR1 report:  eval/report-hr1-ch01.json (partial — Day 2 evening)

  Last commit: 4aec560 (Day 3 prep)
  Day 2 final: 32bf7e0 (HR1 partial + hang diagnosis)
  BBB full report: 41b0af0
  Vision pipeline: fd6dbb2