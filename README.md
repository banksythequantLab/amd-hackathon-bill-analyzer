# AMD Hackathon — Bill Analyzer

Multi-agent legislative bill analyzer running on AMD MI300X (192 GB VRAM).

Submitted to the [lablab.ai AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer), May 2026.

---

## What it does

Ingests a 2,700-page U.S. legislative bill (Big Beautiful Bill, ~1M tokens) and produces a structured analysis: USC cross-references, fiscal impact, pork detection, stakeholder mapping, and citation validation.

## Why MI300X

192 GB VRAM lets us run **three Qwen models concurrently** on a single GPU:

- **Qwen3-VL-8B-Thinking** (~8 GB FP8) — extracts charts/tables from the bill
- **Qwen3.6-35B-A3B** (~35 GB FP8) — long-context spine, holds 250K-token chunks
- **Qwen3-32B** (~64 GB BF16) — full-precision reasoner for citations + math

Combined with vLLM's Automatic Prefix Caching on ROCm, eight specialist agents share a single chunk's KV cache — turning what would be 8 expensive prefills on a 5090 into 1 prefill + 7 cache hits.

## Architecture

7-agent LangGraph DAG:

| # | Agent | Model | Tools |
|---|---|---|---|
| 1 | Plain-English Summarizer | Qwen3.6 spine | — |
| 2 | Conflict Spotter | Qwen3.6 spine | — |
| 3 | USC Cross-Reference | Qwen3-32B | `fetch_usc` |
| 4 | Citation Validator | Qwen3.6 spine | `fetch_usc` |
| 5 | Fiscal Impact Estimator | Qwen3-32B | vision JSON |
| 6 | Pork Finder | Qwen3-32B | `fetch_usc`, `census_lookup` |
| 7 | Stakeholder Tracer | Qwen3-32B | — |

Plus a Final Report Synthesizer node that consumes all agent outputs.

## Quick start

```bash
# (filled in on Day 6)
./infra/one_click_run.sh path/to/bill.pdf
```

## Status

🚧 **In active development** — May 4–10, 2026. Track progress via Build-in-Public posts on X/LinkedIn under `#AMDDevHackathon`.

## License

MIT — see [LICENSE](./LICENSE).
