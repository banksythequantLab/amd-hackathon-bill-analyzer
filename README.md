---
title: Bill Analyzer (AMD Hackathon)
emoji: 📜
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: true
license: mit
short_description: Multi-agent legislative bill analyzer + podcast pipeline on AMD MI300X
---

# AMD Hackathon — Bill Analyzer + Podcast Studio

End-to-end legislative analysis pipeline for the [lablab.ai AMD Developer Hackathon](https://lablab.ai/ai-hackathons/amd-developer), May 2026. Runs entirely on a single AMD **MI300X** (192 GB VRAM) using vLLM + ROCm + ComfyUI.

Drop a U.S. bill PDF in. Get back a structured analysis (USC cross-references, pork detection, conflict spotting, plain-English summary, ranked podcast headlines), then optionally turn the winning headline into a 2–3 minute podcast video with AI hosts, generated slides, motion animation, and TTS narration. **All-Qwen** stack, no proprietary models.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
  - [Pipeline 1 — Bill Analysis](#pipeline-1--bill-analysis-cpu-orchestration--mi300x-inference)
  - [Pipeline 2 — Podcast Studio](#pipeline-2--podcast-studio-all-qwen-media-generation-on-mi300x)
- [Why MI300X](#why-mi300x)
- [Multi-chunk handling](#multi-chunk-handling)
- [Quality gate — dual-call slide critic](#quality-gate--dual-call-slide-critic)
- [Repo layout](#repo-layout)
- [Quick start](#quick-start)
- [Demo bills](#demo-bills)

---

## What it does

The Gradio UI has **two complementary surfaces** running on a shared MI300X backend:

### Bill Analysis surface

Upload a bill PDF (or pick a pre-processed canonical bill from the gallery). The pipeline:

1. **Smart-chunks** the PDF on `DIVISION` / `TITLE` / `Subtitle` boundaries so cross-references stay intact (max 220K cl100k tokens per chunk).
2. **Runs 6 specialist agents per chunk** sequentially against the Qwen3-30B FP8 spine:
   - Plain-English Summarizer
   - USC Cross-Reference (with live LMDB enrichment)
   - Pork Finder
   - Conflict Spotter
   - Podcast Headlines Generator (10 candidates per chunk)
   - Headline Ranker (composite score across newsworthiness / specificity / appeal)
3. **Aggregates across chunks** into a single canonical-shaped report — bullets are prefixed by chunk, citations/items/conflicts/headlines are concatenated, the global winner is re-ranked across all chunks' candidates.
4. **Saves** per-chunk JSONs (`{bill}-ch01.json` ... `{bill}-chNN.json`) plus a merged report (`{bill}-merged.json`) into `eval/canonical/` so the Bills Lookup gallery and Podcast Studio pick them up.

The UI streams agent-by-agent progress events to a live log panel so you can watch the spine work.

### Podcast Studio surface

For any pre-processed bill:

1. Pick the bill from a dropdown.
2. The **headline picker** auto-populates with the 10 ranked candidates from `headline_ranker` (e.g. `#1 (27/30) Hidden Clause: Forests Must Now Protect Water`).
3. Click any headline → it loads into an **editable text box**. Tweak it freely or type your own.
4. Optional: add **creative direction** (a free-form prompt) — e.g. *"Focus on the surveillance angle and the 4th Amendment risks. Keep tone dry and journalistic."*
5. Click **🎙️ Generate Podcast Video**. The full all-Qwen media pipeline runs on MI300X and a 2-3 minute mp4 streams back into the page.

Custom runs (different headline or non-blank direction) get their own subfolder `eval/{bill}-cloud-custom-{8-char-hash}/`, so you can have multiple variants of the same bill side by side without clobbering the canonical render.

---

## Architecture

### Pipeline 1 — Bill Analysis (CPU orchestration → MI300X inference)

```mermaid
flowchart TD
    PDF[Bill PDF] --> CHUNK[Smart Chunker<br/>TITLE / Subtitle / DIVISION boundaries<br/>cap 220K cl100k tokens]
    CHUNK --> CHUNKS[N chunks]

    subgraph PerChunk["Per-chunk: 6 agents sequential, share KV cache via vLLM APC"]
        direction TB
        A1[1. Summarizer] --> A2[2. USC Cross-Ref]
        A2 --> A3[3. Pork Finder]
        A3 --> A4[4. Conflict Spotter]
        A4 --> A5[5. Podcast Headlines<br/>10 candidates]
        A5 --> A6[6. Headline Ranker<br/>composite score]
    end

    CHUNKS --> PerChunk
    A2 -.fetch_usc.-> LMDB[(USC LMDB<br/>60k+ sections<br/>~10 µs hits)]

    PerChunk --> MERGE[Multi-Chunk Merger<br/>src/multichunk.py]
    MERGE --> CANON[(eval/canonical/<br/>{bill}-merged.json<br/>+ per-chunk JSONs)]

    classDef qwen fill:#1f4e79,stroke:#fff,color:#fff
    classDef tool fill:#5b8c5a,stroke:#fff,color:#fff
    classDef io fill:#8b4513,stroke:#fff,color:#fff
    class A1,A2,A3,A4,A5,A6 qwen
    class LMDB tool
    class PDF,CANON io
```

### Pipeline 2 — Podcast Studio (all-Qwen media generation on MI300X)

```mermaid
flowchart TD
    CANON[(canonical report<br/>+ chosen headline<br/>+ optional creative direction)] --> SCRIPT[1. PodcastScriptWriter<br/>Qwen3-30B spine<br/>19-line Alex/Jordan dialog]
    SCRIPT --> SLIDES[2. SlidePromptGenerator<br/>Qwen3-30B<br/>19 image prompts]
    SCRIPT --> WAN_PROMPTS[3. WanMotionPromptGenerator<br/>Qwen3-30B<br/>19 pre+post motion prompts]

    SLIDES --> QIMG[Qwen-Image-2512 FP8<br/>+ Lightning 4-step LoRA<br/>1280x720 slide PNGs]
    QIMG --> CRITIC[4. SlideCritic<br/>Qwen3-VL-8B-Thinking<br/>dual-call OCR + judgment]
    CRITIC -- pass --> WAN
    CRITIC -- fail<br/>retry up to 4x w/ new seeds --> QIMG

    WAN_PROMPTS --> WAN[Wan 2.2 i2v 14B FP8<br/>+ LightX2V 4-step LoRA<br/>832x480 81-frame clips]
    SCRIPT --> TTS[Qwen3-TTS-12Hz-1.7B-CustomVoice<br/>Alex=Ryan, Jordan=Ono_anna]

    WAN --> COMPOSE[FFmpeg compose<br/>scene-NN.mp4 = pre + speaker-clip + post]
    TTS --> COMPOSE
    COMPOSE --> MASTER[final-{bill}-cloud-podcast.mp4<br/>~2-3 min]

    classDef qwen fill:#1f4e79,stroke:#fff,color:#fff
    classDef render fill:#7c2d12,stroke:#fff,color:#fff
    classDef io fill:#8b4513,stroke:#fff,color:#fff
    class SCRIPT,SLIDES,WAN_PROMPTS,CRITIC qwen
    class QIMG,WAN,TTS render
    class CANON,MASTER io
```

**Per-step compute** (proven on MI300X):

| Step | Model | Time/unit |
|---|---|--:|
| Slide gen | Qwen-Image-2512 FP8 + Lightning 4-step LoRA, 1280×720 | 4–19 s |
| Slide critique | Qwen3-VL-8B, dual-call OCR + independent judgment | ~2.0 s |
| Wan i2v animation | Wan 2.2 i2v 14B + LightX2V LoRA, 832×480, 81 frames | 25–49 s |
| TTS line | Qwen3-TTS-12Hz-1.7B, custom voice | 12–30 s |

A full ~2-minute podcast (border25, 19 scenes) ships in ~24 minutes cold; cached re-composes finish in ~30 s.

---

## Why MI300X

192 GB VRAM lets the **same GPU** hold the entire all-Qwen stack at once — no model swapping mid-pipeline:

| Slot | Model | Mem |
|---|---|---:|
| Spine reasoning | Qwen3-30B-A3B-Instruct-2507-FP8 | ~35 GB |
| Vision OCR + judgment | Qwen3-VL-8B-Thinking-FP8 | ~9 GB |
| Image generation | Qwen-Image-2512 FP8 + 4-step Lightning LoRA | ~22 GB |
| Image-to-video | Wan 2.2 i2v 14B FP8 (high+low noise + LightX2V LoRAs) | ~30 GB |
| TTS | Qwen3-TTS-12Hz-1.7B + CustomVoice + Tokenizer | ~5 GB |
| KV cache + scratch | (250K-token chunks, ROCm scratch, etc) | ~50 GB |
| **Total** | | **~150 GB** |

Combined with vLLM's Automatic Prefix Caching on ROCm 7.2.3, all 6 specialist agents share a single chunk's KV cache — turning what would be N expensive prefills on a smaller GPU into 1 prefill + (N-1) cache hits.

---

## Multi-chunk handling

Bills like the Build Back Better Act are 540+ pages and ~933K cl100k tokens — they don't fit in one chunk. The chunker splits BBB into 5 chunks at structural boundaries:

```
ch01: pp.3-542    | 199,381 tok | TITLE I—AGRICULTURE
ch02: pp.542-1114 | 217,853 tok | Subtitle G—Medicaid
ch03: pp.1114-1696| 218,139 tok | Subtitle C—Encouraging Small Business
ch04: pp.1696-2110| 154,567 tok | Subtitle G—Green Energy
ch05: pp.2110-2468| 137,352 tok | Subtitle I—Responsibly Funding
```

`analyze_pdf` runs all 6 agents on every chunk, then `src/multichunk.merge_chunk_reports` aggregates per-chunk outputs into a single canonical-shaped report:

- **summarizer**: bullets prefixed by chunk (`[ch02] ...`); per-chunk one-sentence summaries concatenated
- **usc_cross_ref / pork_finder / conflict_spotter / podcast_headlines**: list fields concatenated, each item annotated with its source `_chunk`
- **headline_ranker**: rankings union'd across chunks, re-sorted by composite score, global winner picked
- **totals**: `wall_clock_s`, `prompt_tokens_total`, `completion_tokens_total` summed
- **pages**: `[min(start), max(end)]` (full bill range)

Single-chunk bills round-trip identical to before (no breaking change to existing renderers).

---

## Quality gate — dual-call slide critic

`SlideCritic` runs **two independent Qwen3-VL calls** per slide and requires both to vote pass:

1. **OCR call** — system prompt forces character-by-character transcription, no auto-correction. Programmatic char-level normalize + match against the expected headline. Catches typos like `JUDICAL`, `SUEKERS`, `COLECTION` that a single yes/no judgment call rubber-stamps.
2. **Judgment call** — independent visual review with the expected headline **withheld** so the model can't be primed. Checks legibility, well-formed text, on-brand style, and absence of artifacts.

Final verdict: pass requires `ocr_pass AND judgment_pass`. Disagreement is recorded as `agreement = ocr_only | judgment_only | both_pass | both_fail` for diagnostics. On the border25 19-slide run this caught 2 typos a single-call critic missed.

---

## Repo layout

```
.
├── app.py                          # Gradio UI (Bill Analyzer + Podcast Studio)
├── scripts/
│   └── make_podcast_cloud.py       # Stage 5 cloud podcast pipeline orchestrator
├── src/
│   ├── multichunk.py               # Multi-chunk report merger
│   ├── chunking/
│   │   └── smart_chunker.py        # PDF -> structural chunks
│   ├── agents/
│   │   ├── summarizer.py           # 6 analysis agents
│   │   ├── usc_xref.py
│   │   ├── pork_finder.py
│   │   ├── conflict_spotter.py
│   │   ├── podcast_headlines_generator.py
│   │   ├── headline_ranker.py
│   │   ├── podcast_script_writer.py     # 5 media agents
│   │   ├── slide_prompt_generator.py
│   │   ├── wan_motion_prompt_generator.py
│   │   ├── slide_critic.py
│   │   └── youtube_metadata_generator.py
│   └── tools/
│       └── http_fetch_usc.py       # USC LMDB HTTP client
├── comfy/                          # ComfyUI workflows for Qwen-Image, Wan 2.2 i2v, LTX 2.3, etc.
│   ├── workflows/
│   └── render_*.py                 # Local render driver scripts
├── eval/
│   └── canonical/                  # {bill}-ch01.json + {bill}-merged.json (committed)
├── infra/
│   ├── usc_corpus_build.py
│   └── vllm_serve.sh
└── README.md
```

---

## Quick start

### Local dev (Windows)

```powershell
# Set up Python env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Point at the cloud spine (or run vLLM locally on an MI300X)
$env:SPINE_ENDPOINT = "http://your-mi300x:8001/v1"
$env:USC_LMDB_HTTP  = "http://your-mi300x:8004"

# Launch the Gradio UI
python app.py
# -> http://localhost:7860
```

### Generate a podcast directly via CLI

```powershell
# Default (uses auto-ranked winner headline):
python scripts/make_podcast_cloud.py --bill border25

# With overrides:
python -c "from scripts.make_podcast_cloud import run_full_pipeline; run_full_pipeline('border25', override_headline='My Custom Headline', creative_direction='Focus on the surveillance angle.')"
```

### MI300X side (one-time setup)

```bash
# Build USC corpus (LMDB) — one-time, ~35s on 8-core box
python infra/usc_corpus_build.py \
    --xml-dir   ./data/xml \
    --lmdb-path ./data/usc.lmdb \
    --release   119-36

# Launch the Qwen endpoints
./infra/vllm_serve.sh all
# Brings up:
#   :8001  vllm-spine    (Qwen3-30B-A3B-Instruct-2507-FP8)
#   :8002  vllm-vision   (Qwen3-VL-8B-Thinking-FP8)
#   :8004  usc-lmdb-srv
#   :8188  comfyui       (Qwen-Image + Wan 2.2 + Qwen3-TTS workflows)
```

---

## Demo bills

Six U.S. bills are pre-processed and live in `eval/canonical/` — they appear in the Bills Lookup gallery and the Podcast Studio dropdown without any compute:

| Short | Bill | Pages | Tokens | Chunks |
|---|---|--:|--:|--:|
| `laken`     | Laken Riley Act (S.5, P.L. 119-1)               | 4     | 1,978   | 1 |
| `ndaa26`    | FY26 NDAA (House)                               | 13    | 4,507   | 1 |
| `border25`  | Secure the Border Act (HR 2)                    | 214   | 59,735  | 1 |
| `israel24`  | Israel Security Supplemental Appropriations     | 110   | 69,579  | 1 |
| `capr26`    | Continuing Appropriations 2026                  | 161   | 114,441 | 1 |
| `bbb`       | Build Back Better Act 2021 (HR 5376)            | 2,468 | 927,292 | 5 |

The first canonical podcast video shipped is for `border25` (`DNA Collected from Every Alien` — the auto-ranked winner) and lives at `eval/border25-cloud/final-border25-cloud-podcast.mp4` after a successful pipeline run.

---

## Status

🚧 **Active development** — May 4–10, 2026.

Build progress is published in real time as Build-in-Public posts on X / LinkedIn under `#AMDDevHackathon`.

## License

MIT — see [LICENSE](./LICENSE).
