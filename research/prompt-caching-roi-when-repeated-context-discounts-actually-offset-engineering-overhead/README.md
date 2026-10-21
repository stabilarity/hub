# Prompt Caching ROI: When Repeated-Context Discounts Actually Offset Engineering Overhead

Code and data for the Stabilarity Research Hub article of the same title (Cost-Effective Enterprise AI series).

## Questions

- **RQ1.** At list prices read on 2026-10-21, how many reuses within the time-to-live (TTL) does each provider's
  prompt cache need before it is cheaper than no caching, and how often must requests arrive for that to happen?
- **RQ2.** How much of the list-price discount survives real arrival timing and reuse structure, and which TTL is
  cheapest on real traces?
- **RQ3.** Once engineering time is counted, what monthly input-token volume makes caching work pay back within three
  months?

## Data (all public)

| File | Content | Source |
|---|---|---|
| `data/prices.csv` | Base input, cache write, cache read and output prices (USD per million tokens), TTL and minimum cacheable length for 18 model/mode pairs | Anthropic https://platform.claude.com/docs/en/about-claude/pricing and https://platform.claude.com/docs/en/build-with-claude/prompt-caching; OpenAI https://developers.openai.com/api/docs/pricing and https://developers.openai.com/api/docs/guides/prompt-caching; Google https://ai.google.dev/gemini-api/docs/pricing and https://ai.google.dev/gemini-api/docs/caching; DeepSeek https://api-docs.deepseek.com/quick_start/pricing and https://api-docs.deepseek.com/guides/kv_cache. All read 2026-10-21. |
| `data/tracelab_rounds.csv.gz` | 357,161 LLM requests from 4,265 Claude Code and Codex sessions: request time, input, provider-reported cached and output tokens | TraceLab v0.0.1, SyFI Lab, University of Washington, CC BY 4.0, https://github.com/uw-syfi/TraceLab (arXiv:2606.30560). Changes: one row per request with timing and token fields only; session ids re-indexed; user, project and tool fields dropped. |
| `data/mooncake_conversation_trace.jsonl.gz`, `data/mooncake_toolagent_trace.jsonl.gz` | One hour of Kimi conversation (12,031 requests) and tool/agent (23,608 requests) traffic with 512-token prefix-block hashes | Mooncake FAST'25 traces, Apache-2.0, https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release (Qin et al., USENIX FAST 2025). Unchanged, gzip-compressed. |

The engineering day rate uses the US median annual wage for software developers, $135,980 in May 2025, from the
Bureau of Labor Statistics Occupational Outlook Handbook
(https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm).

## How to run

```bash
python3 analysis.py      # offline, deterministic; writes charts/chart1.png ... chart4.png and results.json
```

Python 3.10+, numpy, matplotlib. `build_data.py` rebuilds the compact trace files from the original downloads.

## Method

All costs are multiples of the model's base input price. A cache write costs `w`, a read `r`, an uncached token 1.

- **Break-even reads (RQ1).** One write followed by `k` reads beats `k + 1` uncached requests when
  `w + k r <= 1 + k`, so `k* = ceil((w - 1) / (1 - r))`. With no write premium (`w = 1`, implicit caching), any hit is
  a gain and a miss costs nothing extra.
- **Poisson model (RQ1, RQ2).** Requests sharing a prefix arrive at rate `lambda`; every request refreshes the TTL `T`.
  The hit probability is `h = 1 - exp(-lambda T)` and the expected cost per request is `h r + (1 - h) w`. Caching is
  net-positive when `h > (w - 1) / (w - r)`, which gives a break-even mean inter-arrival time.
- **Storage-billed caches.** For Gemini explicit caches the hits per hour needed to cover storage are
  `storage / (input price - cached price)`.
- **TraceLab replay (RQ2).** Within each session, request `k` can reuse the previous request's prompt if its own prompt
  is at least as long (append-only context; a shorter prompt is treated as no reuse) and if the gap since the previous
  request is within the TTL. A hit pays `r` on the reused prefix and `w` on the appended tokens; a miss pays `w` on the
  whole prompt (automatic caching writes everything). Prompts below the model's minimum are uncached. Every request is
  repriced at the policy's multipliers, whichever model produced it. Validation: the replayed hit-token share of
  Claude Code requests (92.76% at 5 minutes, 98.02% at 1 hour) brackets the 95.82% that the provider actually reported
  for those requests. A hindsight dynamic programme chooses between the 5-minute and 1-hour TTL for every write and
  bounds what any TTL-selection heuristic could gain.
- **Mooncake replay (RQ2).** A service-wide block cache: a request hits its leading blocks that were touched within the
  TTL; all its blocks are then touched. Hit tokens are `min(512 x blocks, input length)`; non-hit tokens are written at
  `w`. The trace covers one hour, so TTLs of one hour or more give identical hit shares, and the replay starts cold.
- **Engineering payback (RQ3, our assumptions).** Day rate = $135,980 x 1.4 (benefits and overhead) / 230 working days
  = $827.70. Three effort tiers: switch-on (0.5 days once), prefix-stable (5 days once + 1 day a month for monitoring and
  regressions), full programme (20 days once + 2 days a month). Required monthly saving = one-off cost / 3 months +
  monthly cost. Required monthly input tokens = required saving / (input price x realised input saving from the
  Claude Code replay for that model's multipliers and TTL). Sensitivity: day rate x0.5 and x1.5, 1- and 12-month
  horizons, and a halved realised saving.

`results.json` uses `-1` for `poisson_breakeven_mean_interarrival_min` when there is no threshold (no write premium).

## Outputs

- `charts/chart1.png`: Poisson model, saving on the cached prefix against mean time between requests
- `charts/chart2.png`: distribution of gaps between consecutive requests in TraceLab sessions, against TTLs
- `charts/chart3.png`: realised input saving per policy on four traces, against the list-price ceiling `1 - r`
- `charts/chart4.png`: monthly input tokens needed for a three-month payback, by base input price and effort tier
- `results.json`: every number used in the article

## Limitations

List prices on one day; implicit-cache TTLs for OpenAI pre-GPT-5.6 models, Gemini and DeepSeek are not fixed by the
providers (5 minutes, 5 minutes and 3 hours are modelled); implicit caching is best effort, and the replay assumes every
eligible request hits. The TraceLab users are developers using coding agents, not an enterprise fleet. Mooncake hashes
are service-wide, which overstates the reuse a single API customer would see, and cover one hour. Engineering effort
figures are assumptions, not measurements. Latency, rate-limit and throughput benefits of caching are not priced.
