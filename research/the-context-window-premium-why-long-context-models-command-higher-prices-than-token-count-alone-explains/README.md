# The Context-Window Premium

Code and data for the Stabilarity Research Hub article *The Context-Window Premium: Why Long-Context
Models Command Higher Prices Than Token Count Alone Explains* (AI Economics series).

## Question

Is long-context capability priced as a distinct feature, or is it bundled into ordinary per-token rates?

- **RQ1.** After controlling for model capability, does a larger maximum context window raise the list
  price per input token, and how common are explicit long-context surcharges?
- **RQ2.** What does a single long request cost at 32K, 128K, 200K, 500K and 990K prompt tokens, and where
  do step functions appear?
- **RQ3.** How much of the long-context premium survives prompt caching and batch discounts, and what does
  a full window cost per token that the model reliably uses?

## Data (all public, accessed 2026-10-20)

| File | Content | Sources |
|---|---|---|
| `data/pricing.csv` | 83 provider-direct rows (81 with prices): context window, input/output/cached price per 1M tokens, long-context tiers, batch discount, source URL and access date per row | Anthropic, OpenAI, Google Gemini API, xAI, Alibaba Cloud Model Studio, DeepSeek, Mistral, Moonshot, Together AI, Groq, AWS Bedrock (Amazon Nova, Meta Llama), Cohere pricing pages and model docs (URL in each row) |
| `data/openrouter_models_2026-10-20.json.gz` | raw snapshot of `https://openrouter.ai/api/v1/models` | OpenRouter models API |
| `data/openrouter_models.csv` | flattened snapshot (built by `data/extract_openrouter.py`), including the Artificial Analysis Intelligence Index carried by the API | OpenRouter; Artificial Analysis |
| `data/effective_context.csv` | claimed vs effective context lengths | NoLiMa (https://github.com/adobe-research/NoLiMa, arXiv:2502.05167), RULER (https://github.com/NVIDIA/RULER, arXiv:2404.06654) |

Rules used when building `pricing.csv`: only values read on a page are recorded; an unreadable or
unpublished value is left empty and explained in `notes`. Tier thresholds are strict (`>`) unless
`threshold_inclusive=1` (xAI bills `>= 200k`). All tokens in a request are billed at the tier of that
request, as every tiered provider in the sample states. Prices are list prices in USD for the default
(global or international) deployment. DeepSeek rows are peak-hour rates (off-peak is half). Gemini 4 Argon
and the Gemini 3.x Flash rows carry promotional prices with announced increases (see `notes`).

## How to run

```bash
python3 data/extract_openrouter.py   # optional: rebuilds data/openrouter_models.csv from the snapshot
python3 analysis.py                  # writes charts/chart1.png ... chart4.png and results.json
```

Python 3.10+, numpy, matplotlib. Fully offline and deterministic (no random numbers).

## Method

- Request cost = prompt tokens x input rate (at the request's tier) + 2,000 output tokens x output rate.
  A request must fit the window including its output, so the 1M-class point is 990K prompt tokens.
- Cached scenario: 90% of the prompt billed at the cache-hit rate (steady state; cache writes and storage
  fees ignored). Batch scenario: published batch discount. Cache and batch are combined only for Anthropic
  and OpenAI, whose pages state that the discounts stack.
- Surcharge = cost at tiered rates minus cost of the same tokens at the short-tier rates.
- Regressions: OLS of log10 input price on log10 context window, with HC1 robust standard errors; variants
  add the Artificial Analysis Intelligence Index and provider fixed effects.
- Cost per reliably used token = list input price x claimed window / effective window (NoLiMa definition:
  longest context retaining at least 85% of the model's short-context score).

## Outputs

- `charts/chart1.png` input price vs context window (market and provider-direct)
- `charts/chart2.png` cost per prompt token vs prompt length, indexed to a 32K request
- `charts/chart3.png` long-context surcharge on one 990K request: standard, batch, cached
- `charts/chart4.png` list price vs price per reliably used token at a full window
- `results.json` every number quoted in the article

## Limitations

Prices are a snapshot of one day. List prices ignore negotiated discounts, regional uplifts, priority or
flex tiers and cache storage fees. The capability index covers 213 of 365 market listings and 49 of 67
first-party rows. Effective-context measurements exist only for older models, so RQ3's usable-token
figures describe that cohort, not the newest releases.
