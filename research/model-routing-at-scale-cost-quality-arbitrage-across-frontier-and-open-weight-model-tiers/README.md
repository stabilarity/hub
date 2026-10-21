# Model Routing at Scale: Cost-Quality Arbitrage Across Frontier and Open-Weight Model Tiers

Code and data for the Stabilarity Research Hub article of the same title (Cost-Effective Enterprise AI series).

## Questions

- **RQ1.** How much of a frontier model's cost can be avoided in principle by sending each query to the cheapest
  adequate model in a frontier / small-closed / open-weight pool, and how much of that headroom is visible per query?
- **RQ2.** How much does a trained router actually save at quality targets of 95%, 98% and 100% of the frontier
  model's score, compared with a random traffic split and with the oracle, and what happens when the task family
  was not seen in training?
- **RQ3.** How sensitive are realised savings to router discrimination (AUC) and to the price gap between tiers at
  current list prices?

## Data (all public)

| File | Content | Source |
|---|---|---|
| `data/sprout_scores.csv.gz` | 44,241 prompts (30,968 train, 6,636 validation, 6,637 test): task family, and for each of 13 models the judged score (0-1), input tokens and output tokens. No prompt or response text. | SPROUT dataset, https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT (Somerstep et al., CARROT, arXiv:2502.03261) |
| `data/router_predictions.csv.gz` | Predicted score per model for every validation and test prompt from a TF-IDF + ridge router trained on the train split, in two variants: `in_dist` (all train families) and `lofo` (the prompt's task family removed from training) | built by `build_data.py` from the SPROUT train split |
| `data/prices.csv` | USD per 1M input/output tokens for the four pool models | https://developers.openai.com/api/docs/pricing (GPT-4o, GPT-4o-mini), https://www.together.ai/pricing (Llama 3.3 70B; Llama 3 8B Instruct Lite used as proxy for Llama 3.1 8B), accessed 2026-10-21 |
| `data/tier_pairs.csv` | five current frontier / cheap-tier price pairs | OpenAI, Anthropic (https://platform.claude.com/docs/en/about-claude/pricing), Google (https://ai.google.dev/gemini-api/docs/pricing) and Together pricing pages, accessed 2026-10-21 |

Task families in SPROUT: chat instructions (OpenHermes), MMLU-Pro, MATH, GPQA, MuSR and RAGBench.

## How to run

```bash
python3 analysis.py      # offline, deterministic; writes charts/chart1.png ... chart4.png and results.json
```

Python 3.10+, numpy, matplotlib. `build_data.py` (needs the SPROUT parquet files, pyarrow and scikit-learn) rebuilds
the two `.csv.gz` files; it is deterministic.

## Method

- **Pool.** GPT-4o (frontier), GPT-4o-mini (small closed), Llama-3.3-70B (large open-weight), Llama-3.1-8B (small
  open-weight). Cost per query uses each model's own measured input and output tokens times its list price, so verbose
  models pay for their verbosity. Quality is the mean judged score on the 6,637 test prompts.
- **Pool router.** For each prompt, choose `argmax_m predicted_score_m - lambda * expected_cost_m`, where the expected
  cost is the model's mean cost on the train split. For each quality target, lambda is the largest value whose
  validation quality reaches the target fraction of GPT-4o's validation score; it is then applied unchanged to test
  ("validation-calibrated, realised"). A second view picks the cheapest lambda whose *test* quality meets the target
  ("matched quality", ideal calibration). A matched-quality saving of `-1.0` in `results.json` means no lambda met the target.
- **Two-tier router.** Escalate from a cheap model to GPT-4o when the predicted score gap exceeds a threshold chosen
  the same way.
- **Oracle.** The same rules with the true test scores and costs. It is an upper bound, not a deployable router.
- **Random split.** The best random allocation of traffic between any two pool models at matched cost.
- **Router-quality sensitivity.** For the GPT-4o / Llama-3.3-70B pair, the escalation score is the true score gap plus
  Gaussian noise (seed 20261021); for each noise level we report the AUC for "GPT-4o scores higher" and the saving at
  98% of GPT-4o's score (threshold set on test for every point).
- **Price-gap sensitivity.** Saving = 1 - e - (1 - e) / r for escalation rate e and blended price ratio r, with r
  computed at the mean SPROUT token mix (221 input, 575 output tokens). Equal tokens per query across tiers are assumed.

## Limitations

The models are 2024-generation; SPROUT scores come from an LLM judge; the router is deliberately simple (TF-IDF +
ridge); prices are list prices on one day; the Llama 3.1 8B price is a proxy.

## Outputs

- `charts/chart1.png` cost-quality plane: single models, random split, trained routers, oracle
- `charts/chart2.png` savings at matched test quality for 95%, 98% and 100% targets
- `charts/chart3.png` saving at 98% vs router AUC (two-tier GPT-4o / Llama-3.3-70B)
- `charts/chart4.png` saving vs frontier-to-cheap price ratio, with five current tier pairs
- `results.json` every number used in the article
