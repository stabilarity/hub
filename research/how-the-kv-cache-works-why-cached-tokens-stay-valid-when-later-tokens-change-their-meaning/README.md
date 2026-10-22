# How the KV cache works: why cached tokens stay valid when later tokens change their meaning

Reproducible code for the Stabilarity Research Hub article of the same title (Cost-Effective Enterprise AI series).

## Question

A decoder-only language model caches the key (K) and value (V) vectors of every token it has read. Later tokens can
change what an earlier word means ("The bank ... of the river" vs "The bank ... approved the loan"). Why is the cache
still correct, where does the re-interpretation happen, and where does the cache stop being exact?

## Experiments (`analysis.py`)

| ID | What it tests | Models |
|---|---|---|
| E1 exactness | Greedy decoding with the KV cache vs full recomputation without it, 24 prompts x 64 new tokens; max logit difference and identical sequences | Pythia-70M/160M/410M (fp32), Pythia-160M (fp64) |
| E2 immutability | K, V and hidden states of a prefix computed alone vs inside two continuations that give it different meanings; the same prefix in BERT | Pythia-160M/410M, bert-base-uncased |
| E3 agility | Nearest-centroid sense probe at the old (cached) position, at a new position that repeats the word after the context, and at the original position in BERT; attention from the new position to the old one | Pythia-160M, bert-base-uncased |
| E4 cost | CPU decode latency per token with and without the cache (timing only), analytical FLOPs, and KV bytes per token for public model configurations | Pythia-160M; configs in `data/model_configs.json` |
| E5 approximation | Teacher-forced decoding with a modified cache: fp16/bf16 casting, naive int8/int4/int2 quantisation, sliding-window eviction with and without 4 attention-sink tokens, and reuse of a chunk's cache computed without its preceding context | Pythia-160M |

## Data

- `data/ambiguity_pairs.json`: 16 hand-written ambiguous words with two disambiguating continuations and three
  unambiguous reference sentences per sense, plus 2 garden-path prefixes. Written for this study.
- `data/passages.json`: 24 documents from the Pile test split via
  [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) (same selection as
  `research/open-source-model-auditing-infrastructure-third-party-verification-of-training-data-claims`).
- `data/model_configs.json`: layer, head and head-dimension counts of public models, checked against their
  Hugging Face `config.json` files and technical reports (sources inside the file).
- Models: [EleutherAI Pythia](https://huggingface.co/EleutherAI) 70M/160M/410M (arXiv:2304.01373) and
  [bert-base-uncased](https://huggingface.co/bert-base-uncased) (arXiv:1810.04805), downloaded from Hugging Face.

## Run

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python analysis.py      # CPU only; writes charts/chart1..5.png and results.json
```

Everything in `results.json` except the `timing` block is deterministic (fixed seed, deterministic algorithms,
6 CPU threads). Float32 results can differ in the last bits on other CPUs or library versions; the float64 runs
are included to show that the remaining differences are rounding.
The fp16 and bf16 rows of E5 were added after the first full run and computed separately with the same
`run_variant` code and passages (3 processes x 2 threads); each is compared with an exact-cache run from the same process.

## Outputs

- `chart1.png` cosine distance of the ambiguous word's state between the two contexts, by layer (BERT vs Pythia)
- `chart2.png` sense-probe accuracy by layer at the old cached position, a new position and BERT's position
- `chart3.png` CPU time per new token with and without the cache vs context length
- `chart4.png` KV-cache bytes per token for public model configurations (16-bit)
- `chart5.png` top-1 agreement with the exact cache for lower-precision, quantised, evicted and reused caches
- `results.json` all numbers quoted in the article
