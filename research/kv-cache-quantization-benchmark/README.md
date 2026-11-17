# KV cache quantization and pruning benchmark (CPU, two small open models)

Hands-on test of what compressing the key-value (KV) cache costs a real model, measured on an ordinary
6-vCPU x86 server without a GPU.

| Key | Model (repo used, pinned revision in `analysis.py`) | Licence |
|---|---|---|
| qwen2.5-0.5b | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 |
| qwen3-0.6b | `Qwen/Qwen3-0.6B` (chat template with `enable_thinking=False`) | Apache-2.0 |

Llama-3.2-1B-Instruct was tried first; at fp32 (~5 GB of weights) it was killed by the out-of-memory killer on this
12 GB machine, so Qwen3-0.6B took its place. Qwen2.5-0.5B has 2 KV heads of 64 dims in 24 layers
(12 KiB of BF16 KV per token); Qwen3-0.6B has 8 KV heads of 128 dims in 28 layers (112 KiB per token).

## Questions
1. **Quantization:** how much needle-in-a-haystack (NIAH) retrieval accuracy and perplexity do INT8, INT4 and
   INT2 KV caches lose, with keys and values quantized per token versus the KIVI layout
   (keys per channel, values per token)?
2. **Pruning / eviction:** at a 50% and a 25% token budget, how do a sliding window, StreamingLLM
   (4 attention sinks + window), an H2O-style heavy-hitter cache and a SnapKV-style question-aware cache compare?
3. **Bytes:** which method gives the most retrieval accuracy per byte of KV memory?

## Protocol
- **NIAH (passkey style):** a chat prompt holds filler text (HellaSwag contexts) with the needle
  "The pass key is NNNNN. Remember it. NNNNN is the pass key." inserted at 5%, 25%, 50%, 75% or 95% depth,
  at a total prompt length of 1,024 and 2,048 tokens. Qwen2.5: 2 random passkeys per cell (20 prompts);
  Qwen3: 1 per cell (10 prompts), to stay inside the compute budget.
- The context (everything before the question) is prefilled once in full precision. The cache is then compressed,
  and only then are the question and answer fed in: the **compress-then-ask** setting of cached documents and
  multi-turn reuse. Question and answer tokens stay uncompressed (like KIVI's full-precision residual window).
- **Scoring:** the 5-digit answer is teacher-forced; a prompt counts as correct only if the greedy
  (argmax) prediction matches every answer token, which is exactly when greedy decoding would print the passkey.
  The answer log-probability is stored too.
- **Perplexity:** WikiText-2 (test). The first 1,024 tokens are cached and compressed; the NLL of the next 256
  tokens is measured. "No context" NLL is measured on those 256 tokens alone as a reference.
- **Decode latency:** 16 greedy single-token steps after a 2k-token prefill, full cache vs 50% and 25% window,
  3 repetitions, median of medians. Real (the cache is genuinely shorter), but noisy on a shared server.
- **KV bytes** are computed: BF16 = 16 bits/value; INT-b = b bits + an fp16 scale and zero per group of
  32 values (1 bit/value overhead); eviction = kept fraction of BF16.

## Methods
| Name | What it does |
|---|---|
| bf16 | cache rounded to bfloat16 (baseline) |
| tok-int{8,4,2} | asymmetric min-max fake quantization, keys and values per token (groups of 32 channels) |
| kivi-int{8,4,2} | keys per channel (groups of 32 tokens), values per token |
| window-{50,25} | keep the most recent 50% / 25% of context tokens |
| sink-{50,25} | 4 first tokens + most recent tokens |
| h2o-{50,25} | half recent, half heavy hitters by attention accumulated over all context queries, per KV head (one-shot, question-agnostic) |
| snapkv-{50,25} | top tokens by attention from the question tokens (max-pool kernel 7), per KV head (question-aware) |

## Honest caveats
- Quantization is **simulated** (quantize -> dequantize in fp32). It measures accuracy, not the speed of
  real INT4/INT2 kernels, and the memory numbers are computed, not allocated.
- Weights and activations run in fp32 (bf16 matmuls are emulated and slow on AVX2); the baseline cache is
  rounded to bf16.
- Keys are quantized after RoPE (KVQuant reports pre-RoPE key quantization is better).
- Small models, short contexts (1k/2k) and 10-20 prompts per method per model: wide confidence intervals.

## Data
- HellaSwag validation contexts (`Rowan/hellaswag`, MIT), `data/hellaswag_val_first100.json`, as haystack filler.
- WikiText-2 raw test (`Salesforce/wikitext`, CC BY-SA), `data/wikitext2_test_first100.json`.
Both were fetched from the Hugging Face datasets-server `rows` API and are committed here.

## How to run
```bash
python analysis.py --run qwen2.5-0.5b   # network for the weights; streamed into RAM, nothing written to disk
python analysis.py --run qwen3-0.6b
python analysis.py                      # offline: raw/*.json -> results.json + charts/chart1-5.png
```
Environment: Python 3.12, torch 2.10.0 (CPU), transformers 5.2, matplotlib for the charts.
Correctness checks stored per model (`checks` in `results.json`): cached vs single-pass forward logits, and the
eviction gather path with every token kept vs the baseline.

## Outputs
- `raw/<model>.json`: every prompt, method, prediction and timing
- `results.json`: aggregates (accuracy with 95% Wilson intervals, by depth and length; perplexity; KV bytes;
  quantization error; latency)
- `charts/chart1.png`: NIAH accuracy heatmap per method, depth and length
- `charts/chart2.png`: NIAH accuracy vs KV bytes
- `charts/chart3.png`: perplexity ratio vs BF16 cache
- `charts/chart4.png`: key vs value channel magnitudes and quantization error per layout
- `charts/chart5.png`: eviction, needle retention vs accuracy
