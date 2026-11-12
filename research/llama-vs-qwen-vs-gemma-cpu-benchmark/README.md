# Llama vs Qwen vs Gemma on a CPU

Hands-on, measured comparison of the current ~1B-class open models from three families on an ordinary
6-vCPU x86 server (AVX2, no AVX-512, no AMX, no GPU):

| Key | Model (repo used, pinned revision in `bench.py`) | Official repo | Licence |
|---|---|---|---|
| llama-3.2-1b | `unsloth/Llama-3.2-1B-Instruct` (byte-identical safetensors size to the official file) | `meta-llama/Llama-3.2-1B-Instruct` | Llama 3.2 Community License |
| gemma-3-1b | `unsloth/gemma-3-1b-it` | `google/gemma-3-1b-it` | Gemma Terms of Use |
| qwen3-0.6b | `Qwen/Qwen3-0.6B` | same | Apache-2.0 |
| qwen2.5-0.5b | `Qwen/Qwen2.5-0.5B-Instruct` | same | Apache-2.0 |

The official Meta and Google repos are gated, so the ungated mirrors were used. Mistral is absent on
purpose: its smallest current open model (Ministral 3 3B, about 3.85 B parameters with the vision encoder)
needs about 7.7 GB in bfloat16 and does not fit this machine's RAM or disk.

## Questions
1. Accuracy: zero-shot log-likelihood multiple choice (lm-eval-harness style, byte-length normalised
   `acc_norm`, no chat template) on the first 30 items of ARC-Easy (test), ARC-Challenge (test) and
   HellaSwag (validation), with 95% Wilson intervals. The subsets are small on purpose (compute budget):
   read them as a sanity check, not as a leaderboard.
2. Speed: prefill tokens/s on a 512-token prompt and greedy decode tokens/s for 128 new tokens, in
   bfloat16, at `torch.set_num_threads(6)` and `(2)`.
3. Memory: peak resident set size (VmHWM) of the benchmark process, on-disk size of the weights, and the
   KV cache per token from each config (`layers x kv_heads x head_dim x 2 x 2 bytes`, with Gemma's
   sliding-window layers capped at their window), checked against the cache actually allocated after the
   512-token prefill.
4. Derived: accuracy points above chance per GB of RAM and per billion parameters, correct answers per
   decode-second.

## Data
- ARC (AI2 Reasoning Challenge), `allenai/ai2_arc`, CC BY-SA 4.0: `data/arc_*_first100.json`
- HellaSwag, `Rowan/hellaswag`, MIT: `data/hellaswag_val_first100.json`

Both were fetched once from the Hugging Face datasets-server `rows` API and are committed here.

## How to run
```bash
# 1) measurements (network needed for weights; about 18 min wall time for all four on 6 vCPUs)
python bench.py qwen2.5-0.5b --items 30
python bench.py qwen3-0.6b --items 30
python bench.py gemma-3-1b --items 30
python bench.py llama-3.2-1b --items 30
# 2) aggregation and charts (offline, deterministic)
python analysis.py
```
`bench.py` streams `model.safetensors` from the Hub directly into the parameter tensors in one pass and
checks its SHA-256 against the Hub's LFS hash, so no weight file is written to disk. It also checks the
shared-prefix scorer against a naive one-forward-per-choice scorer on two items per model
(`check_shared_kv_vs_naive` in `results.json`).

Environment: Python 3.12, torch 2.10.0 (CPU), transformers 5.2, matplotlib 3.10 for the charts.

## Outputs
- `raw/<model>.json`: everything measured per model
- `results.json`: aggregated numbers, derived metrics and the exact values plotted
- `charts/chart1.png`: accuracy per task with 95% Wilson intervals
- `charts/chart2.png`: prefill and decode tokens/s at 2 and 6 threads
- `charts/chart3.png`: KV-cache MiB at 4k and 32k context
- `charts/chart4.png`: pooled accuracy vs peak RAM
