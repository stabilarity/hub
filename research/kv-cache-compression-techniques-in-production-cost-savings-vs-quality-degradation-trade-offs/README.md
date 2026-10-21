# KV-Cache Compression in Production: Cost Savings vs Quality Degradation

Code and data for the Stabilarity Research Hub article *KV-Cache Compression Techniques in Production: Cost
Savings vs Quality Degradation Trade-offs* (Cost-Effective Enterprise AI series).

## Question

- **RQ1.** How much KV-cache memory does each production model configuration need per token, and how many
  concurrent 8K, 32K and 128K sequences fit on one serving unit at BF16, FP8, 4-bit and 2-bit precision?
- **RQ2.** What do those capacity gains imply for decode cost per million output tokens, and when does the
  saving fall short of the compression ratio?
- **RQ3.** How much task quality do representative methods give up at their realised compression ratio on
  LongBench workloads, and which ones sit on the cost-quality frontier?

## Data (all public)

| File | Content | Source |
|---|---|---|
| `data/models.csv` | layers, KV heads, head dimension, MLA latent sizes, parameter counts | Llama 3 report (arXiv:2407.21783, Table 3), Qwen3 report (arXiv:2505.09388, Table 1), DeepSeek-V3 report (arXiv:2412.19437, Sec. 4.2), Hugging Face `config.json` and model API (`safetensors.total`) |
| `data/gpus.csv` | HBM capacity, bandwidth, on-demand price per GPU-hour | https://www.nvidia.com/en-us/data-center/h100/, https://www.nvidia.com/en-us/data-center/h200/, https://www.runpod.io/pricing (Secure Cloud, accessed 2026-10-21) |
| `data/kv_formats.csv` | compression factor vs BF16 for each KV format | vLLM quantized KV cache docs (FP8), NVIDIA NVFP4 KV cache blog (about 50% of FP8), Agrawal and Mayer arXiv:2607.05399 Table 4 (realised ratios) |
| `data/table2_longbench.csv` | LongBench scores per method for Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 | arXiv:2607.05399, Table 2 |
| `data/table4_compression_llama.csv` | realised compression ratio per dataset and context bucket (Llama-3.1-8B) | arXiv:2607.05399, Table 4 |

## How to run

```bash
python3 analysis.py   # writes charts/chart1.png ... chart4.png and results.json
```

Python 3.10+, numpy, matplotlib. Offline and deterministic (no random numbers).

## Method (our estimate, an idealised upper bound)

- KV bytes per token: GQA `2 x layers x kv_heads x head_dim x 2 bytes`; MLA `layers x (latent + rope) x 2 bytes`.
- KV budget per serving unit: `0.92 x HBM x n_gpu - BF16 weight bytes` (0.92 is vLLM's default
  `--gpu-memory-utilization`). Activations and runtime buffers are ignored.
- Batch = `min(floor(budget / per-sequence KV), 256)`; the 256 cap is our assumption for a scheduler limit.
- Decode is treated as memory-bandwidth bound: step time = (weight bytes + KV bytes read) / aggregate bandwidth.
  Cost per 1M output tokens = USD per GPU-hour x n_gpu / tokens per hour x 1e6.
- Compression factor `c` divides KV bytes per token. Dequantisation compute, scale metadata, kernel
  inefficiency, prefill cost and response-length changes are not modelled, so savings are upper bounds.
- Quality retention = method score / uncompressed (All-KV) score per dataset; pooled over the two models
  and six datasets (12 model-task pairs). The cost of each method uses its mean realised compression
  ratio for Llama-3.1-8B on one H100 at 32K tokens.

## Serving units

| Unit | Model | GPUs |
|---|---|---|
| Llama-3.1-8B on 1x H100 | Llama-3.1-8B-Instruct, BF16 weights | 1 x H100 SXM 80 GB |
| Qwen3-32B on 1x H200 | Qwen3-32B, BF16 weights | 1 x H200 141 GB |
| Llama-3.3-70B on 2x H200 | Llama-3.3-70B-Instruct, BF16 weights | 2 x H200 141 GB, tensor parallel |

## Outputs

- `charts/chart1.png` KV memory per 128K-token sequence by model and format
- `charts/chart2.png` max concurrent sequences vs context, Llama-3.1-8B on one H100
- `charts/chart3.png` decode cost per 1M output tokens vs context for three serving units
- `charts/chart4.png` cost vs quality retention on LongBench tasks for six methods
- `results.json` every number used in the article
