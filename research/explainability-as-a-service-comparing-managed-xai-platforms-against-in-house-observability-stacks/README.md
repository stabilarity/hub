# Explainability-as-a-Service: Comparing Managed XAI Platforms Against In-House Observability Stacks

Code and data for the Stabilarity Research Hub article of the same title (AI Observability & Monitoring series).

## Questions

- **RQ1.** What latency does each explanation estimator add to a single online prediction, comparing the sampling
  estimators that managed services expose (Kernel SHAP in SageMaker Clarify; sampled Shapley with a path count in
  Vertex Explainable AI) with what an in-house stack can run (exact TreeSHAP, gradient attributions)?
- **RQ2.** How close are the sampling estimators to exact Shapley values at the sample budgets those services expose?
- **RQ3.** At what traffic volume does a self-hosted open-source LLM-observability stack become cheaper than
  span-priced managed plans, and which plans document trace retention of six months or more?

## Data (all public)

| File | Content | Source |
|---|---|---|
| `data/adult.data.gz` | UCI Adult census income, training file (32,561 rows, 14 features) | https://archive.ics.uci.edu/dataset/2/adult |
| (embedded) | Wisconsin diagnostic breast cancer (569 rows, 30 features), shipped with scikit-learn | `sklearn.datasets.load_breast_cancer` |
| `data/managed_pricing.csv` | List prices of span- or unit-priced managed plans | https://www.datadoghq.com/pricing/?product=llm-observability, https://langfuse.com/pricing (read 2026-10-21) |
| `data/retention_coverage.csv` | Default and longest listed trace retention per plan | the two pages above and https://www.langchain.com/pricing (read 2026-10-21) |
| `data/infra_prices.csv` | AWS EC2 on-demand Linux prices, us-east-1 | https://aws.amazon.com/ec2/pricing/on-demand/ (price file published 2026-10-15) |

## How to run

```bash
pip install "numpy<2.4" shap==0.54.1 captum==0.9.0 xgboost torch scikit-learn scipy matplotlib
python3 analysis.py      # offline; writes charts/chart1.png ... chart4.png and results.json
```

Timings were taken on an AMD EPYC virtual machine (2.0 GHz), one thread (`OMP_NUM_THREADS=1`), Python 3.12,
shap 0.54.1, xgboost 3.2.0, torch 2.10, captum 0.9.0. Attribution values and fidelity metrics are deterministic
(fixed seed 20261021); latencies vary by a few tens of percent between runs and machines.

## Method

- **Models.** XGBoost, 200 trees, depth 6, on Adult (ordinal-encoded categoricals) and on breast cancer; a
  64-64 ReLU MLP (PyTorch) on standardised Adult features. 75/25 stratified split.
- **Baseline and target.** All estimators explain the log-odds output relative to one baseline row (the training
  median). The exact reference is interventional TreeSHAP with that single baseline; on Adult it was checked against
  brute-force enumeration of all 2^14 coalitions (max abs difference 6.4e-7). For the MLP the reference is the
  brute-force exact Shapley value.
- **Estimators.** Kernel SHAP (`shap.KernelExplainer`, `l1_reg=False`) at 64, 256, 1,024 and 4,096 samples, plus
  the library default `l1_reg` (`num_features(10)` since shap 0.47) at 1,024; permutation-sampling Shapley with path counts 1, 5, 10, 25 and 50 (the
  range Vertex accepts; 25 is its suggested value); path-dependent TreeSHAP as built into XGBoost; Captum
  Saliency, Integrated Gradients (32 steps), GradientSHAP (16 samples); input times gradient relative to the baseline.
- **Metrics.** Median and p95 latency per single-row explanation over 40 test rows (median of 3 repeats per row);
  latency as a multiple of one prediction; normalised mean absolute error against the exact values; Spearman
  correlation of |attribution|; overlap of the top-3 features. Compute cost per million explanations uses the
  per-vCPU share of an m7i.xlarge on-demand price (our estimate).
- **TCO model.** Each traced request = 1 trace, 4 LLM spans, 6 other spans, 1 score. Datadog bills LLM spans only;
  Langfuse bills every trace, observation and score as a unit, at the graduated rates of its pricing calculator
  ($8, $7, $6.50 and $6 per 100k units above 100k, 1M, 10M and 50M units). Self-hosted: 3 m7i.xlarge nodes plus 1 m7i.2xlarge for each started
  block of 100M units per month, +25% for storage and backups, and 0.25 FTE of a software developer at the BLS May 2025 median wage
  ($135,980) times a 1.3 loading factor. Sensitivity: 0.10 and 0.50 FTE; 2 and 8 LLM spans per request.
  All self-hosted figures are our estimates.

## Outputs

- `charts/chart1.png` latency per explanation by estimator (both tree models)
- `charts/chart2.png` cost-fidelity frontier of Kernel SHAP and sampled Shapley
- `charts/chart3.png` monthly cost of managed plans vs self-hosted stack across traffic volumes
- `charts/chart4.png` documented retention vs the six-month floor in Regulation (EU) 2024/1689 Art. 19 and 26(6)
- `results.json` every number used in the article

## Limitations

Two tabular datasets and one tree configuration; single-thread CPU timing in Python, which inflates fixed
overheads of Kernel SHAP; our sampled-Shapley implementation follows the published algorithm, not either vendor's
code; list prices on one day; infrastructure sizing and engineering effort are assumptions, varied in the sensitivity
analysis.
