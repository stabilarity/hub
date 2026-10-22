# Third-party verification of training-data claims: membership and dataset inference on Pythia

Companion code for the Stabilarity Research Hub article
"Open-Source Model Auditing Infrastructure: Third-Party Verification of Training Data Claims".

## Question

An auditor holds only the weights of an open model. How well can they verify whether a given
document, or a given collection of documents, was in its training data? And how much does the usual
shortcut (using text published after the model's cutoff as "known non-members") inflate the answer?

## Data (all public, committed under `data/`)

| File | What it is | Source |
|---|---|---|
| `pile_train_sample.json` | first 400 documents of >= 1,500 characters from each of Wikipedia (en), ArXiv and Pile-CC in Pile train shard 00 (members of Pythia's training data) | [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted), `train/00.jsonl.zst` |
| `pile_test_sample.json` | the same selection from the Pile test split (never trained on by Pythia; same distribution) | same, `test.jsonl.zst` |
| `wikipedia_fresh_2026.json` | 400 English Wikipedia articles created 13-21 October 2026 (MediaWiki API `list=recentchanges`, `rctype=new`, >= 5,000 bytes, plaintext extracts) | en.wikipedia.org (CC BY-SA) |
| `wikimia_64.json`, `wikimia_128.json` | the WikiMIA benchmark (members: events before 2017; non-members: events after 2023) | [swj0419/WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA) |
| `scores.csv` | per-document membership signals for three Pythia models (output of `score.py`) | this repo |
| `scoring_env.json` | versions, hardware and timing of the scoring run | this repo |

Texts are truncated to 2,500 characters; at most the first 256 tokens are scored. Section-heading lines
(`== ... ==`) are removed from the fresh Wikipedia articles because Pile Wikipedia text has none, so the
fresh set differs from the Pile set in date rather than in markup.

Pile: Gao et al., [arXiv:2101.00027](https://arxiv.org/abs/2101.00027). Pythia: Biderman et al.,
[arXiv:2304.01373](https://arxiv.org/abs/2304.01373). WikiMIA: Shi et al., [arXiv:2310.16789](https://arxiv.org/abs/2310.16789).

## Method

* `score.py` runs EleutherAI/pythia-70m, -160m and -410m (fp32, CPU) over every text and records four
  standard reference-free signals: mean token loss, zlib ratio (loss over zlib-compressed length),
  Min-K% Prob and Min-K%++ (k = 20%).
* `analysis.py` (offline) computes ROC AUC with 1,000-sample bootstrap CIs and TPR at 1% FPR for each
  comparison, two model-free "blind" baselines (a 5-fold cross-validated TF-IDF bag-of-words classifier, and
  the latest four-digit year mentioned in the text), and a Maini et al.-style dataset-inference test
  (logistic weighting of the four signals learnt on one half, one-sided Welch t-test on the other half,
  500 trials per set size) with matched and with date-shifted validation sets.

## Run

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python score.py      # downloads the three Pythia checkpoints once; CPU only
./venv/bin/python analysis.py   # charts/chart1-4.png and results.json
```

The committed `data/scores.csv` was produced on 22 October 2026 on a 6-core x86-64 CPU (no GPU), fp32,
Python 3.12.3, torch 2.14.1+cpu, transformers 5.20.2. Scoring took 494 s (Pythia-70M), 923 s (160M) and
2,100 s (410M); `data/scoring_env.json` records this. `analysis.py` needs only `scores.csv` and the text
samples, runs offline in about 5 minutes, and is deterministic (fixed seed 20261021; two consecutive runs
produce a byte-identical `results.json`).

## Outputs

* `charts/chart1.png` - matched member vs non-member AUC per attack and domain (Pythia-410M)
* `charts/chart2.png` - date-shifted sets: best attack vs blind baselines
* `charts/chart3.png` - AUC vs model size, matched vs shifted
* `charts/chart4.png` - dataset-inference power and false-positive rate vs set size, with a matched validation set and with a validation set of Wikipedia articles created in October 2026
* `results.json` - every number used in the article
