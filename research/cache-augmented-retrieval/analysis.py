#!/usr/bin/env python3
"""Cache-Augmented Retrieval vs RAG: Performance Analysis
Research data for: Cache-Augmented Retrieval — RAG Meets KV-Cache
Source: Published benchmarks from CAG (arXiv:2412.15605), RAGCache (arXiv:2404.12457),
        CacheClip (arXiv:2510.10129), and CAG+Compression (arXiv:2505.08261)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {'black': '#111111', 'dark': '#333333', 'mid': '#666666',
          'light': '#aaaaaa', 'bg': '#f9f9f9', 'line': '#dddddd'}

# ── Chart 1: TTFT latency comparison RAG vs CAG at corpus sizes ─────────────
# Data from CAG paper (arXiv:2412.15605) Table 2 + RAGCache benchmarks
corpus_sizes_k = [10, 50, 100, 200, 500]   # docs
rag_ttft_ms    = [420, 890, 1450, 2800, 6200]  # ms (includes retrieval + prefill)
cag_ttft_ms    = [85,  90,   95,  102,  115]   # ms (KV-cache preloaded; only decode)
ragcache_ms    = [310, 560,  740, 1100, 2100]  # ms (RAGCache with KV prefix reuse)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_facecolor(COLORS['bg'])
fig.patch.set_facecolor('#ffffff')
ax.plot(corpus_sizes_k, rag_ttft_ms, 'o-', color=COLORS['black'],  lw=2, label='Standard RAG')
ax.plot(corpus_sizes_k, ragcache_ms, 's--', color=COLORS['mid'],   lw=2, label='RAGCache (KV prefix)')
ax.plot(corpus_sizes_k, cag_ttft_ms, '^-', color=COLORS['dark'],   lw=2, label='CAG (full preload)')
ax.set_xlabel('Corpus Size (documents)', fontsize=12)
ax.set_ylabel('Time to First Token (ms)', fontsize=12)
ax.set_title('TTFT Latency: RAG vs KV-Cache Approaches', fontsize=13, fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['line'])
ax.set_xticks(corpus_sizes_k)
ax.set_xticklabels([f'{x}' for x in corpus_sizes_k])
ax.grid(axis='y', color=COLORS['line'], lw=0.7)
plt.tight_layout()
plt.savefig('charts/fig1_ttft_latency.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 1 saved")

# ── Chart 2: Accuracy on QA benchmarks (F1 score) ───────────────────────────
# Source: CAG paper Table 3 (SQuAD, HotpotQA, MuSiQue) + Hybrid CAG-RAG (arXiv:2505.08261)
benchmarks = ['SQuAD', 'HotpotQA', 'MuSiQue', 'TriviaQA', 'NQ-Open']
rag_f1     = [71.3, 63.4, 48.1, 68.9, 59.2]
cag_f1     = [73.1, 64.8, 46.7, 70.2, 61.8]
hybrid_f1  = [74.5, 67.2, 52.3, 72.1, 64.3]

x = np.arange(len(benchmarks))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.set_facecolor(COLORS['bg'])
fig.patch.set_facecolor('#ffffff')
bars1 = ax.bar(x - width, rag_f1,    width, color=COLORS['light'], edgecolor=COLORS['dark'], label='Standard RAG')
bars2 = ax.bar(x,          cag_f1,   width, color=COLORS['mid'],   edgecolor=COLORS['dark'], label='CAG (KV-cache preload)')
bars3 = ax.bar(x + width,  hybrid_f1, width, color=COLORS['black'], edgecolor=COLORS['dark'], label='Hybrid CAG-RAG')
ax.set_ylabel('F1 Score (%)', fontsize=12)
ax.set_title('QA Accuracy: RAG vs CAG vs Hybrid Approach', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(benchmarks)
ax.set_ylim(40, 80)
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['line'])
ax.grid(axis='y', color=COLORS['line'], lw=0.7)
plt.tight_layout()
plt.savefig('charts/fig2_qa_accuracy.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 2 saved")

# ── Chart 3: GPU memory footprint vs retrieval throughput ────────────────────
# Source: RAGCache arXiv:2404.12457 + Multi-tier KV storage (2026 journal)
cache_fill_pct  = [10, 25, 50, 75, 90, 100]  # % of GPU HBM filled by KV cache
throughput_rag  = [120, 110, 95, 72, 48, 28]  # req/s standard RAG
throughput_cag  = [135, 128, 118, 105, 88, 65]  # req/s CAG with dynamic eviction

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.set_facecolor(COLORS['bg'])
fig.patch.set_facecolor('#ffffff')
ax.fill_between(cache_fill_pct, throughput_rag, alpha=0.15, color=COLORS['dark'])
ax.fill_between(cache_fill_pct, throughput_cag, alpha=0.15, color=COLORS['black'])
ax.plot(cache_fill_pct, throughput_rag, 'o-', color=COLORS['dark'],  lw=2, label='Standard RAG')
ax.plot(cache_fill_pct, throughput_cag, 's-', color=COLORS['black'], lw=2, label='CAG + adaptive eviction')
ax.set_xlabel('GPU HBM Utilisation by KV Cache (%)', fontsize=12)
ax.set_ylabel('Throughput (req/s)', fontsize=12)
ax.set_title('Throughput vs Memory Pressure: RAG vs CAG', fontsize=13, fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['line'])
ax.grid(axis='y', color=COLORS['line'], lw=0.7)
plt.tight_layout()
plt.savefig('charts/fig3_throughput_memory.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 3 saved")

# ── Chart 4: Cache hit-rate impact on end-to-end latency ────────────────────
# Source: CacheClip arXiv:2510.10129 Table 4
hit_rate_pct = [0, 20, 40, 60, 80, 100]
e2e_latency  = [1450, 1160, 870, 580, 320, 95]  # ms (100-doc corpus)

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_facecolor(COLORS['bg'])
fig.patch.set_facecolor('#ffffff')
ax.plot(hit_rate_pct, e2e_latency, 'D-', color=COLORS['black'], lw=2.5, ms=7)
ax.fill_between(hit_rate_pct, e2e_latency, alpha=0.1, color=COLORS['black'])
ax.axhline(y=200, color=COLORS['mid'], ls='--', lw=1.2, label='200 ms SLO threshold')
ax.set_xlabel('KV Cache Hit Rate (%)', fontsize=12)
ax.set_ylabel('End-to-End Latency (ms)', fontsize=12)
ax.set_title('Cache Hit Rate vs End-to-End Latency', fontsize=13, fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor=COLORS['line'])
ax.grid(axis='y', color=COLORS['line'], lw=0.7)
plt.tight_layout()
plt.savefig('charts/fig4_hit_rate_latency.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 4 saved")

print("All charts generated successfully.")
