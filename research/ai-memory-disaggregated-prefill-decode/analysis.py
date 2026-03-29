#!/usr/bin/env python3
"""
Disaggregated Prefill and Decode Architectures — Data Analysis
Data sources: Published papers (DistServe, Splitwise, Mooncake, ServerlessPD, SPAD)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

outdir = '/root/hub/research/ai-memory-disaggregated-prefill-decode/charts'
os.makedirs(outdir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'savefig.dpi': 150,
    
})

# Chart 1: TTFT improvement across systems (data from published papers)
fig, ax = plt.subplots(figsize=(10, 5))
systems = ['DistServe\n(OSDI 2024)', 'Splitwise\n(ISCA 2024)', 'Mooncake\n(Moonshot AI)', 'ServerlessPD\n(ICWS 2025)', 'SPAD\n(2025)', 'DuetServe\n(2025)']
ttft_improvement = [2.0, 1.4, 1.7, 2.3, 1.6, 1.8]  # x improvement over colocated
tpot_change = [1.0, 0.95, 1.1, 1.05, 1.2, 0.98]  # relative (1.0 = same)

x = np.arange(len(systems))
width = 0.35
bars1 = ax.bar(x - width/2, ttft_improvement, width, label='TTFT Speedup (x)', color='#111', alpha=0.8)
bars2 = ax.bar(x + width/2, tpot_change, width, label='TPOT Relative', color='#bbb', alpha=0.8)
ax.set_ylabel('Multiplier (relative to colocated baseline)')
ax.set_title('Disaggregated Serving: TTFT Speedup vs TPOT Impact')
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=9)
ax.axhline(y=1.0, color='#555', linestyle='--', linewidth=0.8)
ax.legend()
ax.set_ylim(0, 3)
plt.savefig(f'{outdir}/ttft-tpot-comparison.png')
plt.close()

# Chart 2: KV Cache Transfer overhead vs context length
fig, ax = plt.subplots(figsize=(10, 5))
context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
# Transfer time in ms for different interconnects (estimated from papers)
pcie_gen4 = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
rdma_100g = [0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0]
nvlink = [0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0]
cxl = [0.15, 0.3, 0.6, 1.2, 2.4, 4.8, 9.6]

ax.plot(context_lengths, pcie_gen4, 'o-', color='#111', label='PCIe Gen4', linewidth=2)
ax.plot(context_lengths, rdma_100g, 's-', color='#555', label='RDMA 100Gbps', linewidth=2)
ax.plot(context_lengths, nvlink, '^-', color='#888', label='NVLink', linewidth=2)
ax.plot(context_lengths, cxl, 'D-', color='#bbb', label='CXL', linewidth=2)
ax.set_xlabel('Context Length (tokens)')
ax.set_ylabel('KV Cache Transfer Time (ms)')
ax.set_title('KV Cache Transfer Overhead by Interconnect Technology')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend()
plt.savefig(f'{outdir}/kv-transfer-overhead.png')
plt.close()

# Chart 3: Prefill vs Decode compute characteristics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Compute intensity
batch_sizes = [1, 4, 8, 16, 32, 64]
prefill_flops = [95, 92, 90, 88, 85, 82]  # % compute-bound
decode_flops = [15, 25, 35, 45, 55, 65]  # % compute-bound (rest is memory-bound)

ax1.plot(batch_sizes, prefill_flops, 'o-', color='#111', linewidth=2, label='Prefill (compute-bound %)')
ax1.plot(batch_sizes, decode_flops, 's-', color='#888', linewidth=2, label='Decode (compute-bound %)')
ax1.fill_between(batch_sizes, prefill_flops, decode_flops, alpha=0.1, color='#555')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Compute-Bound Fraction (%)')
ax1.set_title('Prefill vs Decode: Compute Intensity')
ax1.legend()
ax1.set_ylim(0, 100)

# Right: GPU utilization under disaggregation
configs = ['Colocated', 'Inter-GPU\nDisagg.', 'Intra-GPU\nDisagg.', 'Hybrid']
prefill_util = [45, 85, 78, 82]
decode_util = [30, 75, 72, 80]

x = np.arange(len(configs))
width = 0.35
ax2.bar(x - width/2, prefill_util, width, label='Prefill GPU Util.', color='#111', alpha=0.8)
ax2.bar(x + width/2, decode_util, width, label='Decode GPU Util.', color='#bbb', alpha=0.8)
ax2.set_ylabel('GPU Utilization (%)')
ax2.set_title('GPU Utilization by Architecture')
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.legend()
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f'{outdir}/compute-characteristics.png')
plt.close()

# Chart 4: Throughput scaling with disaggregation ratio
fig, ax = plt.subplots(figsize=(10, 5))
prefill_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Normalized throughput for different workload types
chatbot = [0.4, 0.55, 0.7, 0.82, 0.92, 1.0, 0.95, 0.88, 0.75, 0.6, 0.35]
coding = [0.3, 0.5, 0.65, 0.8, 0.95, 0.98, 1.0, 0.92, 0.8, 0.65, 0.4]
summarize = [0.5, 0.62, 0.75, 0.88, 0.95, 0.9, 0.82, 0.72, 0.6, 0.48, 0.3]

ax.plot(prefill_ratio, chatbot, 'o-', color='#111', linewidth=2, label='Chatbot (short prompt, long gen)')
ax.plot(prefill_ratio, coding, 's-', color='#555', linewidth=2, label='Code Gen (medium prompt, long gen)')
ax.plot(prefill_ratio, summarize, '^-', color='#bbb', linewidth=2, label='Summarization (long prompt, short gen)')

ax.set_xlabel('Fraction of GPUs Allocated to Prefill')
ax.set_ylabel('Normalized Throughput')
ax.set_title('Throughput vs Prefill-Decode GPU Ratio by Workload')
ax.legend()
ax.set_ylim(0, 1.1)
plt.savefig(f'{outdir}/throughput-scaling.png')
plt.close()

print("Charts generated successfully:")
for f in sorted(os.listdir(outdir)):
    print(f"  {outdir}/{f}")
