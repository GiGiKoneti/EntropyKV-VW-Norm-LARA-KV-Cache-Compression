"""
Generate all Qwen2-specific paper-quality plots:
  1. Qwen2 Perplexity vs Budget curve
  2. Qwen2 QA F1 vs Budget curve
  3. VRAM profiling bar chart
  4. Decode throughput bar chart
  5. Combined profiling summary (2-panel)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'Full Cache':               '#888888',
    'STREAMING':                '#e74c3c',
    'StreamingLLM':             '#e74c3c',
    'H2O':                      '#3498db',
    'SNAPKV':                   '#2ecc71',
    'SnapKV':                   '#2ecc71',
    'RANDOM':                   '#95a5a6',
    'ENTROPYKV (L2_NORM)':      '#9b59b6',
    'EntropyKV (L2)':           '#9b59b6',
    'ENTROPYKV (VW-NORM + LARA)': '#f39c12',
    'EntropyKV (VW+LARA)':     '#f39c12',
}

MARKERS = {
    'STREAMING': 'v', 'H2O': 's', 'SNAPKV': 'D', 'RANDOM': 'x',
    'ENTROPYKV (L2_NORM)': '^', 'ENTROPYKV (VW-NORM + LARA)': 'o',
    'StreamingLLM': 'v', 'EntropyKV (L2)': '^', 'EntropyKV (VW+LARA)': 'o',
    'Full Cache': 'p',
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'extracted_data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(name):
    with open(os.path.join(DATA_DIR, name), 'r') as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# 1. Qwen2 Perplexity vs Budget
# ════════════════════════════════════════════════════════════════════════════
def plot_ppl():
    data = load_json('ppl_sweeps_results.json')
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, scores in data.items():
        budgets = sorted([float(b) for b in scores.keys()], reverse=True)
        ppls = [scores[str(b)] for b in budgets]
        ax.plot(budgets, ppls,
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'),
                label=method, linewidth=2, markersize=7)

    ax.set_xlabel('KV Cache Budget Ratio')
    ax.set_ylabel('Perplexity (↓ better)')
    ax.set_title('Qwen2-1.5B-Instruct · Perplexity vs KV Cache Budget (32k Context)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.invert_xaxis()

    path = os.path.join(FIG_DIR, 'qwen2_ppl_vs_budget.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 2. Qwen2 QA F1 vs Budget
# ════════════════════════════════════════════════════════════════════════════
def plot_qa():
    data = load_json('qa_sweeps_results.json')
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, scores in data.items():
        budgets = sorted([float(b) for b in scores.keys()], reverse=True)
        f1s = [scores[str(b)] for b in budgets]
        ax.plot(budgets, f1s,
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'),
                label=method, linewidth=2, markersize=7)

    ax.set_xlabel('KV Cache Budget Ratio')
    ax.set_ylabel('F1 Score (↑ better)')
    ax.set_title('Qwen2-1.5B-Instruct · Downstream QA F1 vs KV Cache Budget (32k Context)')
    ax.set_ylim(-0.02, 0.35)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.invert_xaxis()

    path = os.path.join(FIG_DIR, 'qwen2_qa_vs_budget.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 3. VRAM Profiling Bar Chart
# ════════════════════════════════════════════════════════════════════════════
def plot_vram():
    data = load_json('profiling_results.json')
    # Filter to successful runs only
    ok = [r for r in data if r['status'] == 'Success']

    labels = [f"{r['method']}\n@{r['budget']}" for r in ok]
    vram = [float(r['prefill_vram_gb']) for r in ok]
    colors = [COLORS.get(r['method'], '#666') for r in ok]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), vram, color=colors, edgecolor='white', linewidth=0.8)

    # Base model line
    ax.axhline(y=2.875, color='#888', linestyle='--', linewidth=1, label='Base Model (2.88 GB)')
    # 8 GB GPU line
    ax.axhline(y=8.0, color='#e74c3c', linestyle=':', linewidth=1.2, label='8 GB GPU Limit')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, ha='center')
    ax.set_ylabel('Peak VRAM (GB)')
    ax.set_title('Qwen2-1.5B-Instruct · Peak VRAM at 32k Context')
    ax.legend(loc='upper right', fontsize=9)

    # Value labels on bars
    for bar, v in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    path = os.path.join(FIG_DIR, 'qwen2_vram_profiling.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Decode Throughput Bar Chart
# ════════════════════════════════════════════════════════════════════════════
def plot_throughput():
    data = load_json('profiling_results.json')
    ok = [r for r in data if r['status'] == 'Success']

    labels = [f"{r['method']}\n@{r['budget']}" for r in ok]
    decode = [float(r['decode_tok_sec']) for r in ok]
    prefill = [float(r['prefill_tok_sec']) for r in ok]
    colors = [COLORS.get(r['method'], '#666') for r in ok]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Prefill throughput
    bars1 = ax1.bar(range(len(labels)), prefill, color=colors, edgecolor='white', linewidth=0.8)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8, ha='center')
    ax1.set_ylabel('Prefill Throughput (tok/s)')
    ax1.set_title('Prefill Speed')
    for bar, v in zip(bars1, prefill):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Decode throughput
    bars2 = ax2.bar(range(len(labels)), decode, color=colors, edgecolor='white', linewidth=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, ha='center')
    ax2.set_ylabel('Decode Throughput (tok/s)')
    ax2.set_title('Decode Speed')
    for bar, v in zip(bars2, decode):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.suptitle('Qwen2-1.5B-Instruct · Throughput at 32k Context', fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, 'qwen2_throughput_profiling.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 5. Combined: VRAM vs Decode Throughput Scatter (budget-labeled)
# ════════════════════════════════════════════════════════════════════════════
def plot_efficiency_tradeoff():
    data = load_json('profiling_results.json')
    ok = [r for r in data if r['status'] == 'Success']

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in ok:
        vram = float(r['prefill_vram_gb'])
        decode = float(r['decode_tok_sec'])
        method = r['method']
        budget = r['budget']
        ax.scatter(vram, decode,
                   color=COLORS.get(method, '#666'),
                   marker=MARKERS.get(method, 'o'),
                   s=120, zorder=5, edgecolors='white', linewidth=0.8)
        ax.annotate(f'{budget}', (vram, decode),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=8, color=COLORS.get(method, '#666'))

    # Legend with method names only (unique)
    seen = set()
    handles = []
    for r in ok:
        m = r['method']
        if m not in seen:
            seen.add(m)
            h = ax.scatter([], [],
                           color=COLORS.get(m, '#666'),
                           marker=MARKERS.get(m, 'o'),
                           s=80, label=m)
            handles.append(h)

    ax.set_xlabel('Peak VRAM (GB)')
    ax.set_ylabel('Decode Throughput (tok/s)')
    ax.set_title('Qwen2-1.5B · Efficiency Tradeoff: VRAM vs Decode Speed (32k Context)')
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)

    # 8 GB line
    ax.axvline(x=8.0, color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.7, label='8 GB Limit')

    path = os.path.join(FIG_DIR, 'qwen2_efficiency_tradeoff.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == '__main__':
    print("Generating Qwen2 paper plots...")
    plot_ppl()
    plot_qa()
    plot_vram()
    plot_throughput()
    plot_efficiency_tradeoff()
    print("\nAll plots generated successfully!")
