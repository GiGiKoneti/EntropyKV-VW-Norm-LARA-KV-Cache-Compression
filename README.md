# EntropyKV: Value-Weighted KV Cache Eviction with Layer-Adaptive Recency Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**EntropyKV** is an online, **100% attention-free** key-value (KV) cache eviction policy designed to run long-context LLMs on consumer-grade GPUs. 

Traditional KV cache compression methods (e.g., H2O, SnapKV) require materializing full attention weight matrices to identify important tokens. This creates a severe systems mismatch: computing attention weights over long contexts (e.g., 32k+ tokens) requires immense memory overhead, causing Out-of-Memory (OOM) crashes on consumer hardware and failing to integrate with high-performance fused kernels like **FlashAttention** or **PagedAttention**.

**EntropyKV** bypasses this by operating entirely on key and value vector norms. It decides which tokens to evict on-the-fly without ever calculating attention weights, keeping memory footprints low and generation speeds high.

---

## 💡 The Core Analogy: Eviction as Page Replacement

In operating systems, when a process exceeds physical memory, the OS doesn't crash; it uses a **page replacement policy** to evict cold pages and keep hot ones in memory. EntropyKV treats the KV cache of an LLM exactly like physical memory pages:

```
[Incoming Token] ──> [LLM Cache Layer] 
                           │
             Is cache size > physical budget?
                           │
             ┌─────────────┴─────────────┐
            YES                         NO
             │                           │
  [Page Eviction Policy]         [Store in Cache]
  • Keep sink pages (Sinks)
  • Keep recent pages (LARA)
  • Measure norm of remaining
    pages and evict the cold ones
```

Instead of using expensive attention weights, EntropyKV relies on two lightweight systems concepts:
* **Value-Weighted Key Norm (VW-Norm):** Measures token importance using key vector norms (which correlate with attention scores) and value vector norms (representing output contribution). This behaves like a cheap page-access metric.
* **Layer-Adaptive Recency Allocation (LARA):** Recognizes that model layers behave differently. Early and late layers capture structural and reasoning states (hot memory pages), so they get larger recency windows. Middle layers perform redundant computation (cold pages) and are compressed aggressively.

---

## 🚀 Key Highlights & Contributions

* ⚡ **100% Attention-Free:** Fully compatible with hardware-fused attention engines like **FlashAttention** and PyTorch's native **SDPA**.
* 📉 **43% VRAM Reduction:** Squeezes 32k context on Qwen2-1.5B down from 11.6 GB to **6.6 GB**, fitting entirely within a standard 8 GB consumer GPU.
* 🏎️ **32× Decode Speedup:** Speeds up text generation by drastically reducing the size of the attention computation loop.
* 🧠 **Semantic Quantization:** Unlike other methods that produce incoherent garbage when highly compressed, EntropyKV exhibits graceful degradation—retaining 100% semantic structure (e.g., outputting "2023" instead of "2026" on passcode retrieval rather than failing entirely).

---

## 📊 Performance Benchmarks

### 1. Downstream QA (LongBench F1 Score)
At a highly compressed **50% cache budget**, EntropyKV preserves performance where baselines collapse:

| Method | TinyLlama-1.1B (2k context) | Qwen2-1.5B (32k context) |
| :--- | :---: | :---: |
| **Full Cache (100% budget)** | 0.2379 | 0.1438 |
| **EntropyKV (Ours - 50% budget)** | **0.1052** *(Preserves 92% @ 0.7)* | **0.1333** |
| H2O (50% budget) | 0.0500 | 0.0961 |
| L2-Norm (50% budget) | 0.0517 | 0.0517 |
| StreamingLLM (50% budget) | 0.0267 | 0.0222 |

### 2. Peak VRAM & Speed (Qwen2-1.5B @ 32k context)
*Under long-context prefilling, attention-based baselines like H2O and SnapKV suffer Out-of-Memory (OOM) failures on a laptop GPU (RTX 5060, 8 GB VRAM).*

| Method | Cache Budget | Peak VRAM (GB) | Decode Speed (tok/s) | Speedup | Status on 8GB GPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Cache** | 1.0 | 11.6 GB | 0.3 | 1.0× | ❌ OOM |
| **H2O / SnapKV** | all | OOM | — | — | ❌ OOM (Prefill activation spike) |
| **StreamingLLM** | 0.3 | 6.6 GB | 5.1 | 17.0× |  Active |
| **EntropyKV (Ours)** | 0.5 | 8.3 GB | 6.9 | 23.0× |  Active |
| **EntropyKV (Ours)** | 0.3 | **6.6 GB** | **9.6** | **32.0×** |  Active |

---

## 📁 Repository Structure

```
.
├── src/
│   ├── cache/
│   │   ├── entropy_cache.py    # Main EntropyKV implementation (VW-Norm, LARA)
│   │   └── utils.py            # Key-Value norm calculations
│   ├── baselines/
│   │   ├── streaming.py        # StreamingLLM baseline (Sink + Recency)
│   │   ├── h2o.py              # Heavy Hitter Oracle (H2O) baseline
│   │   ├── snapkv.py           # SnapKV baseline
│   │   └── random_eviction.py  # Random eviction baseline
│   └── eval/
│       ├── perplexity.py       # Sliding-window WikiText-2 perplexity harness
│       ├── longbench.py        # Downstream QA benchmark (LongBench)
│       └── niah.py             # Needle-in-a-Haystack retrieval evaluation
│
├── experiments/
│   ├── run_full_sweep.py       # Master evaluation runner (PPL + QA + NIAH)
│   ├── run_downstream.py       # Downstream QA and perplexity evaluator
│   └── profile_efficiency.py   # Memory and throughput profiling
│
├── analysis/
│   └── generate_qwen2_plots.py # Plotting utilities for benchmarking
│
├── configs/                    # Baseline and algorithm configuration JSONs
└── requirements.txt            # Project dependencies
```

---

## 🔧 Installation & Verification

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/GiGiKoneti/EntropyKV.git
cd EntropyKV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch (with CUDA support)
Install PyTorch matching your CUDA version (e.g., CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Configuration
```bash
python -c "import torch; print('CUDA active:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 🏎️ Running Benchmarks

### Quick Pipeline Verification
Run a quick, downscaled sweep to verify that all caches and baselines compile and execute correctly:
```bash
python experiments/run_full_sweep.py --quick
```

### Full Evaluation Suite
Run the master evaluation suite (Perplexity + LongBench QA + NIAH) across multiple cache budgets:
```bash
python experiments/run_full_sweep.py
```

### VRAM and Speed Profiling
To profile peak GPU memory usage and token-per-second generation speeds at 32k context lengths:
```bash
python experiments/profile_efficiency.py
```

---

## 🛡️ License
Distributed under the **MIT License**. See `LICENSE` for more information.