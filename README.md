# EntropyKV: Value-Weighted KV Cache Compression with Layer-Adaptive Recency Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**EntropyKV** is an online, **100% attention-free** key-value (KV) cache compression framework. Traditional KV cache eviction methods (e.g., H2O, SnapKV) require materializing and reading full attention matrices to determine token importance. This requirement creates an architectural mismatch with high-performance fused kernels like **FlashAttention** or **PagedAttention**, resulting in high memory overhead, slowdowns, and Out-of-Memory (OOM) errors during long-context prefilling.

In contrast, **EntropyKV** operates directly on the key and value states, completely bypassing the need to materialize attention weights. By leveraging key-vector entropy, variance, and value-vector norms, it determines token significance on-the-fly, achieving extreme compression with minimal quality degradation.

---

## 🚀 Key Highlights & Contributions

* 🌟 **100% Attention-Free Eviction**: Zero dependency on attention matrices. Fully compatible with native PyTorch SDPA, FlashAttention, and hardware-fused attention engines.
* 🧠 **Value-Weighted Key Norm (VW-Norm)**: Combines key-vector variance (entropy proxies) with value-vector magnitudes, selectively protecting highly active "outlier" and high-entropy states.
* 📈 **Layer-Adaptive Recency Allocation (LARA)**: Uses a U-shaped allocation function across network depth, reserving larger recency windows for attention-sink layers (bottom) and semantic consolidation layers (top).
* ⚙️ **Hardware-Safe Chunked Prefill**: Implements memory-safe context processing, allowing 32k context lengths on consumer GPUs (e.g., NVIDIA RTX 5060 Laptop 8GB) without OOM.

---

## 📊 Summary of Downstream Results

### 1. TinyLlama-1.1B-Chat (2k Context Limit)
* **Downstream QA F1 Score**: At 50% KV cache budget, our method preserves **0.1052 F1** (outperforming StreamingLLM by **3.9×**, and doubling both H2O and L2-Norm). At budget 0.7, it retains **92%** of the full cache baseline.
* **Needle-in-a-Haystack (NIAH)**: Achieves **37.5% average retrieval accuracy** at budget 0.5, uniquely preserving early-context recall where standard recency methods (StreamingLLM) suffer 0% retrieval.

### 2. Qwen2-1.5B-Instruct (32k Context Limit)
* **Downstream QA F1 Score**: At budget 0.5, our method retains **0.1333 F1** — **6×** StreamingLLM, **1.4×** H2O, and **2.6×** L2-Norm.
* **Efficiency & Scalability**: Cuts peak VRAM by **43%** (saving 5.0 GB) and delivers **32× decoding speedup** under tight budgets, while H2O and SnapKV suffer from Out-of-Memory (OOM) failures due to prefill activation spikes.

---

## 📁 Repository Structure

```
.
├── src/
│   ├── cache/
│   │   ├── entropy_cache.py       # Main EntropyKV cache implementation (VW-Norm, LARA)
│   │   └── utils.py               # Value-weighted norm computation routines
│   ├── baselines/
│   │   ├── streaming.py           # StreamingLLM baseline
│   │   ├── h2o.py                 # Heavy Hitter Oracle (H2O) baseline
│   │   ├── snapkv.py              # SnapKV prefill baseline
│   │   └── random_eviction.py     # Uniform random eviction baseline
│   └── eval/
│       ├── perplexity.py          # WikiText-2 sliding-window perplexity harness
│       ├── longbench.py           # LongBench downstream QA benchmark
│       └── niah.py                # Needle-in-a-Haystack (NIAH) context evaluation
│
├── experiments/
│   ├── run_full_sweep.py          # Master sweep pipeline (PPL + QA + NIAH)
│   ├── run_downstream.py          # Dedicated downstream QA and PPL sweeps
│   └── profile_efficiency.py      # VRAM and throughput profiling script
│
├── analysis/
│   ├── correlation_analysis.py    # Per-layer Spearman rank correlation analysis
│   └── instrument_forward_pass.py # Key-state and attention distribution extraction
│
├── ThingsForPaper/                # ★ Collection of all paper drafts, LaTeX source,
│                                  #   figures, and raw results (see section below)
├── configs/                       # Hyperparameter configuration JSONs
└── paper/                         # Original LaTeX draft templates
```

---

## 📦 Windows + CUDA Setup & Installation

### 1. Prerequisites
* **Python 3.10+** (Ensure it is in your system `PATH`).
* **Git** (For cloning and pushing).
* **NVIDIA CUDA Toolkit** (Compatible with PyTorch, e.g., CUDA 12.1 or 12.4).

### 2. Set Up Virtual Environment
Open PowerShell or Command Prompt in the repository root:
```powershell
# Create the virtual environment
python -m venv venv

# Activate the virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate the virtual environment (CMD)
.\venv\Scripts\activate.bat
```

### 3. Install PyTorch with CUDA Support
Ensure PyTorch is installed with GPU acceleration:
```bash
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify CUDA Support
```bash
python -c "import torch; print('CUDA active:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 📈 Running Sweeps & Profiling

### 🧪 Sanity Check / Quick Sweep
Verify the installation and pipeline using a subset of data (fast CPU/GPU check):
```bash
python experiments/run_full_sweep.py --quick
```

### 🔬 Full Benchmark Suite
Execute the entire evaluation suite (PPL, QA, NIAH) across multiple budget ratios:
```bash
python experiments/run_full_sweep.py
```

### ⚡ VRAM & Throughput Profiling
To measure peak VRAM and generation speed at 32k context length:
```bash
python experiments/profile_efficiency.py
```

---

## 📝 Writing the Paper? Go to `ThingsForPaper/`!

For end-to-end academic paper drafting, we compiled all resources into the **`ThingsForPaper/`** folder:
* 📊 **`01_Figures/`**: 32 publication-ready plots (PPL curves, QA histograms, 13 NIAH heatmaps).
* 🔢 **`02_Raw_Data/`**: Raw JSON and NPZ files for plotting or statistical validation.
* ✍️ **`06_Research_Notes/`**: Contains a fully drafted paper draft (`paper_draft.md`), a pre-written LaTeX caption document (`figure_captions.md`), and a quick-reference cheat sheet (`cheat_sheet.md`) with all key numbers.
* 📄 **`07_Paper_LaTeX/`**: Complete `main.tex` and bibliography `references.bib` ready for compilation.

Refer to [ThingsForPaper/README.md](file:///c:/Users/gigik/OneDrive/Desktop/Attention-Free-Key-Vector-Entropy-Eviction/ThingsForPaper/README.md) for more details.