# EntropyKV: Attention-Free Key-Vector Entropy Eviction

This repository implements and evaluates **EntropyKV**, an attention-free KV cache eviction strategy that leverages the structural properties of key vectors (such as L2 norm, variance, and Shannon entropy) across the head dimension to predict token importance.

Unlike traditional eviction policies (such as H2O or SnapKV) which require materializing and reading attention matrices—creating a system-algorithm mismatch with fused attention kernels like FlashAttention or PagedAttention—**EntropyKV is 100% attention-free**, enabling direct compatibility with high-performance, fused inference engines.

---

## 📂 Repository Structure

- **`src/`**: Core library code.
  - [`src/cache/entropy_cache.py`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/src/cache/entropy_cache.py): The main `EntropyCache` implementation extending Hugging Face's `DynamicCache`.
  - [`src/baselines/`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/src/baselines): Re-implementations of comparative KV cache eviction baselines:
    - `streaming.py` (StreamingLLM)
    - `h2o.py` (Heavy Hitter Oracle)
    - `snapkv.py` (SnapKV)
    - `random_eviction.py` (Random Eviction baseline)
  - [`src/eval/`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/src/eval): Evaluation modules:
    - `perplexity.py`: Sliding-window WikiText-2 perplexity evaluation (supporting token-by-token autoregressive decoding mode and chunk/prefill mode).
    - `longbench.py`: Downstream long-context QA harness.
    - `niah.py`: Needle-in-a-Haystack (NIAH) retrieval test.
- **`analysis/`**: Hypotheses testing and Phase 0 correlation validation.
  - [`analysis/instrument_forward_pass.py`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/analysis/instrument_forward_pass.py): Extracts raw key-states and GQA attention distributions.
  - [`analysis/correlation_analysis.py`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/analysis/correlation_analysis.py): Performs statistical analysis (Spearman Rank Correlation) and generates plots.
- **`experiments/`**: Execution wrappers for sweep automation.
  - [`experiments/run_full_sweep.py`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/experiments/run_full_sweep.py): Runs WikiText-2 PPL sweeps and downstream QA sweeps.
  - [`experiments/run_downstream.py`](file:///Users/mynimbus/Attention-Free-Key-Vector-Entropy-Eviction/experiments/run_downstream.py): Runs QA and NIAH sweeps.
- **`configs/`**: JSON configuration parameters for baseline and proposed methods.
- **`paper/`**: LaTeX draft template (`main.tex`) for academic publication.

---

## 💻 Windows + CUDA Setup & Installation

Follow these steps to set up this repository on your Windows laptop with NVIDIA GPU support:

### 1. Prerequisites
- **Python 3.10+**: Ensure Python is added to your system `PATH`.
- **Git**: Installed and configured to pull/push from GitHub.
- **CUDA Toolkit**: Match the PyTorch CUDA installation (typically CUDA 12.1 or 12.4).

### 2. Virtual Environment Setup
Open PowerShell or Command Prompt in the repository folder:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# In PowerShell:
.\venv\Scripts\Activate.ps1
# In CMD:
.\venv\Scripts\activate.bat
```

### 3. Install PyTorch with CUDA Support
To leverage your GPU, install the CUDA-enabled version of PyTorch first:
```bash
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify GPU/CUDA Activation
Verify that PyTorch can successfully communicate with your GPU:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 📈 Running Evaluations & Sweeps

All experiment execution scripts have been made fully cross-platform (using `sys.executable` internally instead of hardcoded `python3` paths) to run seamlessly on both Unix-like and Windows shells.

### 🧪 Quick Verification Run
To verify the entire environment and execution pipelines on a subset of data (fast CPU/GPU-friendly sanity check):
```bash
python experiments/run_full_sweep.py --quick
```
This runs a micro-sweep of perplexity (using 512 evaluation tokens) and downstream tasks (2 samples) across a subset of budget ratios, producing verification plots in `analysis/figures/`.

### 🔬 Full Benchmark Suite
To run the full evaluation suite as described in our research draft:
```bash
python experiments/run_full_sweep.py
```
This performs a thorough sliding-window WikiText-2 perplexity sweep, downstream long-context QA benchmark, and Needle-in-a-Haystack (NIAH) retrieval maps across all budgets and baseline algorithms, saving results and figures under:
- `analysis/extracted_data/` (JSON and compressed `.npz` arrays)
- `analysis/figures/` (Matplotlib/Seaborn Heatmaps and Line Plots)

---

## 📊 Phase 0: Hypothesis Validation (Optional Re-Run)
If you wish to re-extract key states and re-run the correlation analysis:
1. **Instrument Forward Pass**:
   ```bash
   python analysis/instrument_forward_pass.py
   ```
2. **Compute Statistics & Heatmaps**:
   ```bash
   python analysis/correlation_analysis.py
   ```
This will regenerate the Spearman correlation Heatmaps mapping layer-wise attention-sink preferences, demonstrating the negative correlation ($r \approx -0.72$) between key-vector L2 norms and future attention accumulation.