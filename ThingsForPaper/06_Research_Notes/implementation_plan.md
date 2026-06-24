# Implementation Plan: Qwen2 Long-Context Baseline Sweeps, Downstream Evaluations, and Profiling

This plan outlines the next steps to complete the evaluation of **Qwen2-1.5B-Instruct** up to its **32,000 native context limit** across:
1. Baseline NIAH comparative sweeps (StreamingLLM, H2O, SnapKV).
2. Downstream perplexity and QA sweeps.
3. VRAM and speed/throughput profiling.

---

## User Review Required

> [!WARNING]
> **H2O and SnapKV Qwen2 Attention Compatibility:**
> H2O and SnapKV require extracting attention weights during the prefill phase, forcing us to load the model with `attn_implementation="eager"`. Because Hugging Face's Qwen2 eager attention implementation has known numerical instability (NaN/repetitive loop generation bugs in FP16), we may need to debug or apply a mathematical patch to Qwen2's eager attention class if these baseline sweeps collapse.

> [!IMPORTANT]
> **Downstream Evaluation Scale & Latency Constraints:**
> Running full-context QA sweeps (e.g. LongBench QA) up to 32k context on a laptop GPU (RTX 5060) can take several hours if run on the entire dataset. We propose:
> * Running the perplexity sweep on WikiText-2 (2048 validation tokens) as done previously.
> * Running the downstream QA sweep on a **5-sample representative subset** (matching the TinyLlama evaluation) to keep execution time within a reasonable window (~15-30 minutes) while still capturing clear F1 score curves.

---

## Open Questions

> [!IMPORTANT]
> **1. LongBench QA Scale:** Do you approve running the downstream QA sweeps on the 5-sample subset to prevent multi-hour runs, or should we target a larger sample size?
> **2. SnapKV Import Verification:** If SnapKV continues to experience silent execution termination, do you approve replacing/simplifying the SnapKV import/implementation loop?

---

## Proposed Changes

### Downstream Code Changes

We need to ensure that the position ID bug fix we implemented for chunked prefill in NIAH is also propagated to the downstream evaluation harnesses.

#### [MODIFY] [longbench.py](file:///C:/Users/gigik/OneDrive/Desktop/Attention-Free-Key-Vector-Entropy-Eviction/src/eval/longbench.py)
* Inspect and modify the chunked prefill loop inside `longbench.py` to construct and pass correct absolute `position_ids` based on the sequence offset `i`.

#### [MODIFY] [harness.py](file:///C:/Users/gigik/OneDrive/Desktop/Attention-Free-Key-Vector-Entropy-Eviction/src/eval/harness.py)
* Inspect and modify the prefill loop inside `harness.py` (perplexity sweep) to verify that absolute `position_ids` are explicitly passed if chunking is enabled.

---

## Verification Plan

### 1. Comparative NIAH Sweeps (Budget = 0.5, Context up to 32k)
We will execute the remaining baseline sweeps on Qwen2:
```bash
# StreamingLLM sweep
.\venv\Scripts\python.exe -m src.eval.niah --model Qwen/Qwen2-1.5B-Instruct --method streaming --budget_ratio 0.5 --chunk_size 2048

# H2O sweep (pending eager attention verification)
.\venv\Scripts\python.exe -m src.eval.niah --model Qwen/Qwen2-1.5B-Instruct --method h2o --budget_ratio 0.5 --chunk_size 2048

# SnapKV sweep (pending crash diagnostics)
.\venv\Scripts\python.exe -m src.eval.niah --model Qwen/Qwen2-1.5B-Instruct --method snapkv --budget_ratio 0.5 --chunk_size 2048
```

### 2. Downstream QA Sweeps (Budgets = [1.0, 0.7, 0.5, 0.3])
Verify Qwen2 performance on long document QA using the updated harness:
```bash
.\venv\Scripts\python.exe -m src.eval.longbench --model Qwen/Qwen2-1.5B-Instruct --methods full,streaming,h2o,entropykv --budgets 1.0,0.7,0.5,0.3 --chunk_size 2048
```

### 3. Perplexity Sweeps (Budgets = [1.0, 0.9, 0.7, 0.5, 0.3])
Verify online perplexity stability:
```bash
.\venv\Scripts\python.exe -m src.eval.harness --model Qwen/Qwen2-1.5B-Instruct --methods full,streaming,h2o,entropykv --budgets 1.0,0.9,0.7,0.5,0.3 --chunk_size 2048
```
