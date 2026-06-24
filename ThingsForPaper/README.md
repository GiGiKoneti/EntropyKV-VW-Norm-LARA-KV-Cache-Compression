# ThingsForPaper — Complete Paper Resource Collection

> **Project**: EntropyKV — Value-Weighted KV Cache Compression with Layer-Adaptive Recency Allocation
> **Authors**: [Your Name]
> **Date**: June 2026
> **Status**: All experiments complete, ready for paper writing

---

## 📁 Folder Structure

```
ThingsForPaper/
│
├── 01_Figures/                          ← All publication-ready plots
│   ├── TinyLlama/                       ← TinyLlama-1.1B evaluation plots
│   │   ├── ppl_vs_budget.png            ← PPL vs KV Cache Budget curve
│   │   ├── qa_performance_vs_budget.png ← QA F1 vs Budget curve
│   │   ├── niah_heatmap_*.png           ← NIAH retrieval heatmaps (5 methods)
│   │   └── niah_heatmap_tinyllama_*.png ← Full-cache baseline heatmap
│   ├── Qwen2/                           ← Qwen2-1.5B evaluation plots
│   │   ├── qwen2_ppl_vs_budget.png      ← PPL vs Budget (32k context)
│   │   ├── qwen2_qa_vs_budget.png       ← QA F1 vs Budget (32k context)
│   │   ├── qwen2_vram_profiling.png     ← VRAM bar chart
│   │   ├── qwen2_throughput_profiling.png ← Prefill + Decode throughput bars
│   │   ├── qwen2_efficiency_tradeoff.png ← VRAM vs Decode speed scatter
│   │   └── niah_heatmap_qwen2_*.png     ← NIAH heatmaps (6 variants)
│   └── Correlation_Analysis/            ← Key norm ↔ attention correlation
│       ├── correlation_heatmap.png      ← Per-layer correlation matrix
│       ├── correlation_by_layer.png     ← Layer-wise correlation bars
│       └── scatter_score_vs_attention.png ← Score vs attention scatter
│
├── 02_Raw_Data/                         ← Raw numerical results (JSON, NPZ)
│   ├── NIAH_Matrices/
│   │   ├── TinyLlama/                   ← 5 .npz files (accuracy grids)
│   │   └── Qwen2/                       ← 6 .npz files (accuracy grids)
│   ├── Perplexity/
│   │   └── ppl_sweeps_results.json      ← 6 methods × 5 budgets
│   ├── QA_Downstream/
│   │   └── qa_sweeps_results.json       ← 4 methods × 4 budgets
│   ├── Profiling/
│   │   └── profiling_results.json       ← VRAM + throughput (32k context)
│   └── Attention_Samples/
│       └── sample_0..4.npz             ← Raw attention weight snapshots
│
├── 03_Source_Code/                      ← Core implementation
│   ├── cache/
│   │   ├── entropy_cache.py            ← EntropyKV cache (VW-Norm, LARA)
│   │   └── utils.py                    ← Value-weighted norm computation
│   ├── baselines/
│   │   ├── streaming.py                ← StreamingLLM implementation
│   │   ├── h2o.py                      ← H2O implementation
│   │   ├── snapkv.py                   ← SnapKV implementation
│   │   └── random_eviction.py          ← Random eviction baseline
│   └── eval/
│       ├── niah.py                     ← Needle-in-a-Haystack evaluation
│       ├── harness.py                  ← Perplexity sweep harness
│       ├── longbench.py                ← LongBench QA evaluation
│       └── perplexity.py               ← Sliding-window perplexity
│
├── 04_Experiment_Scripts/               ← Orchestration scripts
│   ├── run_full_sweep.py               ← Master sweep (all methods)
│   ├── run_downstream.py               ← Downstream PPL + QA sweep
│   ├── run_ablation.py                 ← Ablation study runner
│   ├── run_baseline.py                 ← Single baseline runner
│   ├── run_entropykv.py                ← EntropyKV standalone runner
│   └── profile_efficiency.py           ← VRAM/throughput profiler
│
├── 05_Analysis_Scripts/                 ← Data analysis & plotting
│   ├── generate_qwen2_plots.py         ← Qwen2 plot generator
│   ├── correlation_analysis.py         ← Key norm ↔ attention analysis
│   └── instrument_forward_pass.py      ← Attention weight extraction
│
├── 06_Research_Notes/                   ← Journey documentation
│   ├── paper_results_summary.md        ← ★ CENTRAL DOCUMENT: All numbers
│   ├── walkthrough.md                  ← Chronological experiment log
│   ├── implementation_plan.md          ← Original implementation plan
│   ├── proof_of_hypothesis.md          ← Early hypothesis validation
│   ├── kv_cache_eval_report.md         ← Initial evaluation report
│   ├── EntropyKV_Implementation_Plan.md ← Detailed design document
│   ├── Layer Adaptive KV Cache Research.md ← Background research
│   └── paper_draft.md                  ← ★ PAPER DRAFT (see below)
│
├── 07_Paper_LaTeX/                      ← LaTeX source files
│   ├── main.tex                        ← Paper template
│   ├── neurips_2024.sty                ← NeurIPS style
│   └── references.bib                  ← Bibliography
│
├── 08_Experiment_Logs/                  ← Raw conversation/execution logs
│   └── transcript.jsonl                ← Full agent conversation log
│
└── README.md                           ← This file
```

---

## 🏁 Quick Start Guide for Paper Writing

### Step 1: Read the Central Results Summary
Start with `06_Research_Notes/paper_results_summary.md` — it has every number, table, and figure you need.

### Step 2: Review the Paper Draft
`06_Research_Notes/paper_draft.md` contains a complete first draft of the paper with all sections.

### Step 3: Grab Figures
All publication-ready PNG figures are in `01_Figures/`, organized by model.

### Step 4: Verify Numbers
Raw JSON data in `02_Raw_Data/` can be used to regenerate any table or plot.

### Step 5: Regenerate Plots
Run `05_Analysis_Scripts/generate_qwen2_plots.py` to regenerate all Qwen2 plots.

---

## 📊 Key Results at a Glance

| Metric | Our Method (VW+LARA) | Best Baseline | Improvement |
|---|---|---|---|
| QA F1 @ Budget 0.5 (TinyLlama) | **0.1052** | H2O: 0.0500 | 2.1× |
| QA F1 @ Budget 0.5 (Qwen2) | **0.1333** | H2O: 0.0961 | 1.4× |
| NIAH Semantic Accuracy (Qwen2) | **100%** | SnapKV: 25% | 4× |
| VRAM Savings @ Budget 0.3 | **43%** (11.6→6.6 GB) | Same | — |
| Decode Speedup @ Budget 0.3 | **32×** (0.3→9.6 tok/s) | Same | — |
| Attention Weights Required? | **No** (SDPA-compatible) | H2O/SnapKV: Yes (OOM) | ∞ |

---

## 🧪 Experimental Configuration

| Parameter | TinyLlama | Qwen2 |
|---|---|---|
| Model | TinyLlama-1.1B-Chat-v1.0 | Qwen2-1.5B-Instruct |
| Context Length | 2,048 | 32,000 |
| GPU | RTX 5060 Laptop (8 GB) | RTX 5060 Laptop (8 GB) |
| Chunk Size | 512 | 2,048 |
| Budget Ratios | 1.0, 0.9, 0.7, 0.5, 0.3 | 1.0, 0.9, 0.7, 0.5, 0.3 |
| Methods | 6 (Full, Stream, H2O, Snap, Random, Ours) | 6 (same) |
