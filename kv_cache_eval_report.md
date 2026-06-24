# KV-Cache Compression Evaluation Report
**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0  
**Dataset:** WikiText-2 (validation split, 2048 tokens)  
**Methods evaluated:** Full Cache (baseline), StreamingLLM, H2O, EntropyKV (L2-Norm, VW-Norm + LARA)  
**Budget ratios tested:** 1.0, 0.9, 0.7, 0.5, 0.3  

---

## 1. Perplexity Results

### 1.1 Full Cache (Ground Truth)

| Budget | Tokens | PPL | VRAM (GB) | Throughput (tok/s) |
|--------|--------|-----|-----------|-------------------|
| 1.00 | 1024 | **6.2258** | 2.0865 | 30.88 |

### 1.2 StreamingLLM

| Budget | Tokens | PPL | Δ vs Full (×) | VRAM (GB) | Throughput (tok/s) |
|--------|--------|-----|---------------|-----------|-------------------|
| 0.90 | 921 | 9.5811 | 1.54× | 2.0836 | 29.69 |
| 0.70 | 716 | 25.1740 | 4.05× | 2.0777 | 29.70 |
| 0.50 | 512 | 85.6797 | 13.8× | 2.0719 | 28.72 |
| 0.30 | 307 | 202.9577 | 32.6× | 2.0660 | 28.40 |

### 1.3 H2O (Sinks=4, Recency=32)

| Budget | Tokens | PPL | Δ vs Full (×) | VRAM (GB) | Throughput (tok/s) |
|--------|--------|-----|---------------|-----------|-------------------|
| 0.90 | 921 | 9.4818 | 1.52× | 2.0862 | 26.80 |
| 0.70 | 716 | 23.2933 | 3.74× | 2.0797 | 25.48 |
| 0.50 | 512 | 56.2398 | 9.03× | 2.0732 | 22.14 |
| 0.30 | 307 | 102.8538 | 16.5× | 2.0668 | 22.00 |

### 1.4 Key Observation: H2O vs STREAMING

H2O consistently outperforms StreamingLLM on PPL at every budget, with the gap widening at aggressive compression:

| Budget | STREAMING PPL | H2O PPL | H2O improvement |
|--------|--------------|---------|----------------|
| 0.90 | 9.58 | 9.48 | ~1% |
| 0.70 | 25.17 | 23.29 | 7.5% |
| 0.50 | 85.68 | 56.24 | 34.4% |
| 0.30 | 202.96 | 102.85 | 49.3% |

This is the **only clean, publishable finding** from this sweep.

---

## 2. Throughput Analysis

H2O's attention-score accumulation introduces measurable overhead vs StreamingLLM at equivalent budgets:

| Budget | STREAMING (tok/s) | H2O (tok/s) | H2O slowdown |
|--------|------------------|-------------|--------------|
| 0.90 | 29.69 | 26.80 | −9.7% |
| 0.70 | 29.70 | 25.48 | −14.2% |
| 0.50 | 28.72 | 22.14 | −22.9% |
| 0.30 | 28.40 | 22.00 | −22.5% |

H2O pays a significant latency cost for its quality advantage. At budget 0.3, H2O is ~29% slower than STREAMING. This quality/latency tradeoff must be explicitly reported in any comparison.

---

## 3. Downstream QA Results

### 3.1 Summary

**All EM and F1 scores are 0 across every method and every budget.** This eval is non-functional.

| Method | Budget | Avg EM | Avg F1 |
|--------|--------|--------|--------|
| STREAMING | 1.0 / 0.7 / 0.5 / 0.3 | 0.0000 | 0.0000 |
| H2O | 1.0 / 0.7 / 0.5 / 0.3 | 0.0000 | 0.0000 |
| EntropyKV (L2-Norm) | 1.0 | 0.0000 | **0.0267** |
| EntropyKV (L2-Norm) | 0.7 / 0.5 / 0.3 | 0.0000 | 0.0000 |

### 3.2 Root Causes

1. **The fallback QA task is degenerate.** The `allenai/qasper` dataset failed to load (dataset scripts deprecated), triggering a synthetic fallback. The fallback repeats the *same* question 5 times — `"How many times weaker is gravity on this planet according to the result?"` — with incrementing reference answers (42, 59, 76, 93, 110 times). No WikiText-2 passage contains this answer; the task is semantically disconnected from the context.

2. **Model outputs are degenerate loops.** Predictions like `"andSidenote andSidenote andSidenote"`, `"oidaceoidaceoidace"`, and `"the the the the the the"` indicate the generation is producing repetitive garbage regardless of KV method or budget, which rules out eviction as the sole cause.

3. **EntropyKV's marginal F1=0.0267** at budget 1.0 is the only non-zero score. It is noise over 5 samples, but it is a weak signal that L2-norm importance scoring preserves marginally more relevant tokens. Not actionable until the harness is fixed.

---

## 4. NIAH (Needle-in-a-Haystack) Results

### 4.1 Summary

**Success rate is 0.0 at every context length, depth, and method.** No needle was retrieved by any configuration.

| Method | Context range | Depths tested | Success rate |
|--------|--------------|---------------|-------------|
| STREAMING | 1024–8192 | 0.0–1.0 | 0/24 = 0.0% |
| H2O | 1024–2048 | 0.0–1.0 | 0/12 = 0.0% |
| H2O | 4096–8192 | all | OOM — not run |

### 4.2 Root Causes

**Primary cause — model context limit exceeded.** TinyLlama-1.1B has a trained context window of 2048 tokens. The NIAH sweep tests contexts up to 8192 tokens, producing the warning:

```
Token indices sequence length is longer than the specified maximum sequence length (297365 > 2048)
```

Beyond 2048 tokens, TinyLlama's RoPE positional embeddings are extrapolating into an undefined regime. The degenerate looping outputs (`"lolololololololo"`, `"ententent"`, `"gamisteriatepperialiate"`) are classic RoPE-overflow artifacts — **this is model failure, not KV eviction failure.** The NIAH sweep cannot differentiate between compression methods under these conditions.

**Secondary cause — likely broken needle insertion.** Even at context=1024 (within TinyLlama's trained window), success is 0.0. This strongly suggests the needle string is either not being inserted into the context correctly, or the prompt format does not match TinyLlama's chat template, causing the model to never attend to the needle region.

### 4.3 H2O OOM at Context ≥ 4096

H2O throws `cudaErrorMemoryAllocation` at context=4096+ despite a budget-0.5 eviction policy. STREAMING at the same budget runs without OOM. Likely cause: H2O materializes the full attention score matrix before performing eviction — this is O(n²) in memory and negates the KV budget savings during the prefill phase. The recency buffer initialization may also be allocating extra tensors. This is a real implementation bug that should be diagnosed with `torch.cuda.memory_summary()` before and after the prefill step.

---

## 5. Critical Issues — Prioritised

### P0 — Blocking (results are invalid until resolved)

- **Fix NIAH needle insertion.** Before running generation, assert `needle_string in constructed_context` and log a sample prompt for manual inspection. Until verified, all NIAH numbers are meaningless.
- **Fix QA harness.** Replace the synthetic gravity-question loop with a real long-context QA benchmark: SCROLLS, LongBench-QA, or a properly constructed WikiText retrieval task where the answer token is provably present in the context.
- **Diagnose SnapKV truncation.** The log cuts off immediately after `>>> Running SNAPKV at budget 0.9...` with no output, error, or result. Check for silent import errors, missing dependencies, or OOM. SnapKV (local window + k-means clustering) is the most semantically-aware baseline and its absence makes the comparison incomplete.

### P1 — Important (limits validity of perplexity results)

- **Switch base model.** TinyLlama-1.1B's 2048-token ceiling invalidates any experiment past that context. Replace with a model that natively supports ≥ 8192 tokens: Mistral-7B-v0.3, LLaMA-3-8B-Instruct, or Qwen2-7B-Instruct. For a compute-constrained environment, Mistral-7B with 4-bit quantization fits in the same VRAM envelope.
- **Fix H2O OOM.** Add `torch.cuda.empty_cache()` between runs, profile peak memory during prefill, and verify the eviction is happening before (not after) the full attention matrix is materialized.
- **Add budget=1.0 baseline for every method.** Currently only STREAMING has a full-budget run (and it matches FULL at 6.23 PPL via the identity case). H2O and EntropyKV at budget=1.0 should reproduce FULL PPL — verifying this is a basic sanity check for each implementation.

### P2 — Polish (improves result quality)

- **Report mean ± std over multiple seeds.** A single 2048-token WikiText-2 pass is high-variance. Run 3–5 random contiguous segments and report variance. A single PPL number has no error bar.
- **Plot PPL vs budget and throughput vs budget jointly.** The quality-efficiency tradeoff is the core story; surfacing it as a curve rather than a table makes the H2O vs STREAMING tradeoff immediately visible.
- **Report VRAM savings explicitly.** VRAM differences between methods are currently small (~15 MB across the full budget range), likely because 2048 tokens is too short for the KV cache to be the dominant memory consumer. At longer contexts with a larger model, this gap will be the primary motivation for compression.

---

## 6. Conclusions

### What can be concluded from this sweep

1. **H2O outperforms StreamingLLM on perplexity at every budget**, with the advantage growing at aggressive compression (49% better PPL at budget 0.3). Attention-score-guided eviction preserves language model quality better than pure recency.
2. **H2O is 10–29% slower than StreamingLLM** at equivalent budgets due to attention score accumulation overhead.
3. **PPL degrades nonlinearly below budget 0.7** for both methods. The jump from 0.7 → 0.5 is steep (STREAMING: 25 → 86; H2O: 23 → 56). Budgets below 0.5 are operationally non-viable for coherent generation.

### What cannot be concluded

- Nothing about retrieval quality (NIAH eval is broken).
- Nothing about downstream task performance (QA eval is broken).
- Nothing about behavior at context lengths > 2048 tokens (model ceiling).
- Nothing about SnapKV (missing results).
- Nothing about EntropyKV beyond the weakest budget signal (5 samples, non-functional harness).

---

## 7. Recommended Next Steps

```
Priority  Action
P0        Fix NIAH insertion + verify with assert before generation
P0        Replace synthetic QA with LongBench / SCROLLS
P0        Debug SnapKV — import error or OOM?
P1        Swap model → Mistral-7B or LLaMA-3-8B (native long context)
P1        Profile H2O prefill memory → fix OOM at context 4096+
P1        Add budget=1.0 identity check for H2O and EntropyKV
P2        3-seed PPL averaging with std reported
P2        PPL vs budget + throughput vs budget plots
P2        VRAM analysis at longer contexts (2048→32k)
```

---

*Generated from evaluation log analysis. All PPL figures taken directly from reported outputs. QA and NIAH failure analysis based on zero-score outputs and logged warnings.*
