# EntropyKV: Implementation Plan
### Attention-Free Key-Vector Entropy Eviction for Efficient Long-Context LLM Inference

> **Document type:** Research implementation roadmap — text plan only, no code  
> **Author context:** Solo researcher, consumer GPU, targeting arXiv cs.CL / cs.LG  
> **Core claim to validate:** Key-vector entropy is a reliable, attention-free proxy for token importance in transformer KV caches

---

## Table of Contents

1. [Project Philosophy](#1-project-philosophy)
2. [Research Hypothesis (Formal)](#2-research-hypothesis-formal)
3. [Environment Setup](#3-environment-setup)
4. [Phase 0 — Hypothesis Validation (Before Writing Any Method)](#4-phase-0--hypothesis-validation)
5. [Phase 1 — Baseline Implementation](#5-phase-1--baseline-implementation)
6. [Phase 2 — EntropyKV Core Method](#6-phase-2--entropykv-core-method)
7. [Phase 3 — Evaluation Suite](#7-phase-3--evaluation-suite)
8. [Phase 4 — Ablation Studies](#8-phase-4--ablation-studies)
9. [Phase 5 — Paper Writing](#9-phase-5--paper-writing)
10. [Experiment Logging Protocol](#10-experiment-logging-protocol)
11. [Risk Register](#11-risk-register)
12. [Timeline](#12-timeline)
13. [File & Directory Structure](#13-file--directory-structure)

---

## 1. Project Philosophy

### The One Rule

**Validate before you build.**

Before writing a single line of eviction logic, you must confirm that the core hypothesis is empirically true. If key-vector entropy does not correlate with token importance, the entire method collapses. Run Phase 0 first. If the correlation holds, proceed. If it does not, you have already learned something publishable — negative results with good methodology still have value, and you will have found a new direction.

### What This Paper Is

This paper is a **systems-algorithm paper**. It is not a theory paper. It is not a training paper. It is not a benchmark paper. The contribution is:

> A practical eviction strategy that works inside the constraints of modern inference engines, uses only quantities already computed during the forward pass, and achieves competitive compression ratios without attention score extraction.

Every decision — which models to use, which datasets to benchmark on, which baselines to compare against — should be evaluated against this framing. If a choice does not serve the story, cut it.

### What This Paper Is NOT

- It is not a new attention mechanism
- It is not a new training objective
- It is not a speculative decoding paper
- It is not a quantization paper
- It does not require CUDA kernel modifications
- It does not require distributed training

If you find yourself going in any of those directions, stop and return to this section.

### The Core Story in Five Sentences

1. KV cache memory grows linearly with context length, creating a bottleneck in long-context inference.
2. Existing eviction methods rely on attention scores, but modern inference engines use fused attention kernels that do not expose the full attention matrix.
3. We observe that the entropy of key vectors in each cache position is a reliable proxy for token redundancy.
4. We propose EntropyKV, an attention-free eviction strategy computed directly from key vectors with no kernel modification required.
5. EntropyKV achieves X% memory reduction with less than Y perplexity increase, and is compatible with FlashAttention and paged KV layouts out of the box.

This five-sentence structure is your north star. Every experiment you run, every graph you make, every section you write serves exactly one of these five sentences.

---

## 2. Research Hypothesis (Formal)

### Primary Hypothesis

**H1:** The Shannon entropy of a key vector at position *i*, computed across the head dimension, negatively correlates with that token's future attention weight accumulation. That is: low-entropy key vectors correspond to tokens that receive less future attention, and are therefore safer to evict.

### Secondary Hypotheses

**H2:** This correlation is stronger in upper transformer layers than in lower layers. Lower layers encode more syntactic, positional information and have more diffuse, uniform key vectors. Upper layers encode more semantic, task-specific information with more structured, peaked key distributions.

**H3:** The correlation is model-architecture-agnostic. It holds across TinyLlama, Phi-3-mini, and Qwen2-1.5B despite differences in tokenizers, positional encodings, and head dimensions.

**H4:** An eviction strategy based on H1 achieves competitive perplexity preservation compared to attention-based methods (H2O, SnapKV) at equal KV budget ratios, while requiring zero attention matrix materialization.

### What "Validated" Means

H1 is validated if: Pearson or Spearman correlation between key-vector entropy and cumulative attention weight is statistically significant (|r| > 0.3, p < 0.01) across at least two models and two datasets.

H2 is validated if: correlation coefficient is measurably higher in layers 75–100% of depth compared to layers 0–25% of depth, across at least two models.

H3 is validated if: H1 holds on TinyLlama, Phi-3-mini, and Qwen2-1.5B with similar correlation strength.

H4 is validated if: EntropyKV perplexity at 50% KV budget is within 0.5 of H2O and SnapKV at 50% KV budget, with lower or equal latency overhead.

---

## 3. Environment Setup

### Hardware Requirements

This research is designed to run on consumer hardware. The following configurations are sufficient:

- **Minimum:** 8GB VRAM GPU (RTX 3060, RTX 4060) — can run TinyLlama and Qwen2-0.5B at 2K–4K context
- **Recommended:** 16–24GB VRAM GPU (RTX 3090, RTX 4090, A10) — can run Phi-3-mini and Qwen2-1.5B at 4K–8K context
- **Alternative:** Apple M2/M3 Pro with 32GB unified memory — slower but sufficient for all target models
- **CPU fallback:** All Phase 0 correlation analysis can run on CPU with small batch sizes; it is slow but functional

### Software Stack

The entire project runs on a deliberately minimal software stack. The guiding principle is: use the highest-level abstraction that still lets you access what you need.

**Core framework:** HuggingFace `transformers` library. This is the right choice because:
- It exposes the `Cache` abstraction (`DynamicCache`) that can be subclassed and modified
- All target small models are available without custom loading code
- The community has extensive documentation and examples
- It is the standard in which baselines (kvpress, SnapKV implementations) are written

**Experiment logging:** Weights & Biases (wandb). Every single experiment run must be logged. Never run an experiment without logging. More on this in Section 10.

**Numerical analysis:** NumPy and SciPy for correlation analysis in Phase 0. Matplotlib and seaborn for all plots.

**Evaluation:** The `lm-evaluation-harness` library for standardized benchmark evaluation. WikiText-2 perplexity evaluation with the standard stride method.

**Baseline reference implementations:**
- `kvpress` library (wraps several eviction methods including H2O, SnapKV) — this is your primary comparison harness
- StreamingLLM reference implementation from MIT
- Review the H2O paper's official GitHub for their exact evaluation protocol

### Models to Use

All target models are chosen for three reasons: (1) they fit in consumer VRAM, (2) they use GQA or MHA that gives interesting per-head behavior, (3) they are widely used in the compression literature so your results are directly comparable.

**Primary models (must appear in paper):**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — smallest, fastest, good for rapid iteration
- `microsoft/Phi-3-mini-4k-instruct` — strong small model, GQA, widely cited
- `Qwen/Qwen2-1.5B-Instruct` — different architecture, tests generalizability

**Secondary models (include if time/compute allows):**
- `google/gemma-2b-it` — another popular small model
- `Qwen/Qwen2-0.5B-Instruct` — extreme small case, stress test

### What NOT to Use

- Do not attempt to run Llama-2-7B or larger for primary experiments. It is unnecessary for the paper and will consume disproportionate time.
- Do not integrate vLLM or llama.cpp during the research phase. These are deployment targets, not research tools. Add a "compatibility analysis" section in the paper instead of actually integrating.
- Do not use any fine-tuned or instruction-tuned variants for perplexity evaluation — use base models. Instruction-tuned models are appropriate for downstream task evaluation only.

---

## 4. Phase 0 — Hypothesis Validation

**Duration estimate:** 1–2 weeks  
**Goal:** Confirm or deny H1, H2, H3 before any method is built  
**Output:** A correlation analysis notebook with saved figures, and a go/no-go decision

This phase is the most important phase in the entire project. Do not rush it. Do not skip it.

### Step 0.1 — Instrument the Forward Pass

To validate H1, you need two quantities for every token position, every layer, every head:

1. **Key-vector entropy** — the Shannon entropy of the key vector at that position, computed across the head dimension
2. **Cumulative attention weight** — the sum of attention weights that token receives from all future tokens during generation

The challenge is that (2) requires materializing attention weights, which you deliberately avoid in your method. In Phase 0, you will do this anyway — but only in a controlled, measurement-only setting, not in production inference. You will use standard HuggingFace `output_attentions=True` mode to extract attention weights. This is slow and memory-intensive, which is exactly why you want to avoid it at inference time — but for measurement purposes on short sequences it is fine.

Write a measurement harness that:
- Loads a model in standard HuggingFace format
- Runs a forward pass on a fixed input sequence with `output_attentions=True`
- At each layer, for each head, extracts: (a) all key vectors, (b) the full attention weight matrix
- Computes key-vector entropy for each position
- Computes cumulative attention received by each position (sum down each column of the attention matrix)
- Saves both arrays to disk for analysis

### Step 0.2 — Entropy Computation Details

Key-vector entropy must be computed carefully. There are several reasonable choices and you should test all of them:

**Option A — Direct Shannon entropy of the softmax-normalized key vector**
Normalize the key vector values to a probability distribution via softmax, then compute H = -Σ p log p. This treats the head dimension as a categorical distribution.

**Option B — Variance of the key vector**
Compute the variance of key vector values across the head dimension. This is not strictly entropy but captures similar information (high variance = more peaked/structured = lower entropy in the information-theoretic sense).

**Option C — L2 norm of the key vector**
Simple magnitude. Hypothesis: redundant tokens have lower-magnitude key vectors.

**Option D — Entropy of the absolute-value-normalized key vector**
Take absolute values of key vector components, normalize to sum to 1, then compute Shannon entropy. Avoids the sign ambiguity in direct softmax normalization.

Test all four in Phase 0. Report which correlates best with cumulative attention weight. If Option B (variance) correlates as well as Option A (true entropy) but is cheaper to compute, use variance and rename the paper "Variance-Based Attention-Free KV Eviction" — simplicity is a feature, not a weakness.

### Step 0.3 — Correlation Analysis Protocol

For each combination of (model, dataset_sample, layer, head):
1. Extract key-vector entropy scores and cumulative attention weights for all positions
2. Compute Spearman rank correlation (not Pearson — the relationship may be monotonic but nonlinear)
3. Record: correlation coefficient r, p-value, number of positions analyzed
4. Aggregate: mean r per layer, mean r per head, mean r per model

Produce the following plots:
- Scatter plot: key-vector entropy vs cumulative attention weight, colored by layer depth (one plot per model)
- Heatmap: mean correlation coefficient across layers × heads (one plot per model)
- Line plot: mean correlation per layer depth (x-axis: layer index, y-axis: mean r) — this is your H2 test
- Box plot: distribution of r values across all layers, grouped by model — this is your H3 test

### Step 0.4 — Go/No-Go Decision

After Phase 0 analysis, make a formal decision:

**GO conditions:** H1 validated (mean |r| > 0.3 across layers on at least two models)

**CONDITIONAL GO:** H1 weakly validated (mean |r| > 0.15) but H2 strongly validated (upper layers show |r| > 0.4). In this case, scope the method to upper-layer-only eviction and present this as a feature, not a limitation.

**NO-GO:** H1 fails (mean |r| < 0.15 everywhere). In this case: (a) write up the negative result and the analysis as-is, it is still a contribution, OR (b) pivot to a different eviction signal. Candidates: key-vector cosine similarity to neighboring tokens (redundancy-based), key-vector norm (magnitude-based), key-vector change rate (delta between positions).

If you reach a NO-GO, do not abandon the project. The infrastructure you built in Phase 0 is reusable for any alternative signal. Simply swap the scoring function and re-run the correlation analysis.

---

## 5. Phase 1 — Baseline Implementation

**Duration estimate:** 1–2 weeks  
**Goal:** Establish a clean, reproducible evaluation harness with working baseline methods  
**Output:** Perplexity and throughput numbers for full cache, H2O, StreamingLLM, SnapKV on all target models

### Step 1.1 — Evaluation Harness Design

Before implementing anything custom, build the evaluation harness. This is the scaffolding that every method — including yours — will plug into. Designing it first prevents you from having to rewrite evaluation code three times.

The harness takes as inputs:
- Model name (HuggingFace model ID)
- Dataset name (WikiText-2, PG-19, or a custom sample)
- Context length (2048, 4096, or 8192 tokens)
- KV budget ratio (0.1 to 1.0, where 1.0 = full cache = no eviction)
- Eviction method name (full, h2o, streaming, snapkv, entropykv)
- Random seed

The harness outputs and logs to wandb:
- Perplexity (PPL) on the evaluation set using the standard stride evaluation
- Peak VRAM usage in GB
- Average tokens per second during decoding
- Time to first token (TTFT) in milliseconds
- KV cache size at the end of generation in MB
- Actual KV retention percentage (may differ from budget due to rounding)

Design this harness so that adding a new eviction method requires changing only one pluggable component — the scoring function. Everything else (model loading, dataset loading, metric computation, logging) stays fixed. This discipline will save enormous time during ablations.

### Step 1.2 — Full Cache Baseline

Run the harness with no eviction (KV budget = 1.0) on all three primary models at all three context lengths. These numbers are your ground truth.

Record and log:
- PPL on WikiText-2 test set
- PPL on PG-19 test set (first 1000 documents, standard subset)
- Peak VRAM
- Tokens/sec

These numbers should approximately match what is reported in the original model papers and in existing compression papers. If they do not, something is wrong with your evaluation setup. Fix it before proceeding.

### Step 1.3 — H2O Baseline

Implement or integrate H2O using the `kvpress` library if available, or from the reference implementation. H2O evicts tokens with the lowest cumulative attention score, retaining a fixed budget of the highest-scoring tokens plus the most recent tokens.

Run H2O at KV budget ratios: 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2

Log all metrics for each configuration. Produce a plot: PPL vs KV budget ratio for H2O (and later, all methods on the same axes).

### Step 1.4 — StreamingLLM Baseline

StreamingLLM always retains the first K "sink" tokens and the most recent W tokens, discarding everything in between. It is not truly compressive (evicted tokens are gone forever) but it is a widely-cited baseline.

Run at equivalent "effective budget" levels. The budget here is sink_size + window_size. Vary window_size while keeping sink_size = 4 (standard).

### Step 1.5 — SnapKV Baseline

SnapKV selects which KV entries to retain during the prefill phase, based on a one-shot attention observation. It does not evict during decoding.

Note carefully: SnapKV and H2O have different eviction timing. SnapKV is prefill-time. H2O is decode-time (online). Your method will likely be decode-time (online) like H2O, which means the comparison is fairest against H2O. SnapKV comparison is still valuable but note the timing difference explicitly in the paper.

### Step 1.6 — Baseline Summary Table

After running all baselines, produce the master comparison table. Rows = methods. Columns = PPL at budget 100%, 70%, 50%, 30%; tokens/sec at budget 50%; peak VRAM at budget 50%.

This table will appear nearly verbatim in your paper. Getting it right now saves time later.

---

## 6. Phase 2 — EntropyKV Core Method

**Duration estimate:** 2–3 weeks  
**Goal:** Implement, validate, and tune the EntropyKV eviction strategy  
**Output:** Working implementation with performance competitive to or better than H2O on PPL/memory tradeoff

### Step 2.1 — Method Specification

Based on Phase 0 results, select the best entropy proxy (Shannon entropy, variance, L2 norm, or absolute-value entropy). The method specification is then:

**EntropyKV eviction policy:**

At each decode step (or every K steps — see Step 2.4 on eviction frequency), compute an importance score for every position currently in the KV cache. The importance score for position i is the key-vector entropy (or selected proxy) of that position's key vector, averaged across all heads (or a selected subset of heads — see Step 2.3 on head selection). Evict the lowest-scoring positions until the cache size meets the target budget.

Always retain:
- The first S tokens (attention sinks, typically S = 4)
- The most recent R tokens (recency window, typically R = 32 or 64)
- The current token being generated

Everything else is subject to eviction based on entropy score.

### Step 2.2 — Where to Hook Into HuggingFace

The correct integration point is the `DynamicCache` class in HuggingFace `transformers`. Specifically:

The `DynamicCache` class stores key and value states as lists of tensors, one per layer. You will subclass this class and override the update method to run eviction after each decode step (or at a configurable interval). Your subclass has access to all key tensors at the moment of insertion — this is where you compute entropy scores.

You do not need to modify any model code. You do not need to modify any attention code. You only need to control what goes into and comes out of the cache. This is why the approach is "attention-free" — you compute your scoring signal from the key tensors that are already in your hands before they go into the attention computation.

Verify: after subclassing, does the model still produce identical outputs to the original when eviction is disabled (budget = 1.0)? This must be true before proceeding.

### Step 2.3 — Per-Head vs Averaged Scoring

A key design decision: should entropy be computed per-head and averaged, or globally across all heads?

**Option A — Average across all heads:** Simple. For each position, compute entropy of that position's key vector in each head, then average. The average score determines eviction priority.

**Option B — Head-wise eviction:** Each head independently decides which positions to evict. More complex — different heads may disagree about which tokens are important. Allows per-head budget allocation.

**Option C — Weighted average by layer position:** Upper layers' head votes count more than lower layers' head votes, motivated by H2's validation (if confirmed).

Start with Option A. If it underperforms H2O significantly (more than 0.5 PPL gap at equal budget), experiment with Options B and C. Do not start with the complex version.

### Step 2.4 — Eviction Frequency

When exactly does eviction happen?

**Option A — Every decode step:** Most conservative memory usage. Highest overhead (scoring runs every step). Cache never exceeds budget by more than 1 token.

**Option B — Threshold-based:** Run eviction only when cache size exceeds budget by more than X tokens. Reduces overhead, allows small temporary overrun.

**Option C — Periodic:** Run eviction every K decode steps. K is a hyperparameter.

**Option D — Batch eviction at prefill:** Like SnapKV, compute scores once after prefill and pre-select which positions to retain. Simplest, lowest overhead, but not adaptive to what gets generated.

Implement Options A and D first. Option D is the simplest and may already be competitive. Option A is the most thorough. The difference in PPL between A and D, and the difference in overhead, is itself a finding worth reporting.

### Step 2.5 — Layer Handling

Two approaches:

**Approach A — Uniform across layers:** Apply the same KV budget ratio to every layer. Simple. Start here.

**Approach B — Layer-adaptive budget:** If H2 is validated (upper layers have higher correlation), allocate more KV budget to upper layers and less to lower layers. For a 12-layer model with 50% total budget, you might give lower layers 40% budget and upper layers 60% budget. The exact allocation curve is a hyperparameter.

Approach B is your differentiated contribution if the correlation is layer-dependent. However, implementing it before validating H2 is premature. Start with Approach A.

### Step 2.6 — The Position Bias Problem

When you evict tokens from middle positions, the remaining tokens have non-contiguous original positions. With RoPE positional encoding (used in LLaMA, Phi, Qwen), position information is embedded in the key vectors via rotation at query time. Evicting a middle token does not change the positions of remaining tokens' keys — they retain their original position encodings.

This is actually fine for RoPE: the position encoding is applied during attention computation (queries are rotated to match keys), not stored in the key vectors themselves. Eviction simply removes a key-value pair; the remaining pairs retain their original positions, and the model attends to the remaining positions as if the evicted positions never existed.

However, you must verify this does not introduce artifacts. The sanity check: run EntropyKV with budget = 0.99 (evict only 1% of tokens) and confirm PPL is within 0.01 of full cache. If it is not, there is a position encoding problem.

### Step 2.7 — Tuning the Hyperparameters

The method has the following hyperparameters:

- **Entropy metric:** Shannon entropy / variance / L2 norm / abs-entropy (chosen in Phase 0)
- **Sink token count S:** Number of first tokens always retained. Default 4. Test 0, 2, 4, 8.
- **Recency window R:** Number of most recent tokens always retained. Default 32. Test 16, 32, 64, 128.
- **Eviction frequency:** Every step vs periodic vs prefill-only
- **Layer budget allocation:** Uniform vs layer-adaptive

Tune hyperparameters on TinyLlama only, using a validation slice of WikiText-2 (not the test set). Fix these hyperparameters and do not re-tune them on other models. Report the fixed hyperparameter values in the paper. Do not tune on the test set — reviewers will notice if your hyperparameters seem suspiciously optimal.

---

## 7. Phase 3 — Evaluation Suite

**Duration estimate:** 1–2 weeks  
**Goal:** Full evaluation across all models, datasets, budgets, and downstream tasks  
**Output:** All tables and plots needed for the paper

### Step 3.1 — Perplexity Evaluation (Primary)

Run EntropyKV and all baselines on:
- Models: TinyLlama-1.1B, Phi-3-mini, Qwen2-1.5B
- Datasets: WikiText-2, PG-19
- Context lengths: 2048, 4096 tokens (8192 if VRAM allows)
- KV budget ratios: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2

This is a 3 × 2 × 2 × 9 = 108-cell matrix per method. With 4 methods (full, H2O, StreamingLLM, EntropyKV), that is 432 experiment configurations. Most are fast (TinyLlama at 2K context runs in minutes). Budget your time accordingly and prioritize: Phi-3-mini at 4K context is the most important configuration for the paper.

### Step 3.2 — Throughput and Memory Profiling

For each method at each budget ratio on Phi-3-mini at 4K context:
- Measure: peak VRAM (GB), average tokens/sec during decode, TTFT (ms)
- Report the overhead of entropy computation as a percentage of total decode time
- This overhead is a key differentiator: if EntropyKV overhead is <5% of decode time, it is negligible and should be highlighted

This section is critical for the "systems" positioning of the paper. You are not just showing PPL; you are showing that your method has real-world inference cost benefits.

### Step 3.3 — Long-Context Downstream Tasks

Perplexity alone is not sufficient. Reviewers will ask about downstream task performance. Run evaluation on:

**LongBench tasks (select a subset):**
- Single-doc QA (e.g., NarrativeQA or Qasper) — tests whether the model can find answers in long documents
- Multi-doc QA (e.g., HotpotQA) — tests multi-hop reasoning across long context
- Summarization (e.g., GovReport) — tests whether key information is retained

**Needle-in-a-Haystack (NIAH):**
- Place a specific fact at various positions in a long document, ask the model to retrieve it
- Vary: (a) context length 1K–8K, (b) needle position (beginning/middle/end), (c) KV budget ratio
- This directly tests whether your eviction strategy incorrectly evicts the needle
- This produces a visually striking 2D heatmap (context length × needle position, colored by retrieval accuracy) that works extremely well in papers

NIAH is arguably the most important evaluation for a KV eviction paper because it directly measures the failure mode: evicting important tokens. If EntropyKV does well on NIAH relative to H2O and StreamingLLM, that is strong evidence the entropy signal is identifying genuinely unimportant tokens.

### Step 3.4 — Attention-Free Compatibility Demonstration

This section establishes the core systems claim. You need to demonstrate — not just claim — that EntropyKV requires no attention matrix access.

The demonstration is straightforward: show a diagram (in the paper) of the forward pass computation graph, annotating which tensors are accessed by H2O (attention matrix — requires kernel modification or non-fused mode) versus EntropyKV (key tensors only — available before attention, no modification needed).

Additionally: benchmark EntropyKV with FlashAttention-2 enabled versus disabled. Show that performance is identical (since EntropyKV never touches the attention matrix). Show that H2O requires disabling FlashAttention-2 to function. This is a decisive systems result.

If you cannot run FlashAttention-2 on your hardware, note this limitation and make the argument theoretically with reference to the FlashAttention paper's memory access pattern analysis.

---

## 8. Phase 4 — Ablation Studies

**Duration estimate:** 1 week  
**Goal:** Isolate the contribution of each design choice  
**Output:** Ablation table showing what each component adds

### Ablation 1 — Entropy Metric Choice

Run all four entropy proxy options (Shannon entropy, variance, L2 norm, abs-entropy) at KV budget 0.5 on TinyLlama, WikiText-2. Report PPL for each. This justifies your choice of the best metric.

### Ablation 2 — Sink Token Count

Fix everything else. Vary S (sink token count): 0, 2, 4, 8, 16. Show that S = 4 (or whatever you found optimal) is important — removing sinks (S = 0) degrades PPL significantly. This validates the attention sink phenomenon and shows your design decision is principled.

### Ablation 3 — Recency Window Size

Fix everything else. Vary R: 0, 16, 32, 64, 128. Show the tradeoff: too small and the model loses recent context; too large and memory savings diminish.

### Ablation 4 — Eviction Frequency

Compare: every-step eviction vs prefill-only eviction vs periodic (every 10 steps). Report PPL and overhead for each. This is a practically important result — if prefill-only eviction is within 0.2 PPL of every-step eviction, the simpler version is preferable.

### Ablation 5 — Layer-Adaptive vs Uniform Budget (if H2 validated)

If Phase 0 confirmed that upper layers have higher correlation, compare: uniform budget allocation vs pyramidal (more budget to upper layers) vs inverted pyramidal (more budget to lower layers). Show that the layer-adaptive allocation that matches the correlation pattern outperforms uniform allocation.

### Ablation 6 — Entropy vs Random Eviction

This is the most important ablation and must be included. Compare EntropyKV against random eviction at the same budget. If entropy-based scoring does not outperform random eviction, the entropy signal is not providing useful information. Conversely, if it does outperform random significantly, this is strong evidence that the entropy signal is meaningful.

---

## 9. Phase 5 — Paper Writing

**Duration estimate:** 2–3 weeks  
**Goal:** A complete, submission-ready arXiv paper  
**Output:** PDF + source .tex files

### Paper Structure

**Title:**
`EntropyKV: Attention-Free Key-Vector Entropy Eviction for Efficient Long-Context LLM Inference`

or alternative:

`Attention-Free KV Cache Compression via Key-Vector Entropy Estimation`

Choose based on which sounds stronger after seeing your results. If layer-adaptivity is a major contribution, add it: `Layer-Adaptive Attention-Free KV Cache Compression via Key-Vector Entropy`.

**Abstract (target: 200 words)**

Structure: (1) problem statement — KV cache memory bottleneck, (2) limitation of existing work — attention score dependency, (3) your observation — key-vector entropy as proxy, (4) your method — EntropyKV, (5) results — X% memory, Y PPL, Z% speedup vs H2O-compatible baseline.

Write the abstract last, after all results are finalized.

**Section 1 — Introduction (target: 1.5 pages)**

Open with the inference memory problem. State the scale: at 8K context length, KV cache for a 7B model exceeds X GB. For small models on edge devices, this is prohibitive. Existing solutions (H2O, SnapKV) require attention scores. Explain the FlashAttention incompatibility problem — this is your key motivation. State your contributions as a bulleted list. Contributions are:
1. Empirical analysis showing key-vector entropy correlates with token importance
2. EntropyKV, an attention-free eviction strategy
3. Demonstration of FlashAttention compatibility
4. Competitive results on WikiText-2, PG-19, LongBench, NIAH across three small models

**Section 2 — Related Work (target: 1 page)**

Cover: KV cache eviction methods (H2O, StreamingLLM, SnapKV, PyramidKV, ScissorHands), KV cache quantization (KIVI, KVQuant), and efficient attention (FlashAttention, PagedAttention). For each method, state in one sentence what it does and why EntropyKV differs. Do not over-cite. 15–25 references is appropriate for this type of paper.

**Section 3 — Background (target: 0.5 pages)**

Formally define: the KV cache, the eviction problem (given a budget B, select which B tokens to retain), the attention computation graph (showing where attention scores are computed in fused kernels and why they are inaccessible).

Include a diagram: show the standard transformer forward pass, highlight in red where attention scores live (inside the fused kernel), and highlight in green where key vectors are accessible (before attention). One figure, very clean.

**Section 4 — Analysis: Key-Vector Entropy as Importance Proxy (target: 1 page)**

This is your Phase 0 results section. Present the correlation analysis. Show the scatter plot of entropy vs cumulative attention weight. Show the per-layer correlation heatmap. Present H1, H2, H3 as findings with quantitative support. This section is what makes the paper more than just an engineering contribution — it provides empirical grounding for the method.

**Section 5 — Method: EntropyKV (target: 1.5 pages)**

Present the method precisely. Include a pseudocode-style algorithm block (written in algorithmic notation, not Python). Specify: (1) entropy computation formula, (2) always-retain conditions (sinks, recency), (3) eviction decision rule, (4) layer handling, (5) integration point (DynamicCache).

Include a system diagram showing EntropyKV in the inference pipeline: model → key tensor extracted → entropy scored → cache updated → next decode step. Compare against H2O's pipeline (same but requires attention matrix → incompatible with fused attention).

**Section 6 — Experiments (target: 3 pages)**

Subsections:
- 6.1 Setup: models, datasets, baselines, hyperparameters, hardware
- 6.2 Perplexity results (main table + PPL vs budget curve plot)
- 6.3 Memory and throughput results
- 6.4 Long-context evaluation (LongBench table + NIAH heatmap)
- 6.5 FlashAttention compatibility demonstration

**Section 7 — Ablation Studies (target: 1 page)**

Present ablation table. One row per ablation configuration. Columns: PPL@50%, overhead, notes. Discuss what each ablation reveals.

**Section 8 — Discussion (target: 0.5 pages)**

Address limitations honestly:
- Entropy computation adds some overhead (quantify it)
- Not tested on models >3B (acknowledge, note future work)
- Eviction is irreversible during generation (note this is shared by all online eviction methods)
- Performance on very long contexts (>16K) not tested

**Section 9 — Conclusion (target: 0.3 pages)**

Restate: problem, approach, key result. One sentence on future directions (RL-based adaptive budgeting, integration with speculative decoding).

### Figures to Include

The following figures are mandatory. Every one tells part of the story.

1. **Motivation figure** (intro): diagram showing KV cache memory growth vs context length for all three models. Makes the problem visceral.
2. **System compatibility figure** (section 3): side-by-side pipeline — H2O requires attention extraction (red X on fused kernel) vs EntropyKV uses key tensors only (green check).
3. **Correlation analysis figure** (section 4): scatter plot of entropy vs attention weight, per-layer correlation heatmap.
4. **PPL vs KV budget curves** (section 6): one line per method, x-axis = budget ratio, y-axis = PPL. This is the main result figure.
5. **NIAH heatmap** (section 6): 2D grid, x-axis = needle position, y-axis = context length, color = retrieval accuracy. One panel per method.
6. **Memory and throughput comparison** (section 6): bar chart, methods on x-axis, peak VRAM / tokens-per-second on y-axis.
7. **Ablation figure** (section 7): bar chart showing PPL at 50% budget for each ablation variant.

All figures must be publication-quality vector graphics (PDF or SVG output from matplotlib). Set font sizes to match the paper's body text. Use colorblind-safe palettes (matplotlib's `tab10` or seaborn's `colorblind`).

---

## 10. Experiment Logging Protocol

This section is non-negotiable. Losing experimental results is catastrophic. The following protocol must be followed from Day 1.

### wandb Configuration

Every experiment run logs the following fields automatically:
- `run_id`: unique identifier (timestamp + random suffix)
- `model_name`: HuggingFace model ID
- `method`: one of {full, h2o, streaming, snapkv, entropykv, entropykv_ablation_*}
- `dataset`: one of {wikitext2, pg19, longbench, niah}
- `context_length`: integer
- `kv_budget_ratio`: float 0.0–1.0
- `entropy_metric`: one of {shannon, variance, l2norm, abs_entropy}
- `sink_tokens`: integer
- `recency_window`: integer
- `eviction_frequency`: one of {every_step, prefill_only, periodic_K}
- `layer_budget_mode`: one of {uniform, pyramidal, inverse_pyramidal}
- `seed`: integer
- `ppl_wikitext2`: float (if applicable)
- `ppl_pg19`: float (if applicable)
- `peak_vram_gb`: float
- `tokens_per_sec`: float
- `ttft_ms`: float
- `kv_size_final_mb`: float
- `actual_retention_pct`: float
- `hardware`: string (GPU name or "CPU" or "Apple M-series")

### Naming Convention

Experiment runs are named: `{method}_{model_short}_{dataset}_{budget}_{seed}`

Example: `entropykv_tinyllama_wikitext2_050_42`

### Reproducibility

Every run saves:
- The exact config dict as a JSON file
- The random seed used
- The HuggingFace model revision hash
- The library versions (transformers, torch, kvpress)

Never run an experiment without a fixed seed. Use seeds 42, 123, and 7 for all primary results and report mean ± standard deviation.

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| H1 fails — entropy does not correlate with attention weight | Medium | High | Phase 0 is designed to catch this early. Pivot to variance, L2, or cosine similarity signal. |
| EntropyKV PPL is worse than H2O at all budgets | Low-Medium | High | Debug eviction timing, sink token count, recency window. Try prefill-only mode. |
| Entropy computation overhead is too high (>20% of decode time) | Low | Medium | Switch to approximate entropy (variance). Compute only every K steps. |
| Position encoding artifacts after eviction | Low | Medium | Validate with budget=0.99 sanity check. RoPE should be safe. |
| VRAM insufficient for Phi-3-mini at 4K on your GPU | Medium | Medium | Use 4-bit model quantization for the model weights (not the KV cache). Use Qwen2-1.5B as primary model instead. |
| Paper scope creep — too many features added | High | High | Return to Section 1 (Project Philosophy) whenever you feel the urge to add a feature. |
| Baseline numbers do not match published papers | Medium | Medium | Check evaluation protocol (stride, batch size, context handling) carefully. Differences up to 0.3 PPL are acceptable if documented. |
| FlashAttention not available on your hardware | Medium | Low | Make the compatibility argument theoretically. Cite FlashAttention paper's memory access analysis. |

---

## 12. Timeline

| Week | Phase | Key Deliverables |
|---|---|---|
| Week 1 | Phase 0 — setup | Environment working, models loading, forward pass instrumented |
| Week 2 | Phase 0 — analysis | Correlation analysis complete, Phase 0 figures saved, go/no-go decision |
| Week 3 | Phase 1 — baselines | Evaluation harness working, full-cache and H2O baselines complete |
| Week 4 | Phase 1 + Phase 2 start | StreamingLLM + SnapKV baselines done, EntropyKV prototype running |
| Week 5 | Phase 2 — EntropyKV | EntropyKV working, hyperparameter tuning on TinyLlama done |
| Week 6 | Phase 3 — evaluation | PPL evaluation matrix complete across all models and budgets |
| Week 7 | Phase 3 — downstream | LongBench + NIAH evaluation complete |
| Week 8 | Phase 4 — ablations | All ablation experiments complete |
| Week 9 | Phase 5 — writing | Introduction + Related Work + Background drafted |
| Week 10 | Phase 5 — writing | Method + Experiments + Ablations drafted, all figures finalized |
| Week 11 | Phase 5 — writing | Full draft complete, internal review |
| Week 12 | Phase 5 — submission | Revision, proofread, arXiv submission |

**Total estimated timeline: 12 weeks (3 months)**

This is aggressive but realistic for a focused solo researcher. The most common reason timelines slip is scope creep and re-running experiments due to poor logging. Following the logging protocol in Section 10 from Day 1 prevents most timeline slippage.

---

## 13. File & Directory Structure

```
entropykv/
│
├── README.md                          # Project overview, reproduction instructions
│
├── configs/                           # All experiment configurations as JSON
│   ├── baseline_full.json
│   ├── baseline_h2o.json
│   ├── baseline_streaming.json
│   ├── baseline_snapkv.json
│   └── entropykv_default.json
│
├── analysis/                          # Phase 0 — hypothesis validation
│   ├── instrument_forward_pass.py
│   ├── compute_entropy_scores.py
│   ├── correlation_analysis.py
│   └── figures/                       # Saved Phase 0 plots
│
├── src/
│   ├── cache/
│   │   ├── entropy_cache.py           # EntropyKV DynamicCache subclass
│   │   └── utils.py                   # Entropy computation utilities
│   ├── eval/
│   │   ├── harness.py                 # Main evaluation harness
│   │   ├── perplexity.py              # PPL computation with stride
│   │   ├── longbench.py               # LongBench evaluation
│   │   └── niah.py                    # Needle-in-a-haystack
│   └── baselines/
│       ├── h2o.py                     # H2O integration
│       ├── streaming.py               # StreamingLLM integration
│       └── snapkv.py                  # SnapKV integration
│
├── experiments/
│   ├── run_baseline.py                # Run any baseline method
│   ├── run_entropykv.py               # Run EntropyKV
│   ├── run_ablation.py                # Run ablation study
│   └── results/                       # Saved result CSVs (backed up to wandb)
│
├── paper/
│   ├── main.tex                       # Main paper LaTeX source
│   ├── references.bib                 # BibTeX references
│   ├── figures/                       # Publication-quality figures
│   └── neurips_2024.sty               # Template file
│
└── requirements.txt                   # Pinned library versions
```

---

*This document is a living plan. Update it as the project evolves. If a phase produces unexpected results that change the research direction, revise the relevant downstream phases before proceeding. The plan serves you — you do not serve the plan.*
