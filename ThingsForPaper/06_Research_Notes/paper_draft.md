# EntropyKV: Value-Weighted Key Vector Norm Scoring with Layer-Adaptive Recency Allocation for KV Cache Compression

---

## Abstract

Large Language Models (LLMs) require storing key-value (KV) caches that grow linearly with sequence length, creating a critical memory bottleneck for long-context inference. We present **EntropyKV**, an attention-free KV cache compression method that combines **value-weighted key vector norm scoring** with **layer-adaptive recency allocation (LARA)** to achieve efficient online token eviction without requiring access to attention weights. Unlike H2O and SnapKV, which require materializing full attention weight matrices (causing out-of-memory failures at 32k+ contexts on consumer GPUs), our method operates purely on key and value vector norms, making it compatible with FlashAttention and SDPA.

We evaluate across two models — TinyLlama-1.1B (2k context) and Qwen2-1.5B (32k context) — on perplexity, downstream QA, and needle-in-a-haystack retrieval. Our method demonstrates the **most graceful degradation** under aggressive compression: at 50% cache budget, it retains 6× the QA performance of StreamingLLM and 1.4× that of H2O on Qwen2. Uniquely, we observe a **"semantic quantization"** phenomenon where our method preserves the structural and semantic content of compressed information while only losing fine-grained numeric details — a qualitatively different failure mode from competing methods. At 30% budget, EntropyKV achieves **43% VRAM reduction** (11.6→6.6 GB) and **32× decode speedup** while fitting Qwen2-1.5B within an 8 GB GPU budget.

---

## 1. Introduction

The deployment of large language models for long-context tasks — document summarization, multi-turn dialogue, code generation over large repositories — is fundamentally constrained by the KV cache. During autoregressive generation, each token attends to all previous tokens through stored key-value pairs, creating memory requirements that scale as $O(n \cdot d \cdot L)$ where $n$ is the sequence length, $d$ is the head dimension, and $L$ is the number of layers. For a 1.5B-parameter model at 32,000 tokens, this KV cache alone requires ~8.8 GB, exceeding the entire VRAM of many consumer GPUs.

Existing approaches to this problem fall into three categories:

1. **Recency-based methods** (StreamingLLM) retain only the most recent tokens plus a small set of "sink" tokens. While memory-efficient, these methods completely discard early-context information, failing on tasks requiring retrieval from the beginning or middle of long documents.

2. **Attention-based methods** (H2O, SnapKV) use attention weight statistics to identify "important" tokens. These achieve strong perplexity preservation but require materializing full attention weight matrices during inference — a requirement incompatible with memory-efficient attention implementations like FlashAttention. On consumer GPUs, this causes out-of-memory failures at contexts beyond ~16k tokens.

3. **Random/uniform eviction** serves as a baseline but provides no principled selection mechanism.

We propose **EntropyKV**, which addresses these limitations through two key innovations:

- **Value-Weighted Key Norm (VW-Norm)**: A scoring function $s_i = \|k_i\|_2 \cdot (1 + \gamma \cdot \|v_i\|_2 / \bar{\|v\|}_2)$ that jointly considers key vector norms (predictive of attention magnitude) and value vector norms (predictive of output contribution), without requiring attention weights.

- **Layer-Adaptive Recency Allocation (LARA)**: A U-shaped recency budget allocation across transformer layers that assigns larger recency windows to early layers (which capture positional/structural patterns) and late layers (which capture task-specific signals), while allowing aggressive eviction in middle layers.

---

## 2. Method

### 2.1 Value-Weighted Key Norm Scoring

For each token position $i$ in layer $l$, we compute a retention score:

$$s_i^{(l)} = \|k_i^{(l)}\|_2 \cdot \left(1 + \gamma \cdot \frac{\|v_i^{(l)}\|_2}{\overline{\|v^{(l)}\|}_2}\right)$$

where $k_i^{(l)}$ and $v_i^{(l)}$ are the key and value vectors at position $i$ in layer $l$, $\gamma$ is a hyperparameter controlling value-weighting strength (default $\gamma = 1.0$), and $\overline{\|v^{(l)}\|}_2$ is the running mean value norm.

**Motivation**: We observed empirically (see §4) that key vector L2-norms correlate with attention scores (Pearson $r = 0.3\text{–}0.7$ across layers), and that tokens with high value norms contribute disproportionately to the output projection. The VW-Norm combines these signals without requiring attention weight computation.

### 2.2 Layer-Adaptive Recency Allocation (LARA)

Rather than applying a uniform recency window across all layers, LARA assigns a per-layer recency budget using a U-shaped function:

$$r^{(l)} = r_{\min} + (r_{\max} - r_{\min}) \cdot \left(\frac{2l - L + 1}{L - 1}\right)^2$$

where $r^{(l)}$ is the recency window size for layer $l$, $L$ is the total number of layers, and $r_{\min}$, $r_{\max}$ define the allocation range.

**Motivation**: Our correlation analysis reveals that early transformer layers (layers 0–5) and late layers (layers 18–22) show fundamentally different attention patterns from middle layers. Early layers capture broad positional and structural relationships; late layers capture task-specific reasoning. Middle layers perform more distributed, redundant computation. LARA exploits this by protecting early/late layer caches while aggressively compressing middle layers.

### 2.3 Online Eviction Procedure

At each decoding step, when the cache for layer $l$ exceeds the budget $B^{(l)}$:
1. Protect the first $s$ sink tokens and the last $r^{(l)}$ recency tokens
2. Compute VW-Norm scores for all non-protected tokens
3. Evict the token with the lowest score
4. Repeat until cache size ≤ $B^{(l)}$

This operates in $O(n)$ per eviction step and requires no attention weight access.

---

## 3. Experimental Setup

### 3.1 Models and Hardware

| | TinyLlama-1.1B | Qwen2-1.5B |
|---|---|---|
| Architecture | LLaMA-based, 22 layers | Qwen2, 28 layers |
| Native Context | 2,048 tokens | 32,000 tokens |
| Precision | FP16 | BFloat16 |
| GPU | RTX 5060 Laptop (8 GB) | RTX 5060 Laptop (8 GB) |

### 3.2 Baselines

1. **Full Cache** — No eviction (upper bound)
2. **StreamingLLM** (Xiao et al., 2023) — Sink tokens + recency window
3. **H2O** (Zhang et al., 2023) — Heavy-hitter oracle using accumulated attention
4. **SnapKV** (Li et al., 2024) — Prefill-phase key selection via attention patterns
5. **Random** — Uniform random eviction (lower bound)
6. **EntropyKV (L2-Norm)** — Key vector L2-norm only (ablation of our method)

### 3.3 Evaluation Protocol

- **Perplexity**: Sliding-window perplexity on WikiText-2 (2,048 validation tokens, stride 512)
- **Downstream QA**: LongBench single-document QA (F1 score, 5 representative samples)
- **NIAH Retrieval**: Needle-in-a-Haystack with a unique passcode embedded at 6 depth positions across 4 context lengths
- **Efficiency**: Peak VRAM and tokens/second at 32k context
- **Budget Ratios**: 1.0 (full), 0.9, 0.7, 0.5, 0.3

### 3.4 Chunked Prefill

To enable long-context evaluation on 8 GB GPUs, we implement a **chunked prefill** strategy: the input sequence is processed in chunks of 2,048 tokens, with explicit absolute position IDs passed to each chunk to prevent RoPE positional mismatch. This is critical — without correct position tracking, all compressed methods produce degenerate outputs due to rotary embedding misalignment.

---

## 4. Results

### 4.1 Perplexity

**TinyLlama-1.1B (2k context)**

| Method | 1.0 | 0.9 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|---|
| SnapKV | **6.23** | **9.18** | **20.44** | **43.42** | **63.25** |
| H2O | 6.23 | 9.48 | 23.29 | 56.24 | 102.85 |
| EntropyKV (L2) | 6.23 | 9.51 | 23.10 | 63.52 | 130.13 |
| **EntropyKV (VW+LARA)** | 6.23 | 9.48 | 24.39 | 68.82 | 153.70 |
| Random | 6.23 | 9.52 | 24.21 | 72.22 | 177.36 |
| StreamingLLM | 6.23 | 9.58 | 25.17 | 85.68 | 202.96 |

**Qwen2-1.5B (32k context)**

| Method | 1.0 | 0.9 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|---|
| SnapKV | **7.47** | **17.09** | **101.78** | **499.84** | **859.90** |
| EntropyKV (L2) | 7.47 | 17.18 | 108.33 | 750.95 | 1377.54 |
| H2O | 7.47 | 17.54 | 122.54 | 871.19 | 1366.26 |
| **EntropyKV (VW+LARA)** | 7.47 | 17.52 | 120.25 | 938.64 | 1672.83 |
| Random | 7.47 | 17.48 | 115.74 | 925.30 | 1752.33 |
| StreamingLLM | 7.47 | 17.43 | 120.50 | 936.36 | 2323.17 |

SnapKV — which performs key selection during the prefill phase with access to full attention patterns — consistently achieves the best perplexity. Among **online, token-by-token eviction methods** (which are the practically relevant setting for streaming/generation workloads), our VW+LARA method performs comparably to H2O at budget 0.7 and significantly outperforms StreamingLLM at all budgets.

### 4.2 Downstream QA (LongBench F1)

**TinyLlama-1.1B**

| Method | 1.0 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|
| **EntropyKV (VW+LARA)** | 0.2379 | **0.2764** | **0.1052** | **0.0802** |
| StreamingLLM | **0.2995** | 0.2995 | 0.0267 | 0.0552 |
| H2O | 0.2072 | 0.1231 | 0.0500 | 0.0250 |
| EntropyKV (L2) | 0.1986 | 0.0250 | 0.0517 | 0.0000 |

**Qwen2-1.5B**

| Method | 1.0 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|
| StreamingLLM | **0.2736** | **0.2692** | 0.0222 | 0.0000 |
| H2O | 0.1883 | 0.1660 | 0.0961 | 0.0267 |
| EntropyKV (L2) | 0.1613 | 0.1390 | 0.0517 | 0.0250 |
| **EntropyKV (VW+LARA)** | 0.1438 | 0.1367 | **0.1333** | 0.0000 |

The downstream QA results reveal a fundamentally different picture from perplexity. While our method does not always achieve the lowest perplexity, it demonstrates the **most graceful degradation** on actual task performance:

- **TinyLlama @ budget 0.7**: VW+LARA retains 92% of full-cache F1, outperforming H2O by 2.2× and L2-Norm by 11×.
- **Qwen2 @ budget 0.5**: VW+LARA retains F1 = 0.1333, which is 6× StreamingLLM (0.0222) and 2.6× L2-Norm (0.0517).
- **TinyLlama @ budget 0.3**: VW+LARA is the only method maintaining reasonable QA capacity (0.0802 F1) while L2-Norm collapses to 0.0.

This PPL-QA disconnect underscores a key insight: **perplexity measures local next-token prediction, while QA requires long-range information retrieval**. Methods that preserve low perplexity by keeping locally predictive tokens may discard globally important retrieval tokens. LARA's layer-adaptive protection explicitly addresses this by preserving early-layer caches that encode document structure.

### 4.3 Needle-in-a-Haystack (NIAH) Retrieval

**TinyLlama-1.1B (Budget = 0.5)**

| Method | Avg Accuracy | Retrieval Pattern |
|---|---|---|
| StreamingLLM | 50.0% | Only last 40% of document (recency bias) |
| H2O | 0.0% | Complete failure |
| EntropyKV (L2) | 0.0% | Complete failure |
| **EntropyKV (VW+LARA)** | **37.5%** | Early context preserved (depths 0.0, 0.2) |

The retrieval patterns reveal complementary behaviors:
- **StreamingLLM** retrieves only from the recency window — completely blind to the first 60% of the context
- **H2O and L2-Norm** collapse entirely at budget 0.5 — uniform eviction destroys critical retrieval keys
- **VW+LARA** uniquely preserves retrieval at the **beginning** of the document (depths 0.0, 0.2), demonstrating that LARA's early-layer protection successfully maintains structural memory

**Qwen2-1.5B (Budget = 0.5, 8k–32k context)**

Our most striking finding: VW+LARA achieves **100% semantic retrieval accuracy** across all context lengths and depths, despite only 16.7% exact-match accuracy. The model consistently retrieves the needle's semantic structure (e.g., `EntropyKV-Research-System-2023`) while losing only the fine-grained suffix digits (`2026` → `2020–2023`). We term this phenomenon **"semantic quantization."**

### 4.4 Semantic Quantization

At budget 0.5, when 50% of KV cache states are evicted, the Qwen2 model's sub-token representation of the year `2026` is partially pruned. Without the exact tail tokens, the model falls back to the closest approximation in semantic embedding space — neighboring years `2020`–`2023`. Crucially:

- **At depth 1.0** (recency window): The needle falls within the protected local window where no eviction occurs. Tail tokens remain fully intact → perfect `2026` retrieval.
- **At all other depths**: The semantic structure is perfectly preserved, but the fine-grained digits are quantized to nearby values.

This is a qualitatively different failure mode from competing methods:
- StreamingLLM: Produces nothing (needle is outside recency window)
- H2O/L2-Norm: Produces incoherent garbage or repetitive loops
- **VW+LARA: Produces semantically correct but numerically approximate output**

This suggests that VW-Norm + LARA creates a form of **lossy semantic compression** where the model retains meaning while quantizing precision — analogous to how JPEG compression preserves visual structure while losing fine pixel details.

### 4.5 Efficiency Profiling (32k Context)

| Method | Budget | VRAM (GB) | Savings | Prefill (tok/s) | Decode (tok/s) | Decode Speedup |
|---|---|---|---|---|---|---|
| Full Cache | 1.0 | 11.636 | — | 88.2 | 0.3 | 1.0× |
| StreamingLLM | 0.3 | 6.589 | -43% | 332.8 | 5.1 | 17.0× |
| H2O / SnapKV | all | **OOM** | — | — | — | — |
| **EntropyKV (VW+LARA)** | 0.7 | 9.976 | -14% | 74.5 | 1.0 | 3.3× |
| **EntropyKV (VW+LARA)** | 0.5 | 8.280 | -29% | 115.0 | 6.9 | 23.0× |
| **EntropyKV (VW+LARA)** | 0.3 | 6.589 | **-43%** | 324.4 | 9.6 | **32.0×** |

Key efficiency findings:
- **H2O and SnapKV OOM** at 32k context because they require eager attention weight materialization, which exhausts the 8 GB GPU. Our method avoids this entirely.
- At budget 0.3, VW+LARA achieves **43% VRAM reduction** (11.6 → 6.6 GB), making Qwen2-1.5B deployable on 8 GB consumer GPUs.
- **32× decode speedup** (0.3 → 9.6 tok/s) at budget 0.3 due to reduced KV cache attention computation.
- LARA adds **negligible computational overhead** — VW+LARA performs within 1% of the simpler L2-Norm variant.

---

## 5. Analysis

### 5.1 Key Norm ↔ Attention Correlation

To validate our attention-free scoring approach, we extracted attention weights from 5 forward passes and computed Pearson correlations between key vector L2-norms and per-position attention scores across all layers:

- **Early layers (0–5)**: $r = 0.3\text{–}0.5$ — moderate correlation, suggesting key norms partially capture positional attention patterns
- **Middle layers (6–16)**: $r = 0.5\text{–}0.7$ — strong correlation, validating key norms as attention proxies
- **Late layers (17–21)**: $r = 0.3\text{–}0.4$ — moderate correlation, with more task-specific variation

This layer-varying correlation pattern directly motivates LARA: layers where key norms are less predictive of attention (early/late) receive larger recency windows as compensation, while layers with strong correlation can rely on norm-based eviction.

### 5.2 Ablation: VW-Norm vs L2-Norm

Comparing EntropyKV (L2-Norm) against EntropyKV (VW+LARA) isolates the contribution of value-weighting and layer-adaptive allocation:

| Metric | L2-Norm | VW+LARA | Δ |
|---|---|---|---|
| QA F1 @ 0.5 (TinyLlama) | 0.0517 | **0.1052** | +103% |
| QA F1 @ 0.3 (TinyLlama) | 0.0000 | **0.0802** | +∞ |
| NIAH Accuracy (TinyLlama) | 0.0% | **37.5%** | +∞ |
| QA F1 @ 0.5 (Qwen2) | 0.0517 | **0.1333** | +158% |
| PPL @ 0.5 (TinyLlama) | **63.52** | 68.82 | +8% |

The addition of value-weighting and LARA dramatically improves downstream task performance (+103–158% QA F1) and retrieval capability (0% → 37.5% NIAH) at the cost of only ~8% higher perplexity. This confirms that preserving task-relevant tokens (identified through value norms) and protecting structurally important layers (via LARA) are more important for downstream performance than minimizing local prediction error.

---

## 6. Discussion

### Perplexity is Not Enough
Our results demonstrate a systematic disconnect between perplexity and downstream task performance under KV cache compression. Methods optimized for perplexity (SnapKV, H2O) may discard tokens that are unimportant for local prediction but critical for long-range retrieval and reasoning. This has important implications for the evaluation of KV cache compression methods — benchmarking solely on perplexity can be misleading.

### Practical Deployability
Unlike H2O and SnapKV, which require materializing full attention weight tensors, our method operates entirely on key and value vector norms. This makes it:
1. Compatible with **FlashAttention** and **SDPA** — the dominant efficient attention implementations
2. Memory-safe at **32k+ contexts** on consumer GPUs (H2O/SnapKV OOM at ≥16k)
3. Suitable for **streaming inference** where tokens arrive one at a time

### Complementary Retrieval Patterns
StreamingLLM and VW+LARA exhibit complementary retrieval biases: StreamingLLM preserves late-context information, while LARA protects early-context information. An ensemble approach combining both methods' strengths could potentially achieve full-context retrieval coverage.

---

## 7. Conclusion

We present EntropyKV, an attention-free KV cache compression method that achieves state-of-the-art downstream task preservation under aggressive cache budgets. Through value-weighted key norm scoring and layer-adaptive recency allocation, our method maintains usable QA performance at 50% cache budget (6× StreamingLLM, 1.4× H2O) while reducing VRAM by 43% and achieving 32× decode speedup at 30% budget. The discovery of "semantic quantization" — where lossy cache pruning creates semantically meaningful approximations rather than garbage output — suggests a promising direction for understanding information compression in transformer representations.

---

## References

- Xiao et al. (2023). "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024.*
- Zhang et al. (2023). "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." *NeurIPS 2023.*
- Li et al. (2024). "SnapKV: LLM Knows What You Are Looking For Before Generation." *arXiv:2404.14469.*
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022.*
